const std = @import("std");
const testing = std.testing;
const atomic = std.atomic;
const math = std.math;
const Allocator = std.mem.Allocator;
const Tensor = @import("tensor.zig").Tensor;
const Slice = @import("tensor.zig").Slice;
const time = std.time;

const T: usize = 160;
const V: usize = 8;
const CACHE_LINE_SIZE: usize = atomic.cache_line;
const CHUNK_SIZE: usize = 1;
const AVX2_ALIGNMENT = 32;
const MICRO_KERNEL_SIZE: usize = std.simd.suggestVectorLength(f32) orelse 8;
const Vec8f = @Vector(8, f32);

const ThreadLocalData = struct {
    current_index: atomic.Value(usize) align(CACHE_LINE_SIZE),
    _padding: [CACHE_LINE_SIZE - @sizeOf(atomic.Value(usize))]u8 = undefined,
};

const mode = std.builtin.FloatMode.optimized;
comptime {
    @setFloatMode(mode);
}

const ThreadContext = struct {
    A: []const f16,
    B: []const f16,
    C: []f32,
    M: usize,
    N: usize,
    K: usize,
    tiles_M: usize,
    tiles_N: usize,
    total_tiles: usize,
    shared_counter: *ThreadLocalData,
};

fn workerThread(ctx: ThreadContext) void {
    var local_C: [T][T]f32 align(AVX2_ALIGNMENT) = undefined;

    while (true) {
        const start_idx = ctx.shared_counter.current_index.fetchAdd(CHUNK_SIZE, .seq_cst);
        if (start_idx >= ctx.total_tiles) break;

        const end_idx = @min(start_idx + CHUNK_SIZE, ctx.total_tiles);
        var idx = start_idx;

        while (idx < end_idx) : (idx += 1) {
            const i = idx / ctx.tiles_N;
            const j = idx % ctx.tiles_N;

            const i_start = i * T;
            const j_start = j * T;
            const i_end = @min(i_start + T, ctx.M);
            const j_end = @min(j_start + T, ctx.N);

            for (0..T) |x| {
                @memset(&local_C[x], 0);
            }

            var k: usize = 0;
            while (k < ctx.K) : (k += T) {
                const k_end = @min(k + T, ctx.K);
                computeTile(ctx, &local_C, i_start, j_start, k, i_end, j_end, k_end);
            }

            for (i_start..i_end) |ii| {
                const row_offset = ii * ctx.N;
                const local_row = ii - i_start;
                for (j_start..j_end) |jj| {
                    const local_col = jj - j_start;
                    ctx.C[row_offset + jj] += local_C[local_row][local_col];
                }
            }
        }
    }
}

fn computeTile(
    ctx: ThreadContext,
    local_C: *[T][T]f32,
    i_start: usize,
    j_start: usize,
    k_start: usize,
    i_end: usize,
    j_end: usize,
    k_end: usize,
) void {
    var A_local: [T][T]f32 align(32) = undefined;
    var B_local: [T][T]f32 align(32) = undefined;

    const k_size = k_end - k_start;
    const i_size = i_end - i_start;
    const j_size = j_end - j_start;

    for (0..i_size) |i| {
        const row_offset = (i_start + i) * ctx.K;
        for (0..k_size) |k| {
            const a_idx = row_offset + k_start + k;
            A_local[i][k] = @floatCast(ctx.A[a_idx]);
        }
    }

    for (0..k_size) |k| {
        const row_offset = (k_start + k) * ctx.N;
        for (0..j_size) |j| {
            const b_idx = row_offset + j_start + j;
            B_local[k][j] = @floatCast(ctx.B[b_idx]);
        }
    }

    var i: usize = 0;
    while (i + 8 <= i_size) : (i += 8) {
        var j: usize = 0;
        while (j + 8 <= j_size) : (j += 8) {
            var acc: [8][8]f32 align(32) = [_][8]f32{[_]f32{0} ** 8} ** 8;

            for (0..k_size) |k| {
                const a_vec = Vec8f{
                    A_local[i][k],     A_local[i + 1][k],
                    A_local[i + 2][k], A_local[i + 3][k],
                    A_local[i + 4][k], A_local[i + 5][k],
                    A_local[i + 6][k], A_local[i + 7][k],
                };

                const b_vec = Vec8f{
                    B_local[k][j],     B_local[k][j + 1],
                    B_local[k][j + 2], B_local[k][j + 3],
                    B_local[k][j + 4], B_local[k][j + 5],
                    B_local[k][j + 6], B_local[k][j + 7],
                };

                inline for (0..8) |bi| {
                    const a_broadcast: Vec8f = @splat(a_vec[bi]);
                    const c_vec = Vec8f{
                        acc[bi][0], acc[bi][1], acc[bi][2], acc[bi][3],
                        acc[bi][4], acc[bi][5], acc[bi][6], acc[bi][7],
                    };
                    const prod = @mulAdd(Vec8f, a_broadcast, b_vec, c_vec);
                    inline for (0..8) |bj| {
                        acc[bi][bj] = prod[bj];
                    }
                }
            }

            for (0..8) |bi| {
                for (0..8) |bj| {
                    local_C[i + bi][j + bj] += acc[bi][bj];
                }
            }
        }

        while (j < j_size) : (j += 1) {
            for (0..8) |bi| {
                var sum: f32 = 0;
                for (0..k_size) |k| {
                    sum = @mulAdd(f32, A_local[i + bi][k], B_local[k][j], sum);
                }
                local_C[i + bi][j] += sum;
            }
        }
    }

    while (i < i_size) : (i += 1) {
        for (0..j_size) |j| {
            var sum: f32 = 0;
            for (0..k_size) |k| {
                sum = @mulAdd(f32, A_local[i][k], B_local[k][j], sum);
            }
            local_C[i][j] += sum;
        }
    }
}

pub fn matmul(
    allocator: Allocator,
    A: Tensor(f16),
    B: Tensor(f16),
    C: Tensor(f32),
) !void {
    const A_shape = A.shape;
    const B_shape = B.shape;
    const C_shape = C.shape;

    if (A_shape.len != 2 or B_shape.len != 2 or C_shape.len != 2) {
        std.log.err("Incompatible Tensor Shapes, A shape : {any}, B shape {any}, C shape {any}", .{ A_shape, B_shape, C_shape });
        return error.InvalidTensorDimension;
    }

    const M = A_shape[0];
    const K = A_shape[1];
    const N = B_shape[1];

    if (B_shape[0] != K or C_shape[0] != M or C_shape[1] != N) {
        std.log.err("Incompatible shapes, A shape : {any}, B shape {any}, C shape {any}", .{ A_shape, B_shape, C_shape });
        return error.IncompatibleTensorShapes;
    }

    const A_data = A.getSlice();
    const B_data = B.getSlice();
    const C_data = C.getSlice();

    const tiles_M = (M + T - 1) / T;
    const tiles_N = (N + T - 1) / T;
    const total_tiles = tiles_M * tiles_N;

    var shared_data = ThreadLocalData{ .current_index = atomic.Value(usize).init(0) };

    const num_threads = try std.Thread.getCpuCount();
    var thread_pool = try std.ArrayList(std.Thread).initCapacity(allocator, num_threads);
    defer thread_pool.deinit();

    const context = ThreadContext{
        .A = A_data,
        .B = B_data,
        .C = C_data,
        .M = M,
        .N = N,
        .K = K,
        .tiles_M = tiles_M,
        .tiles_N = tiles_N,
        .total_tiles = total_tiles,
        .shared_counter = &shared_data,
    };

    for (0..num_threads) |_| {
        try thread_pool.append(try std.Thread.spawn(.{}, workerThread, .{context}));
    }

    for (thread_pool.items) |thread| thread.join();
}
//// benchmarking code ////

fn benchmarkMatMul(allocator: std.mem.Allocator, M: usize, N: usize, K: usize, num_runs: usize) !f64 {
    // Create input tensors
    var A = try Tensor(f16).init(allocator, &[_]usize{ M, K });
    defer A.deinit();
    var B = try Tensor(f16).init(allocator, &[_]usize{ K, N });
    defer B.deinit();
    var C = try Tensor(f32).init(allocator, &[_]usize{ M, N });
    defer C.deinit();

    // Initialize with some values (random pattern)
    var rng = std.rand.DefaultPrng.init(42);
    const random = rng.random();
    for (A.data) |*val| {
        val.* = @floatCast(random.float(f32) * 2.0 - 1.0);
    }
    for (B.data) |*val| {
        val.* = @floatCast(random.float(f32) * 2.0 - 1.0);
    }

    // Warmup run
    try matmul(allocator, A, B, C);

    // Timing runs
    var total_time: u64 = 0;
    var timer = try time.Timer.start();

    for (0..num_runs) |_| {
        @memset(C.data, 0);
        timer.reset();
        try matmul(allocator, A, B, C);
        total_time += timer.read();
    }

    const avg_nanos = @as(f64, @floatFromInt(total_time)) / @as(f64, @floatFromInt(num_runs));
    const avg_secs = avg_nanos / 1e9;

    // Calculate GFLOPS
    // For matrix multiplication, num_operations = 2 * M * N * K
    const ops = 2 * M * N * K;
    const gflops = (@as(f64, @floatFromInt(ops)) / avg_secs) / 1e9;

    return gflops;
}

fn printBenchmarkResults(label: []const u8, gflops: f64, M: usize, N: usize, K: usize) void {
    std.debug.print("{s:<20} Size: [{d:>4}x{d:>4}x{d:>4}] Performance: {d:>6.2} GFLOPS\n", .{ label, M, N, K, gflops });
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const num_runs = 5; // Number of timing runs for each size

    // Test different matrix sizes
    const sizes = [_][3]usize{
        .{ 512, 512, 512 }, // Square
        .{ 1024, 1024, 1024 }, // Larger square
        .{ 768, 512, 256 }, // Rectangular
        .{ 2048, 1024, 512 }, // Large rectangular
        .{ 160, 160, 160 }, // Tile size
        .{ 2048, 2048, 2048 }, // Large square
    };

    std.debug.print("\nMatrix Multiplication Performance Benchmark\n", .{});
    std.debug.print("=========================================\n\n", .{});

    for (sizes) |size| {
        const M = size[0];
        const N = size[1];
        const K = size[2];

        const gflops = try benchmarkMatMul(allocator, M, N, K, num_runs);

        var label_buf: [32]u8 = undefined;
        const label = try std.fmt.bufPrint(&label_buf, "Case {d}x{d}x{d}", .{ M, N, K });
        printBenchmarkResults(label, gflops, M, N, K);
    }

    // Additional benchmarks for tile size multiples
    std.debug.print("\nTile Size Multiple Tests:\n", .{});
    std.debug.print("======================\n\n", .{});

    const tile_multiples = [_]usize{ 1, 2, 3, 4 };

    for (tile_multiples) |multiple| {
        const size = T * multiple;
        const gflops = try benchmarkMatMul(allocator, size, size, size, num_runs);

        var label_buf: [32]u8 = undefined;
        const label = try std.fmt.bufPrint(&label_buf, "{}xTile Size", .{multiple});
        printBenchmarkResults(label, gflops, size, size, size);
    }
}

/////
/////
////.
///..

// Helper function to compare results with naive implementation
fn naiveMatMul(A: *const Tensor(f16), B: *const Tensor(f16), C: *Tensor(f32)) !void {
    const M = A.shape[0];
    const K = A.shape[1];
    const N = B.shape[1];

    if (C.shape[0] != M or C.shape[1] != N) {
        return error.IncompatibleTensorShapes;
    }

    @memset(C.data, 0);

    for (0..M) |i| {
        for (0..N) |j| {
            var sum: f32 = 0;
            for (0..K) |k| {
                const a_val: f32 = @floatCast(A.data[i * K + k]);
                const b_val: f32 = @floatCast(B.data[k * N + j]);
                sum = @mulAdd(f32, a_val, b_val, sum);
            }
            C.data[i * N + j] = sum;
        }
    }
}

// Helper to fill tensor with test pattern
fn fillTestPattern(tensor: anytype, pattern: enum { Identity, Sequential, Random }) void {
    const shape = tensor.shape;
    const rows = shape[0];
    const cols = shape[1];

    switch (pattern) {
        .Identity => {
            for (0..rows) |i| {
                for (0..cols) |j| {
                    tensor.data[i * cols + j] = if (i == j) 1.0 else 0.0;
                }
            }
        },
        .Sequential => {
            for (tensor.data, 0..) |*val, i| {
                val.* = @floatCast(@as(f32, @floatFromInt(i)) * 0.01);
            }
        },
        .Random => {
            var rng = std.rand.DefaultPrng.init(42);
            const random = rng.random();
            for (tensor.data) |*val| {
                val.* = @floatCast(random.float(f32) * 2.0 - 1.0);
            }
        },
    }
}

// Helper to compare tensors with epsilon
fn compareResults(expected: *const Tensor(f32), actual: *const Tensor(f32), epsilon: f32) !void {
    try testing.expectEqual(expected.shape.len, actual.shape.len);
    for (expected.shape, actual.shape) |s1, s2| {
        try testing.expectEqual(s1, s2);
    }

    var max_diff: f32 = 0;
    for (expected.data, actual.data) |e, a| {
        const diff = @abs(e - a);
        max_diff = @max(max_diff, diff);
        try testing.expect(diff <= epsilon);
    }
}

test "MatMul - Basic functionality" {
    const allocator = testing.allocator;

    // Small matrix test
    {
        const shape_a = [_]usize{ 4, 3 };
        const shape_b = [_]usize{ 3, 4 };

        var A = try Tensor(f16).init(allocator, &shape_a);
        defer A.deinit();
        var B = try Tensor(f16).init(allocator, &shape_b);
        defer B.deinit();

        fillTestPattern(&A, .Sequential);
        fillTestPattern(&B, .Sequential);

        var C = try Tensor(f32).init(allocator, &[_]usize{ shape_a[0], shape_b[1] });
        defer C.deinit();

        var expected = try Tensor(f32).init(allocator, &[_]usize{ shape_a[0], shape_b[1] });
        defer expected.deinit();

        // Compute with both implementations
        try matmul(allocator, A, B, C);
        try naiveMatMul(&A, &B, &expected);

        // Compare results
        try compareResults(&expected, &C, 0.001);
    }
}

test "MatMul - Large matrices" {
    const allocator = testing.allocator;

    // Test with matrices larger than tile size
    {
        const shape_a = [_]usize{ 2000, 180 };
        const shape_b = [_]usize{ 180, 1600 };

        var A = try Tensor(f16).init(allocator, &shape_a);
        defer A.deinit();
        var B = try Tensor(f16).init(allocator, &shape_b);
        defer B.deinit();

        fillTestPattern(&A, .Random);
        fillTestPattern(&B, .Random);

        var C = try Tensor(f32).init(allocator, &[_]usize{ shape_a[0], shape_b[1] });
        defer C.deinit();

        var expected = try Tensor(f32).init(allocator, &[_]usize{ shape_a[0], shape_b[1] });
        defer expected.deinit();

        try matmul(allocator, A, B, C);
        try naiveMatMul(&A, &B, &expected);

        try compareResults(&expected, &C, 0.01);
    }
}

test "MatMul - Identity matrix" {
    const allocator = testing.allocator;

    // Test multiplication with identity matrix
    {
        const size = 32;
        const shape = [_]usize{ size, size };

        var A = try Tensor(f16).init(allocator, &shape);
        defer A.deinit();
        var I = try Tensor(f16).init(allocator, &shape);
        defer I.deinit();

        fillTestPattern(&A, .Random);
        fillTestPattern(&I, .Identity);

        var C = try Tensor(f32).init(allocator, &shape);
        defer C.deinit();

        // A * I should equal A
        try matmul(allocator, A, I, C);

        // Convert A to f32 for comparison
        var A_f32 = try Tensor(f32).init(allocator, &shape);
        defer A_f32.deinit();
        for (A.data, A_f32.data) |a, *c| {
            c.* = @floatCast(a);
        }

        try compareResults(&A_f32, &C, 0.001);
    }
}

test "MatMul - Non-square matrices" {
    const allocator = testing.allocator;

    // Test various non-square matrix combinations
    const test_shapes = [_][2][2]usize{
        .{ .{ 50, 30 }, .{ 30, 70 } },
        .{ .{ 25, 80 }, .{ 80, 35 } },
        .{ .{ 100, 20 }, .{ 20, 100 } },
    };

    for (test_shapes) |shapes| {
        const shape_a = shapes[0];
        const shape_b = shapes[1];

        var A = try Tensor(f16).init(allocator, &shape_a);
        defer A.deinit();
        var B = try Tensor(f16).init(allocator, &shape_b);
        defer B.deinit();

        fillTestPattern(&A, .Random);
        fillTestPattern(&B, .Random);

        var C = try Tensor(f32).init(allocator, &[_]usize{ shape_a[0], shape_b[1] });
        defer C.deinit();

        var expected = try Tensor(f32).init(allocator, &[_]usize{ shape_a[0], shape_b[1] });
        defer expected.deinit();

        try matmul(allocator, A, B, C);
        try naiveMatMul(&A, &B, &expected);

        try compareResults(&expected, &C, 0.01);
    }
}

test "MatMul - Error cases" {
    const allocator = testing.allocator;

    // Test incompatible shapes
    {
        var A = try Tensor(f16).init(allocator, &[_]usize{ 3, 4 });
        defer A.deinit();
        var B = try Tensor(f16).init(allocator, &[_]usize{ 5, 6 }); // Wrong inner dimension
        defer B.deinit();
        var C = try Tensor(f32).init(allocator, &[_]usize{ 3, 6 });
        defer C.deinit();

        try testing.expectError(error.IncompatibleTensorShapes, matmul(allocator, A, B, C));
    }

    // Test wrong output shape
    {
        var A = try Tensor(f16).init(allocator, &[_]usize{ 3, 4 });
        defer A.deinit();
        var B = try Tensor(f16).init(allocator, &[_]usize{ 4, 5 });
        defer B.deinit();
        var C = try Tensor(f32).init(allocator, &[_]usize{ 3, 6 }); // Wrong output size
        defer C.deinit();

        try testing.expectError(error.IncompatibleTensorShapes, matmul(allocator, A, B, C));
    }
}

test "MatMul - Edge cases" {
    const allocator = testing.allocator;

    // Test matrices with dimensions near tile boundaries
    const test_sizes = [_]usize{ 158, 159, 160, 161, 162 };

    for (test_sizes) |size| {
        var A = try Tensor(f16).init(allocator, &[_]usize{ size, size });
        defer A.deinit();
        var B = try Tensor(f16).init(allocator, &[_]usize{ size, size });
        defer B.deinit();

        fillTestPattern(&A, .Random);
        fillTestPattern(&B, .Random);

        var C = try Tensor(f32).init(allocator, &[_]usize{ size, size });
        defer C.deinit();

        var expected = try Tensor(f32).init(allocator, &[_]usize{ size, size });
        defer expected.deinit();

        try matmul(allocator, A, B, C);
        try naiveMatMul(&A, &B, &expected);

        try compareResults(&expected, &C, 0.01);
    }
}

test "MatMul - Numerical stability" {
    const allocator = testing.allocator;

    // Test with very small and very large numbers
    {
        const shape = [_]usize{ 32, 32 };
        var A = try Tensor(f16).init(allocator, &shape);
        defer A.deinit();
        var B = try Tensor(f16).init(allocator, &shape);
        defer B.deinit();

        // Fill with very small numbers
        fillTestPattern(&A, .Random);
        fillTestPattern(&B, .Random);

        for (A.data) |*val| val.* = 1e-4;
        for (B.data) |*val| val.* = 1e-4;

        var C = try Tensor(f32).init(allocator, &shape);
        defer C.deinit();

        var expected = try Tensor(f32).init(allocator, &shape);
        defer expected.deinit();

        try matmul(allocator, A, B, C);
        try naiveMatMul(&A, &B, &expected);

        try compareResults(&expected, &C, 0.001);

        // Fill with larger numbers (but within f16 range)
        for (A.data) |*val| val.* = 10.0;
        for (B.data) |*val| val.* = 10.0;

        @memset(C.data, 0);
        @memset(expected.data, 0);

        try matmul(allocator, A, B, C);
        try naiveMatMul(&A, &B, &expected);

        try compareResults(&expected, &C, 0.1);
    }
}
