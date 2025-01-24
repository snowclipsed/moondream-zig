const std = @import("std");
const atomic = std.atomic;
const Tensor = @import("tensor.zig").Tensor;
const ops = @import("ops.zig");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const testing = std.testing;

comptime {
    @setFloatMode(.optimized);
}

// tuning parameters
pub const Tile: usize = 168; // Tile size for matrix blocking
pub const Vec: usize = 8; // Vector size for SIMD operations

const CACHE_LINE_SIZE: usize = atomic.cache_line;
const CHUNK_SIZE: usize = 1;
const AVX2_ALIGNMENT = 32;
const MICRO_KERNEL_SIZE: usize = Vec; // Match micro-kernel to vector size

const Vec8f = @Vector(8, f32);

const ThreadLocalData = struct {
    current_index: atomic.Value(usize) align(CACHE_LINE_SIZE),
    _padding: [CACHE_LINE_SIZE - @sizeOf(atomic.Value(usize))]u8 = undefined,
};

const ThreadContext = struct {
    a: []const f32,
    b: []const f32,
    c: []f32,
    M: usize,
    N: usize,
    K: usize,
    tiles_M: usize,
    tiles_N: usize,
    total_tiles: usize,
    shared_counter: *ThreadLocalData,
};

pub fn matmul(comptime T: type, a: Tensor(T), b: Tensor(T), allocator: Allocator) !Tensor(T) {
    if (a.shape.len != 2 or b.shape.len != 2) {
        return error.InvalidDimensions;
    }
    if (a.shape[1] != b.shape[0]) {
        return error.ShapeMismatch;
    }

    // Use optimized implementation for f32
    if (T == f32) {
        return optimizedMatmulF32(a, b, allocator);
    }

    // Simple implementation for other types
    const M = a.shape[0];
    const N = b.shape[1];
    const K = a.shape[1];

    const result_shape = [_]usize{ M, N };
    var result = try Tensor(T).init(allocator, &result_shape);
    errdefer result.deinit();

    // Initialize result to zero
    @memset(std.mem.sliceAsBytes(result.data), 0);

    // Simple triple-loop matrix multiplication
    for (0..M) |i| {
        for (0..N) |j| {
            var sum: T = 0;
            for (0..K) |k| {
                sum += a.data[i * K + k] * b.data[k * N + j];
            }
            result.data[i * N + j] = sum;
        }
    }

    return result;
}

fn optimizedMatmulF32(a: Tensor(f32), b: Tensor(f32), allocator: Allocator) !Tensor(f32) {
    const M = a.shape[0];
    const N = b.shape[1];
    const K = a.shape[1];

    const result_shape = [_]usize{ M, N };
    var result = try Tensor(f32).init(allocator, &result_shape);
    errdefer result.deinit();

    // Initialize result to zero
    @memset(result.data, 0);

    // Calculate tile grid dimensions
    const tiles_M = (M + Tile - 1) / Tile;
    const tiles_N = (N + Tile - 1) / Tile;
    const total_tiles = tiles_M * tiles_N;

    // Initialize shared atomic counter
    var shared_data = ThreadLocalData{ .current_index = atomic.Value(usize).init(0) };

    // Get number of CPU cores
    const num_threads = try std.Thread.getCpuCount();

    // Create thread pool
    var thread_pool = try std.ArrayList(std.Thread).initCapacity(allocator, num_threads);
    defer thread_pool.deinit();

    // Create thread context
    const context = ThreadContext{
        .a = a.data,
        .b = b.data,
        .c = result.data,
        .M = M,
        .N = N,
        .K = K,
        .tiles_M = tiles_M,
        .tiles_N = tiles_N,
        .total_tiles = total_tiles,
        .shared_counter = &shared_data,
    };

    // Spawn worker threads
    const WorkerFn = struct {
        fn worker(ctx: ThreadContext) void {
            workerThread(ctx);
        }
    };

    for (0..num_threads) |_| {
        try thread_pool.append(try std.Thread.spawn(.{}, WorkerFn.worker, .{context}));
    }

    // Wait for all threads to complete
    for (thread_pool.items) |thread| {
        thread.join();
    }

    return result;
}

fn workerThread(ctx: ThreadContext) void {
    var local_C: [Tile][Tile]f32 align(AVX2_ALIGNMENT) = undefined;

    while (true) {
        const start_idx = ctx.shared_counter.current_index.fetchAdd(CHUNK_SIZE, .seq_cst);
        if (start_idx >= ctx.total_tiles) break;

        const end_idx = @min(start_idx + CHUNK_SIZE, ctx.total_tiles);
        var idx = start_idx;

        while (idx < end_idx) : (idx += 1) {
            const i = idx / ctx.tiles_N;
            const j = idx % ctx.tiles_N;

            const i_start = i * Tile;
            const j_start = j * Tile;
            const i_end = @min(i_start + Tile, ctx.M);
            const j_end = @min(j_start + Tile, ctx.N);

            @memset(std.mem.sliceAsBytes(&local_C), 0);

            var k: usize = 0;
            while (k < ctx.K) : (k += Tile) {
                const k_end = @min(k + Tile, ctx.K);
                microKernelAVX2(ctx, &local_C, i_start, j_start, k, i_end, j_end, k_end);
            }

            for (i_start..i_end) |ii| {
                const row_offset = ii * ctx.N;
                const local_row = ii - i_start;

                var j_idx = j_start;
                while (j_idx + Vec <= j_end) : (j_idx += Vec) {
                    const vec_idx = j_idx - j_start;

                    const c_vec = Vec8f{
                        local_C[local_row][vec_idx],
                        local_C[local_row][vec_idx + 1],
                        local_C[local_row][vec_idx + 2],
                        local_C[local_row][vec_idx + 3],
                        local_C[local_row][vec_idx + 4],
                        local_C[local_row][vec_idx + 5],
                        local_C[local_row][vec_idx + 6],
                        local_C[local_row][vec_idx + 7],
                    };

                    const dest_vec = Vec8f{
                        ctx.c[row_offset + j_idx],
                        ctx.c[row_offset + j_idx + 1],
                        ctx.c[row_offset + j_idx + 2],
                        ctx.c[row_offset + j_idx + 3],
                        ctx.c[row_offset + j_idx + 4],
                        ctx.c[row_offset + j_idx + 5],
                        ctx.c[row_offset + j_idx + 6],
                        ctx.c[row_offset + j_idx + 7],
                    };

                    const result = dest_vec + c_vec;
                    for (0..8) |offset| {
                        ctx.c[row_offset + j_idx + offset] = result[offset];
                    }
                }

                while (j_idx < j_end) : (j_idx += 1) {
                    ctx.c[row_offset + j_idx] += local_C[local_row][j_idx - j_start];
                }
            }
        }
    }
}

fn microKernelAVX2(
    ctx: ThreadContext,
    local_C: *[Tile][Tile]f32,
    i_start: usize,
    j_start: usize,
    k_start: usize,
    i_end: usize,
    j_end: usize,
    k_end: usize,
) void {
    var A_local: [Tile][Tile]f32 align(32) = undefined;
    var B_local: [Tile][Tile]f32 align(32) = undefined;

    const k_size = k_end - k_start;
    const i_size = i_end - i_start;
    const j_size = j_end - j_start;

    for (0..i_size) |i| {
        const src_idx = (i_start + i) * ctx.K + k_start;
        for (0..k_size) |k| {
            A_local[i][k] = ctx.a[src_idx + k];
        }
    }

    for (0..k_size) |k| {
        const src_idx = (k_start + k) * ctx.N + j_start;
        for (0..j_size) |j| {
            B_local[k][j] = ctx.b[src_idx + j];
        }
    }

    var i: usize = 0;
    while (i + MICRO_KERNEL_SIZE <= i_size) : (i += MICRO_KERNEL_SIZE) {
        var j: usize = 0;
        while (j + MICRO_KERNEL_SIZE <= j_size) : (j += MICRO_KERNEL_SIZE) {
            var acc: [8][8]f32 align(32) = [_][8]f32{[_]f32{0} ** 8} ** 8;

            var k: usize = 0;
            while (k < k_size) : (k += 1) {
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
// test "optimized matmul benchmark" {
//     var arena = std.heap.ArenaAllocator.init(testing.allocator);
//     defer arena.deinit();
//     const allocator = arena.allocator();

//     const sizes = [_]struct { m: usize, n: usize, k: usize }{
//         .{ .m = 256, .n = 256, .k = 256 },
//         .{ .m = 512, .n = 512, .k = 512 },
//         .{ .m = 1024, .n = 1024, .k = 1024 },
//         .{ .m = 1024, .n = 2048, .k = 1024 },
//         .{ .m = 2048, .n = 2048, .k = 2048 },
//     };
//     const iterations = 5;

//     std.debug.print("\nT = {d} \nV = {d}\n", .{ T, V });
//     std.debug.print("Number of threads = {d}\n", .{try std.Thread.getCpuCount()});

//     for (sizes) |size| {
//         const shape_a = [_]usize{ size.m, size.k };
//         const shape_b = [_]usize{ size.k, size.n };

//         var a = try Tensor(f32).init(allocator, &shape_a);
//         defer a.deinit();
//         var b = try Tensor(f32).init(allocator, &shape_b);
//         defer b.deinit();

//         // Initialize with random data
//         var prng = std.rand.DefaultPrng.init(0);
//         var random = prng.random();
//         for (a.data) |*val| val.* = random.float(f32);
//         for (b.data) |*val| val.* = random.float(f32);

//         // Warmup run
//         {
//             var warmup = try matmul(f32, a, b, allocator);
//             defer warmup.deinit();
//         }

//         var gflops_array = try allocator.alloc(f64, iterations);
//         defer allocator.free(gflops_array);

//         for (0..iterations) |i| {
//             var timer = try std.time.Timer.start();
//             var result = try matmul(f32, a, b, allocator);
//             defer result.deinit();
//             const elapsed_ns = timer.read();

//             const opers = 2.0 * @as(f64, @floatFromInt(size.m * size.n * size.k));
//             const seconds = @as(f64, @floatFromInt(elapsed_ns)) / 1e9;
//             gflops_array[i] = opers / seconds / 1e9;
//         }

//         // Calculate average GFLOPS
//         var total_gflops: f64 = 0;
//         for (gflops_array) |gflops| {
//             total_gflops += gflops;
//         }
//         const avg_gflops = total_gflops / @as(f64, @floatFromInt(iterations));

//         std.debug.print("Matrix size: {d}x{d}x{d}, GFLOPS: {d:.2}\n", .{ size.m, size.n, size.k, avg_gflops });
//     }
// }

test "matmul basic functionality" {
    const allocator = testing.allocator;

    // Test case 1: Square matrices
    {
        var a = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        defer a.deinit();
        var b = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        defer b.deinit();

        a.data[0] = 1.0;
        a.data[1] = 2.0;
        a.data[2] = 3.0;
        a.data[3] = 4.0;

        b.data[0] = 5.0;
        b.data[1] = 6.0;
        b.data[2] = 7.0;
        b.data[3] = 8.0;

        var result = try matmul(f32, a, b, allocator);
        defer result.deinit();

        // Compare with known result
        try testing.expectApproxEqAbs(result.data[0], 19.0, 1e-6);
        try testing.expectApproxEqAbs(result.data[1], 22.0, 1e-6);
        try testing.expectApproxEqAbs(result.data[2], 43.0, 1e-6);
        try testing.expectApproxEqAbs(result.data[3], 50.0, 1e-6);
    }
}

test "matmul edge cases" {
    const allocator = testing.allocator;

    // Test case 1: 1x1 matrices
    {
        var a = try Tensor(f32).init(allocator, &[_]usize{ 1, 1 });
        defer a.deinit();
        var b = try Tensor(f32).init(allocator, &[_]usize{ 1, 1 });
        defer b.deinit();

        a.data[0] = 3.0;
        b.data[0] = 4.0;

        var result = try matmul(f32, a, b, allocator);
        defer result.deinit();

        try testing.expectApproxEqAbs(result.data[0], 12.0, 1e-6);
    }

    // Test case 2: Tall matrix × Wide matrix
    {
        var a = try Tensor(f32).init(allocator, &[_]usize{ 3, 1 });
        defer a.deinit();
        var b = try Tensor(f32).init(allocator, &[_]usize{ 1, 3 });
        defer b.deinit();

        a.data[0] = 1.0;
        a.data[1] = 2.0;
        a.data[2] = 3.0;

        b.data[0] = 4.0;
        b.data[1] = 5.0;
        b.data[2] = 6.0;

        var result = try matmul(f32, a, b, allocator);
        defer result.deinit();

        try testing.expectEqual(result.shape[0], @as(usize, 3));
        try testing.expectEqual(result.shape[1], @as(usize, 3));
    }

    // Test case 3: Zero matrices
    {
        var a = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        defer a.deinit();
        var b = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        defer b.deinit();

        @memset(a.data, 0);
        @memset(b.data, 0);

        var result = try matmul(f32, a, b, allocator);
        defer result.deinit();

        for (result.data) |val| {
            try testing.expectEqual(val, 0);
        }
    }
}

test "matmul error cases" {
    const allocator = testing.allocator;

    // Test case 1: Mismatched dimensions
    {
        var a = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
        defer a.deinit();
        var b = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        defer b.deinit();

        try testing.expectError(error.ShapeMismatch, matmul(f32, a, b, allocator));
    }

    // Test case 2: Invalid dimensions
    {
        var a = try Tensor(f32).init(allocator, &[_]usize{2});
        defer a.deinit();
        var b = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        defer b.deinit();

        try testing.expectError(error.InvalidDimensions, matmul(f32, a, b, allocator));
    }
}

test "matmul correctness against reference" {
    const allocator = testing.allocator;
    const test_sizes = [_][3]usize{
        .{ 32, 32, 32 }, // Small square
        .{ 47, 35, 23 }, // Odd sizes
        .{ 128, 64, 96 }, // Rectangular
        .{ 1, 64, 128 }, // Single row × wide
        .{ 128, 1, 64 }, // Tall × single column
        .{ Tile - 1, Tile + 1, Tile }, // Around tile size
        .{ Tile, Tile, Tile }, // Exactly tile size
        .{ Tile + 1, Tile - 1, Tile }, // Around tile size
    };

    for (test_sizes) |size| {
        const M = size[0];
        const N = size[1];
        const K = size[2];

        // Create random input tensors
        var a = try ops.createRandomTensor(f32, allocator, &[_]usize{ M, K }, 42);
        defer a.deinit();
        var b = try ops.createRandomTensor(f32, allocator, &[_]usize{ K, N }, 43);
        defer b.deinit();

        // Compute using tiled matmul
        var result = try matmul(f32, a, b, allocator);
        defer result.deinit();

        // Compute using reference matmul
        var expected = try ops.matmul(f32, &a, b);
        defer expected.deinit();

        // Compare results
        const eps: f32 = 1e-4; // Allow for some floating-point error
        for (result.data, expected.data) |val, exp| {
            try testing.expectApproxEqAbs(val, exp, eps);
        }

        std.debug.print("Test passed for size: M={}, N={}, K={}\n", .{ M, N, K });
    }
}

test "matmul numerical stability" {
    const allocator = testing.allocator;

    // Test case 1: Moderately large numbers
    {
        var a = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        defer a.deinit();
        var b = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        defer b.deinit();

        // Using smaller values to avoid overflow
        const large: f32 = 1e3;
        @memset(a.data, large);
        @memset(b.data, large);

        var result = try matmul(f32, a, b, allocator);
        defer result.deinit();

        // Check results
        for (result.data) |val| {
            // Verify no infinity and reasonable magnitude
            try testing.expect(!std.math.isInf(val));
            try testing.expect(!std.math.isNan(val));
            // For 2x2 matrices filled with 1e3, each element should be 2 * (1e3 * 1e3) = 2e6
            try testing.expectApproxEqAbs(val, 2e6, 1e-6);
        }
    }

    // Test case 2: Small but non-zero numbers
    {
        var a = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        defer a.deinit();
        var b = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        defer b.deinit();

        const small: f32 = 1e-3;
        @memset(a.data, small);
        @memset(b.data, small);

        var result = try matmul(f32, a, b, allocator);
        defer result.deinit();

        // Check results
        for (result.data) |val| {
            try testing.expect(!std.math.isNan(val));
            try testing.expect(val > 0); // Should be positive
            // For 2x2 matrices filled with 1e-3, each element should be 2 * (1e-3 * 1e-3) = 2e-6
            try testing.expectApproxEqAbs(val, 2e-6, 1e-9);
        }
    }

    // Test case 3: Mixed magnitudes
    {
        var a = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        defer a.deinit();
        var b = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        defer b.deinit();

        // First row large, second row small
        a.data[0] = 1e3;
        a.data[1] = 1e3;
        a.data[2] = 1e-3;
        a.data[3] = 1e-3;

        b.data[0] = 1e-3;
        b.data[1] = 1e3;
        b.data[2] = 1e-3;
        b.data[3] = 1e3;

        var result = try matmul(f32, a, b, allocator);
        defer result.deinit();

        // Check results
        for (result.data) |val| {
            try testing.expect(!std.math.isInf(val));
            try testing.expect(!std.math.isNan(val));
        }
    }
}
pub fn calculateGflops(allocator: std.mem.Allocator, M: usize, N: usize, K: usize, iterations: usize) !f64 {
    const shape_a = [_]usize{ M, K };
    const shape_b = [_]usize{ K, N };

    var a = try Tensor(f32).init(allocator, &shape_a);
    defer a.deinit();
    var b = try Tensor(f32).init(allocator, &shape_b);
    defer b.deinit();

    // Initialize with random data
    var prng = std.rand.DefaultPrng.init(0);
    var random = prng.random();
    for (a.data) |*val| val.* = random.float(f32);
    for (b.data) |*val| val.* = random.float(f32);

    // Warmup run
    {
        var warmup = try matmul(f32, a, b, allocator);
        defer warmup.deinit();
    }

    var gflops_array = try allocator.alloc(f64, iterations);
    defer allocator.free(gflops_array);

    for (0..iterations) |i| {
        var timer = try std.time.Timer.start();
        var result = try matmul(f32, a, b, allocator);
        defer result.deinit();
        const elapsed_ns = timer.read();

        const opers = 2 * M * N * K; // multiply-add is 2 operations
        const seconds = @as(f64, @floatFromInt(elapsed_ns)) / 1e9;
        gflops_array[i] = @as(f64, @floatFromInt(opers)) / seconds / 1e9;
    }

    // Calculate average GFLOPS
    var total_gflops: f64 = 0;
    for (gflops_array) |gflops| {
        total_gflops += gflops;
    }
    return total_gflops / @as(f64, @floatFromInt(iterations));
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Define test sizes
    const sizes = [_]struct { m: usize, n: usize, k: usize }{
        .{ .m = 256, .n = 256, .k = 256 },
        .{ .m = 512, .n = 512, .k = 512 },
        .{ .m = 1024, .n = 1024, .k = 1024 },
        .{ .m = 1024, .n = 2048, .k = 1024 },
        .{ .m = 2048, .n = 2048, .k = 2048 },
        .{ .m = 2048, .n = 4096, .k = 2048 },
        .{ .m = 4096, .n = 4096, .k = 4096 },
        .{ .m = 8192, .n = 2048, .k = 8192 },
        .{ .m = 1152, .n = 4304, .k = 1152 },
    };

    const iterations = 5;

    try std.io.getStdOut().writer().print("\nRunning MatMul Benchmark\n", .{});
    try std.io.getStdOut().writer().print("T = {d} \nV = {d} \n", .{ Tile, Vec });
    try std.io.getStdOut().writer().print("Number of threads = {d}\n", .{try std.Thread.getCpuCount()});

    for (sizes) |size| {
        const avg_gflops = try calculateGflops(allocator, size.m, size.n, size.k, iterations);
        try std.io.getStdOut().writer().print("Matrix size: {d}x{d}x{d}, GFLOPS: {d:.2}\n", .{ size.m, size.n, size.k, avg_gflops });
    }
}
