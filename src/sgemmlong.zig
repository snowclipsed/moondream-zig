const std = @import("std");
const atomic = std.atomic;
const Tensor = @import("tensor.zig").Tensor;
const Allocator = std.mem.Allocator;

// Optimized parameters for skinny matrices
pub const VERTICAL_TILE: usize = 64;
pub const HORIZONTAL_TILE: usize = 8;
pub const Vec: usize = 8;
pub const CACHE_LINE_SIZE: usize = atomic.cache_line;

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

pub fn skinnyMatmul(comptime T: type, a: Tensor(T), b: Tensor(T), allocator: Allocator) !Tensor(T) {
    if (a.shape.len != 2 or b.shape.len != 2) {
        return error.InvalidDimensions;
    }
    if (a.shape[1] != b.shape[0]) {
        return error.ShapeMismatch;
    }

    // Specialized implementation for f32
    if (T == f32) {
        return optimizedSkinnyMatmulF32(a, b, allocator);
    }

    // Fallback implementation for other types
    const M = a.shape[0];
    const N = b.shape[1];
    const K = a.shape[1];

    const result_shape = [_]usize{ M, N };
    var result = try Tensor(T).init(allocator, &result_shape);
    errdefer result.deinit();

    @memset(result.data, 0);

    // Basic implementation for non-f32 types
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

fn optimizedSkinnyMatmulF32(a: Tensor(f32), b: Tensor(f32), allocator: Allocator) !Tensor(f32) {
    @setRuntimeSafety(false);

    const M = a.shape[0];
    const N = b.shape[1];
    const K = a.shape[1];

    const result_shape = [_]usize{ M, N };
    var result = try Tensor(f32).init(allocator, &result_shape);
    errdefer result.deinit();

    @memset(result.data, 0);

    // Calculate tile dimensions
    const tiles_M = (M + VERTICAL_TILE - 1) / VERTICAL_TILE;
    const tiles_N = (N + HORIZONTAL_TILE - 1) / HORIZONTAL_TILE;
    const total_tiles = tiles_M * tiles_N;

    var shared_data = ThreadLocalData{ .current_index = atomic.Value(usize).init(0) };

    const num_threads = try std.Thread.getCpuCount();
    var thread_pool = try std.ArrayList(std.Thread).initCapacity(allocator, num_threads);
    defer thread_pool.deinit();

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
    for (0..num_threads) |_| {
        try thread_pool.append(try std.Thread.spawn(.{}, workerThread, .{context}));
    }

    // Wait for completion
    for (thread_pool.items) |thread| {
        thread.join();
    }

    return result;
}

fn workerThread(ctx: ThreadContext) void {
    @setRuntimeSafety(false);

    var local_buffer: [VERTICAL_TILE * HORIZONTAL_TILE]f32 align(32) = undefined;

    while (true) {
        const tile_idx = ctx.shared_counter.current_index.fetchAdd(1, .seq_cst);
        if (tile_idx >= ctx.total_tiles) break;

        const tile_i = tile_idx / ctx.tiles_N;
        const tile_j = tile_idx % ctx.tiles_N;

        const i_start = tile_i * VERTICAL_TILE;
        const j_start = tile_j * HORIZONTAL_TILE;
        const i_end = @min(i_start + VERTICAL_TILE, ctx.M);
        const j_end = @min(j_start + HORIZONTAL_TILE, ctx.N);

        // Zero the local buffer
        @memset(&local_buffer, 0);

        // Process the tile
        for (i_start..i_end) |i| {
            const local_i = i - i_start;
            for (j_start..j_end) |j| {
                const local_j = j - j_start;
                var sum: f32 = 0;

                // Process K dimension in vector chunks
                var k: usize = 0;
                while (k + Vec <= ctx.K) : (k += Vec) {
                    const a_vec = Vec8f{
                        ctx.a[i * ctx.K + k],
                        ctx.a[i * ctx.K + k + 1],
                        ctx.a[i * ctx.K + k + 2],
                        ctx.a[i * ctx.K + k + 3],
                        ctx.a[i * ctx.K + k + 4],
                        ctx.a[i * ctx.K + k + 5],
                        ctx.a[i * ctx.K + k + 6],
                        ctx.a[i * ctx.K + k + 7],
                    };

                    const b_vec = Vec8f{
                        ctx.b[k * ctx.N + j],
                        ctx.b[(k + 1) * ctx.N + j],
                        ctx.b[(k + 2) * ctx.N + j],
                        ctx.b[(k + 3) * ctx.N + j],
                        ctx.b[(k + 4) * ctx.N + j],
                        ctx.b[(k + 5) * ctx.N + j],
                        ctx.b[(k + 6) * ctx.N + j],
                        ctx.b[(k + 7) * ctx.N + j],
                    };

                    const prod = a_vec * b_vec;
                    sum += @reduce(.Add, prod);
                }

                // Handle remaining elements
                while (k < ctx.K) : (k += 1) {
                    sum += ctx.a[i * ctx.K + k] * ctx.b[k * ctx.N + j];
                }

                local_buffer[local_i * HORIZONTAL_TILE + local_j] = sum;
            }
        }

        // Write results back to global memory
        for (i_start..i_end) |i| {
            const local_i = i - i_start;
            for (j_start..j_end) |j| {
                const local_j = j - j_start;
                const idx = local_i * HORIZONTAL_TILE + local_j;
                const val = local_buffer[idx];
                _ = @atomicRmw(f32, &ctx.c[i * ctx.N + j], .Add, val, .seq_cst);
            }
        }
    }
}

const testing = std.testing;
const math = std.math;

// Helper function to create a tensor with sequential values
fn createSequentialTensor(allocator: Allocator, shape: []const usize) !Tensor(f32) {
    var tensor = try Tensor(f32).init(allocator, shape);
    errdefer tensor.deinit();

    for (tensor.data, 0..) |*val, i| {
        val.* = @floatFromInt(i);
    }
    return tensor;
}

// Helper function to create a tensor filled with a specific value
fn createFilledTensor(allocator: Allocator, shape: []const usize, value: f32) !Tensor(f32) {
    var tensor = try Tensor(f32).init(allocator, shape);
    errdefer tensor.deinit();

    for (tensor.data) |*val| {
        val.* = value;
    }
    return tensor;
}

// Helper function to verify matrix multiplication results
fn verifyMatmul(a: Tensor(f32), b: Tensor(f32), c: Tensor(f32)) !void {
    const M = a.shape[0];
    const N = b.shape[1];
    const K = a.shape[1];

    for (0..M) |i| {
        for (0..N) |j| {
            var expected: f32 = 0;
            for (0..K) |k| {
                expected += a.data[i * K + k] * b.data[k * N + j];
            }
            const actual = c.data[i * N + j];
            // Use relative tolerance for large numbers
            const tolerance = if (@abs(expected) > 1e4) @abs(expected) * 1e-4 else 1e-2;
            try testing.expectApproxEqAbs(expected, actual, tolerance);
        }
    }
}

test "skinnyMatmul - tall matrix (M >> N)" {
    const allocator = testing.allocator;

    // Create a tall 1000x10 matrix A and a 10x10 matrix B
    const a_shape = [_]usize{ 1000, 10 };
    const b_shape = [_]usize{ 10, 10 };

    var a = try createSequentialTensor(allocator, &a_shape);
    defer a.deinit();

    var b = try createSequentialTensor(allocator, &b_shape);
    defer b.deinit();

    var result = try skinnyMatmul(f32, a, b, allocator);
    defer result.deinit();

    // Verify dimensions
    try testing.expectEqual(@as(usize, 1000), result.shape[0]);
    try testing.expectEqual(@as(usize, 10), result.shape[1]);

    // Verify results
    try verifyMatmul(a, b, result);
}

test "skinnyMatmul - wide matrix (N >> M)" {
    const allocator = testing.allocator;

    // Create a 10x100 matrix A and a 100x1000 matrix B
    const a_shape = [_]usize{ 10, 100 };
    const b_shape = [_]usize{ 100, 1000 };

    var a = try createSequentialTensor(allocator, &a_shape);
    defer a.deinit();

    var b = try createSequentialTensor(allocator, &b_shape);
    defer b.deinit();

    var result = try skinnyMatmul(f32, a, b, allocator);
    defer result.deinit();

    // Verify dimensions
    try testing.expectEqual(@as(usize, 10), result.shape[0]);
    try testing.expectEqual(@as(usize, 1000), result.shape[1]);

    // Verify results
    try verifyMatmul(a, b, result);
}

test "skinnyMatmul - edge cases" {
    const allocator = testing.allocator;

    // Test 1x1 matrices
    {
        const shape = [_]usize{ 1, 1 };
        var a = try createFilledTensor(allocator, &shape, 2);
        defer a.deinit();
        var b = try createFilledTensor(allocator, &shape, 3);
        defer b.deinit();

        var result = try skinnyMatmul(f32, a, b, allocator);
        defer result.deinit();

        try testing.expectEqual(@as(f32, 6), result.data[0]);
    }

    // Test matrices with dimension size equal to VERTICAL_TILE
    {
        const a_shape = [_]usize{ 32, 8 };
        const b_shape = [_]usize{ 8, 32 };

        var a = try createSequentialTensor(allocator, &a_shape);
        defer a.deinit();
        var b = try createSequentialTensor(allocator, &b_shape);
        defer b.deinit();

        var result = try skinnyMatmul(f32, a, b, allocator);
        defer result.deinit();

        try verifyMatmul(a, b, result);
    }
}

test "skinnyMatmul - invalid inputs" {
    const allocator = testing.allocator;

    // Test invalid dimensions
    {
        const a_shape = [_]usize{ 10, 20, 30 }; // 3D tensor
        const b_shape = [_]usize{ 20, 30 };

        var a = try Tensor(f32).init(allocator, &a_shape);
        defer a.deinit();
        var b = try Tensor(f32).init(allocator, &b_shape);
        defer b.deinit();

        try testing.expectError(error.InvalidDimensions, skinnyMatmul(f32, a, b, allocator));
    }

    // Test mismatched dimensions
    {
        const a_shape = [_]usize{ 10, 20 };
        const b_shape = [_]usize{ 30, 40 }; // Should be {20, 40}

        var a = try Tensor(f32).init(allocator, &a_shape);
        defer a.deinit();
        var b = try Tensor(f32).init(allocator, &b_shape);
        defer b.deinit();

        try testing.expectError(error.ShapeMismatch, skinnyMatmul(f32, a, b, allocator));
    }
}

test "skinnyMatmul - non-f32 types" {
    const allocator = testing.allocator;

    // Test with integers
    {
        const a_shape = [_]usize{ 5, 3 };
        const b_shape = [_]usize{ 3, 4 };

        var a = try Tensor(i32).init(allocator, &a_shape);
        defer a.deinit();
        var b = try Tensor(i32).init(allocator, &b_shape);
        defer b.deinit();

        // Fill with test data
        for (a.data, 0..) |*val, i| {
            val.* = @intCast(i);
        }
        for (b.data, 0..) |*val, i| {
            val.* = @intCast(i);
        }

        var result = try skinnyMatmul(i32, a, b, allocator);
        defer result.deinit();

        // Verify dimensions
        try testing.expectEqual(@as(usize, 5), result.shape[0]);
        try testing.expectEqual(@as(usize, 4), result.shape[1]);
    }
}

test "skinnyMatmul - stress test" {
    const allocator = testing.allocator;

    // Test various matrix size combinations
    const sizes = [_]usize{ 8, 16, 32, 64, 128 };

    for (sizes) |m| {
        for (sizes) |k| {
            for (sizes) |n| {
                const a_shape = [_]usize{ m, k };
                const b_shape = [_]usize{ k, n };

                var a = try createSequentialTensor(allocator, &a_shape);
                defer a.deinit();
                var b = try createSequentialTensor(allocator, &b_shape);
                defer b.deinit();

                var result = try skinnyMatmul(f32, a, b, allocator);
                defer result.deinit();

                try verifyMatmul(a, b, result);
            }
        }
    }
}

fn calculateGflops(allocator: std.mem.Allocator, M: usize, N: usize, K: usize, iterations: usize) !struct { avg: f64, min: f64, max: f64, std_dev: f64 } {
    const shape_a = [_]usize{ M, K };
    const shape_b = [_]usize{ K, N };

    var a = try Tensor(f32).init(allocator, &shape_a);
    defer a.deinit();
    var b = try Tensor(f32).init(allocator, &shape_b);
    defer b.deinit();

    // Initialize with random data
    var prng = std.rand.DefaultPrng.init(0);
    var random = prng.random();
    for (a.data) |*val| val.* = random.float(f32) * 2 - 1; // Range [-1, 1]
    for (b.data) |*val| val.* = random.float(f32) * 2 - 1;

    // Warmup runs
    for (0..2) |_| {
        var warmup = try skinnyMatmul(f32, a, b, allocator);
        defer warmup.deinit();
    }

    var gflops_array = try allocator.alloc(f64, iterations);
    defer allocator.free(gflops_array);

    // Main benchmark loop
    for (0..iterations) |i| {
        var timer = try std.time.Timer.start();
        var result = try skinnyMatmul(f32, a, b, allocator);
        defer result.deinit();
        const elapsed_ns = timer.read();

        const opers = 2 * M * N * K; // multiply-add is 2 operations
        const seconds = @as(f64, @floatFromInt(elapsed_ns)) / 1e9;
        gflops_array[i] = @as(f64, @floatFromInt(opers)) / seconds / 1e9;
    }

    // Calculate statistics
    var total: f64 = 0;
    var min: f64 = gflops_array[0];
    var max: f64 = gflops_array[0];

    for (gflops_array) |gflops| {
        total += gflops;
        min = @min(min, gflops);
        max = @max(max, gflops);
    }

    const avg = total / @as(f64, @floatFromInt(iterations));

    // Calculate standard deviation
    var sum_squared_diff: f64 = 0;
    for (gflops_array) |gflops| {
        const diff = gflops - avg;
        sum_squared_diff += diff * diff;
    }
    const std_dev = @sqrt(sum_squared_diff / @as(f64, @floatFromInt(iterations)));

    return .{ .avg = avg, .min = min, .max = max, .std_dev = std_dev };
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Define test sizes focusing on skinny matrices
    const sizes = [_]struct { m: usize, n: usize, k: usize, desc: []const u8 }{
        .{ .m = 800, .n = 6142, .k = 2048, .desc = "Actual scenario" },
        .{ .m = 1, .n = 6142, .k = 2048, .desc = "Actual scenario 2" },
        // Square matrices for baseline
        .{ .m = 1024, .n = 1024, .k = 1024, .desc = "Square (baseline)" },
        .{ .m = 2048, .n = 2048, .k = 2048, .desc = "Square (large)" },

        // Tall skinny matrices
        .{ .m = 4096, .n = 256, .k = 256, .desc = "Tall skinny" },
        .{ .m = 8192, .n = 128, .k = 128, .desc = "Very tall skinny" },
        .{ .m = 16384, .n = 64, .k = 64, .desc = "Extremely tall skinny" },

        // Wide skinny matrices
        .{ .m = 256, .n = 4096, .k = 256, .desc = "Wide skinny" },

        .{ .m = 64, .n = 16384, .k = 64, .desc = "Extremely wide skinny" },

        // Mixed dimensions
        .{ .m = 2048, .n = 64, .k = 2048, .desc = "Tall with large K" },
        .{ .m = 64, .n = 2048, .k = 2048, .desc = "Wide with large K" },

        // Edge cases
        .{ .m = VERTICAL_TILE, .n = HORIZONTAL_TILE, .k = 1024, .desc = "Exact tile size" },
        .{ .m = VERTICAL_TILE - 1, .n = HORIZONTAL_TILE - 1, .k = 1024, .desc = "Just under tile size" },
        .{ .m = VERTICAL_TILE + 1, .n = HORIZONTAL_TILE + 1, .k = 1024, .desc = "Just over tile size" },
    };

    const iterations = 5;
    var stdout = std.io.getStdOut().writer();

    try stdout.print("\nSkinny MatMul Benchmark\n", .{});
    try stdout.print("Configuration:\n", .{});
    try stdout.print("  VERTICAL_TILE: {d}\n", .{VERTICAL_TILE});
    try stdout.print("  HORIZONTAL_TILE: {d}\n", .{HORIZONTAL_TILE});
    try stdout.print("  Vector size: {d}\n", .{Vec});
    try stdout.print("  Number of threads: {d}\n", .{try std.Thread.getCpuCount()});
    try stdout.print("  Iterations per test: {d}\n\n", .{iterations});

    // Print header
    try stdout.print("{s:<20} {s:<15} {s:<15} {s:<15} {s:<12} {s:<12} {s:<12} {s:<12}\n", .{ "Description", "Size", "Memory (MB)", "GFLOPS/iter", "Min", "Max", "StdDev", "GB/s" });
    try stdout.print("{s}\n", .{"-" ** 110});

    for (sizes) |size| {
        const memory_mb = @as(f64, @floatFromInt(size.m * size.n + size.n * size.k + size.m * size.k)) * 4.0 / (1024 * 1024);
        const result = try calculateGflops(allocator, size.m, size.n, size.k, iterations);

        // Calculate memory bandwidth (GB/s)
        const elements_accessed = size.m * size.k + size.k * size.n + size.m * size.n;
        const bytes_accessed = elements_accessed * @sizeOf(f32);
        const gb_per_s = @as(f64, @floatFromInt(bytes_accessed)) * @as(f64, @floatFromInt(iterations)) / 1e9;

        try stdout.print("{s:<20} {d}x{d}x{d:<7} {d:>8.1} {d:>14.1} {d:>11.1} {d:>11.1} {d:>11.2} {d:>11.1}\n", .{ size.desc, size.m, size.n, size.k, memory_mb, result.avg, result.min, result.max, result.std_dev, gb_per_s });
    }
}
