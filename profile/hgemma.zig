const std = @import("std");
const testing = std.testing;
const atomic = std.atomic;
const math = std.math;
const Allocator = std.mem.Allocator;
const Tensor = @import("tensor.zig").Tensor;
const Slice = @import("tensor.zig").Slice;
const broadcast_add_simd = @import("ops.zig").broadcast_add_simd;
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

            // Initialize local_C with bias if provided
            if (ctx.bias) |bias| {
                if (ctx.is_vector_bias) {
                    // Vector bias: broadcast across rows
                    for (0..T) |x| {
                        for (0..T) |y| {
                            if (i_start + x < i_end and j_start + y < j_end) {
                                const bias_idx = j_start + y;
                                local_C[x][y] = @floatCast(bias[bias_idx]);
                            } else {
                                local_C[x][y] = 0;
                            }
                        }
                    }
                } else {
                    // Matrix bias
                    for (0..T) |x| {
                        for (0..T) |y| {
                            if (i_start + x < i_end and j_start + y < j_end) {
                                const bias_idx = (i_start + x) * ctx.N + (j_start + y);
                                local_C[x][y] = @floatCast(bias[bias_idx]);
                            } else {
                                local_C[x][y] = 0;
                            }
                        }
                    }
                }
            } else {
                for (0..T) |x| {
                    @memset(&local_C[x], 0);
                }
            }

            var k: usize = 0;
            while (k < ctx.K) : (k += T) {
                const k_end = @min(k + T, ctx.K);
                computeTile(ctx, &local_C, i_start, j_start, k, i_end, j_end, k_end);
            }

            // Write back results
            for (i_start..i_end) |ii| {
                const row_offset = ii * ctx.N;
                const local_row = ii - i_start;
                for (j_start..j_end) |jj| {
                    const local_col = jj - j_start;
                    ctx.C[row_offset + jj] = @floatCast(local_C[local_row][local_col]);
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

    // Load and convert A to f32
    for (0..i_size) |i| {
        const row_offset = (i_start + i) * ctx.K;
        for (0..k_size) |k| {
            const a_idx = row_offset + k_start + k;
            A_local[i][k] = @floatCast(ctx.A[a_idx]);
        }
    }

    // Load and convert B to f32
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

pub fn matmul(a: Tensor(f16), b: Tensor(f16), bias: ?Tensor(f16), allocator: Allocator) !Tensor(f16) {
    const A_shape = a.shape;
    const B_shape = b.shape;

    if (A_shape.len != 2 or B_shape.len != 2) {
        std.log.err("Incompatible Tensor Shapes, A shape : {any}, B shape {any}", .{ A_shape, B_shape });
        return error.InvalidTensorDimension;
    }

    const M = A_shape[0];
    const K = A_shape[1];
    const N = B_shape[1];

    if (B_shape[0] != K) {
        std.log.err("Incompatible shapes, A shape : {any}, B shape {any}", .{ A_shape, B_shape });
        return error.IncompatibleTensorShapes;
    }

    // Validate bias shape if provided
    if (bias) |c| {
        const C_shape = c.shape;
        if (C_shape.len == 2) {
            // Full matrix bias
            if (C_shape[0] != M or C_shape[1] != N) {
                std.log.err("Incompatible bias shape: {any}", .{C_shape});
                return error.IncompatibleBiasShape;
            }
        } else if (C_shape.len == 1) {
            // Vector bias for broadcasting
            if (C_shape[0] != N) {
                std.log.err("Incompatible bias shape for broadcasting: {any}", .{C_shape});
                return error.IncompatibleBiasShape;
            }
        } else {
            std.log.err("Unsupported bias shape: {any}", .{C_shape});
            return error.IncompatibleBiasShape;
        }
    }

    // Create output tensor
    var result = try Tensor(f16).init(allocator, &[_]usize{ M, N });
    errdefer result.deinit();

    const A_data = a.getSlice();
    const B_data = b.getSlice();
    const C_data = result.getSlice();
    const bias_data = if (bias) |c| c.getSlice() else null;

    const tiles_M = (M + T - 1) / T;
    const tiles_N = (N + T - 1) / T;
    const total_tiles = tiles_M * tiles_N;

    var shared_data = ThreadLocalData{ .current_index = atomic.Value(usize).init(0) };

    const num_threads = try std.Thread.getCpuCount();
    var thread_pool = try std.ArrayList(std.Thread).initCapacity(allocator, num_threads);
    defer thread_pool.deinit();

    const is_vector_bias = if (bias) |c| c.shape.len == 1 else false;

    const context = ThreadContext{
        .A = A_data,
        .B = B_data,
        .C = C_data,
        .bias = bias_data,
        .M = M,
        .N = N,
        .K = K,
        .tiles_M = tiles_M,
        .tiles_N = tiles_N,
        .total_tiles = total_tiles,
        .shared_counter = &shared_data,
        .is_vector_bias = is_vector_bias,
    };

    for (0..num_threads) |_| {
        try thread_pool.append(try std.Thread.spawn(.{}, workerThread, .{context}));
    }

    for (thread_pool.items) |thread| thread.join();

    return result;
}

const ThreadContext = struct {
    A: []const f16,
    B: []const f16,
    C: []f16,
    bias: ?[]const f16,
    M: usize,
    N: usize,
    K: usize,
    tiles_M: usize,
    tiles_N: usize,
    total_tiles: usize,
    shared_counter: *ThreadLocalData,
    is_vector_bias: bool,
};

// Test configurations
// Test configurations
const MatrixConfig = struct {
    M: usize,
    N: usize,
    K: usize,
    name: []const u8,
};

const configs = [_]MatrixConfig{
    // Small matrices
    .{ .M = 32, .N = 32, .K = 32, .name = "small_square" },
    .{ .M = 64, .N = 64, .K = 64, .name = "medium_square" },
    .{ .M = 128, .N = 128, .K = 128, .name = "large_square" },

    // Rectangular matrices
    .{ .M = 64, .N = 32, .K = 128, .name = "rectangular_1" },
    .{ .M = 128, .N = 64, .K = 256, .name = "rectangular_2" },

    // Tile boundary cases
    .{ .M = 159, .N = 159, .K = 159, .name = "tile_boundary_below" },
    .{ .M = 160, .N = 160, .K = 160, .name = "tile_boundary_exact" },
    .{ .M = 161, .N = 161, .K = 161, .name = "tile_boundary_above" },

    // Large matrices for throughput testing
    .{ .M = 512, .N = 512, .K = 512, .name = "very_large_square" },
    .{ .M = 1024, .N = 1024, .K = 1024, .name = "huge_square" },
};

const WARMUP_RUNS = 3;
const BENCH_RUNS = 10;

pub fn main() !void {
    // Use arena allocator for benchmarking
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Print header
    std.debug.print("\nMatrix Multiplication Benchmark\n", .{});
    std.debug.print("================================================\n", .{});
    std.debug.print("Configuration: {} warmup runs, {} benchmark runs\n", .{ WARMUP_RUNS, BENCH_RUNS });
    std.debug.print("================================================\n\n", .{});

    // Run benchmarks for each configuration
    for (configs) |config| {
        try benchmarkConfig(allocator, config);
    }
}

fn benchmarkConfig(allocator: Allocator, config: MatrixConfig) !void {
    std.debug.print("Testing {s} ({d}x{d}x{d}):\n", .{ config.name, config.M, config.N, config.K });

    // Create test matrices
    var a = try createRandomTensor(allocator, &[_]usize{ config.M, config.K });
    defer a.deinit();

    var b = try createRandomTensor(allocator, &[_]usize{ config.K, config.N });
    defer b.deinit();

    var bias = try createRandomTensor(allocator, &[_]usize{ config.M, config.N });
    defer bias.deinit();

    // Warm up
    for (0..WARMUP_RUNS) |_| {
        // Plain matmul
        var result1 = try matmul(a, b, null, allocator);
        result1.deinit();

        // Fused matmul + bias
        var result2 = try matmul(a, b, bias, allocator);
        result2.deinit();

        // Separate matmul + bias_add
        var result3 = try matmul(a, b, null, allocator);
        try broadcast_add_simd(&result3, bias);
        result3.deinit();
    }

    // Benchmark without bias
    var times_no_bias: [BENCH_RUNS]u64 = undefined;
    for (0..BENCH_RUNS) |i| {
        const start = time.nanoTimestamp();
        var result = try matmul(a, b, null, allocator);
        const end = time.nanoTimestamp();
        result.deinit();
        times_no_bias[i] = @intCast(end - start);
    }

    // Benchmark with fused bias
    var times_fused_bias: [BENCH_RUNS]u64 = undefined;
    for (0..BENCH_RUNS) |i| {
        const start = time.nanoTimestamp();
        var result = try matmul(a, b, bias, allocator);
        const end = time.nanoTimestamp();
        result.deinit();
        times_fused_bias[i] = @intCast(end - start);
    }

    // Benchmark with separate bias add
    var times_separate_bias: [BENCH_RUNS]u64 = undefined;
    for (0..BENCH_RUNS) |i| {
        const start = time.nanoTimestamp();
        var result = try matmul(a, b, null, allocator);
        try broadcast_add_simd(&result, bias);
        const end = time.nanoTimestamp();
        result.deinit();
        times_separate_bias[i] = @intCast(end - start);
    }

    // Calculate statistics
    const stats_no_bias = calculateStats(times_no_bias[0..]);
    const stats_fused_bias = calculateStats(times_fused_bias[0..]);
    const stats_separate_bias = calculateStats(times_separate_bias[0..]);

    // Convert to seconds for GFLOPS calculation
    const mean_seconds_no_bias = @as(f64, @floatFromInt(stats_no_bias.mean)) / 1_000_000_000.0;
    const mean_seconds_fused_bias = @as(f64, @floatFromInt(stats_fused_bias.mean)) / 1_000_000_000.0;
    const mean_seconds_separate_bias = @as(f64, @floatFromInt(stats_separate_bias.mean)) / 1_000_000_000.0;

    // Calculate FLOPS (include the bias add operation in the count)
    const ops_per_matmul = @as(f64, @floatFromInt(config.M)) *
        @as(f64, @floatFromInt(config.N)) *
        @as(f64, @floatFromInt(config.K)) * 2.0;
    const ops_per_bias_add = @as(f64, @floatFromInt(config.M)) *
        @as(f64, @floatFromInt(config.N));

    const gflops_no_bias = ops_per_matmul / mean_seconds_no_bias / 1_000_000_000.0;
    const gflops_fused_bias = (ops_per_matmul + ops_per_bias_add) / mean_seconds_fused_bias / 1_000_000_000.0;
    const gflops_separate_bias = (ops_per_matmul + ops_per_bias_add) / mean_seconds_separate_bias / 1_000_000_000.0;

    // Print results
    std.debug.print("Without bias:\n", .{});
    std.debug.print("  Mean: {d:.2}ms (±{d:.2}ms)\n", .{
        @as(f64, @floatFromInt(stats_no_bias.mean)) / 1e6,
        @as(f64, @floatFromInt(stats_no_bias.stddev)) / 1e6,
    });
    std.debug.print("  Performance: {d:.2} GFLOPS\n", .{gflops_no_bias});

    std.debug.print("With fused bias:\n", .{});
    std.debug.print("  Mean: {d:.2}ms (±{d:.2}ms)\n", .{
        @as(f64, @floatFromInt(stats_fused_bias.mean)) / 1e6,
        @as(f64, @floatFromInt(stats_fused_bias.stddev)) / 1e6,
    });
    std.debug.print("  Performance: {d:.2} GFLOPS\n", .{gflops_fused_bias});
    std.debug.print("  Overhead vs no bias: {d:.1}%\n", .{((@as(f64, @floatFromInt(stats_fused_bias.mean)) /
        @as(f64, @floatFromInt(stats_no_bias.mean))) - 1.0) * 100.0});

    std.debug.print("With separate bias add:\n", .{});
    std.debug.print("  Mean: {d:.2}ms (±{d:.2}ms)\n", .{
        @as(f64, @floatFromInt(stats_separate_bias.mean)) / 1e6,
        @as(f64, @floatFromInt(stats_separate_bias.stddev)) / 1e6,
    });
    std.debug.print("  Performance: {d:.2} GFLOPS\n", .{gflops_separate_bias});
    std.debug.print("  Overhead vs no bias: {d:.1}%\n", .{((@as(f64, @floatFromInt(stats_separate_bias.mean)) /
        @as(f64, @floatFromInt(stats_no_bias.mean))) - 1.0) * 100.0});
    std.debug.print("  Overhead vs fused: {d:.1}%\n\n", .{((@as(f64, @floatFromInt(stats_separate_bias.mean)) /
        @as(f64, @floatFromInt(stats_fused_bias.mean))) - 1.0) * 100.0});
}

const Stats = struct {
    mean: u64,
    stddev: u64,
};

fn calculateStats(times: []const u64) Stats {
    var sum: u64 = 0;
    for (times) |time_val| {
        sum += time_val;
    }
    const mean = sum / times.len;

    var sum_sq_diff: u64 = 0;
    for (times) |time_val| {
        const diff = if (time_val > mean) time_val - mean else mean - time_val;
        sum_sq_diff += diff * diff;
    }
    const variance = sum_sq_diff / times.len;
    const stddev = @as(u64, @intFromFloat(@sqrt(@as(f64, @floatFromInt(variance)))));

    return .{
        .mean = mean,
        .stddev = stddev,
    };
}

fn createRandomTensor(allocator: Allocator, shape: []const usize) !Tensor(f16) {
    var tensor = try Tensor(f16).init(allocator, shape);
    errdefer tensor.deinit();

    const data = tensor.getSlice();
    var rng = std.rand.DefaultPrng.init(@intCast(time.nanoTimestamp()));

    for (data) |*val| {
        // Generate values between -1 and 1
        const random_f32 = rng.random().float(f32) * 2.0 - 1.0;
        val.* = @floatCast(random_f32);
    }

    return tensor;
}
fn approximatelyEqual(a: f16, b: f16, epsilon: f16) bool {
    return @abs(a - b) <= epsilon;
}

fn verifyMatmulResult(
    allocator: Allocator,
    a: Tensor(f16),
    b: Tensor(f16),
    bias: ?Tensor(f16),
    expected: Tensor(f16),
) !void {
    var result = try matmul(a, b, bias, allocator);
    defer result.deinit();

    const result_data = result.getSlice();
    const expected_data = expected.getSlice();

    // Compare shape lengths
    try testing.expectEqual(result.shape.len, expected.shape.len);

    // Compare each dimension
    for (result.shape, expected.shape) |r_dim, e_dim| {
        try testing.expectEqual(r_dim, e_dim);
    }

    // Use a relatively large epsilon due to f16 precision and accumulated error
    const epsilon: f16 = 0.1;

    for (result_data, 0..) |val, i| {
        try testing.expect(approximatelyEqual(val, expected_data[i], epsilon));
    }
}

fn createTensor(allocator: Allocator, shape: []const usize, data: []const f16) !Tensor(f16) {
    var tensor = try Tensor(f16).init(allocator, shape);
    errdefer tensor.deinit();

    const tensor_data = tensor.getSlice();
    @memcpy(tensor_data, data);

    return tensor;
}

test "matmul - 2x2 matrices without bias" {
    const allocator = testing.allocator;

    // A = [1 2]
    //     [3 4]
    const a_data = [_]f16{ 1, 2, 3, 4 };
    var a = try createTensor(allocator, &[_]usize{ 2, 2 }, &a_data);
    defer a.deinit();

    // B = [5 6]
    //     [7 8]
    const b_data = [_]f16{ 5, 6, 7, 8 };
    var b = try createTensor(allocator, &[_]usize{ 2, 2 }, &b_data);
    defer b.deinit();

    // Expected = [19 22]
    //            [43 50]
    const expected_data = [_]f16{ 19, 22, 43, 50 };
    var expected = try createTensor(allocator, &[_]usize{ 2, 2 }, &expected_data);
    defer expected.deinit();

    try verifyMatmulResult(allocator, a, b, null, expected);
}

test "matmul - 2x2 matrices with bias" {
    const allocator = testing.allocator;

    // A = [1 2]
    //     [3 4]
    const a_data = [_]f16{ 1, 2, 3, 4 };
    var a = try createTensor(allocator, &[_]usize{ 2, 2 }, &a_data);
    defer a.deinit();

    // B = [5 6]
    //     [7 8]
    const b_data = [_]f16{ 5, 6, 7, 8 };
    var b = try createTensor(allocator, &[_]usize{ 2, 2 }, &b_data);
    defer b.deinit();

    // Bias = [0.1 0.2]
    //        [0.3 0.4]
    const bias_data = [_]f16{ 0.1, 0.2, 0.3, 0.4 };
    var bias = try createTensor(allocator, &[_]usize{ 2, 2 }, &bias_data);
    defer bias.deinit();

    // Expected = [19.1 22.2]
    //            [43.3 50.4]
    const expected_data = [_]f16{ 19.1, 22.2, 43.3, 50.4 };
    var expected = try createTensor(allocator, &[_]usize{ 2, 2 }, &expected_data);
    defer expected.deinit();

    try verifyMatmulResult(allocator, a, b, bias, expected);
}

test "matmul - rectangular matrices without bias" {
    const allocator = testing.allocator;

    // A = [1 2 3]
    //     [4 5 6]
    const a_data = [_]f16{ 1, 2, 3, 4, 5, 6 };
    var a = try createTensor(allocator, &[_]usize{ 2, 3 }, &a_data);
    defer a.deinit();

    // B = [7  8]
    //     [9  10]
    //     [11 12]
    const b_data = [_]f16{ 7, 8, 9, 10, 11, 12 };
    var b = try createTensor(allocator, &[_]usize{ 3, 2 }, &b_data);
    defer b.deinit();

    // Expected = [58  64]
    //            [139 154]
    const expected_data = [_]f16{ 58, 64, 139, 154 };
    var expected = try createTensor(allocator, &[_]usize{ 2, 2 }, &expected_data);
    defer expected.deinit();

    try verifyMatmulResult(allocator, a, b, null, expected);
}

test "matmul - large matrices" {
    const allocator = testing.allocator;

    // Test with matrices larger than the tile size
    const M: usize = 200;
    const K: usize = 100;
    const N: usize = 150;

    // Create matrices with a simple pattern
    var a = try Tensor(f16).init(allocator, &[_]usize{ M, K });
    defer a.deinit();
    const a_data = a.getSlice();
    for (a_data, 0..) |*val, i| {
        val.* = @floatCast(@as(f32, @floatFromInt(i % 7)) / 7.0);
    }

    var b = try Tensor(f16).init(allocator, &[_]usize{ K, N });
    defer b.deinit();
    const b_data = b.getSlice();
    for (b_data, 0..) |*val, i| {
        val.* = @floatCast(@as(f32, @floatFromInt(i % 5)) / 5.0);
    }

    var bias = try Tensor(f16).init(allocator, &[_]usize{ M, N });
    defer bias.deinit();
    const bias_data = bias.getSlice();
    for (bias_data, 0..) |*val, i| {
        val.* = @floatCast(@as(f32, @floatFromInt(i % 3)) / 3.0);
    }

    // Compute result and verify basic properties
    var result = try matmul(a, b, bias, allocator);
    defer result.deinit();

    try testing.expectEqual(result.shape[0], M);
    try testing.expectEqual(result.shape[1], N);
}

test "matmul - error cases" {
    const allocator = testing.allocator;

    // Test incompatible matrix dimensions
    {
        // A is 2x3, B is 2x2 - this should fail
        var a = try Tensor(f16).init(allocator, &[_]usize{ 2, 3 });
        defer a.deinit();
        var b = try Tensor(f16).init(allocator, &[_]usize{ 2, 2 });
        defer b.deinit();

        // Expect specific error for incompatible shapes
        try testing.expectError(error.IncompatibleTensorShapes, matmul(a, b, null, allocator));
    }

    // Test incompatible bias dimensions
    {
        // Result would be 2x3 but bias is 2x2
        var a = try Tensor(f16).init(allocator, &[_]usize{ 2, 2 });
        defer a.deinit();
        var b = try Tensor(f16).init(allocator, &[_]usize{ 2, 3 });
        defer b.deinit();
        var bias = try Tensor(f16).init(allocator, &[_]usize{ 2, 2 });
        defer bias.deinit();

        try testing.expectError(error.IncompatibleBiasShape, matmul(a, b, bias, allocator));
    }

    // Test invalid tensor dimensions
    {
        // 3D tensor not allowed
        var a = try Tensor(f16).init(allocator, &[_]usize{ 2, 2, 2 });
        defer a.deinit();
        var b = try Tensor(f16).init(allocator, &[_]usize{ 2, 2 });
        defer b.deinit();

        try testing.expectError(error.InvalidTensorDimension, matmul(a, b, null, allocator));
    }
}

test "matmul - tile boundary cases" {
    const allocator = testing.allocator;

    // Test matrices with dimensions near tile boundaries
    const tile_sizes = [_]usize{ 159, 160, 161 };

    for (tile_sizes) |size| {
        var a = try Tensor(f16).init(allocator, &[_]usize{ size, size });
        defer a.deinit();
        const a_data = a.getSlice();
        for (a_data, 0..) |*val, i| {
            val.* = @floatCast(@as(f32, @floatFromInt(i % 7)) / 7.0);
        }

        var b = try Tensor(f16).init(allocator, &[_]usize{ size, size });
        defer b.deinit();
        const b_data = b.getSlice();
        for (b_data, 0..) |*val, i| {
            val.* = @floatCast(@as(f32, @floatFromInt(i % 5)) / 5.0);
        }

        var result = try matmul(a, b, null, allocator);
        defer result.deinit();

        try testing.expectEqual(result.shape[0], size);
        try testing.expectEqual(result.shape[1], size);
    }
}

test "matmul - numerical stability" {
    const allocator = testing.allocator;

    // Test with very small and very large numbers
    const a_data = [_]f16{ 1e-3, 1e-3, 1e-3, 1e-3 };
    var a = try createTensor(allocator, &[_]usize{ 2, 2 }, &a_data);
    defer a.deinit();

    const b_data = [_]f16{ 1e3, 1e3, 1e3, 1e3 };
    var b = try createTensor(allocator, &[_]usize{ 2, 2 }, &b_data);
    defer b.deinit();

    const bias_data = [_]f16{ 1.0, 1.0, 1.0, 1.0 };
    var bias = try createTensor(allocator, &[_]usize{ 2, 2 }, &bias_data);
    defer bias.deinit();

    var result = try matmul(a, b, bias, allocator);
    defer result.deinit();

    // Verify that results are in reasonable range
    const result_data = result.getSlice();
    for (result_data) |val| {
        try testing.expect(val >= 1.0 and val <= 1e4);
    }
}

test "matmul - identity matrix test" {
    const allocator = testing.allocator;

    // Create a 4x4 identity matrix
    const identity_data = [_]f16{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };
    var identity = try createTensor(allocator, &[_]usize{ 4, 4 }, &identity_data);
    defer identity.deinit();

    // Create a test matrix
    const test_data = [_]f16{
        1,  2,  3,  4,
        5,  6,  7,  8,
        9,  10, 11, 12,
        13, 14, 15, 16,
    };
    var test_matrix = try createTensor(allocator, &[_]usize{ 4, 4 }, &test_data);
    defer test_matrix.deinit();

    // Verify A * I = A
    try verifyMatmulResult(allocator, test_matrix, identity, null, test_matrix);

    // Verify I * A = A
    try verifyMatmulResult(allocator, identity, test_matrix, null, test_matrix);
}
