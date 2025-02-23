const std = @import("std");
const Allocator = std.mem.Allocator;
const atomic = std.atomic;
const math = @import("std").math;
const ArrayList = std.ArrayList;

const Tensor = @import("../core/tensor.zig").Tensor;
const ops = @import("../ops/ops.zig");

const testing = std.testing;
const time = @import("std").time;

comptime {
    @setFloatMode(.optimized);
}

// SIMD and optimization constants
const Vec8f = @Vector(8, f32);
const T: usize = 168; // Tile size optimized for cache
const CACHE_LINE_SIZE: usize = 128;

// Thread-local data structure aligned to cache line
const ThreadLocalData = struct {
    current_index: atomic.Value(usize) align(64),
    _padding: [64 - @sizeOf(atomic.Value(usize))]u8 = undefined,
};

// Context for regular matrix multiplication
const ThreadContext = struct {
    A: []const f16,
    B_t: []const f16,
    C: []f16,
    M: usize,
    N: usize,
    K: usize,
    tiles_M: usize,
    tiles_N: usize,
    total_tiles: usize,
    shared_counter: *ThreadLocalData,
};

// Context for vector-matrix multiplication
const VectorThreadContext = struct {
    A: []const f16,
    B_t: []const f16,
    C: []f16,
    K: usize,
    N: usize,
    total_chunks: usize,
    chunk_size: usize,
    shared_counter: *ThreadLocalData,
};

comptime {
    @setFloatMode(.optimized);
}

// Core computation for vector chunk
fn processVectorChunk(start_n: usize, end_n: usize, a: []const f16, b_t: []const f16, c: []f16, K: usize) void {
    @setRuntimeSafety(false);

    for (start_n..end_n) |n| {
        var sum: f32 = 0;
        const row = n * K;

        var k: usize = 0;
        const k_aligned = K - (K % 8);

        while (k < k_aligned) : (k += 8) {
            const a_vec = Vec8f{
                @as(f32, @floatCast(a[k + 0])),
                @as(f32, @floatCast(a[k + 1])),
                @as(f32, @floatCast(a[k + 2])),
                @as(f32, @floatCast(a[k + 3])),
                @as(f32, @floatCast(a[k + 4])),
                @as(f32, @floatCast(a[k + 5])),
                @as(f32, @floatCast(a[k + 6])),
                @as(f32, @floatCast(a[k + 7])),
            };

            const b_vec = Vec8f{
                @as(f32, @floatCast(b_t[row + k + 0])),
                @as(f32, @floatCast(b_t[row + k + 1])),
                @as(f32, @floatCast(b_t[row + k + 2])),
                @as(f32, @floatCast(b_t[row + k + 3])),
                @as(f32, @floatCast(b_t[row + k + 4])),
                @as(f32, @floatCast(b_t[row + k + 5])),
                @as(f32, @floatCast(b_t[row + k + 6])),
                @as(f32, @floatCast(b_t[row + k + 7])),
            };

            sum += @reduce(.Add, a_vec * b_vec);
        }

        while (k < K) : (k += 1) {
            sum += @as(f32, @floatCast(a[k])) * @as(f32, @floatCast(b_t[row + k]));
        }

        c[n] = @floatCast(sum);
    }
}

// Vector-matrix multiplication worker
fn vectorWorker(ctx: VectorThreadContext) void {
    while (true) {
        const chunk_idx = ctx.shared_counter.current_index.fetchAdd(1, .seq_cst);
        if (chunk_idx >= ctx.total_chunks) break;

        const start_n = chunk_idx * ctx.chunk_size;
        const end_n = @min(start_n + ctx.chunk_size, ctx.N);

        processVectorChunk(start_n, end_n, ctx.A, ctx.B_t, ctx.C, ctx.K);
    }
}

// Core tile computation for matrix multiplication
fn computeTile(
    A: []const f16,
    B_t: []const f16,
    C: []f16,
    i_start: usize,
    j_start: usize,
    k_start: usize,
    i_end: usize,
    j_end: usize,
    k_end: usize,
    K: usize,
    N: usize,
) void {
    @setRuntimeSafety(false);
    var local_C: [T][T]f32 align(32) = undefined;
    var A_local: [T][T]f32 align(32) = undefined;
    var B_local: [T][T]f32 align(32) = undefined;

    const k_size = k_end - k_start;
    const i_size = i_end - i_start;
    const j_size = j_end - j_start;

    // Clear accumulators
    for (0..T) |x| {
        @memset(&local_C[x], 0);
    }

    // Load and convert input tiles
    for (0..i_size) |i| {
        const row_offset = (i_start + i) * K;
        for (0..k_size) |k| {
            A_local[i][k] = @floatCast(A[row_offset + k_start + k]);
        }
    }

    for (0..j_size) |j| {
        const col_offset = (j_start + j) * K;
        for (0..k_size) |k| {
            B_local[k][j] = @floatCast(B_t[col_offset + k_start + k]);
        }
    }

    // Main computation loop with SIMD
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

        // Handle remaining columns
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

    // Handle remaining rows
    while (i < i_size) : (i += 1) {
        for (0..j_size) |j| {
            var sum: f32 = 0;
            for (0..k_size) |k| {
                sum = @mulAdd(f32, A_local[i][k], B_local[k][j], sum);
            }
            local_C[i][j] += sum;
        }
    }

    // Store results
    for (i_start..i_end) |ii| {
        const row_offset = ii * N;
        const local_row = ii - i_start;
        for (j_start..j_end) |jj| {
            const local_col = jj - j_start;
            C[row_offset + jj] = @floatCast(local_C[local_row][local_col]);
        }
    }
}

fn computeTileTransposed(
    ctx: ThreadContext,
    local_C: *[T][T]f32,
    i_start: usize,
    j_start: usize,
    k_start: usize,
    i_end: usize,
    j_end: usize,
    k_end: usize,
) void {
    @setRuntimeSafety(false);
    var A_local: [T][T]f32 align(32) = undefined;
    var B_local: [T][T]f32 align(32) = undefined;

    const k_size = k_end - k_start;
    const i_size = i_end - i_start;
    const j_size = j_end - j_start;

    // Load and convert A to f32
    for (0..i_size) |i| {
        const row_offset = (i_start + i) * ctx.K;
        for (0..k_size) |k| {
            A_local[i][k] = @floatCast(ctx.A[row_offset + k_start + k]);
        }
    }

    // Load and convert B_t to f32
    for (0..j_size) |j| {
        const col_offset = (j_start + j) * ctx.K;
        for (0..k_size) |k| {
            B_local[k][j] = @floatCast(ctx.B_t[col_offset + k_start + k]);
        }
    }

    // Main computation loop with SIMD
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

        // Handle remaining columns
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

    // Handle remaining rows
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
// Regular matrix multiplication worker
fn matmulWorker(ctx: ThreadContext) void {
    @setRuntimeSafety(false);
    var local_C: [T][T]f32 align(32) = undefined;

    while (true) {
        const tile_idx = ctx.shared_counter.current_index.fetchAdd(1, .seq_cst);
        if (tile_idx >= ctx.total_tiles) break;

        const i = tile_idx / ctx.tiles_N;
        const j = tile_idx % ctx.tiles_N;

        const i_start = i * T;
        const j_start = j * T;
        const i_end = @min(i_start + T, ctx.M);
        const j_end = @min(j_start + T, ctx.N);

        // Clear local_C for this tile
        for (0..T) |x| {
            @memset(&local_C[x], 0);
        }

        var k: usize = 0;
        while (k < ctx.K) : (k += T) {
            const k_end = @min(k + T, ctx.K);
            computeTileTransposed(ctx, &local_C, i_start, j_start, k, i_end, j_end, k_end);
        }

        // Store results
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

// Main matrix multiplication function that operates on Tensors
pub fn matmul(a: Tensor(f16), b_transposed: Tensor(f16), allocator: Allocator) !Tensor(f16) {
    @setRuntimeSafety(false);
    const A_shape = a.shape;
    const B_t_shape = b_transposed.shape;

    // Input validation
    if (A_shape.len != 2 or B_t_shape.len != 2) {
        std.log.err("Incompatible Tensor Shapes, A shape : {any}, B_t shape {any}", .{ A_shape, B_t_shape });
        return error.InvalidTensorDimension;
    }

    const M = A_shape[0];
    const K = A_shape[1];
    const N = B_t_shape[0]; // Note: N is first dimension since B is transposed

    if (B_t_shape[1] != K) {
        std.log.err("Incompatible shapes, A shape : {any}, B_t shape {any}", .{ A_shape, B_t_shape });
        return error.IncompatibleTensorShapes;
    }

    var result = try Tensor(f16).init(allocator, &[_]usize{ M, N });
    errdefer result.deinit();

    const A_data = a.getSlice();
    const B_t_data = b_transposed.getSlice();
    const C_data = result.getSlice();

    // Special fast path for vector-matrix multiplication
    if (M == 1) {
        const chunk_size = 32;
        var shared_data = ThreadLocalData{ .current_index = atomic.Value(usize).init(0) };

        const vec_context = VectorThreadContext{
            .A = A_data,
            .B_t = B_t_data,
            .C = C_data,
            .K = K,
            .N = N,
            .total_chunks = (N + chunk_size - 1) / chunk_size,
            .chunk_size = chunk_size,
            .shared_counter = &shared_data,
        };

        const num_threads = try std.Thread.getCpuCount();
        var thread_pool = try std.ArrayList(std.Thread).initCapacity(allocator, num_threads);
        defer thread_pool.deinit();

        for (0..num_threads) |_| {
            try thread_pool.append(try std.Thread.spawn(.{}, vectorWorker, .{vec_context}));
        }

        for (thread_pool.items) |thread| thread.join();
        return result;
    }

    // Regular matrix multiplication path
    const tiles_M = (M + T - 1) / T;
    const tiles_N = (N + T - 1) / T;
    const total_tiles = tiles_M * tiles_N;

    var shared_data = ThreadLocalData{ .current_index = atomic.Value(usize).init(0) };

    const context = ThreadContext{
        .A = A_data,
        .B_t = B_t_data,
        .C = C_data,
        .M = M,
        .N = N,
        .K = K,
        .tiles_M = tiles_M,
        .tiles_N = tiles_N,
        .total_tiles = total_tiles,
        .shared_counter = &shared_data,
    };

    const num_threads = try std.Thread.getCpuCount();
    var thread_pool = try std.ArrayList(std.Thread).initCapacity(allocator, num_threads);
    defer thread_pool.deinit();

    for (0..num_threads) |_| {
        try thread_pool.append(try std.Thread.spawn(.{}, matmulWorker, .{context}));
    }

    for (thread_pool.items) |thread| thread.join();

    return result;
}

// Internal implementation that works on slices
fn matmulImpl(allocator: Allocator, a: []const f16, b_t: []const f16, c: []f16, M: usize, K: usize, N: usize) !void {
    @setRuntimeSafety(false);

    // Fast path for vector-matrix multiplication
    if (M == 1) {
        const chunk_size = 32;
        var shared_data = ThreadLocalData{ .current_index = atomic.Value(usize).init(0) };

        const vec_context = VectorThreadContext{
            .A = a,
            .B_t = b_t,
            .C = c,
            .K = K,
            .N = N,
            .total_chunks = (N + chunk_size - 1) / chunk_size,
            .chunk_size = chunk_size,
            .shared_counter = &shared_data,
        };

        const num_threads = try std.Thread.getCpuCount();
        var thread_pool = try std.ArrayList(std.Thread).initCapacity(allocator, num_threads);
        defer thread_pool.deinit();

        for (0..num_threads) |_| {
            try thread_pool.append(try std.Thread.spawn(.{}, vectorWorker, .{vec_context}));
        }

        for (thread_pool.items) |thread| thread.join();
        return;
    }

    // Regular matrix multiplication path
    const tiles_M = (M + T - 1) / T;
    const tiles_N = (N + T - 1) / T;
    const total_tiles = tiles_M * tiles_N;

    var shared_data = ThreadLocalData{ .current_index = atomic.Value(usize).init(0) };

    const context = ThreadContext{
        .A = a,
        .B_t = b_t,
        .C = c,
        .M = M,
        .N = N,
        .K = K,
        .tiles_M = tiles_M,
        .tiles_N = tiles_N,
        .total_tiles = total_tiles,
        .shared_counter = &shared_data,
    };

    const num_threads = try std.Thread.getCpuCount();
    var thread_pool = try std.ArrayList(std.Thread).initCapacity(allocator, num_threads);
    defer thread_pool.deinit();

    for (0..num_threads) |_| {
        try thread_pool.append(try std.Thread.spawn(.{}, matmulWorker, .{context}));
    }

    for (thread_pool.items) |thread| thread.join();
}
fn transpose(allocator: std.mem.Allocator, tensor: Tensor(f16)) !Tensor(f16) {
    const shape = tensor.shape;
    var result = try Tensor(f16).init(allocator, &[_]usize{ shape[1], shape[0] });
    errdefer result.deinit();

    const src = tensor.getSlice();
    const dst = result.getSlice();

    for (0..shape[0]) |i| {
        for (0..shape[1]) |j| {
            dst[j * shape[0] + i] = src[i * shape[1] + j];
        }
    }
    return result;
}

const print = std.debug.print;

// Convert bytes to gigabytes
fn bytesToGB(bytes: f64) f64 {
    return bytes / (1024 * 1024 * 1024);
}

// Calculate GFLOPS (Giga Floating Point Operations per Second)
fn calculateGFLOPS(m: usize, n: usize, k: usize, elapsed_ns: u64) f64 {
    // Each element requires 2 operations (multiply and add)
    const total_ops = @as(f64, @floatFromInt(m * n * k * 2));
    const seconds = @as(f64, @floatFromInt(elapsed_ns)) / 1e9;
    return (total_ops / seconds) / 1e9;
}

// Calculate memory bandwidth in GB/s
fn calculateBandwidth(m: usize, n: usize, k: usize, elapsed_ns: u64) f64 {
    // Memory access: A (m*k), B (k*n), C (m*n)
    const bytes_accessed = @as(f64, @floatFromInt((m * k + k * n + m * n) * @sizeOf(f16)));
    const seconds = @as(f64, @floatFromInt(elapsed_ns)) / 1e9;
    return bytesToGB(bytes_accessed) / seconds;
}

pub fn main() !void {
    const sizes = [_][3]usize{
        .{ 1, 2048, 51200 },
        .{ 800, 2048, 51200 },
        .{ 1, 2048, 6144 }, // Vector
        .{ 800, 2048, 6144 }, // Vector
        .{ 512, 512, 512 }, // Square
        .{ 1024, 1024, 1024 }, // Larger square
        .{ 768, 512, 256 }, // Rectangular
        .{ 2048, 1024, 512 }, // Large rectangular
        .{ 160, 160, 160 }, // Tile size
        .{ 2048, 2048, 2048 }, // Large square
    };

    // Initialize allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Print header
    print("\nMatrix Multiplication Benchmark\n", .{});
    print("================================\n", .{});
    print("Size (MxKxN) | Time (ms) | GFLOPS | GB/s | Avg Time (ms) | Best GFLOPS | Notes\n", .{});
    print("-------------+-----------|--------|------|---------------|-------------|-------\n", .{});

    // Number of iterations for each size
    const iterations = 5;

    for (sizes) |size| {
        const M = size[0];
        const K = size[1];
        const N = size[2];

        // Create input tensors
        var a = try Tensor(f16).init(allocator, &[_]usize{ M, K });
        defer a.deinit();
        var b = try Tensor(f16).init(allocator, &[_]usize{ K, N });
        defer b.deinit();

        // Initialize with random data
        const a_data = a.getSlice();
        const b_data = b.getSlice();
        var rng = std.rand.DefaultPrng.init(0);

        for (a_data) |*val| {
            val.* = @floatCast(rng.random().float(f32) * 2.0 - 1.0);
        }
        for (b_data) |*val| {
            val.* = @floatCast(rng.random().float(f32) * 2.0 - 1.0);
        }

        // Pre-transpose B
        var b_t = try transpose(allocator, b);
        defer b_t.deinit();

        // Warmup run
        {
            var result = try matmul(a, b_t, allocator);
            result.deinit();
        }

        // Benchmark runs
        var total_time: u64 = 0;
        var min_time: u64 = std.math.maxInt(u64);
        var max_gflops: f64 = 0;
        var max_bandwidth: f64 = 0;

        for (0..iterations) |i| {
            const start = time.nanoTimestamp();
            var result = try matmul(a, b_t, allocator);
            const end = time.nanoTimestamp();
            result.deinit();

            const elapsed = @as(u64, @intCast(end - start));
            total_time += elapsed;
            min_time = @min(min_time, elapsed);

            const gflops = calculateGFLOPS(M, N, K, elapsed);
            const bandwidth = calculateBandwidth(M, N, K, elapsed);
            max_gflops = @max(max_gflops, gflops);
            max_bandwidth = @max(max_bandwidth, bandwidth);

            // Print individual iteration results
            print("{d}x{d}x{d} | {d:8.2} | {d:6.1} | {d:4.1} | ", .{
                M, K, N,
                @as(f64, @floatFromInt(elapsed)) / 1e6, // ms
                gflops,
                bandwidth,
            });

            if (i == 0) {
                print("{d:8.2} | {d:6.1} | ", .{
                    @as(f64, @floatFromInt(total_time)) / (1e6 * @as(f64, @floatFromInt(i + 1))), // avg ms
                    max_gflops,
                });
            } else {
                print("           |           | ", .{});
            }

            // Add notes about the matrix type
            if (i == 0) {
                if (M == 1) {
                    print("Vector-Matrix\n", .{});
                } else if (M == N and N == K) {
                    print("Square\n", .{});
                } else if (M == 160 and N == 160 and K == 160) {
                    print("Tile-sized\n", .{});
                } else {
                    print("Rectangular\n", .{});
                }
            } else {
                print("\n", .{});
            }
        }

        print("-------------+-----------|--------|------|---------------|-------------|-------\n", .{});
        print("Average: {d:8.2} ms, Best: {d:6.1} GFLOPS, {d:4.1} GB/s\n\n", .{
            @as(f64, @floatFromInt(total_time)) / (1e6 * @as(f64, @floatFromInt(iterations))),
            max_gflops,
            max_bandwidth,
        });
    }
}
