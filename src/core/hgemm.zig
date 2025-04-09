const std = @import("std");
const Allocator = std.mem.Allocator;

const atomic = std.atomic;
const math = std.math;
const Tensor = @import("tensor.zig").Tensor;
const SlabReusingAllocator = @import("slab_reusing_allocator.zig").SlabReusingAllocator;
const ThreadPool = @import("thread_pool.zig").ThreadPool;
const global_thread_pool = @import("thread_pool.zig");

const time = std.time;
const testing = std.testing;

// Compile-time optimizations for tuning parameters
pub const T: usize = blk: {
    const ideal_tile = 168; // Original tile size
    break :blk (ideal_tile + 7) & ~@as(usize, 7); // Ensure multiple of 8
};

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

const TileTaskContext = struct {
    A: []const f16,
    B: []const f16,
    C: []f16,
    M: usize,
    N: usize,
    K: usize,
    tiles_M: usize,
    tiles_N: usize,
    total_tiles: usize,
    shared_counter: *ThreadLocalData,
    chunk_size: usize,
};

// New task function that will be submitted to the global thread pool
fn tileProcessingTask(ctx: *TileTaskContext) void {
    processTiles(ctx.*);
}

// Worker logic extracted into a separate function
fn processTiles(ctx: TileTaskContext) void {
    @setFloatMode(.optimized);
    @setRuntimeSafety(false);

    var local_C: [T][T]f32 align(AVX2_ALIGNMENT) = undefined;

    while (true) {
        const start_idx = ctx.shared_counter.current_index.fetchAdd(ctx.chunk_size, .seq_cst);
        if (start_idx >= ctx.total_tiles) break;

        const end_idx = @min(start_idx + ctx.chunk_size, ctx.total_tiles);
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

            // Write directly to f16 output with runtime safety disabled
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

// Unchanged computeTile function
fn computeTile(
    ctx: TileTaskContext,
    local_C: *[T][T]f32,
    i_start: usize,
    j_start: usize,
    k_start: usize,
    i_end: usize,
    j_end: usize,
    k_end: usize,
) void {
    @setFloatMode(.optimized);
    @setRuntimeSafety(false);

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

    // Load and convert B to f32 with runtime safety disabled
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

                // Load B values
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
                    // Store back to accumulator
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

// Modified matmul function to use the global thread pool
pub fn matmul(a: Tensor(f16), b: Tensor(f16), allocator: Allocator) !Tensor(f16) {
    @setFloatMode(.optimized);
    @setRuntimeSafety(false);

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

    var result = try Tensor(f16).initWithoutMemset(allocator, &[_]usize{ M, N });
    errdefer result.deinit();

    const A_data = a.getSlice();
    const B_data = b.getSlice();
    const C_data = result.getSlice();

    const tiles_M = (M + T - 1) / T;
    const tiles_N = (N + T - 1) / T;
    const total_tiles = tiles_M * tiles_N;

    var shared_data = ThreadLocalData{ .current_index = atomic.Value(usize).init(0) };

    // Get the global thread pool instance
    const pool = try global_thread_pool.getInstance();
    const num_threads = pool.workers.len;

    // Create task context
    const context = try allocator.create(TileTaskContext);
    defer allocator.destroy(context);

    context.* = TileTaskContext{
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
        .chunk_size = CHUNK_SIZE,
    };

    // Submit tasks to the thread pool
    const task_count = @min(num_threads, total_tiles);

    for (0..task_count - 1) |_| {
        try global_thread_pool.submitTask(TileTaskContext, tileProcessingTask, context);
    }

    // Use the current thread for one portion of the work
    processTiles(context.*);

    // Wait for all tasks to complete
    try global_thread_pool.waitForAll();

    return result;
}

// Benchmarking code remains unchanged
fn benchmarkMatMul(allocator: std.mem.Allocator, M: usize, N: usize, K: usize, num_runs: usize) !f64 {
    // Create input tensors
    var A = try Tensor(f16).init(allocator, &[_]usize{ M, K });
    defer A.deinit();
    var B = try Tensor(f16).init(allocator, &[_]usize{ K, N });
    defer B.deinit();

    // Initialize with some values (random pattern)
    var rng = std.rand.DefaultPrng.init(42);
    const random = rng.random();
    for (A.data) |*val| {
        val.* = @floatCast(random.float(f32) * 0.1 - 0.05); // Scaled down to avoid overflow
    }
    for (B.data) |*val| {
        val.* = @floatCast(random.float(f32) * 0.1 - 0.05); // Scaled down to avoid overflow
    }

    // Warmup run
    var warmup_C = try matmul(A, B, allocator);
    warmup_C.deinit();

    // Timing runs
    var total_time: u64 = 0;
    var timer = try time.Timer.start();

    for (0..num_runs) |_| {
        timer.reset();
        var C = try matmul(A, B, allocator);
        total_time += timer.read();
        C.deinit();
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
    const gpa_allocator = gpa.allocator();
    var slab_reusing_allocator = SlabReusingAllocator(100).init(gpa_allocator);
    defer slab_reusing_allocator.deinit();
    const allocator = slab_reusing_allocator.allocator();

    // Initialize the global thread pool
    try global_thread_pool.init(allocator);
    defer global_thread_pool.deinit();

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
