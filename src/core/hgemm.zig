const std = @import("std");
const Allocator = std.mem.Allocator;

const atomic = std.atomic;
const math = std.math;
const Tensor = @import("tensor.zig").Tensor;

const time = std.time;
const testing = std.testing;

// comptime {
//     @setFloatMode(.optimized);
// }

//  * HGEMM Implementation Notes
//  * =========================================
//  *
//  * This is an optimized matrix multiplication implementation I've developed over several
//  * iterations. I will try to give a brief explanation of the design decisions and performance considerations
//  * that went into it at different parts of the code.

//  * Parameter Interactions
//  * --------------------
//  * The tuning parameters work together in complex ways:
//  * - Tile size affects cache utilization and memory access patterns
//  * - Vector width determines SIMD efficiency and register usage
//  * - Chunk size impacts threading overhead and load balancing
//  * - Alignment requirements influence memory access performance
//  * - All of this is explained a little more in the comments where these parameters are used
//  * - These parameters are tuned and found to work well with most Intel AVX2 and NEON SIMD consumer CPUs

// Compile-time optimizations for tuning parameters
pub const T: usize = blk: {
    // const cache_line = atomic.cache_line;
    const ideal_tile = 168; // Original tile size
    break :blk (ideal_tile + 7) & ~@as(usize, 7); // Ensure multiple of 8
};

const V: usize = 8;
const CACHE_LINE_SIZE: usize = atomic.cache_line;
const CHUNK_SIZE: usize = 1;
const AVX2_ALIGNMENT = 32;
const MICRO_KERNEL_SIZE: usize = std.simd.suggestVectorLength(f32) orelse 8;
const Vec8f = @Vector(8, f32);

//  * Threading Model
//  * -------------
//  * Modern CPUs have multiple cores, and we can take advantage of this parallelism to speed up
//  * matrix multiplication. I used Zig's standard library to create a thread pool with one
//  * worker thread per core. Each worker thread requests a chunk of work from a shared atomic
//  * counter, which ensures that all cores are kept busy. This approach scales well with the
//  * number of cores and provides good performance on multi-core systems.
//  *
//  * Rather than using static work distribution, I implemented a dynamic work-stealing
//  * approach using atomic counters. Each thread requests the next available chunk of
//  * work, which provides better load balancing across cores. I set CHUNK_SIZE to 1 for
//  * fine-grained distribution, which works well for most workloads despite the slightly
//  * higher synchronization cost.

//  * Cache Alignment and False Sharing Prevention
//  * ------------------------------------------
//  * ThreadLocalData is designed to prevent false sharing, a performance issue where
//  * multiple CPU cores repeatedly invalidate each other's cache lines.
//  * The shared_counter field is an atomic counter that each thread uses to request work.
//  *
//  * The CACHE_LINE_SIZE padding ensures each thread's counter lives on its own cache line,
//  * preventing the performance degradation that occurs when multiple cores invalidate
//  * each other's cache lines.
//
//  * Without this padding, multiple cores updating their own counters would cause
//  * constant cache coherency traffic, significantly degrading performance.

const ThreadLocalData = struct {
    current_index: atomic.Value(usize) align(CACHE_LINE_SIZE),
    _padding: [CACHE_LINE_SIZE - @sizeOf(atomic.Value(usize))]u8 = undefined,
};

// * Thread Context Data Structure
// * ----------------------------
// * The ThreadContext struct holds all the data needed by each worker thread to perform
// * matrix multiplication. This includes the input matrices A and B, the output matrix C,
// * the dimensions of the matrices, and the tile sizes for tiling the computation.
// * The shared_counter field is a pointer to the shared atomic counter used for work distribution.
// * This design ensures that each thread has its own copy of the data it needs to perform
// * the computation, reducing contention and improving cache locality.
const ThreadContext = struct {
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
};

fn workerThread(ctx: ThreadContext) void {
    @setFloatMode(.optimized);
    @setRuntimeSafety(false);

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

//  * Tiling Approach
//  * --------------
//  * This is the heart of the optimized GEMM implementation, containing:
//  * 1. LOCAL MEMORY BUFFERS: A_local and B_local store tile data in stack memory
//  *    for maximum cache efficiency
//  * 2. PRECISION CONVERSION: Input f16 values are converted to f32 for computation
//  * 3. VECTORIZED COMPUTATION: 8x8 mini-matrix multiplications using SIMD instructions
//  * 4. EDGE CASE HANDLING: Special code paths for handling matrix edges

//  * Tile Size Selection
//  * ------------------
//  * The tile size is a critical parameter that balances computation and memory access.
//  * Larger tiles increase computation per memory access but risk cache eviction, while
//  * smaller tiles don't amortize function call overhead effectively. The multiple-of-8
//  * alignment supports the SIMD vectorization strategy.
//  *
//  * After experimenting with various sizes, I settled on a tile size of 160 (rounded to a
//  * multiple of 8). This size works well with typical L2/L3 cache sizes on modern CPUs.
//  * Larger tiles would increase computation per memory access but risk cache eviction,
//  * while smaller tiles don't amortize function call overhead effectively. The multiple-of-8
//  * alignment supports the SIMD vectorization strategy.

//  * Memory Access Optimization
//  * ------------------------
//  * For each tile, we create aligned local buffers that serve multiple purposes:
//  * they convert from f16 to f32, create cache-friendly access patterns, and ensure proper
//  * alignment for SIMD operations. This attention to memory layout helps the hardware
//  * prefetcher predict and fetch data effectively.
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
    @setFloatMode(.optimized);
    @setRuntimeSafety(false);

    var A_local: [T][T]f32 align(32) = undefined;
    var B_local: [T][T]f32 align(32) = undefined;

    const k_size = k_end - k_start;
    const i_size = i_end - i_start;
    const j_size = j_end - j_start;

    // * Mixed Precision Input Processing
    // * -------------------------------
    // * The input matrices A and B are stored in f16 format, but the computation is performed
    // * in f32. This approach reduces memory bandwidth requirements by 50% while maintaining
    // * computational accuracy. The conversion from f16 to f32 happens in the tile computation
    // * loop, where the data is loaded into aligned local buffers. This approach minimizes the
    // * performance impact of the conversion and ensures that the main computation loop operates
    // * on f32 data.
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

    // * Blocked Matrix Multiplication
    // * -----------------------------
    //  * The implementation uses 8-wide SIMD operations (Vec8f) which map to 256-bit AVX2
    //  * registers on x86-64 architectures. The 8×8 micro-kernel is designed to maximize
    //  * register usage without spilling to stack, given the 16 vector registers typically
    //  * available. This block size provides good computational density while staying within
    //  * hardware constraints.
    //  *
    // * The core computation loop performs 8x8 matrix multiplication using SIMD instructions.
    // * The loop is unrolled to eliminate branches and maximize computational throughput.
    // * The inner loop broadcasts elements from A across vector registers and multiplies them
    // * with elements from B, accumulating the results in a local buffer. This approach maximizes
    // * computational density and minimizes memory access patterns.
    var i: usize = 0;
    while (i + 8 <= i_size) : (i += 8) {
        var j: usize = 0;
        while (j + 8 <= j_size) : (j += 8) {

            //  * REGISTER BLOCKING
            //  * ---------------
            //  * This 8x8 accumulator array is kept in registers during computation.
            //  * Register blocking is crucial for:
            //  * - Minimizing memory access during computation
            //  * - Increasing arithmetic intensity (ops per memory access)
            //  * - Enabling instruction-level parallelism
            //  *
            //  * The compiler will try to map these values to CPU registers
            //  * for maximum performance.
            var acc: [8][8]f32 align(32) = [_][8]f32{[_]f32{0} ** 8} ** 8;

            for (0..k_size) |k| {

                // * SIMD Vector Load
                // * ---------------
                // * Load 8 values from A and B into SIMD vectors for computation.
                // * This approach maximizes computational throughput by leveraging
                // * the full width of the SIMD unit.

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

                //  * Outer product with SIMD
                //  * ---------------------------------
                //  * This loop computes the outer product of a_vec and b_vec,
                //  * adding to the accumulated results.
                //  *
                //  * For each iteration:
                //  * 1. Broadcast a single element from a_vec to all lanes
                //  * 2. Multiply this broadcast with all elements of b_vec
                //  * 3. Add to accumulator using Fused Multiply-Add (FMA)
                //  *
                //  * This pattern is highly efficient because:
                //  * - It maximizes data reuse (each loaded value is used multiple times)
                //  * - It enables FMA instructions for 2x theoretical throughput
                //  * - It has high instruction-level parallelism
                //  *
                //  * The 'inline' keyword ensures this loop is fully unrolled during compilation.

                inline for (0..8) |bi| {
                    const a_broadcast: Vec8f = @splat(a_vec[bi]);

                    const c_vec = Vec8f{
                        acc[bi][0], acc[bi][1], acc[bi][2], acc[bi][3],
                        acc[bi][4], acc[bi][5], acc[bi][6], acc[bi][7],
                    };
                    //  * FMA Operation
                    //  * --------------------------------
                    //  * This single instruction performs: result = a*b + c
                    //  *
                    //  * FMA has two major benefits:
                    //  * 1. Double theoretical throughput (2 arithmetic ops in 1 instruction)
                    //  * 2. Higher precision (single rounding step instead of two)
                    //  *
                    //  * Modern CPUs (AVX2+FMA) can execute one FMA per cycle per execution unit,
                    //  * delivering up to 16 floating-point operations per cycle (8 values * 2 ops).
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

        //  * EDGE CASE HANDLING: REMAINDER COLUMNS
        //  * -----------------------------------
        //  * Handle remaining columns that don't fill a complete 8-width vector.
        //  *
        //  * This scalar implementation ensures correctness for matrices whose
        //  * dimensions aren't multiples of 8, without compromising the performance
        //  * of the main vectorized path.

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

//  * Mixed Precision Strategy
//  * -----------------------
//  * I chose to store matrices in f16 but perform computations in f32. This approach reduces
//  * memory bandwidth requirements by 50% while maintaining computational accuracy. For matrix
//  * multiplication, which is often memory-bound, this tradeoff significantly improves
//  * performance on most hardware. The conversion from f16 to f32 happens in the tile computation
//  * loop, where the data is loaded into aligned local buffers. This approach minimizes the
//  * performance impact of the conversion and ensures that the main computation loop operates
//  * on f32 data.
//  * The biggest advantage for us is that moondream's weights are stored in f16, so we can use
//  * this function to multiply them with f16 inputs.

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

    const WorkerFn = struct {
        fn worker(ctx: ThreadContext) void {
            workerThread(ctx);
        }
    };

    for (0..num_threads) |_| {
        try thread_pool.append(try std.Thread.spawn(.{}, WorkerFn.worker, .{context}));
    }

    for (thread_pool.items) |thread| thread.join();

    return result;
}

//  * Performance Characteristics
//  * -------------------------
//  * This implementation performs best with:
//  * - Matrices large enough to benefit from tiling
//  * - Multi-core CPUs with good SIMD support
//  * - Scenarios where memory bandwidth is a limiting factor
//  *
//  * It may be less optimal for very small matrices where threading overhead dominates,
//  * or on hardware without effective SIMD capabilities.
//  *
//  * The strength of this implementation comes from how these optimizations work together
//  * across the entire memory and computation hierarchy - reducing bandwidth requirements,
//  * maximizing cache usage, utilizing SIMD parallelism, and keeping all cores productive
//  * with balanced work distribution.
//  */

//// benchmarking code ////

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
