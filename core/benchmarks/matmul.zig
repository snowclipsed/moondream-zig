const std = @import("std");
const openblas = @cImport({
    @cInclude("cblas.h");
    @cInclude("openblas_config.h");
});

// 8 x 64
// 16 x 64
//
// Configuration
const T: usize = 64; // Tile size (adjust as necessary)
const V: usize = 32; // Vector size (adjust as necessary)

pub fn tiledMatMul(allocator: std.mem.Allocator, A: []const f32, B: []const f32, C: []f32, M: usize, N: usize, K: usize) !void {
    const num_threads = try std.Thread.getCpuCount();

    // Calculate the number of tiles in each dimension
    const tiles_M = (M + T - 1) / T;
    const tiles_N = (N + T - 1) / T;
    // const tiles_K = (K + T - 1) / T;

    // Create a queue of work items
    var work_queue = std.ArrayList(WorkItem).init(allocator);
    defer work_queue.deinit();

    // Populate the work queue
    for (0..tiles_M) |i| {
        for (0..tiles_N) |j| {
            try work_queue.append(.{ .i = i, .j = j });
        }
    }

    // Shuffle the work queue for better load balancing
    var rng = std.rand.DefaultPrng.init(@intCast(std.time.milliTimestamp()));
    rng.random().shuffle(WorkItem, work_queue.items);

    // Create thread pool
    var thread_pool = try std.ArrayList(std.Thread).initCapacity(allocator, num_threads);
    defer thread_pool.deinit();

    // Shared context for all threads
    var context = ThreadContext{
        .A = A,
        .B = B,
        .C = C,
        .M = M,
        .N = N,
        .K = K,
        .work_queue = &work_queue,
        .mutex = std.Thread.Mutex{},
    };

    // Spawn threads
    for (0..num_threads) |_| {
        try thread_pool.append(try std.Thread.spawn(.{}, workerThread, .{&context}));
    }

    // Wait for all threads to complete
    for (thread_pool.items) |thread| {
        thread.join();
    }
}

const WorkItem = struct {
    i: usize,
    j: usize,
};

const ThreadContext = struct {
    A: []const f32,
    B: []const f32,
    C: []f32,
    M: usize,
    N: usize,
    K: usize,
    work_queue: *std.ArrayList(WorkItem),
    mutex: std.Thread.Mutex,
};

fn workerThread(context: *ThreadContext) void {
    while (true) {
        // Get next work item
        context.mutex.lock();
        const work_item = if (context.work_queue.popOrNull()) |item| item else {
            context.mutex.unlock();
            break;
        };
        context.mutex.unlock();

        // Process the tile
        const i_start = work_item.i * T;
        const j_start = work_item.j * T;
        const i_end = @min(i_start + T, context.M);
        const j_end = @min(j_start + T, context.N);

        var local_C: [T][T]f32 = [_][T]f32{[_]f32{0} ** T} ** T;

        var k: usize = 0;
        while (k < context.K) : (k += T) {
            const k_end = @min(k + T, context.K);
            tiledMultiplyKernel(context.A, context.B, &local_C, context.N, context.K, i_start, j_start, k, i_end, j_end, k_end);
        }

        // Accumulate results to global C
        for (i_start..i_end) |i| {
            for (j_start..j_end) |j| {
                context.C[i * context.N + j] += local_C[i - i_start][j - j_start];
            }
        }
    }
}

fn tiledMultiplyKernel(A: []const f32, B: []const f32, local_C: *[T][T]f32, N: usize, K: usize, i_start: usize, j_start: usize, k_start: usize, i_end: usize, j_end: usize, k_end: usize) void {
    var A_local: [T][T]f32 = undefined;
    var B_local: [T][T]f32 = undefined;

    // Load A and B into local buffers
    for (0..T) |i| {
        for (0..T) |k| {
            if (i_start + i < i_end and k_start + k < k_end) {
                A_local[i][k] = A[(i_start + i) * K + (k_start + k)];
            } else {
                A_local[i][k] = 0;
            }
        }
    }

    for (0..T) |k| {
        for (0..T) |j| {
            if (k_start + k < k_end and j_start + j < j_end) {
                B_local[k][j] = B[(k_start + k) * N + (j_start + j)];
            } else {
                B_local[k][j] = 0;
            }
        }
    }

    // Compute tile
    var i: usize = 0;
    while (i < T) : (i += 1) {
        var j: usize = 0;
        while (j < T) : (j += V) {
            var vec_sum: @Vector(V, f32) = @splat(0);
            var k: usize = 0;
            while (k < T) : (k += 1) {
                const a_val = A_local[i][k];
                const a_vec = @as(@Vector(V, f32), @splat(a_val));
                const b_vec = blk: {
                    var temp: @Vector(V, f32) = undefined;
                    for (0..V) |idx| {
                        temp[idx] = B_local[k][j + idx];
                    }
                    break :blk temp;
                };
                vec_sum += a_vec * b_vec;
            }

            // Accumulate results to local_C
            for (0..V) |idx| {
                local_C[i][j + idx] += vec_sum[idx];
            }
        }
    }
}

pub fn matrixMultiplyOpenBlas(A: []const f32, B: []const f32, C: []f32, m: usize, n: usize, k: usize) void {
    // Ensure input matrices have correct dimensions
    std.debug.assert(A.len == m * k and B.len == k * n and C.len == m * n);

    // Set the number of threads for OpenBLAS
    const num_threads: c_int = 16; // Change this to the desired number of threads
    openblas.openblas_set_num_threads(num_threads);

    // Perform matrix multiplication using OpenBLAS
    openblas.cblas_sgemm(openblas.CblasRowMajor, // Matrix layout
        openblas.CblasNoTrans, // Don't transpose matrix A
        openblas.CblasNoTrans, // Don't transpose matrix B
        @intCast(m), // Number of rows in A and C
        @intCast(n), // Number of columns in B and C
        @intCast(k), // Number of columns in A and rows in B
        1.0, // Alpha scaling factor
        A.ptr, // Pointer to matrix A
        @intCast(k), // Leading dimension of A
        B.ptr, // Pointer to matrix B
        @intCast(n), // Leading dimension of B
        0.0, // Beta scaling factor
        C.ptr, // Pointer to matrix C
        @intCast(n) // Leading dimension of C
    );
}

pub fn calculateGflops(allocator: std.mem.Allocator, M: usize, N: usize, K: usize, iterations: usize) !f64 {
    const A = try allocator.alloc(f32, M * K);
    defer allocator.free(A);
    const B = try allocator.alloc(f32, K * N);
    defer allocator.free(B);
    const C = try allocator.alloc(f32, M * N);
    defer allocator.free(C);

    // Uncomment for deterministic initialization
    // for (A, 0..) |*val, i| {
    //     val.* = @floatFromInt(i % 10);
    // }
    // for (B, 0..) |*val, i| {
    //     val.* = @floatFromInt((i + 1) % 10);
    // }

    // Random init
    var prng = std.rand.DefaultPrng.init(0);
    var random = prng.random();
    for (A) |*a| a.* = random.float(f32);
    for (B) |*b| b.* = random.float(f32);

    // Warmup run
    try tiledMatMul(allocator, A, B, C, M, N, K);

    // Allocate an array to store GFLOPS for each iteration
    var gflops_array = try allocator.alloc(f64, iterations);
    defer allocator.free(gflops_array);

    // Measure GFLOPS for each iteration
    for (0..iterations) |i| {
        var timer = try std.time.Timer.start();
        try tiledMatMul(allocator, A, B, C, M, N, K);
        const elapsed_ns = timer.read();

        // Calculate GFLOPS for this iteration
        const ops = 2 * M * N * K; // multiply-add is 2 operations
        const seconds = @as(f64, @floatFromInt(elapsed_ns)) / 1e9;
        gflops_array[i] = @as(f64, @floatFromInt(ops)) / seconds / 1e9;
        // std.debug.print("\n --- {} | GFLOPs/s = {d:.2}", .{ i, gflops_array[i] });
    }

    // Calculate average GFLOPS
    var total_gflops: f64 = 0;
    for (gflops_array) |gflops| {
        total_gflops += gflops;
    }
    const avg_gflops = total_gflops / @as(f64, @floatFromInt(iterations));

    return avg_gflops;
}

// Test function
test "tiledMatMul correctness" {
    const allocator = std.testing.allocator;

    const test_sizes = [_][3]usize{
        .{ 128, 128, 128 },
        .{ 100, 100, 100 },
        .{ 200, 150, 175 },
        .{ 32, 64, 48 },
        .{ 47, 34, 45 },
    };

    for (test_sizes) |size| {
        const M = size[0];
        const N = size[1];
        const K = size[2];

        const A = try allocator.alloc(f32, M * K);
        defer allocator.free(A);
        const B = try allocator.alloc(f32, K * N);
        defer allocator.free(B);
        const C = try allocator.alloc(f32, M * N);
        defer allocator.free(C);
        const C_ref = try allocator.alloc(f32, M * N);
        defer allocator.free(C_ref);

        // Initialize matrices
        for (A, 0..) |*val, i| {
            val.* = @floatFromInt(i % 10);
        }
        for (B, 0..) |*val, i| {
            val.* = @floatFromInt((i + 1) % 10);
        }
        @memset(C, 0);
        @memset(C_ref, 0);

        // Perform tiled matrix multiplication
        try tiledMatMul(allocator, A, B, C, M, N, K);

        // Perform reference matrix multiplication
        for (0..M) |i| {
            for (0..N) |j| {
                var sum: f32 = 0;
                for (0..K) |k| {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C_ref[i * N + j] = sum;
            }
        }

        // matrixMultiplyOpenBlas(A, B, C_ref, M, N, K);

        // Compare results
        for (C, C_ref) |c, c_ref| {
            try std.testing.expectApproxEqAbs(c, c_ref, 1e-10);
            try std.testing.expectEqual(c, c_ref);
        }

        std.debug.print("Test passed for size: M={}, N={}, K={}\n", .{ M, N, K });
    }
}

test "GFLOPS Benchmark" {
    const allocator = std.testing.allocator;

    const sizes = [_][3]usize{
        .{ 256, 256, 256 },
        .{ 512, 512, 512 },
        .{ 1024, 1024, 1024 },
        .{ 1024, 2048, 1024 },
        .{ 2048, 2048, 2048 },
        .{ 2048, 4096, 2048 },
        .{ 4096, 4096, 4096 },
        .{ 8192, 2048, 8192 },
        .{ 1152, 4304, 1152 },
        .{ 1, 2048, 51200 },
        // .{ 8192, 8192, 8192 },
        // .{ 8192, 16384, 8192 },
        // .{ 16384, 16384, 16384 },
        // .{ 16384, 32768, 16384 },
        // .{ 32768, 32768, 32768 },
        // .{ 32768, 65536, 32768 },
        // .{ 65536, 65536, 65536 },
    };

    const iterations = 5;

    for (sizes) |size| {
        const M = size[0];
        const N = size[1];
        const K = size[2];

        const gflops = try calculateGflops(allocator, M, N, K, iterations);

        std.debug.print("Matrix size: {}x{}x{}, GFLOPS: {d:.2}\n", .{ M, N, K, gflops });
    }
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const sizes = [_][3]usize{
        .{ 256, 256, 256 },
        .{ 512, 512, 512 },
        .{ 1024, 1024, 1024 },
        .{ 1024, 2048, 1024 },
        .{ 2048, 2048, 2048 },
        .{ 2048, 4096, 2048 },
        .{ 4096, 4096, 4096 },
        .{ 8192, 2048, 8192 },
        .{ 1152, 4304, 1152 },
        .{ 1, 2048, 51200 },
    };

    const iterations = 10;

    std.debug.print("T = {} \n V = {} \n", .{ T, V });

    const num_threads = try std.Thread.getCpuCount();
    std.debug.print("Number of threads = {}\n", .{num_threads});

    for (sizes) |size| {
        const M = size[0];
        const N = size[1];
        const K = size[2];

        const gflops = try calculateGflops(allocator, M, N, K, iterations);

        std.debug.print("Matrix size: {}x{}x{}, GFLOPS: {d:.2}\n", .{ M, N, K, gflops });
    }
}
