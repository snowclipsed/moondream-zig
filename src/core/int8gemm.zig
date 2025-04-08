const std = @import("std");
const Allocator = std.mem.Allocator;
const atomic = std.atomic;
const math = std.math;
const Tensor = @import("tensor.zig").Tensor;
const SlabReusingAllocator = @import("slab_reusing_allocator.zig").SlabReusingAllocator;
const time = std.time;
const testing = std.testing;

// PERFORMANCE TUNED PARAMETERS
pub const T: usize = 256; // Tile size (same as HGEMM for consistency)
const V: usize = 128; // Vector width for INT8 (4x wider than f32)
const CACHE_LINE_SIZE: usize = atomic.cache_line;
const CHUNK_SIZE: usize = 2; // Slightly larger chunks for better threading efficiency
const AVX2_ALIGNMENT = 32;
const Vec32i8 = @Vector(32, i8);
const Vec16i16 = @Vector(16, i16);
const Vec8i32 = @Vector(8, i32);
const Vec8f = @Vector(8, f32);

/// Quantization parameters for int8 GEMM
pub const QuantizationParams = struct {
    input_scale: f32, // Scale for input matrix
    input_zero_point: i8, // Zero point for input matrix
    weight_scale: f32, // Scale for weight matrix
    weight_zero_point: i8, // Zero point for weight matrix
    output_scale: f32, // Scale for output matrix
    output_zero_point: i8, // Zero point for output matrix
    output_min: i8, // Minimum output value
    output_max: i8, // Maximum output value
};

/// Thread-local data structure for work distribution
const ThreadLocalData = struct {
    current_index: atomic.Value(usize) align(CACHE_LINE_SIZE),
    padding: [CACHE_LINE_SIZE - @sizeOf(atomic.Value(usize))]u8 = undefined,
};

/// Thread context for int8 GEMM
const ThreadContext = struct {
    A: []const i8, // Input matrix A
    B: []const i8, // Input matrix B
    C: []i8, // Output matrix C
    M: usize, // Number of rows in A and C
    N: usize, // Number of columns in B and C
    K: usize, // Number of columns in A and rows in B
    tiles_M: usize, // Number of tiles in M dimension
    tiles_N: usize, // Number of tiles in N dimension
    total_tiles: usize, // Total number of tiles
    shared_counter: *ThreadLocalData, // Shared atomic counter for work distribution
    quant_params: QuantizationParams, // Quantization parameters
};

/// Packed tile structure for better cache locality and SIMD access
const PackedTile = struct {
    a_tile: [T][T]i8 align(AVX2_ALIGNMENT),
    b_tile: [T][T]i8 align(AVX2_ALIGNMENT),
};

/// Worker thread function for int8 GEMM - optimized for AVX2 and cache efficiency
fn workerThread(ctx: ThreadContext) void {
    @setFloatMode(.optimized);
    @setRuntimeSafety(false);

    // Local accumulator buffer (using int32 for accumulation)
    var local_C: [T][T]i32 align(AVX2_ALIGNMENT) = undefined;

    // Pre-allocate packed tile buffers for better cache behavior
    var packed_tiles = PackedTile{
        .a_tile = undefined,
        .b_tile = undefined,
    };

    // Pre-compute scaling factors for requantization
    const combined_scale = ctx.quant_params.input_scale *
        ctx.quant_params.weight_scale /
        ctx.quant_params.output_scale;

    // Main loop - process tiles
    while (true) {
        // Get next chunk of work using atomic counter
        const start_idx = ctx.shared_counter.current_index.fetchAdd(CHUNK_SIZE, .seq_cst);
        if (start_idx >= ctx.total_tiles) break;

        const end_idx = @min(start_idx + CHUNK_SIZE, ctx.total_tiles);
        var idx = start_idx;

        while (idx < end_idx) : (idx += 1) {
            // Calculate tile indices
            const i = idx / ctx.tiles_N;
            const j = idx % ctx.tiles_N;

            // Calculate tile bounds
            const i_start = i * T;
            const j_start = j * T;
            const i_end = @min(i_start + T, ctx.M);
            const j_end = @min(j_start + T, ctx.N);

            // Initialize accumulator to zero
            for (0..T) |x| {
                @memset(&local_C[x], 0);
            }

            // Process tiles along K dimension
            var k: usize = 0;
            while (k < ctx.K) : (k += T) {
                const k_end = @min(k + T, ctx.K);

                // Pack A and B tiles for better cache locality
                packTiles(ctx, &packed_tiles, i_start, j_start, k, i_end, j_end, k_end);

                // Compute current tile
                computeTile(ctx, &local_C, &packed_tiles, i_end - i_start, j_end - j_start, k_end - k);
            }

            // Directly requantize and store results without using helper functions
            // This ensures we use the correct indices
            const i_size = i_end - i_start;
            const j_size = j_end - j_start;

            // Output parameters
            const output_zp = ctx.quant_params.output_zero_point;
            const output_min = ctx.quant_params.output_min;
            const output_max = ctx.quant_params.output_max;

            // Process all values directly
            for (0..i_size) |i_local| {
                const row_offset = (i_start + i_local) * ctx.N;

                for (0..j_size) |j_local| {
                    // Get accumulated value
                    const int32_val = local_C[i_local][j_local];

                    // Requantize
                    const float_val = @as(f32, @floatFromInt(int32_val));
                    const scaled_val = float_val * combined_scale;
                    const rounded_val = @round(scaled_val) + @as(f32, @floatFromInt(output_zp));

                    // Clamp and convert
                    const clamped_val = @min(@max(rounded_val, @as(f32, @floatFromInt(output_min))), @as(f32, @floatFromInt(output_max)));

                    ctx.C[row_offset + j_start + j_local] = @intFromFloat(clamped_val);
                }
            }
        }
    }
}

/// Pack tiles for better cache locality and SIMD access
fn packTiles(
    ctx: ThreadContext,
    packed_tiles: *PackedTile,
    i_start: usize,
    j_start: usize,
    k_start: usize,
    i_end: usize,
    j_end: usize,
    k_end: usize,
) void {
    @setFloatMode(.optimized);
    @setRuntimeSafety(false);

    const k_size = k_end - k_start;
    const i_size = i_end - i_start;
    const j_size = j_end - j_start;

    // Pack A tile (transposed for better access patterns)
    var i: usize = 0;
    while (i < i_size) : (i += 8) {
        const rows_remaining = i_size - i;
        const rows_to_process = @min(rows_remaining, 8);

        for (0..k_size) |k| {
            // Load 8 rows (or fewer for the last block)
            for (0..rows_to_process) |r| {
                const idx = (i_start + i + r) * ctx.K + k_start + k;
                packed_tiles.a_tile[k][i + r] = ctx.A[idx]; // Note the transpose
            }

            // Pad with zeros if needed
            for (rows_to_process..8) |r| {
                if (i + r < T) {
                    packed_tiles.a_tile[k][i + r] = 0;
                }
            }
        }
    }

    // Pack B tile for better cache locality
    for (0..k_size) |k| {
        const row_offset = (k_start + k) * ctx.N;

        var j: usize = 0;
        while (j < j_size) : (j += 32) {
            const cols_remaining = j_size - j;
            const cols_to_process = @min(cols_remaining, 32);

            // Load 32 columns (or fewer for the last block)
            for (0..cols_to_process) |c| {
                const idx = row_offset + j_start + j + c;
                packed_tiles.b_tile[k][j + c] = ctx.B[idx];
            }

            // Pad with zeros if needed
            for (cols_to_process..32) |c| {
                if (j + c < T) {
                    packed_tiles.b_tile[k][j + c] = 0;
                }
            }
        }
    }
}

/// Compute tile function for int8 GEMM - optimized with AVX2 SIMD
fn computeTile(
    ctx: ThreadContext,
    local_C: *[T][T]i32,
    packed_tiles: *const PackedTile,
    i_size: usize,
    j_size: usize,
    k_size: usize,
) void {
    @setFloatMode(.optimized);
    @setRuntimeSafety(false);

    // Zero points for input and weights
    const input_zp = ctx.quant_params.input_zero_point;
    const weight_zp = ctx.quant_params.weight_zero_point;

    // Process 8 rows at a time
    var i: usize = 0;
    while (i + 8 <= i_size) : (i += 8) {
        // Process columns in chunks of 32 (AVX2 can handle 32 int8 values)
        var j: usize = 0;
        while (j + 32 <= j_size) : (j += 32) {
            // Process this 8x32 block
            processBlock_8x32(local_C, packed_tiles, i, j, k_size, input_zp, weight_zp);
        }

        // Handle remaining columns
        if (j < j_size) {
            processRemainingColumns(local_C, packed_tiles, i, j, j_size - j, k_size, input_zp, weight_zp);
        }
    }

    // Handle remaining rows
    if (i < i_size) {
        processRemainingRows(local_C, packed_tiles, i, i_size - i, j_size, k_size, input_zp, weight_zp);
    }
}

/// Process an 8x32 block with SIMD optimizations
fn processBlock_8x32(
    local_C: *[T][T]i32,
    packed_tiles: *const PackedTile,
    i_offset: usize,
    j_offset: usize,
    k_size: usize,
    input_zp: i8,
    weight_zp: i8,
) void {
    @setFloatMode(.optimized);
    @setRuntimeSafety(false);

    // Process k dimension with aggressive optimization
    for (0..k_size) |k| {
        // Load 8 values from A (transposed during packing)
        var a_vals: [8]i16 = undefined;
        for (0..8) |r| {
            a_vals[r] = @as(i16, packed_tiles.a_tile[k][i_offset + r]) - input_zp;
        }

        // Process in chunks of 16 for better vectorization

        // First 16 columns
        {
            // Load 16 values from B
            var b_vals: [16]i16 = undefined;
            for (0..16) |c| {
                b_vals[c] = @as(i16, packed_tiles.b_tile[k][j_offset + c]) - weight_zp;
            }

            // Compute 8x16 matrix multiplication
            for (0..8) |r| {
                const a_val = a_vals[r];
                for (0..16) |c| {
                    local_C[i_offset + r][j_offset + c] += @as(i32, a_val) * @as(i32, b_vals[c]);
                }
            }
        }

        // Second 16 columns
        {
            // Load next 16 values from B
            var b_vals: [16]i16 = undefined;
            for (0..16) |c| {
                b_vals[c] = @as(i16, packed_tiles.b_tile[k][j_offset + 16 + c]) - weight_zp;
            }

            // Compute 8x16 matrix multiplication
            for (0..8) |r| {
                const a_val = a_vals[r];
                for (0..16) |c| {
                    local_C[i_offset + r][j_offset + 16 + c] += @as(i32, a_val) * @as(i32, b_vals[c]);
                }
            }
        }
    }
}

/// Process remaining columns (less than 32)
fn processRemainingColumns(
    local_C: *[T][T]i32,
    packed_tiles: *const PackedTile,
    i_offset: usize,
    j_offset: usize,
    j_count: usize,
    k_size: usize,
    input_zp: i8,
    weight_zp: i8,
) void {
    @setFloatMode(.optimized);
    @setRuntimeSafety(false);

    // Process k dimension with optimized handling
    for (0..k_size) |k| {
        // Load 8 values from A (transposed during packing)
        var a_vals: [8]i16 = undefined;
        for (0..8) |r| {
            a_vals[r] = @as(i16, packed_tiles.a_tile[k][i_offset + r]) - input_zp;
        }

        // Load and process remaining columns
        var b_vals: [32]i16 = undefined; // Use fixed size for better compiler optimization
        const effective_j_count = @min(j_count, 32);

        // Load all columns at once
        for (0..effective_j_count) |c| {
            b_vals[c] = @as(i16, packed_tiles.b_tile[k][j_offset + c]) - weight_zp;
        }

        // Process each row
        for (0..8) |r| {
            const a_val = a_vals[r];

            // Process all columns
            for (0..effective_j_count) |c| {
                local_C[i_offset + r][j_offset + c] += @as(i32, a_val) * @as(i32, b_vals[c]);
            }
        }
    }
}

/// Process remaining rows (less than 8)
fn processRemainingRows(
    local_C: *[T][T]i32,
    packed_tiles: *const PackedTile,
    i_offset: usize,
    i_count: usize,
    j_size: usize,
    k_size: usize,
    input_zp: i8,
    weight_zp: i8,
) void {
    @setFloatMode(.optimized);
    @setRuntimeSafety(false);

    // Process columns in chunks of 32 for better SIMD utilization
    var j: usize = 0;
    while (j + 32 <= j_size) : (j += 32) {
        // Process each k value
        for (0..k_size) |k| {
            // Load i_count values from A
            var a_vals: [8]i16 = undefined; // Using 8 as max possible i_count
            const effective_i_count = @min(i_count, 8);

            for (0..effective_i_count) |r| {
                a_vals[r] = @as(i16, packed_tiles.a_tile[k][i_offset + r]) - input_zp;
            }

            // Process in chunks of 16 for better vectorization

            // First 16 columns
            {
                // Load 16 values from B
                var b_vals: [16]i16 = undefined;
                for (0..16) |c| {
                    b_vals[c] = @as(i16, packed_tiles.b_tile[k][j + c]) - weight_zp;
                }

                // Process each row
                for (0..effective_i_count) |r| {
                    const a_val = a_vals[r];
                    for (0..16) |c| {
                        local_C[i_offset + r][j + c] += @as(i32, a_val) * @as(i32, b_vals[c]);
                    }
                }
            }

            // Second 16 columns
            {
                // Load 16 values from B
                var b_vals: [16]i16 = undefined;
                for (0..16) |c| {
                    b_vals[c] = @as(i16, packed_tiles.b_tile[k][j + 16 + c]) - weight_zp;
                }

                // Process each row
                for (0..effective_i_count) |r| {
                    const a_val = a_vals[r];
                    for (0..16) |c| {
                        local_C[i_offset + r][j + 16 + c] += @as(i32, a_val) * @as(i32, b_vals[c]);
                    }
                }
            }
        }
    }

    // Handle remaining columns
    if (j < j_size) {
        const j_count = j_size - j;

        // Process each k value
        for (0..k_size) |k| {
            // Load i_count values from A
            var a_vals: [8]i16 = undefined; // Using 8 as max possible i_count
            const effective_i_count = @min(i_count, 8);

            for (0..effective_i_count) |r| {
                a_vals[r] = @as(i16, packed_tiles.a_tile[k][i_offset + r]) - input_zp;
            }

            // Load j_count values from B
            var b_vals: [32]i16 = undefined; // Fixed size for better optimization
            const effective_j_count = @min(j_count, 32);

            for (0..effective_j_count) |c| {
                b_vals[c] = @as(i16, packed_tiles.b_tile[k][j + c]) - weight_zp;
            }

            // Process each row
            for (0..effective_i_count) |r| {
                const a_val = a_vals[r];

                // Process all columns
                for (0..effective_j_count) |c| {
                    local_C[i_offset + r][j + c] += @as(i32, a_val) * @as(i32, b_vals[c]);
                }
            }
        }
    }
}

/// Optimized requantization function that efficiently converts int32 to int8
fn requantizeAndStore(
    ctx: ThreadContext,
    local_C: *[T][T]i32,
    i_start: usize,
    j_start: usize,
    i_end: usize,
    j_end: usize,
    combined_scale: f32,
) void {
    @setFloatMode(.optimized);
    @setRuntimeSafety(false);

    const i_size = i_end - i_start;
    const j_size = j_end - j_start;

    // Output parameters
    const output_zp = ctx.quant_params.output_zero_point;
    const output_min = ctx.quant_params.output_min;
    const output_max = ctx.quant_params.output_max;

    // Process 8 rows at a time
    var i: usize = 0;
    while (i + 8 <= i_size) : (i += 8) {
        // Process columns in chunks of 32
        var j: usize = 0;

        while (j + 32 <= j_size) : (j += 32) {
            // Requantize and store this 8x32 block
            requantizeBlock_8x32(ctx.C, local_C, i_start + i, j_start + j, ctx.N, combined_scale, output_zp, output_min, output_max);
        }

        // Handle remaining columns
        if (j < j_size) {
            requantizeRemainingColumns(ctx.C, local_C, i_start + i, j_start + j, ctx.N, j_size - j, combined_scale, output_zp, output_min, output_max);
        }
    }

    // Handle remaining rows
    if (i < i_size) {
        requantizeRemainingRows(ctx.C, local_C, i_start + i, j_start, ctx.N, i_size - i, j_size, combined_scale, output_zp, output_min, output_max);
    }
}

/// Requantize an 8x32 block with efficient SIMD
fn requantizeBlock_8x32(
    output: []i8,
    local_C: *[T][T]i32,
    i_start: usize,
    j_start: usize,
    row_stride: usize,
    scale: f32,
    zero_point: i8,
    output_min: i8,
    output_max: i8,
) void {
    @setFloatMode(.optimized);
    @setRuntimeSafety(false);

    // Temporary buffer for storing results
    var temp_buffer: [8][32]i8 align(AVX2_ALIGNMENT) = undefined;

    // Requantize values
    for (0..8) |r| {
        for (0..32) |c| {
            // Get accumulated value - use correct offset into local_C
            const int32_val = local_C[r][c];

            // Convert to float, scale, add zero point
            const float_val = @as(f32, @floatFromInt(int32_val));
            const scaled_val = float_val * scale;
            const rounded_val = @round(scaled_val) + @as(f32, @floatFromInt(zero_point));

            // Clamp to output range and convert to int8
            const clamped_val = @min(@max(rounded_val, @as(f32, @floatFromInt(output_min))), @as(f32, @floatFromInt(output_max)));

            temp_buffer[r][c] = @intFromFloat(clamped_val);
        }
    }

    // Store results in output matrix
    for (0..8) |r| {
        const row_offset = (i_start + r) * row_stride;

        // Copy entire row at once for better memory performance
        @memcpy(output[row_offset + j_start .. row_offset + j_start + 32], &temp_buffer[r]);
    }
}

/// Requantize remaining columns (less than 32)
fn requantizeRemainingColumns(
    output: []i8,
    local_C: *[T][T]i32,
    i_start: usize,
    j_start: usize,
    row_stride: usize,
    j_count: usize,
    scale: f32,
    zero_point: i8,
    output_min: i8,
    output_max: i8,
) void {
    @setFloatMode(.optimized);
    @setRuntimeSafety(false);

    for (0..8) |r| {
        const row_offset = (i_start + r) * row_stride;

        for (0..j_count) |c| {
            // Get accumulated value
            const int32_val = local_C[r][j_start + c];

            // Convert to float, scale, add zero point
            const float_val = @as(f32, @floatFromInt(int32_val));
            const scaled_val = float_val * scale;
            const rounded_val = @round(scaled_val) + @as(f32, @floatFromInt(zero_point));

            // Clamp to output range and convert to int8
            const clamped_val = @min(@max(rounded_val, @as(f32, @floatFromInt(output_min))), @as(f32, @floatFromInt(output_max)));

            output[row_offset + j_start + c] = @intFromFloat(clamped_val);
        }
    }
}

/// Requantize remaining rows (less than 8)
fn requantizeRemainingRows(
    output: []i8,
    local_C: *[T][T]i32,
    i_start: usize,
    j_start: usize,
    row_stride: usize,
    i_count: usize,
    j_size: usize,
    scale: f32,
    zero_point: i8,
    output_min: i8,
    output_max: i8,
) void {
    @setFloatMode(.optimized);
    @setRuntimeSafety(false);

    // Process remaining rows
    for (0..i_count) |r| {
        const row_offset = (i_start + r) * row_stride;

        // Process all columns for this row
        for (0..j_size) |c| {
            // Get accumulated value
            const int32_val = local_C[r][c];

            // Convert to float, scale, add zero point
            const float_val = @as(f32, @floatFromInt(int32_val));
            const scaled_val = float_val * scale;
            const rounded_val = @round(scaled_val) + @as(f32, @floatFromInt(zero_point));

            // Clamp to output range and convert to int8
            const clamped_val = @min(@max(rounded_val, @as(f32, @floatFromInt(output_min))), @as(f32, @floatFromInt(output_max)));

            output[row_offset + j_start + c] = @intFromFloat(clamped_val);
        }
    }
}

/// Main int8 matrix multiplication function
pub fn matmul_int8(a: Tensor(i8), b: Tensor(i8), quant_params: QuantizationParams, allocator: Allocator) !Tensor(i8) {
    @setFloatMode(.optimized);
    @setRuntimeSafety(false);

    const A_shape = a.shape;
    const B_shape = b.shape;

    // Check dimensions
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

    // Create result tensor
    var result = try Tensor(i8).initWithoutMemset(allocator, &[_]usize{ M, N });
    errdefer result.deinit();

    // Get slices to the underlying data
    const A_data = a.getSlice();
    const B_data = b.getSlice();
    const C_data = result.getSlice();

    // Calculate tile dimensions
    const tiles_M = (M + T - 1) / T;
    const tiles_N = (N + T - 1) / T;
    const total_tiles = tiles_M * tiles_N;

    // Initialize shared atomic counter for work distribution
    var shared_data = ThreadLocalData{ .current_index = atomic.Value(usize).init(0) };

    // Setup thread pool for parallel execution
    const num_threads = try std.Thread.getCpuCount();
    var thread_pool = try std.ArrayList(std.Thread).initCapacity(allocator, num_threads);
    defer thread_pool.deinit();

    // Create thread context with all necessary data
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
        .quant_params = quant_params,
    };

    // Worker function for thread spawning
    const WorkerFn = struct {
        fn worker(ctx: ThreadContext) void {
            workerThread(ctx);
        }
    };

    // Spawn worker threads
    for (0..num_threads) |_| {
        try thread_pool.append(try std.Thread.spawn(.{}, WorkerFn.worker, .{context}));
    }

    // Wait for all threads to complete
    for (thread_pool.items) |thread| thread.join();

    return result;
}

/// Benchmark function for int8 matrix multiplication
fn benchmarkInt8MatMul(allocator: std.mem.Allocator, M: usize, N: usize, K: usize, num_runs: usize) !f64 {
    // Create quantization parameters for benchmarking
    const quant_params = QuantizationParams{
        .input_scale = 0.1,
        .input_zero_point = 0,
        .weight_scale = 0.1,
        .weight_zero_point = 0,
        .output_scale = 0.01,
        .output_zero_point = 0,
        .output_min = -127,
        .output_max = 127,
    };

    // Create input tensors
    var A = try Tensor(i8).init(allocator, &[_]usize{ M, K });
    defer A.deinit();
    var B = try Tensor(i8).init(allocator, &[_]usize{ K, N });
    defer B.deinit();

    // Initialize with pseudo-random values
    var rng = std.rand.DefaultPrng.init(42);
    const random = rng.random();

    for (A.data) |*val| {
        val.* = @intCast(@mod(random.int(i16), 255) - 127); // Range -127 to 127
    }

    for (B.data) |*val| {
        val.* = @intCast(@mod(random.int(i16), 255) - 127); // Range -127 to 127
    }

    // Warmup run
    var warmup_C = try matmul_int8(A, B, quant_params, allocator);
    warmup_C.deinit();

    // Timing runs
    var total_time: u64 = 0;
    var timer = try time.Timer.start();

    for (0..num_runs) |_| {
        timer.reset();
        var C = try matmul_int8(A, B, quant_params, allocator);
        total_time += timer.read();
        C.deinit();
    }

    const avg_nanos = @as(f64, @floatFromInt(total_time)) / @as(f64, @floatFromInt(num_runs));
    const avg_secs = avg_nanos / 1e9;

    // Calculate GOPS (Giga Operations Per Second)
    // For matrix multiplication, num_operations = 2 * M * N * K
    const ops = 2 * M * N * K;
    const gops = (@as(f64, @floatFromInt(ops)) / avg_secs) / 1e9;

    return gops;
}

/// Helper function to print benchmark results
fn printBenchmarkResults(label: []const u8, gops: f64, M: usize, N: usize, K: usize) void {
    std.debug.print("{s:<20} Size: [{d:>4}x{d:>4}x{d:>4}] Performance: {d:>6.2} GOPS\n", .{ label, M, N, K, gops });
}

/// Test function to verify correctness of int8 GEMM implementation
fn testInt8MatMulCorrectness(allocator: std.mem.Allocator) !bool {
    const M = 16;
    const N = 16;
    const K = 16;

    // Create quantization parameters for testing
    const quant_params = QuantizationParams{
        .input_scale = 1.0,
        .input_zero_point = 0,
        .weight_scale = 1.0,
        .weight_zero_point = 0,
        .output_scale = 1.0,
        .output_zero_point = 0,
        .output_min = -127,
        .output_max = 127,
    };

    // Create input tensors with known values
    var A = try Tensor(i8).init(allocator, &[_]usize{ M, K });
    defer A.deinit();
    var B = try Tensor(i8).init(allocator, &[_]usize{ K, N });
    defer B.deinit();

    // Initialize with simple pattern (identity matrix with some variations)
    for (0..M) |i| {
        for (0..K) |j| {
            A.data[i * K + j] = if (i == j) 1 else 0;
        }
    }

    for (0..K) |i| {
        for (0..N) |j| {
            B.data[i * N + j] = if (i == j) 2 else 0;
        }
    }

    // Compute result using our implementation
    var C = try matmul_int8(A, B, quant_params, allocator);
    defer C.deinit();

    // Verify result (should be diagonal matrix with value 2)
    var is_correct = true;
    for (0..M) |i| {
        for (0..N) |j| {
            var expected: i8 = 0;
            if (i == j) {
                expected = 2;
            }
            if (C.data[i * N + j] != expected) {
                std.debug.print("Error at [{d},{d}]: expected {d}, got {d}\n", .{ i, j, expected, C.data[i * N + j] });
                is_correct = false;
            }
        }
    }

    return is_correct;
}

/// Main function to benchmark int8 GEMM
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const gpa_allocator = gpa.allocator();

    var slab_reusing_allocator = SlabReusingAllocator(100).init(gpa_allocator);
    defer slab_reusing_allocator.deinit();
    const allocator = slab_reusing_allocator.allocator();

    // Test correctness
    std.debug.print("\nInt8 Matrix Multiplication Correctness Test\n", .{});
    std.debug.print("=======================================\n", .{});
    const is_correct = try testInt8MatMulCorrectness(allocator);
    std.debug.print("Test result: {s}\n\n", .{if (is_correct) "PASSED" else "FAILED"});

    if (!is_correct) {
        return error.TestFailed;
    }

    // Benchmark different matrix sizes
    const num_runs = 5; // Number of timing runs for each size

    // Test different matrix sizes
    const sizes = [_][3]usize{
        .{ 512, 512, 512 }, // Square
        .{ 1024, 1024, 1024 }, // Larger square
        .{ 768, 512, 256 }, // Rectangular
        .{ 2048, 1024, 512 }, // Large rectangular
        .{ 168, 168, 168 }, // Tile size
        .{ 2048, 2048, 2048 }, // Large square
        .{ 4096, 4096, 4096 }, // Very large square
    };

    std.debug.print("Int8 Matrix Multiplication Performance Benchmark\n", .{});
    std.debug.print("=========================================\n\n", .{});

    for (sizes) |size| {
        const M = size[0];
        const N = size[1];
        const K = size[2];

        const gops = try benchmarkInt8MatMul(allocator, M, N, K, num_runs);

        var label_buf: [32]u8 = undefined;
        const label = try std.fmt.bufPrint(&label_buf, "Case {d}x{d}x{d}", .{ M, N, K });

        printBenchmarkResults(label, gops, M, N, K);
    }

    // Test tile size multiples
    std.debug.print("\nTile Size Multiple Tests:\n", .{});
    std.debug.print("======================\n\n", .{});

    const tile_multiples = [_]usize{ 1, 2, 3, 4 };

    for (tile_multiples) |multiple| {
        const size = T * multiple;

        const gops = try benchmarkInt8MatMul(allocator, size, size, size, num_runs);

        var label_buf: [32]u8 = undefined;
        const label = try std.fmt.bufPrint(&label_buf, "{}xTile Size", .{multiple});

        printBenchmarkResults(label, gops, size, size, size);
    }
}
