const std = @import("std");
const assert = std.debug.assert;
const mem = std.mem;
const Allocator = mem.Allocator;
const Thread = std.Thread;
const builtin = @import("builtin");
const T: usize = 64; // Tile size (adjust as necessary)
const V: usize = 32; // Vector size (adjust as necessary)
const simd_align = @alignOf(@Vector(V, f32));
const c = @cImport({
    @cInclude("stb_image.h");
    @cInclude("stb_image_resize2.h");
    @cInclude("stdio.h");
});
// TODO : add std.simd.suggestvectorlength

// vector math functions

// function taken from cgbur/llama2.zig
// applies softmax on a vector

// REDO this function with vectors
fn softmax(x: []f32) !void {
    // Find max for numerical stability
    var max: f32 = -std.math.inf(f32);
    for (x) |val| {
        max = @max(max, val);
    }

    // Compute exp(x - max) and sum
    var sum: f32 = 0.0;
    for (x) |*val| {
        val.* = @exp(val.* - max);
        sum += val.*;
    }

    // Normalize
    if (sum != 0) {
        for (x) |*val| {
            val.* /= sum;
        }
    } else {
        // Handle the case where sum is zero - uniform distribution
        const uniform_val = 1.0 / @as(f32, @floatFromInt(x.len));
        for (x) |*val| {
            val.* = uniform_val;
        }
    }

    // std.debug.print("Softmax max value: {}\n", .{max});
    // std.debug.print("After exp, before normalization - first 3: {any}\n", .{x[0..3]});
    // std.debug.print("Sum for normalization: {}\n", .{sum});
}

fn softmax_rows(x: []f32, num_rows: usize, row_size: usize) !void {
    var row: usize = 0;
    while (row < num_rows) : (row += 1) {
        const start = row * row_size;
        const end = start + row_size;
        const row_slice = x[start..end];

        // Find max for this row only
        var max: f32 = -std.math.inf(f32);
        for (row_slice) |val| {
            max = @max(max, val);
        }

        // Compute exp(x - max) and sum for this row
        var sum: f32 = 0.0;
        for (row_slice) |*val| {
            val.* = @exp(val.* - max);
            sum += val.*;
        }

        // Normalize this row
        if (sum != 0) {
            for (row_slice) |*val| {
                val.* /= sum;
            }
        } else {
            // Handle zero sum case
            const uniform_val = 1.0 / @as(f32, @floatFromInt(row_size));
            for (row_slice) |*val| {
                val.* = uniform_val;
            }
        }
    }
}

pub fn gelu(input: []f32) void {
    const sqrt_2_over_pi = std.math.sqrt(2.0 / std.math.pi);
    const factor = 0.044715;

    for (input) |*value| {
        const x = value.*;
        const term = sqrt_2_over_pi * (x + factor * x * x * x);
        const tanh_term = std.math.tanh(term);
        value.* = 0.5 * x * (1.0 + tanh_term);
    }
}

// Returns index of the max value in an f32 array
fn argmax(x: []f32) usize {
    assert(x.len > 0);
    var max: f32 = x[0];
    var maxi: usize = 0;
    for (1..x.len) |i| {
        if (x[i] > max) {
            max = x[i];
            maxi = i;
        }
    }
    return maxi;
}

// taken from cgbur/llama2.zig
// make this work with SIMD
fn rmsnorm(o: []f32, x: []f32, w: []f32) void {
    assert(o.len == x.len);
    assert(o.len == w.len);

    // sum of squares
    var sum: f32 = 0.0;
    for (x) |val| {
        sum += val * val;
    }
    sum /= @floatFromInt(x.len);
    sum += 1e-5;
    sum = 1.0 / std.math.sqrt(sum);

    // normalize and scale
    for (0..o.len) |i| {
        o[i] = x[i] * sum * w[i];
    }
}

pub fn checkVectorStability(vector: []const f32, name: []const u8) !void {
    for (vector, 0..) |value, index| {
        if (std.math.isNan(value)) {
            std.debug.print("Warning: NaN detected in {s} at index {d}\n", .{ name, index });
            return error.NaNDetected;
        }
        if (std.math.isInf(value)) {
            if (value > 0) {
                std.debug.print("Warning: +Inf detected in {s} at index {d}\n", .{ name, index });
            } else {
                std.debug.print("Warning: -Inf detected in {s} at index {d}\n", .{ name, index });
            }
            return error.InfDetected;
        }
    }
}

pub fn cos_2d(in: []const f32, out: []f32) !void {
    if (in.len != out.len) {
        std.debug.print("Length mismatch in cos_2d: in.len={d}, out.len={d}\n", .{ in.len, out.len });
        return error.LengthMismatch;
    }

    const len = in.len;
    const VectorSize = std.simd.suggestVectorLength(f32) orelse 8;
    const VectorType = @Vector(VectorSize, f32);
    var i: usize = 0;

    while (i + VectorSize <= len) : (i += VectorSize) {
        const v_in = @as(*align(4) const VectorType, @ptrCast(in[i..].ptr)).*;
        @as(*align(4) VectorType, @ptrCast(out[i..].ptr)).* = @cos(v_in);
    }

    // Fix the scalar loop - it was using sin instead of cos
    while (i < len) : (i += 1) {
        out[i] = std.math.cos(in[i]);
    }
}

pub fn sin_2d(in: []const f32, out: []f32) !void {
    if (in.len != out.len) {
        std.debug.print("Length mismatch in cos_2d: in.len={d}, out.len={d}\n", .{ in.len, out.len });
        return error.LengthMismatch;
    }

    const len = in.len;
    const VectorSize = std.simd.suggestVectorLength(f32) orelse 8;
    const VectorType = @Vector(VectorSize, f32);
    var i: usize = 0;

    while (i + VectorSize <= len) : (i += VectorSize) {
        const v_in = @as(*align(4) const VectorType, @ptrCast(in[i..].ptr)).*;
        @as(*align(4) VectorType, @ptrCast(out[i..].ptr)).* = @sin(v_in);
    }

    // Fix the scalar loop - it was using sin instead of cos
    while (i < len) : (i += 1) {
        out[i] = std.math.sin(in[i]);
    }
}

pub fn accumulate(accum: []f32, x: []const f32) void {
    std.debug.assert(accum.len == x.len);
    const len = accum.len;

    const VectorSize = std.simd.suggestVectorLength(f32) orelse 8;

    const VectorType = @Vector(VectorSize, f32);
    var i: usize = 0;

    // Process in SIMD-sized chunks
    while (i + VectorSize <= len) : (i += VectorSize) {
        var v_accum = @as(*align(4) VectorType, @ptrCast(accum[i..].ptr)).*;
        const v_x = @as(*align(4) const VectorType, @ptrCast(x[i..].ptr)).*;
        v_accum += v_x;
        @as(*align(4) VectorType, @ptrCast(accum[i..].ptr)).* = v_accum;
    }

    // Handle remaining elements
    while (i < len) : (i += 1) {
        accum[i] += x[i];
    }
}

pub fn broadcast_add(dest: []f32, bias: []const f32, seq_len: usize) void {
    const feat_dim = bias.len;
    const VectorSize = std.simd.suggestVectorLength(f32) orelse 8;
    const VectorType = @Vector(VectorSize, f32);

    // For each sequence position
    for (0..seq_len) |seq| {
        const offset = seq * feat_dim;
        var i: usize = 0;

        // Process in SIMD-sized chunks
        while (i + VectorSize <= feat_dim) : (i += VectorSize) {
            var v_dest = @as(*align(4) VectorType, @ptrCast(dest[offset + i ..].ptr)).*;
            const v_bias = @as(*align(4) const VectorType, @ptrCast(bias[i..].ptr)).*;
            v_dest += v_bias;
            @as(*align(4) VectorType, @ptrCast(dest[offset + i ..].ptr)).* = v_dest;
        }

        // Handle remaining elements
        while (i < feat_dim) : (i += 1) {
            dest[offset + i] += bias[i];
        }
    }
}

fn transposeSimd(allocator: std.mem.Allocator, matrix: []const f32, rows: usize, cols: usize) ![]f32 {
    // Ensure aligned allocation
    const alignment = 16; // For Vector(4, f32)
    const transposed = try allocator.alignedAlloc(f32, alignment, rows * cols);
    errdefer allocator.free(transposed);

    const VectorType = @Vector(4, f32);
    const simd_width = 4;

    var i: usize = 0;
    while (i < rows) : (i += simd_width) {
        var j: usize = 0;
        while (j < cols) : (j += 1) {
            if (i + simd_width <= rows) {
                const v0 = matrix[(i + 0) * cols + j];
                const v1 = matrix[(i + 1) * cols + j];
                const v2 = matrix[(i + 2) * cols + j];
                const v3 = matrix[(i + 3) * cols + j];
                const vec = VectorType{ v0, v1, v2, v3 };

                // Store values individually if alignment can't be guaranteed
                const base_idx = j * rows + i;
                transposed[base_idx + 0] = vec[0];
                transposed[base_idx + 1] = vec[1];
                transposed[base_idx + 2] = vec[2];
                transposed[base_idx + 3] = vec[3];
            } else {
                var k: usize = 0;
                while (k < rows - i) : (k += 1) {
                    transposed[j * rows + (i + k)] = matrix[(i + k) * cols + j];
                }
            }
        }
    }

    return transposed;
}

/// Matrix Multiplication
pub fn matmul(allocator: std.mem.Allocator, A: []const f32, B: []const f32, C: []f32, M: usize, N: usize, K: usize) !void {
    // TODO : add size checking for all the matrices using assert.

    std.debug.assert(A.len == M * K); // A should be M x K
    std.debug.assert(B.len == K * N); // B should be K x N
    std.debug.assert(C.len == M * N); // C should be M x N

    // TODO : Remove this debug print statements once we are clear about all the dims
    // std.debug.print("A len = {any}, M * K = {any} \n ", .{ A.len, M * K });
    // std.debug.print("B len = {any}, K * N = {any} \n ", .{ B.len, K * N });
    // std.debug.print("C len = {any}, M * N = {any} \n ", .{ C.len, M * N });

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

// copied from https://github.com/cgbur/llama2.zig/blob/main/src/main.zig#L489

// not copied to ops yet.
fn vector_dot_product(x: []const f32, y: []const f32) f32 {
    assert(x.len == y.len);
    const vector_width = V;
    const vec_len = x.len / vector_width; // num of SIMD vectors
    const vec_rem = x.len % vector_width; // num of f32 in the last SIMD vector

    // do the bulk of the work with SIMD
    var sum: @Vector(vector_width, f32) = @splat(0.0);
    var offset: usize = 0;
    for (0..vec_len) |_| {
        const xvec: @Vector(vector_width, f32) = x[offset..][0..vector_width].*;
        const yvec: @Vector(vector_width, f32) = y[offset..][0..vector_width].*;
        sum += xvec * yvec;
        offset += vector_width;
    }

    // handle the last few elements with normal scalar ops
    var sum_rem: f32 = 0.0;
    for (0..vec_rem) |i| {
        sum_rem += x[offset + i] * y[offset + i];
    }

    // reduce the SIMD vector to a scalar
    return @reduce(.Add, sum) + sum_rem;
}

fn vector_weighted_sum(xout: []f32, x: []const f32, y: f32) void {
    assert(xout.len == x.len);
    const vector_width = V;
    const vec_len = x.len / vector_width; // num of SIMD vectors
    const vec_rem = x.len % vector_width; // num of f32 in the last SIMD vector

    // do the bulk of the work with SIMD
    var offset: usize = 0;
    const yvector: @Vector(vector_width, f32) = @splat(y);
    for (0..vec_len) |_| {
        var xoutvec: @Vector(vector_width, f32) = xout[offset..][0..vector_width].*;
        const xvec: @Vector(vector_width, f32) = x[offset..][0..vector_width].*;
        xoutvec += xvec * yvector;
        xout[offset..][0..vector_width].* = xoutvec;
        offset += vector_width;
    }

    // handle the last few elements with normal scalar operations
    for (0..vec_rem) |i| {
        xout[offset + i] += x[offset + i] * y;
    }
}
pub fn printMatrix(m: usize, n: usize, matrix: []const f32, rows_to_print: usize, elements_per_row: usize) void {
    const stdout = std.io.getStdOut().writer();

    // Ensure that rows_to_print does not exceed the number of rows in the matrix
    const rows = if (rows_to_print <= m) rows_to_print else m;

    // Ensure that elements_per_row does not exceed the number of columns in the matrix
    const elements = if (elements_per_row <= n) elements_per_row else n;

    // Iterate through the matrix and print elements row by row
    for (0..rows) |i| {
        for (0..elements) |j| {
            const idx = i * n + j;
            // Print each element with 3 decimal precision, aligned width of 8
            stdout.print("{:8.3}", .{matrix[idx]}) catch {};
            stdout.print(" ", .{}) catch {};
        }
        stdout.print("\n", .{}) catch {};
    }

    if (rows < m) {
        stdout.print("\n... ({} more rows not shown)\n", .{m - rows}) catch {};
    }

    if (elements < n) {
        stdout.print("... ({} more elements per row not shown)\n", .{n - elements}) catch {};
    }
}

/// Calculate the outer product of two vectors.
///
/// A: input vector 1     |
/// B: input vector 2     |
/// C: output vector      |
/// M: length of vector A |
/// N: length of vector B |
///
/// The outer product is calculated as follows:
/// C[i * N + j] = A[i] * B[j]
/// We just apply SIMD optimization to the inner loop.
fn outer(A: []const f32, B: []const f32, C: []f32) void {
    std.debug.assert(A.len * B.len == C.len);

    const VectorSize = 32;
    const VectorType = @Vector(VectorSize, f32);

    // operate over A's rows
    for (0..A.len) |i| {
        var j: usize = 0;
        // broadcast the selected element of A to a vector
        const a_splat = @as(VectorType, @splat(A[i]));

        // we then process B in chunks of VectorSize
        while (j + VectorSize <= B.len) : (j += VectorSize) {
            // load B directly as a vector, assuming alignment
            const b_vec = @as(*align(4) const VectorType, @ptrCast(B[j..].ptr)).*;
            // multiply and store directly
            @as(*align(4) VectorType, @ptrCast(C[i * B.len + j ..].ptr)).* = a_splat * b_vec;
        }

        // handle remaining elements
        while (j < B.len) : (j += 1) {
            C[i * B.len + j] = A[i] * B[j];
        }
    }
}

pub fn cat(allocator: Allocator, A: []const f32, B: []const f32, A_M: usize, A_N: usize, B_M: usize, B_N: usize, dim: usize) ![]f32 {

    // we have two cases, we can either concatenate along the rows or the columns
    // first case: concatenate along the rows

    if (dim == 0) {
        // check if the number of columns of the two matrices are the same
        if (A_N != B_N) {
            return error.InvalidInputSize;
        }

        // create a new matrix with the correct dimensions
        // we check whose number of rows is greater
        const M: usize = A_M + B_M;

        const N = A_N;

        const C = try allocator.alloc(f32, M * N);

        @memcpy(C[0 .. A_M * N], A[0..]);
        @memcpy(C[A_M * N ..], B[0..]);

        return C;
    }

    if (dim == 1) {
        // check if the number of rows of the two matrices are the same
        if (A_M != B_M) {
            return error.InvalidInputSize;
        }

        // create a new matrix with the correct dimensions
        // we check whose number of columns is greater
        const M: usize = A_M;

        const N = A_N + B_N;

        const C = try allocator.alloc(f32, M * N);

        for (0..M) |i| {
            // Copy the i-th row of matrix A
            @memcpy(C[i * N .. i * N + A_N], A[i * A_N .. (i + 1) * A_N]);
            // Copy the i-th row of matrix B
            @memcpy(C[i * N + A_N .. (i + 1) * N], B[i * B_N .. (i + 1) * B_N]);
        }

        return C;
    }

    return error.InvalidDimension;
}

pub fn outerConventional(A: []const f32, B: []const f32, C: []f32, M: usize, N: usize) !void {
    // Ensure the input vectors and matrix have the correct sizes
    if (A.len != M or B.len != N) {
        return error.InvalidInputSize;
    }

    // Compute the outer product using conventional method
    var i: usize = 0;
    while (i < M) : (i += 1) {
        var j: usize = 0;
        while (j < N) : (j += 1) {
            C[i * N + j] = A[i] * B[j];
        }
    }
}

//config

const ConfigReader = extern struct {
    const Self = @This();

    // Text model
    vocab: i32, // vocabulary size, usually 256 (byte-level)
    dim: i32, // transformer dimension
    hidden_dim: i32, // for ffn layers
    n_layers: i32, // number of transformer layers, 24 for text model
    n_heads: i32, // number of attn heads per layer
    head_dim: i32, // size of attn heads per layer
    seq_len: i32, // max sequence length
    rope_theta: f32,
    max_pos_embeddings: i32,
    partial_rotary_factor: f32,

    // vision
    img_channels: i32, // number of channels per patch, RGB has 3
    img_dim: i32, // dimension of the the image, 378x378 default
    patch_size: i32, // size of patch, 14x14 default
    vit_embed_len: i32, // vision embed len
    vit_dim: i32, // width of each patch embedding created from linear patch embedding layer, 1152 default
    n_vit_layers: i32,
    n_vit_heads: i32,
    vit_head_dim: i32,
    hidden_features: i32, // the number of hidden features, equivalent to hidden_dim in text model, 4304 default

    fn config(self: Self) Config {
        return Config{
            .vocab = @intCast(self.vocab),
            .dim = @intCast(self.dim),
            .hidden_dim = @intCast(self.hidden_dim),
            .n_layers = @intCast(self.n_layers),
            .n_heads = @intCast(self.n_heads),
            .head_dim = @intCast(self.head_dim),
            .seq_len = @intCast(self.seq_len),
            .rope_theta = self.rope_theta,
            .max_pos_embeddings = @intCast(self.max_pos_embeddings),
            .partial_rotary_factor = self.partial_rotary_factor,
            .img_channels = @intCast(self.img_channels),
            .img_dim = @intCast(self.img_dim),
            .patch_size = @intCast(self.patch_size),
            .num_patches = @intCast(@divTrunc(self.img_dim, self.patch_size) * @divTrunc(self.img_dim, self.patch_size)),
            .vit_embed_len = @intCast(self.vit_embed_len),
            .vit_dim = @intCast(self.vit_dim),
            .n_vit_layers = @intCast(self.n_vit_layers),
            .n_vit_heads = @intCast(self.n_vit_heads),
            .vit_head_dim = @intCast(self.vit_head_dim),
            .hidden_features = @intCast(self.hidden_features),
        };
    }
};

const Config = struct {

    // Text Model
    vocab: usize,
    dim: usize, //text transformer dim, 2048
    hidden_dim: usize, // hidden fc dim
    n_layers: usize, //number of transformer layers, 24 for text model
    n_heads: usize, //number of attn heads per layer
    head_dim: usize, //size of attn heads
    seq_len: usize, // sequence length, 2048
    rope_theta: f32,
    max_pos_embeddings: usize,
    partial_rotary_factor: f32,

    // Vision Model
    img_channels: usize, // number of channels per patch, RGB has 3
    img_dim: usize, // dimension of the the image, 378x378 default
    patch_size: usize, // size of patch, 14x14 default
    num_patches: usize,
    vit_embed_len: usize,
    vit_dim: usize, // width of each patch embedding created from linear patch embedding layer, 1152 default
    n_vit_layers: usize, // number of ViT layers, 27 default for the vision model
    n_vit_heads: usize, // number of attn heads for each attn layer, 16 default
    vit_head_dim: usize, // size of each attn head, 72 default
    hidden_features: usize, // size of hidden features in ViT fc layers, 4304 in length
};

/// Struct defines the weights of moondream
/// All weights are transposed
/// Naming convention :
/// "t_" prefix : model (phi 1.5)
/// "v_" prefix : vision_model (vision encoder)
/// "_w" suffix : weights
/// "_b" suffix : biases
const Weights = struct {
    const Self = @This();

    // Add magic number for format validation
    const WEIGHTS_MAGIC = 0x4D4F4F4E; // "MOON" in ASCII
    const WEIGHTS_VERSION = 1;

    /// Buffer holding all weight data
    buffer: []f32,
    allocator: Allocator,

    // All the weight slices that point into buffer
    word_token_embedding: []f32,
    t_ln_w: []f32,
    t_ln_b: []f32,
    t_Wqkv_w: []f32,
    t_Wqkv_b: []f32,
    t_out_proj_w: []f32,
    t_out_proj_bias: []f32,
    t_fc1_w: []f32,
    t_fc1_b: []f32,
    t_fc2_w: []f32,
    t_fc2_b: []f32,
    t_linear_w: []f32,
    t_linear_b: []f32,
    t_ln_out_w: []f32,
    t_ln_out_b: []f32,
    v_patch_embedding_linear_w: []f32,
    v_patch_embedding_linear_b: []f32,
    v_pos_embedding: []f32,
    v_Wqkv_w: []f32,
    v_Wqkv_b: []f32,
    v_out_proj_w: []f32,
    v_out_proj_b: []f32,
    v_fc1_w: []f32,
    v_fc1_b: []f32,
    v_fc2_w: []f32,
    v_fc2_b: []f32,
    v_norm1_w: []f32,
    v_norm1_b: []f32,
    v_norm2_w: []f32,
    v_norm2_b: []f32,
    v_norm_out_w: []f32,
    v_norm_out_b: []f32,
    v_proj_fc1_w: []f32,
    v_proj_fc1_b: []f32,
    v_proj_fc2_w: []f32,
    v_proj_fc2_b: []f32,

    fn init(config: Config, filename: []const u8, allocator: Allocator) !Self {
        const sizes = calculateSizes(config);
        const num_weights = calculateTotalSize(sizes);

        // Open and validate the file
        const file = try std.fs.cwd().openFile(filename, .{});
        defer file.close();

        const file_size = try file.getEndPos();
        const expected_size = num_weights * @sizeOf(f32);

        if (file_size != expected_size) {
            std.debug.print("Error: Unexpected file size\n", .{});
            std.debug.print("Expected: {}, Got: {}\n", .{ expected_size, file_size });
            return error.UnexpectedFileSize;
        }

        // Allocate buffer for all weights
        const buffer = try allocator.alloc(f32, num_weights);
        errdefer allocator.free(buffer);

        // Read all weight data
        const bytes_read = try file.readAll(std.mem.sliceAsBytes(buffer));
        if (bytes_read != file_size) {
            return error.IncompleteRead;
        }

        // Initialize the struct with our owned buffer
        var self = Self{
            .buffer = buffer,
            .allocator = allocator,
            // Initialize all weight slices as undefined first
            .word_token_embedding = undefined,
            .t_ln_w = undefined,
            .t_ln_b = undefined,
            .t_Wqkv_w = undefined,
            .t_Wqkv_b = undefined,
            .t_out_proj_w = undefined,
            .t_out_proj_bias = undefined,
            .t_fc1_w = undefined,
            .t_fc1_b = undefined,
            .t_fc2_w = undefined,
            .t_fc2_b = undefined,
            .t_linear_w = undefined,
            .t_linear_b = undefined,
            .t_ln_out_w = undefined,
            .t_ln_out_b = undefined,
            .v_patch_embedding_linear_w = undefined,
            .v_patch_embedding_linear_b = undefined,
            .v_pos_embedding = undefined,
            .v_Wqkv_w = undefined,
            .v_Wqkv_b = undefined,
            .v_out_proj_w = undefined,
            .v_out_proj_b = undefined,
            .v_fc1_w = undefined,
            .v_fc1_b = undefined,
            .v_fc2_w = undefined,
            .v_fc2_b = undefined,
            .v_norm1_w = undefined,
            .v_norm1_b = undefined,
            .v_norm2_w = undefined,
            .v_norm2_b = undefined,
            .v_norm_out_w = undefined,
            .v_norm_out_b = undefined,
            .v_proj_fc1_w = undefined,
            .v_proj_fc1_b = undefined,
            .v_proj_fc2_w = undefined,
            .v_proj_fc2_b = undefined,
        };

        // Set up all the slices into our buffer
        var offset: usize = 0;

        // Text model start
        self.word_token_embedding = self.buffer[offset .. offset + sizes.word_token_embedding];
        offset += sizes.word_token_embedding;

        // Text model transformer blocks
        self.t_ln_w = self.buffer[offset .. offset + sizes.t_ln_w];
        offset += sizes.t_ln_w;

        self.t_ln_b = self.buffer[offset .. offset + sizes.t_ln_b];
        offset += sizes.t_ln_b;

        self.t_Wqkv_w = self.buffer[offset .. offset + sizes.t_Wqkv_w];
        offset += sizes.t_Wqkv_w;

        self.t_Wqkv_b = self.buffer[offset .. offset + sizes.t_Wqkv_b];
        offset += sizes.t_Wqkv_b;

        self.t_out_proj_w = self.buffer[offset .. offset + sizes.t_out_proj_w];
        offset += sizes.t_out_proj_w;

        self.t_out_proj_bias = self.buffer[offset .. offset + sizes.t_out_proj_bias];
        offset += sizes.t_out_proj_bias;

        self.t_fc1_w = self.buffer[offset .. offset + sizes.t_fc1_w];
        offset += sizes.t_fc1_w;

        self.t_fc1_b = self.buffer[offset .. offset + sizes.t_fc1_b];
        offset += sizes.t_fc1_b;

        self.t_fc2_w = self.buffer[offset .. offset + sizes.t_fc2_w];
        offset += sizes.t_fc2_w;

        self.t_fc2_b = self.buffer[offset .. offset + sizes.t_fc2_b];
        offset += sizes.t_fc2_b;

        // Text model end
        self.t_linear_w = self.buffer[offset .. offset + sizes.t_linear_w];
        offset += sizes.t_linear_w;

        self.t_linear_b = self.buffer[offset .. offset + sizes.t_linear_b];
        offset += sizes.t_linear_b;

        self.t_ln_out_w = self.buffer[offset .. offset + sizes.t_ln_out_w];
        offset += sizes.t_ln_out_w;

        self.t_ln_out_b = self.buffer[offset .. offset + sizes.t_ln_out_b];
        offset += sizes.t_ln_out_b;

        // Vision model start
        self.v_patch_embedding_linear_w = self.buffer[offset .. offset + sizes.v_patch_embedding_linear_w];
        offset += sizes.v_patch_embedding_linear_w;

        self.v_patch_embedding_linear_b = self.buffer[offset .. offset + sizes.v_patch_embedding_linear_b];
        offset += sizes.v_patch_embedding_linear_b;

        self.v_pos_embedding = self.buffer[offset .. offset + sizes.v_pos_embedding];
        offset += sizes.v_pos_embedding;

        // Vision transformer blocks
        self.v_Wqkv_w = self.buffer[offset .. offset + sizes.v_Wqkv_w];
        offset += sizes.v_Wqkv_w;

        self.v_Wqkv_b = self.buffer[offset .. offset + sizes.v_Wqkv_b];
        offset += sizes.v_Wqkv_b;

        self.v_out_proj_w = self.buffer[offset .. offset + sizes.v_out_proj_w];
        offset += sizes.v_out_proj_w;

        self.v_out_proj_b = self.buffer[offset .. offset + sizes.v_out_proj_b];
        offset += sizes.v_out_proj_b;

        self.v_fc1_w = self.buffer[offset .. offset + sizes.v_fc1_w];
        offset += sizes.v_fc1_w;

        self.v_fc1_b = self.buffer[offset .. offset + sizes.v_fc1_b];
        offset += sizes.v_fc1_b;

        self.v_fc2_w = self.buffer[offset .. offset + sizes.v_fc2_w];
        offset += sizes.v_fc2_w;

        self.v_fc2_b = self.buffer[offset .. offset + sizes.v_fc2_b];
        offset += sizes.v_fc2_b;

        self.v_norm1_w = self.buffer[offset .. offset + sizes.v_norm1_w];
        offset += sizes.v_norm1_w;

        self.v_norm1_b = self.buffer[offset .. offset + sizes.v_norm1_b];
        offset += sizes.v_norm1_b;

        self.v_norm2_w = self.buffer[offset .. offset + sizes.v_norm2_w];
        offset += sizes.v_norm2_w;

        self.v_norm2_b = self.buffer[offset .. offset + sizes.v_norm2_b];
        offset += sizes.v_norm2_b;

        // Vision model end
        self.v_norm_out_w = self.buffer[offset .. offset + sizes.v_norm_out_w];
        offset += sizes.v_norm_out_w;

        self.v_norm_out_b = self.buffer[offset .. offset + sizes.v_norm_out_b];
        offset += sizes.v_norm_out_b;

        // Projection layers
        self.v_proj_fc1_w = self.buffer[offset .. offset + sizes.v_proj_fc1_w];
        offset += sizes.v_proj_fc1_w;

        self.v_proj_fc1_b = self.buffer[offset .. offset + sizes.v_proj_fc1_b];
        offset += sizes.v_proj_fc1_b;

        self.v_proj_fc2_w = self.buffer[offset .. offset + sizes.v_proj_fc2_w];
        offset += sizes.v_proj_fc2_w;

        self.v_proj_fc2_b = self.buffer[offset .. offset + sizes.v_proj_fc2_b];
        offset += sizes.v_proj_fc2_b;

        // Verify we used exactly all the bytes
        if (offset != num_weights) {
            std.debug.print("Error: Buffer not fully utilized\n", .{});
            std.debug.print("Used: {}, Total: {}\n", .{ offset, num_weights });
            return error.BufferSizeMismatch;
        }

        std.debug.print("Loaded weights successfully. \nTotal parameters: {}\n", .{num_weights});
        return self;
    }

    fn calculateSizes(config: Config) struct {
        //text
        word_token_embedding: usize,
        t_ln_w: usize,
        t_ln_b: usize,
        t_Wqkv_w: usize,
        t_Wqkv_b: usize,
        t_out_proj_w: usize,
        t_out_proj_bias: usize,
        t_fc1_w: usize,
        t_fc1_b: usize,
        t_fc2_w: usize,
        t_fc2_b: usize,
        t_linear_w: usize,
        t_linear_b: usize,
        t_ln_out_w: usize,
        t_ln_out_b: usize,
        // vision
        v_patch_embedding_linear_w: usize,
        v_patch_embedding_linear_b: usize,
        v_pos_embedding: usize,
        v_Wqkv_w: usize,
        v_Wqkv_b: usize,
        v_out_proj_w: usize,
        v_out_proj_b: usize,
        v_fc1_w: usize,
        v_fc1_b: usize,
        v_fc2_w: usize,
        v_fc2_b: usize,
        v_norm1_w: usize,
        v_norm1_b: usize,
        v_norm2_w: usize,
        v_norm2_b: usize,
        v_norm_out_w: usize,
        v_norm_out_b: usize,
        v_proj_fc1_w: usize,
        v_proj_fc1_b: usize,
        v_proj_fc2_w: usize,
        v_proj_fc2_b: usize,
    } {
        return .{
            // Text model
            .word_token_embedding = config.dim * config.vocab,
            // Transformer block begins here //
            .t_ln_w = config.n_layers * config.dim,
            .t_ln_b = config.n_layers * config.dim,
            .t_Wqkv_w = config.n_layers * config.dim * config.n_heads * config.head_dim * 3,
            .t_Wqkv_b = config.n_layers * config.n_heads * config.head_dim * 3,
            .t_out_proj_w = config.n_layers * config.dim * config.seq_len,
            .t_out_proj_bias = config.n_layers * config.dim,
            .t_fc1_w = config.n_layers * config.dim * config.hidden_dim,
            .t_fc1_b = config.n_layers * config.hidden_dim,
            .t_fc2_w = config.n_layers * config.hidden_dim * config.dim,
            .t_fc2_b = config.n_layers * config.dim,
            // Transformer block ends here //

            .t_linear_w = config.dim * config.vocab,
            .t_linear_b = config.vocab,
            .t_ln_out_w = config.dim,
            .t_ln_out_b = config.dim,

            //Vision model
            .v_patch_embedding_linear_w = config.patch_size * config.patch_size * config.img_channels * config.vit_dim,
            .v_patch_embedding_linear_b = config.vit_dim,
            .v_pos_embedding = config.vit_dim * ((config.img_dim / config.patch_size) * (config.img_dim / config.patch_size)) * 1,

            // ViT block begins here //
            .v_Wqkv_w = config.n_vit_layers * config.vit_dim * config.n_vit_heads * config.vit_head_dim * 3,
            .v_Wqkv_b = config.n_vit_layers * config.n_vit_heads * config.vit_head_dim * 3,
            .v_out_proj_w = config.n_vit_layers * config.vit_dim * config.vit_dim,
            .v_out_proj_b = config.n_vit_layers * config.vit_dim,
            .v_fc1_w = config.n_vit_layers * config.vit_dim * config.hidden_features,
            .v_fc1_b = config.n_vit_layers * config.hidden_features,
            .v_fc2_w = config.n_vit_layers * config.hidden_features * config.vit_dim,
            .v_fc2_b = config.n_vit_layers * config.vit_dim,
            .v_norm1_w = config.n_vit_layers * config.vit_dim,
            .v_norm1_b = config.n_vit_layers * config.vit_dim,
            .v_norm2_w = config.n_vit_layers * config.vit_dim,
            .v_norm2_b = config.n_vit_layers * config.vit_dim,
            // ViT block ends here //

            .v_norm_out_w = config.vit_dim,
            .v_norm_out_b = config.vit_dim,
            .v_proj_fc1_w = config.vit_head_dim * config.n_heads * config.hidden_dim, //TODO FIGURE OUT WHY THIS IS??
            .v_proj_fc1_b = config.hidden_dim,
            .v_proj_fc2_w = config.hidden_dim * config.dim,
            .v_proj_fc2_b = config.dim,
        };
    }

    fn calculateTotalSize(sizes: anytype) usize {
        var total: usize = 0;
        inline for (std.meta.fields(@TypeOf(sizes))) |field| {
            total += @field(sizes, field.name);
        }
        return total;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.buffer);
        self.* = undefined;
    }
};

// runstate

const RunState = struct {
    const Self = @This();

    img: []align(simd_align) f32,
    patches: []align(simd_align) f32,
    patch_emb: []align(simd_align) f32,
    projection: []align(simd_align) f32,
    v_x: []align(simd_align) f32,
    v_xb: []align(simd_align) f32,
    v_xb2: []align(simd_align) f32,
    v_xb3: []align(simd_align) f32,
    v_qkv: []align(simd_align) f32,
    v_q: []align(simd_align) f32,
    v_k: []align(simd_align) f32,
    v_v: []align(simd_align) f32,
    v_attn: []align(simd_align) f32,
    v_output: []align(simd_align) f32,
    v_proj: []align(simd_align) f32,
    k_cache: []align(simd_align) f32,
    v_cache: []align(simd_align) f32,
    cos_cache: []align(simd_align) f32,
    sin_cache: []align(simd_align) f32,
    // emb: []align(simd_align) f32,
    // ln_in: []align(simd_align) f32,
    // attn_in: []align(simd_align) f32,
    // x: []align(simd_align) f32,
    // xb: []align(simd_align) f32,
    // mlp_in: []align(simd_align) f32,
    // t_qkv: []align(simd_align) f32, // a buffer that holds the combined kqv
    // q: []align(simd_align) f32,
    // k: []align(simd_align) f32,
    // v: []align(simd_align) f32,
    // attn: []align(simd_align) f32,
    // output: []align(simd_align) f32,
    // inv_freq: []align(simd_align) f32,
    // logits: []align(simd_align) f32,

    fn init(allocator: Allocator, config: Config) !Self {
        const batch_size = 2; // Two identical patches
        const total_patches = batch_size * config.num_patches;
        return Self{
            .img = try allocator.alignedAlloc(f32, simd_align, batch_size * config.img_dim * config.img_dim * config.img_channels),

            // Patches buffer needs to handle both patches
            .patches = try allocator.alignedAlloc(f32, simd_align, batch_size * config.img_dim * config.img_dim * config.img_channels),

            // All patch-related buffers need to account for total_patches
            .patch_emb = try allocator.alignedAlloc(f32, simd_align, total_patches * config.vit_dim),
            .projection = try allocator.alignedAlloc(f32, simd_align, config.num_patches * config.dim),
            .v_x = try allocator.alignedAlloc(f32, simd_align, total_patches * config.vit_dim),
            .v_xb = try allocator.alignedAlloc(f32, simd_align, total_patches * config.hidden_features),
            .v_xb2 = try allocator.alignedAlloc(f32, simd_align, total_patches * config.vit_dim),
            .v_xb3 = try allocator.alignedAlloc(f32, simd_align, config.num_patches * config.hidden_dim),
            .v_qkv = try allocator.alignedAlloc(f32, simd_align, total_patches * config.vit_dim * 3),
            .v_q = try allocator.alignedAlloc(f32, simd_align, total_patches * config.vit_dim),
            .v_k = try allocator.alignedAlloc(f32, simd_align, total_patches * config.vit_dim),
            .v_v = try allocator.alignedAlloc(f32, simd_align, total_patches * config.vit_dim),

            // Attention buffers need to be updated for total_patches
            .v_attn = try allocator.alignedAlloc(f32, simd_align, total_patches * total_patches * config.n_vit_heads),
            .v_output = try allocator.alignedAlloc(f32, simd_align, total_patches * config.vit_dim),
            .v_proj = try allocator.alignedAlloc(f32, simd_align, total_patches * config.vit_dim),
            .k_cache = try allocator.alignedAlloc(f32, simd_align, config.n_layers * config.seq_len * config.n_heads * config.head_dim),
            .v_cache = try allocator.alignedAlloc(f32, simd_align, config.n_layers * config.seq_len * config.n_heads * config.head_dim),
            .cos_cache = try allocator.alignedAlloc(f32, simd_align, config.max_pos_embeddings * config.head_dim / 2),
            .sin_cache = try allocator.alignedAlloc(f32, simd_align, config.max_pos_embeddings * config.head_dim / 2), // hardcoding the partial rotary factor as 1/2 here
            // .emb = try allocator.alignedAlloc(f32, simd_align, config.dim),
            // .ln_in = try allocator.alignedAlloc(f32, simd_align, config.dim),
            // .attn_in = try allocator.alignedAlloc(f32, simd_align, config.dim),
            // .x = try allocator.alignedAlloc(f32, simd_align, config.hidden_dim),
            // .xb = try allocator.alignedAlloc(f32, simd_align, config.dim),
            // .mlp_in = try allocator.alignedAlloc(f32, simd_align, config.dim),
            // .t_qkv = try allocator.alignedAlloc(f32, simd_align, config.dim * 3),
            // .q = try allocator.alignedAlloc(f32, simd_align, config.n_heads * config.head_dim),
            // .k = try allocator.alignedAlloc(f32, simd_align, config.n_heads * config.head_dim),
            // .v = try allocator.alignedAlloc(f32, simd_align, config.n_heads * config.head_dim),
            // .attn = try allocator.alignedAlloc(f32, simd_align, config.n_heads * config.seq_len),
            // .output = try allocator.alignedAlloc(f32, simd_align, config.dim),
            // .inv_freq = try allocator.alignedAlloc(f32, simd_align, config.dim / 2),
            // .logits = try allocator.alignedAlloc(f32, simd_align, config.vocab),
        };
    }

    fn deinit(self: *Self, allocator: Allocator) void {
        inline for (std.meta.fields(Self)) |f| {
            allocator.free(@field(self, f.name));
        }
        self.* = undefined;
    }
};

// tokenizer

// Tokens, their scores, and the max token length. Supports initialization
// from a file and encoding text into tokens via the `encode` method.

const SpecialToken = struct {
    id: u32,
    content: []const u8,
    is_special: bool,
    single_word: bool,
    lstrip: bool,
    rstrip: bool,
    normalized: bool,
};

const Tokenizer = struct {
    // TODO : Add handling external tokens to tokenizer
    const Self = @This();
    tokens: std.StringHashMap(u32),
    special_tokens: std.ArrayList(SpecialToken),
    merges: std.ArrayList([]const u8),
    allocator: Allocator,
    eos_token: u32,
    bos_token: u32,
    pad_token: u32,

    fn init(allocator: Allocator) Tokenizer {
        return .{
            .tokens = std.StringHashMap(u32).init(allocator),
            .special_tokens = std.ArrayList(SpecialToken).init(allocator),
            .merges = std.ArrayList([]const u8).init(allocator),
            .allocator = allocator,
            .eos_token = 50256, // These values should match your tokenizer.json
            .bos_token = 50257,
            .pad_token = 50258,
        };
    }

    fn fromFile(filename: []const u8, allocator: Allocator) !Tokenizer {
        var self = Tokenizer.init(allocator);
        errdefer self.deinit();

        const file = try std.fs.cwd().openFile(filename, .{});
        defer file.close();

        var reader = file.reader();

        // Read regular tokens
        const num_tokens = try reader.readInt(u32, .little);
        // std.debug.print("Loading {} regular tokens\n", .{num_tokens});

        for (0..num_tokens) |_| {
            const token_id = try reader.readInt(u32, .little);
            const token_len = try reader.readInt(u32, .little);
            const token_content = try allocator.alloc(u8, token_len);
            errdefer allocator.free(token_content);
            const bytes_read = try reader.readAll(token_content);
            if (bytes_read != token_len) {
                return error.UnexpectedEOF;
            }
            try self.tokens.put(token_content, token_id);

            // if (i % 1000 == 0) {
            //     // std.debug.print("Loaded {} regular tokens...\n", .{i});
            // }
        }

        // Read special tokens
        const num_special = try reader.readInt(u32, .little);
        // std.debug.print("Loading {} special tokens\n", .{num_special});

        for (0..num_special) |_| {
            // Read token metadata
            const token_id = try reader.readInt(u32, .little);
            const token_len = try reader.readInt(u32, .little);

            // Read flags
            const is_special = try reader.readInt(u8, .little) != 0;
            const single_word = try reader.readInt(u8, .little) != 0;
            const lstrip = try reader.readInt(u8, .little) != 0;
            const rstrip = try reader.readInt(u8, .little) != 0;
            const normalized = try reader.readInt(u8, .little) != 0;

            // Read token content
            const content = try allocator.alloc(u8, token_len);
            errdefer allocator.free(content);
            const bytes_read = try reader.readAll(content);
            if (bytes_read != token_len) {
                return error.UnexpectedEOF;
            }

            // Create and store special token
            try self.special_tokens.append(.{
                .id = token_id,
                .content = content,
                .is_special = is_special,
                .single_word = single_word,
                .lstrip = lstrip,
                .rstrip = rstrip,
                .normalized = normalized,
            });

            // std.debug.print("Loaded special token {}: id={}, content='{s}'\n", .{ i, token_id, content });
        }

        // Read merges
        const num_merges = try reader.readInt(u32, .little);
        // std.debug.print("Loading {} merges\n", .{num_merges});

        for (0..num_merges) |_| {
            // Read first part
            const first_len = try reader.readInt(u16, .little);
            const first = try allocator.alloc(u8, first_len);
            errdefer allocator.free(first);
            const first_bytes_read = try reader.readAll(first);
            if (first_bytes_read != first_len) {
                return error.UnexpectedEOF;
            }

            // Read second part
            const second_len = try reader.readInt(u16, .little);
            const second = try allocator.alloc(u8, second_len);
            errdefer allocator.free(second);
            const second_bytes_read = try reader.readAll(second);
            if (second_bytes_read != second_len) {
                return error.UnexpectedEOF;
            }

            // Combine into merge rule
            const merge = try std.fmt.allocPrint(allocator, "{s} {s}", .{ first, second });
            errdefer allocator.free(merge);

            try self.merges.append(merge);

            // Clean up temporary allocations
            allocator.free(first);
            allocator.free(second);
        }

        // Final load summary
        std.debug.print("Tokenizer loaded: {} tokens, {} special tokens, {} merges\n", .{
            self.tokens.count(),
            self.special_tokens.items.len,
            self.merges.items.len,
        });

        return self;
    }

    fn lookup(self: *const Tokenizer, token: []const u8) ?u32 {
        // std.debug.print("Looking up token: '{s}' (len: {})\n", .{ token, token.len });
        var it = self.tokens.iterator();
        while (it.next()) |entry| {
            // std.debug.print("Stored token: '{s}' (len: {})\n", .{ entry.key_ptr.*, entry.key_ptr.len });
            if (std.mem.eql(u8, entry.key_ptr.*, token)) {
                return entry.value_ptr.*;
            }
        }
        return null;
    }

    fn encode(self: *const Tokenizer, text: []const u8) !std.ArrayList(u32) {
        var tokens = std.ArrayList(u32).init(self.allocator);
        errdefer tokens.deinit();

        var current_pos: usize = 0;
        while (current_pos < text.len) {
            // Check for special characters first
            if (text[current_pos] == ' ') {
                try tokens.append(32); // Space token
                current_pos += 1;
                continue;
            }
            if (text[current_pos] == '\n') {
                try tokens.append(10); // Newline token
                current_pos += 1;
                continue;
            }

            // Regular token matching
            var longest_match: ?struct { token: []const u8, id: u32, len: usize } = null;
            var token_it = self.tokens.iterator();
            while (token_it.next()) |entry| {
                const token = entry.key_ptr.*;
                if (current_pos + token.len <= text.len and
                    std.mem.startsWith(u8, text[current_pos..], token))
                {
                    if (longest_match == null or token.len > longest_match.?.len) {
                        longest_match = .{
                            .token = token,
                            .id = entry.value_ptr.*,
                            .len = token.len,
                        };
                    }
                }
            }

            if (longest_match) |match| {
                try tokens.append(match.id);
                current_pos += match.len;
            } else {
                // Handle unknown byte
                try tokens.append(text[current_pos]);
                current_pos += 1;
            }
        }

        return tokens;
    }

    fn info(self: *const Tokenizer) void {
        std.debug.print("vocab size: {}\n", .{self.tokens.count()});
        std.debug.print("merge list size: {}\n", .{self.merges.items.len});
    }

    fn decode(self: *const Tokenizer, tokens: std.ArrayList(u32)) ![]const u8 {
        var decoded_text = std.ArrayList(u8).init(self.allocator);
        errdefer decoded_text.deinit();

        for (tokens.items) |token_id| {
            // Skip BOS/EOS/PAD tokens
            if (token_id == self.bos_token or token_id == self.eos_token or token_id == self.pad_token) {
                continue;
            }

            // Handle special replacements
            switch (token_id) {
                32 => try decoded_text.appendSlice(" "), // Space
                10 => try decoded_text.appendSlice("\n"), // Newline
                else => {
                    // Regular token handling
                    var token_found = false;
                    var token_it = self.tokens.iterator();
                    while (token_it.next()) |entry| {
                        if (entry.value_ptr.* == token_id) {
                            try decoded_text.appendSlice(entry.key_ptr.*);
                            token_found = true;
                            break;
                        }
                    }

                    if (!token_found) {
                        if (token_id < 256) {
                            try decoded_text.append(@intCast(token_id));
                        } else {
                            return error.TokenNotFound;
                        }
                    }
                },
            }
        }

        return decoded_text.toOwnedSlice();
    }

    fn deinit(self: *Self) void {
        var token_it = self.tokens.keyIterator();
        while (token_it.next()) |key| {
            self.allocator.free(key.*);
        }
        self.tokens.deinit();

        for (self.special_tokens.items) |token| {
            self.allocator.free(token.content);
        }
        self.special_tokens.deinit();

        for (self.merges.items) |merge| {
            self.allocator.free(merge);
        }
        self.merges.deinit();
    }
};

// text model
const Model = struct {
    const Self = @This();
    config: Config,
    weights: Weights,
    tokenizer: Tokenizer,
    state: RunState,
    allocator: Allocator,
    freqs_cis: []Complex,

    fn init(config: Config, weights: Weights, tokenizer: Tokenizer, state: RunState, allocator: Allocator) !Model {
        var model = Model{
            .config = config,
            .weights = weights,
            .tokenizer = tokenizer,
            .state = state,
            .allocator = allocator,
            .freqs_cis = undefined,
        };

        // Initialize rotary embeddings
        const theta = 10000.0;
        model.freqs_cis = try model.precompute_freqs_cis(
            model.config.head_dim,
            model.config.max_pos_embeddings,
            theta,
        );

        return model;
    }
    fn text_model(self: Self, embeddings: []const f32, pos: usize) ![]f32 {
        const dim = self.config.dim;
        const q_len = embeddings.len / dim;

        var hidden_states = try self.allocator.alloc(f32, q_len * dim);
        @memcpy(hidden_states, embeddings);

        for (0..self.config.n_layers) |l| {

            // First apply layer norm
            const ln_in = try self.allocator.alloc(f32, q_len * dim);
            defer self.allocator.free(ln_in);
            @memcpy(ln_in, hidden_states);

            // Layer norm before attention
            try layer_norm(ln_in, &.{dim}, self.weights.t_ln_w[l * dim .. (l + 1) * dim], self.weights.t_ln_b[l * dim .. (l + 1) * dim], 1e-5);

            // Attention using normalized input
            const attn_out = try self.allocator.alloc(f32, q_len * dim);
            defer self.allocator.free(attn_out);
            try self.attention_block(ln_in, pos, q_len, l, attn_out);

            // MLP using normalized input
            const mlp_out = try self.allocator.alloc(f32, q_len * dim);
            defer self.allocator.free(mlp_out);
            try self.mlp(ln_in, q_len, l, mlp_out);

            // Residual connections
            for (0..q_len * dim) |i| {
                hidden_states[i] += attn_out[i] + mlp_out[i];
            }
        }
        return hidden_states;
    }

    fn attention_block(self: Self, x: []f32, pos: usize, q_len: usize, layer: usize, output: []f32) !void {
        const dim = self.config.dim;
        const n_heads = self.config.n_heads;
        const head_dim = self.config.head_dim;

        // 1. QKV Projection [seq_len, dim] -> [seq_len, 3*dim]
        const qkv = try self.allocator.alloc(f32, q_len * dim * 3);
        defer self.allocator.free(qkv);

        try matmul(
            self.allocator,
            x,
            self.weights.t_Wqkv_w[layer * dim * 3 * dim .. (layer + 1) * dim * 3 * dim],
            qkv,
            q_len,
            dim * 3,
            dim,
        );

        // Add QKV bias
        const bias = self.weights.t_Wqkv_b[layer * dim * 3 .. (layer + 1) * dim * 3];
        broadcast_add(qkv, bias, q_len);

        // Debug QKV ranges
        {
            var qkv_min: f32 = std.math.inf(f32);
            var qkv_max: f32 = -std.math.inf(f32);
            for (qkv) |v| {
                if (v < qkv_min) qkv_min = v;
                if (v > qkv_max) qkv_max = v;
            }
            std.debug.print("QKV after projection: min={d:.4}, max={d:.4}\n", .{ qkv_min, qkv_max });
        }

        // 2. Split QKV
        const q = try self.allocator.alloc(f32, n_heads * q_len * head_dim);
        const k = try self.allocator.alloc(f32, n_heads * q_len * head_dim);
        const v = try self.allocator.alloc(f32, n_heads * q_len * head_dim);
        const q_r = try self.allocator.alloc(f32, n_heads * q_len * head_dim);
        const k_r = try self.allocator.alloc(f32, n_heads * q_len * head_dim);
        defer self.allocator.free(q);
        defer self.allocator.free(k);
        defer self.allocator.free(v);
        defer self.allocator.free(q_r);
        defer self.allocator.free(k_r);

        try self.split_qkv(qkv, q, k, v, q_len);

        // 3. Position handling and rotary embeddings
        const position_ids = try self.allocator.alloc(usize, q_len);
        defer self.allocator.free(position_ids);

        for (0..q_len) |i| {
            position_ids[i] = pos + i; // Sequential positions regardless of batch/generation
        }

        try self.apply_rotary_emb(q, self.freqs_cis, position_ids, q_r);
        try self.apply_rotary_emb(k, self.freqs_cis, position_ids, k_r);
        @memcpy(q, q_r);
        @memcpy(k, k_r);

        // 4. KV Cache handling
        const past_seq_len = if (pos > 0) pos else 0;
        const total_seq_len = past_seq_len + q_len;

        if (total_seq_len > self.config.seq_len) {
            return error.SequenceTooLong;
        }

        // Update KV cache at the correct positions
        const cache_layer_offset = layer * self.config.seq_len * dim;
        for (0..q_len) |i| {
            const cache_pos = past_seq_len + i;
            const cache_offset = cache_layer_offset + cache_pos * dim;
            const src_offset = i * dim;

            @memcpy(self.state.k_cache[cache_offset .. cache_offset + dim], k[src_offset .. src_offset + dim]);
            @memcpy(self.state.v_cache[cache_offset .. cache_offset + dim], v[src_offset .. src_offset + dim]);
        }

        // Prepare combined K/V buffers
        const k_combined = try self.allocator.alloc(f32, total_seq_len * dim);
        const v_combined = try self.allocator.alloc(f32, total_seq_len * dim);
        defer self.allocator.free(k_combined);
        defer self.allocator.free(v_combined);

        if (past_seq_len > 0) {
            const cache_start = cache_layer_offset;
            @memcpy(k_combined[0 .. past_seq_len * dim], self.state.k_cache[cache_start .. cache_start + past_seq_len * dim]);
            @memcpy(v_combined[0 .. past_seq_len * dim], self.state.v_cache[cache_start .. cache_start + past_seq_len * dim]);
        }

        @memcpy(k_combined[past_seq_len * dim .. total_seq_len * dim], k[0 .. q_len * dim]);
        @memcpy(v_combined[past_seq_len * dim .. total_seq_len * dim], v[0 .. q_len * dim]);

        // 5. Scaled dot product attention
        const attn_out = try self.allocator.alloc(f32, q_len * dim);
        defer self.allocator.free(attn_out);

        try self.scaled_dot_product_attention(q, k_combined, v_combined, q_len, pos, total_seq_len, attn_out);

        // 6. Output projection
        try matmul(
            self.allocator,
            attn_out,
            self.weights.t_out_proj_w[layer * dim * dim .. (layer + 1) * dim * dim],
            output,
            q_len,
            dim,
            dim,
        );

        const out_bias = self.weights.t_out_proj_bias[layer * dim .. (layer + 1) * dim];
        broadcast_add(output, out_bias, q_len);
    }

    fn split_qkv(self: Self, qkv: []const f32, q: []f32, k: []f32, v: []f32, q_len: usize) !void {
        const dim = self.config.dim;
        const head_dim = self.config.head_dim;
        const n_heads = self.config.n_heads;

        // Important: verify dim = n_heads * head_dim
        std.debug.assert(dim == n_heads * head_dim);

        for (0..q_len) |seq_idx| {
            const qkv_start = seq_idx * (dim * 3);
            const seq_out_start = seq_idx * dim;

            // The input qkv is [seq_len, 3*dim]
            // We want to rearrange it to [seq_len, dim] for each of q,k,v
            // While preserving head arrangement within dim
            @memcpy(q[seq_out_start .. seq_out_start + dim], qkv[qkv_start .. qkv_start + dim]);
            @memcpy(k[seq_out_start .. seq_out_start + dim], qkv[qkv_start + dim .. qkv_start + 2 * dim]);
            @memcpy(v[seq_out_start .. seq_out_start + dim], qkv[qkv_start + 2 * dim .. qkv_start + 3 * dim]);
        }

        // Debug print to verify shapes
        std.debug.print("QKV split dims: seq_len={d}, dim={d}, n_heads={d}, head_dim={d}\n", .{ q_len, dim, n_heads, head_dim });
    }

    fn scaled_dot_product_attention(
        self: Self,
        q: []const f32,
        k: []const f32,
        v: []const f32,
        q_len: usize,
        pos: usize,
        total_seq_len: usize,
        output: []f32,
    ) !void {
        const dim = self.config.dim;
        const n_heads = self.config.n_heads;
        const head_dim = self.config.head_dim;
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

        const attn_mask = try self.allocator.alloc(f32, q_len * total_seq_len);
        defer self.allocator.free(attn_mask);
        make_attention_mask(attn_mask, q_len, pos, total_seq_len);

        const scores = try self.allocator.alloc(f32, q_len * total_seq_len);

        std.debug.assert(scores.len == attn_mask.len);
        std.debug.assert(scores.len == q_len * total_seq_len);
        const head_output = try self.allocator.alloc(f32, q_len * dim);
        defer self.allocator.free(scores);
        defer self.allocator.free(head_output);

        @memset(output, 0); // Important: initialize output to zero

        for (0..n_heads) |h| {
            const head_offset = h * head_dim;

            // Compute scores for this head
            for (0..q_len) |i| {
                const q_offset = i * dim + head_offset;
                for (0..total_seq_len) |j| {
                    const k_offset = j * dim + head_offset;
                    var score: f32 = 0;
                    for (0..head_dim) |d| {
                        score += q[q_offset + d] * k[k_offset + d];
                    }
                    scores[i * total_seq_len + j] = score * scale;
                }
            }

            // Apply mask and softmax
            for (0..q_len) |i| {
                const row_start = i * total_seq_len;
                const row = scores[row_start .. row_start + total_seq_len];

                // Apply mask
                for (0..total_seq_len) |j| {
                    if (attn_mask[i * total_seq_len + j] == 0) {
                        scores[row_start + j] = -std.math.inf(f32);
                    }
                }
                try apply_softmax(row);
            }

            // Debug attention pattern
            if (h == 0) { // Print only for first head
                std.debug.print("Head 0 attention pattern (first token): ", .{});
                const first_row = scores[0..@min(total_seq_len, 5)];
                for (first_row) |score| {
                    std.debug.print("{d:.3} ", .{score});
                }
                std.debug.print("\n", .{});
            }

            // Compute head output - IMPORTANT: use += for accumulation
            for (0..q_len) |i| {
                for (0..total_seq_len) |j| {
                    const score = scores[i * total_seq_len + j];
                    const v_offset = j * dim + head_offset;

                    // Accumulate to this head's section
                    for (0..head_dim) |d| {
                        output[i * dim + head_offset + d] += score * v[v_offset + d];
                    }
                }
            }
        }
    }
    fn make_attention_mask(mask: []f32, q_len: usize, pos: usize, total_seq_len: usize) void {
        @memset(mask, 0.0); // Start with all masked
        std.debug.assert(mask.len == q_len * total_seq_len);

        for (0..q_len) |i| {
            const row_start = i * total_seq_len;
            // Allow attention up to current position
            const attn_up_to = pos + i + 1;
            for (0..attn_up_to) |j| {
                mask[row_start + j] = 1.0;
            }
        }

        // Debug print
        std.debug.print("Mask for pos={d} (first row): ", .{pos});
        for (mask[0..@min(total_seq_len, 10)]) |v| {
            std.debug.print("{d:.0} ", .{v});
        }
        std.debug.print("\n", .{});
    }

    const Complex = struct {
        real: f32,
        imag: f32,
    };

    fn precompute_freqs_cis(self: Self, dim: usize, end: usize, theta: f32) ![]Complex {
        const num_freqs = dim / 2; // For dim=8, this gives us 4 base frequencies

        // Calculate base frequencies exactly like PyTorch
        var freqs = try self.allocator.alloc(f32, num_freqs);
        defer self.allocator.free(freqs);

        // Calculate inverse frequencies exactly like PyTorch
        for (0..num_freqs) |i| {
            // Important: use i*2 to match PyTorch's arange(0, dim, 2)
            const exponent = @as(f32, @floatFromInt(i * 2)) / @as(f32, @floatFromInt(dim));
            freqs[i] = 1.0 / std.math.pow(f32, theta, exponent);
        }

        // Allocate space for complex rotations for each position
        const freqs_cis = try self.allocator.alloc(Complex, end * num_freqs);

        // This is equivalent to t.unsqueeze(1) * freqs.unsqueeze(0)
        for (0..end) |t| {
            for (0..num_freqs) |i| {
                const freq = @as(f32, @floatFromInt(t)) * freqs[i];
                freqs_cis[t * num_freqs + i] = Complex{
                    .real = std.math.cos(freq),
                    .imag = std.math.sin(freq),
                };
            }
        }

        return freqs_cis;
    }

    fn apply_rotary_emb(
        self: Self,
        x: []const f32, // [seq_len, dim] format
        freqs_cis: []const Complex,
        position_ids: []const usize,
        out: []f32,
    ) !void {
        const dim = self.config.dim;
        const head_dim = self.config.head_dim;
        const n_heads = dim / head_dim;
        const seq_len = position_ids.len;
        const rot_dim = @min(head_dim, 32);
        const half_rot_dim = rot_dim / 2;

        assert(x.len == out.len);
        assert(x.len == seq_len * dim);

        // If input and output are the same buffer, we need a temporary buffer
        var temp_buffer: ?[]f32 = null;
        if (x.ptr == out.ptr) {
            temp_buffer = try self.allocator.alloc(f32, rot_dim);
            defer if (temp_buffer) |buf| self.allocator.free(buf);
        }

        // Process each sequence position
        for (0..seq_len) |s| {
            const pos = position_ids[s];
            const seq_offset = s * dim;

            // For each head
            for (0..n_heads) |h| {
                const head_offset = seq_offset + h * head_dim;

                // Handle temporary buffer if needed
                if (temp_buffer) |buf| {
                    @memcpy(buf[0..rot_dim], x[head_offset .. head_offset + rot_dim]);
                }

                // Process pairs within this head's chunk
                var i: usize = 0;
                while (i < half_rot_dim) : (i += 1) {
                    const rot = freqs_cis[pos * rot_dim / 2 + i];

                    const x1 = if (temp_buffer) |buf|
                        buf[i]
                    else
                        x[head_offset + i];
                    const x2 = if (temp_buffer) |buf|
                        buf[i + half_rot_dim]
                    else
                        x[head_offset + i + half_rot_dim];

                    // Apply rotation
                    out[head_offset + i] = x1 * rot.real - x2 * rot.imag;
                    out[head_offset + i + half_rot_dim] = x1 * rot.imag + x2 * rot.real;
                }

                // Handle pass-through portion
                if (rot_dim < head_dim) {
                    if (x.ptr != out.ptr) {
                        const pass_start = head_offset + rot_dim;
                        const pass_len = head_dim - rot_dim;
                        @memcpy(out[pass_start .. pass_start + pass_len], x[pass_start .. pass_start + pass_len]);
                    }
                }
            }
        }
    }

    fn apply_softmax(x: []f32) !void {
        // Find max including -inf values (they won't affect the max)
        var max: f32 = x[0];
        for (x[1..]) |val| {
            if (val > max) max = val;
        }

        // Apply exp and sum
        var sum: f32 = 0;
        for (x) |*val| {
            val.* = @exp(val.* - max);
            sum += val.*; // exp(-inf) becomes 0 automatically
        }

        // Normalize
        const inv_sum = 1.0 / sum;
        for (x) |*val| {
            val.* *= inv_sum;
        }
    }

    fn mlp(self: Self, input: []f32, q_len: usize, l: usize, output: []f32) !void {
        const dim = self.config.dim;
        const hidden_dim = self.config.hidden_dim;

        const fc1_out = try self.allocator.alloc(f32, q_len * hidden_dim);
        defer self.allocator.free(fc1_out);

        const fc2_out = try self.allocator.alloc(f32, q_len * dim);
        defer self.allocator.free(fc2_out);

        // First linear layer
        try matmul(
            self.allocator,
            input,
            self.weights.t_fc1_w[l * dim * hidden_dim .. (l + 1) * dim * hidden_dim],
            fc1_out,
            q_len,
            hidden_dim,
            dim,
        );

        broadcast_add(fc1_out, self.weights.t_fc1_b[l * hidden_dim .. (l + 1) * hidden_dim], q_len);

        // GeLU activation
        gelu(fc1_out);

        // Second linear layer
        try matmul(
            self.allocator,
            fc1_out,
            self.weights.t_fc2_w[l * hidden_dim * dim .. (l + 1) * hidden_dim * dim],
            fc2_out,
            q_len,
            dim,
            hidden_dim,
        );

        broadcast_add(fc2_out, self.weights.t_fc2_b[l * dim .. (l + 1) * dim], q_len);

        @memcpy(output, fc2_out);
    }

    fn lm_head(self: Self, hidden_states: []f32) ![]f32 {
        const dim = self.config.dim;
        const vocab_size = self.config.vocab;

        // Only take the last token's hidden state
        // const last_hidden = hidden_states[(q_len - 1) * dim .. q_len * dim];

        // Normalize before projection
        const normalized = try self.allocator.alloc(f32, dim);
        defer self.allocator.free(normalized);
        @memcpy(normalized, hidden_states);

        try layer_norm(normalized, &.{dim}, self.weights.t_ln_out_w, self.weights.t_ln_out_b, 1e-5);

        // Project to vocabulary
        const logits = try self.allocator.alloc(f32, vocab_size);
        try matmul(self.allocator, normalized, self.weights.t_linear_w, logits, 1, vocab_size, dim);

        accumulate(logits, self.weights.t_linear_b);

        return logits;
    }

    pub const LayerNormError = error{
        EmptyInput,
        DimensionMismatch,
        InvalidEpsilon,
        NaNInput,
        InfInput,
    };

    /// Checks if a floating point value is finite and not NaN
    fn isValid(x: f32) bool {
        return !std.math.isNan(x) and !std.math.isInf(x);
    }

    /// Checks if all values in a slice are valid floating point numbers
    fn validateInputs(values: []const f32) LayerNormError!void {
        for (values) |x| {
            if (std.math.isNan(x)) return LayerNormError.NaNInput;
            if (std.math.isInf(x)) return LayerNormError.InfInput;
        }
    }
    // Gradient checking utility
    fn check_gradients(tensor: []const f32, name: []const u8) void {
        var max_grad: f32 = 0;
        var min_grad: f32 = 0;

        if (tensor.len > 1) {
            for (1..tensor.len) |i| {
                const grad = @abs(tensor[i] - tensor[i - 1]);
                if (grad > max_grad) max_grad = grad;
                if (grad < min_grad) min_grad = grad;
            }
        }

        std.debug.print(
            \\Gradient check for {s}:
            \\  Min gradient: {d:0.6}
            \\  Max gradient: {d:0.6}
            \\
        , .{
            name,
            min_grad,
            max_grad,
        });
    }

    pub fn layer_norm(
        inputs: []f32,
        normalized_shape: []const usize,
        weight: ?[]const f32,
        bias: ?[]const f32,
        eps: f32,
    ) LayerNormError!void {
        // Basic input validation
        if (inputs.len == 0) return LayerNormError.EmptyInput;
        if (eps <= 0) return LayerNormError.InvalidEpsilon;
        try validateInputs(inputs);
        if (weight) |w| try validateInputs(w);
        if (bias) |b| try validateInputs(b);

        // Calculate the number of features being normalized over
        var num_features: usize = 1;
        for (normalized_shape) |dim| {
            num_features *= dim;
        }

        // Verify dimensions match
        if (inputs.len % num_features != 0) {
            return LayerNormError.DimensionMismatch;
        }

        // Check weight and bias dimensions if provided
        if (weight) |w| {
            if (w.len != num_features) return LayerNormError.DimensionMismatch;
        }
        if (bias) |b| {
            if (b.len != num_features) return LayerNormError.DimensionMismatch;
        }

        const num_groups = inputs.len / num_features;

        // Process each normalization group
        var group_idx: usize = 0;
        while (group_idx < num_groups) : (group_idx += 1) {
            const start = group_idx * num_features;
            const end = start + num_features;
            const group = inputs[start..end];

            // Calculate mean and variance using Welford's online algorithm
            var mean: f32 = 0.0;
            var m2: f32 = 0.0; // Second moment about the mean
            var count: f32 = 0.0;

            for (group) |x| {
                count += 1;
                const delta = x - mean;
                mean += delta / count;
                const delta2 = x - mean;
                m2 += delta * delta2;
            }

            // Calculate final variance
            const variance = m2 / (count - 1);
            const inv_std = 1.0 / @sqrt(variance + eps);

            // Normalize and apply affine transform if provided
            for (group, 0..num_features) |*x, i| {
                const x_centered = x.* - mean;
                var normalized = x_centered * inv_std;

                // Apply optional weight and bias
                const w = if (weight) |w_slice| w_slice[i] else 1.0;
                const b = if (bias) |b_slice| b_slice[i] else 0.0;

                normalized = normalized * w + b;
                x.* = normalized;
            }
        }
    }

    fn embed_tokens(self: Self, tokens: std.ArrayList(u32)) ![]f32 {
        var text_embed = try self.allocator.alloc(f32, tokens.items.len * self.config.dim);
        errdefer self.allocator.free(text_embed);

        for (tokens.items, 0..) |token, i| {
            if (token >= self.weights.word_token_embedding.len / self.config.dim) {
                return error.TokenOutOfBounds;
            }
            const src_start = token * self.config.dim;
            const src_end = src_start + self.config.dim;
            const dst_start = i * self.config.dim;
            const dst_end = dst_start + self.config.dim;

            @memcpy(text_embed[dst_start..dst_end], self.weights.word_token_embedding[src_start..src_end]);
        }

        std.debug.print("text embed len {any}", .{text_embed.len});
        return text_embed;
    }

    fn merge_embed(self: Self, text_embed: []f32, image_embed: []f32) ![]f32 {
        const total_len = text_embed.len + image_embed.len;
        const embedding = try self.allocator.alloc(f32, total_len);

        // Image embeddings
        @memcpy(embedding[0..image_embed.len], image_embed);

        @memcpy(embedding[image_embed.len..], text_embed);

        std.debug.print("Merged embeddings: text_len={d}, image_len={d}, total_len={d}\n", .{ text_embed.len, image_embed.len, total_len });

        return embedding;
    }
    /// This function will load the images and then preprocess them into the required format
    pub fn preprocess(self: Self, image_path: []const u8, allocator: Allocator) ![]f32 {
        // Load the image
        const target_height = self.config.img_dim;
        const target_width = self.config.img_dim;
        const batch_size = 2;

        const c_image_path = @as([*c]const u8, @ptrCast(image_path.ptr));
        var width: c_int = 0;
        var height: c_int = 0;
        var channels: c_int = 0;

        const img_data = c.stbi_load(c_image_path, &width, &height, &channels, 0);
        if (img_data == null) {
            return error.FailedToLoadImage;
        }
        defer c.stbi_image_free(img_data);

        std.debug.print("Loaded image with width: {d}, height: {d}, channels: {d}\n", .{ width, height, channels });

        const resized_data = try allocator.alloc(u8, target_height * target_width * @as(usize, @intCast(channels)));

        const result = c.stbir_resize_uint8_srgb(
            img_data,
            width,
            height,
            0,
            resized_data.ptr,
            @as(c_int, @intCast(target_width)),
            @as(c_int, @intCast(target_height)),
            0,
            if (channels == 3) c.STBIR_RGB else c.STBIR_RGBA,
        );

        if (result == 0) {
            return error.FailedToResizeImage;
        }

        var float16_image = try allocator.alloc(f16, batch_size * target_width * target_height * @as(usize, @intCast(channels)));
        const mean = [3]f16{ 0.5, 0.5, 0.5 };
        const stddev = [3]f16{ 0.5, 0.5, 0.5 };

        // reorganize the data from (H,W,C) to (C, H, W) format that torch uses
        // initially data is stored in [R,G,B,R,G,B,R,G,B...R,G,B] format
        // now we want to store it as [R,R,R,R,R,R,..G,G,G,G,G..B,B,B,B,B] format where the RGB values are contiguous
        const single_image_size = target_height * target_width * @as(usize, @intCast(channels)); // single image size of 378 x 378 x 3

        defer allocator.free(float16_image);

        // Process in FP16
        for (0..@as(usize, @intCast(channels))) |ch| {
            for (0..target_height) |h| {
                for (0..target_width) |w| {
                    const src_idx = (h * target_width + w) * @as(usize, @intCast(channels)) + ch;
                    const dst_idx = ch * target_height * target_width + h * target_width + w;

                    const pixel_value: u8 = resized_data[src_idx];
                    const scaled_value = @as(f16, @floatFromInt(pixel_value)) / @as(f16, 255.0);
                    const normalized_value = (scaled_value - mean[ch]) / stddev[ch];
                    float16_image[dst_idx] = normalized_value;
                    if (batch_size > 1) {
                        float16_image[single_image_size + dst_idx] = normalized_value;
                    }
                }
            }
        }

        // Convert to FP32 for export
        var float32_image = try allocator.alloc(f32, float16_image.len);
        for (float16_image, 0..) |value, i| {
            float32_image[i] = @floatCast(value);
        }

        return float32_image;
    }

    fn vision_encoder(self: Self) !void {

        // now the vision encoder will take in the float image and divide it into
        // patches of self.patch_size x self.patch_size (14 x 14)
        // we have to rearrange the values of the patches

        // calculate the number of patches along height and width, these are the same for now
        // because we're using tensors which images resized to 378 x 378 through preprocess()

        // just defining constants from the config here
        const channels = self.config.img_channels; // 3 channels
        const img_h = self.config.img_dim;
        const img_w = self.config.img_dim;
        const patch_h = self.config.patch_size;
        const patch_w = self.config.patch_size;
        const patch_elements = patch_h * patch_w * channels;
        // const num_patches_h = img_h / patch_h;
        // const num_patches_w = img_h / patch_w;
        const num_patches = self.config.num_patches;
        const batch_size: usize = 2;
        const total_patches = batch_size * num_patches;

        // we are going to change the format of our image from (B, C, H, W) to (h * w, C * p1 * p2) or (729, 3 * 14 * 14)

        @memcpy(self.state.patches, try rearrangeBCHWtoBTC(
            self.allocator,
            self.state.img,
            batch_size,
            channels,
            img_h,
            img_w,
            patch_h,
        ));

        // NOTE: This transformation above is correct and verified with pytorch version

        // we get the patch embedding by doing matmul with the patch embedding linear layer and adding bias
        // each patch is individually multiplied and then stored into self.state.patches!

        for (0..total_patches) |patch| {

            // for each patch, we do matrix multiplication
            // (1, 3 * 14 * 14)  @ (3 * 14 * 14, 1152)
            // all this is stored in patch_emb which is (729, 1152)
            try matmul(
                self.allocator,
                self.state.patches[patch * patch_elements .. (patch + 1) * patch_elements],
                self.weights.v_patch_embedding_linear_w,
                self.state.patch_emb[patch * self.config.vit_dim .. (patch + 1) * self.config.vit_dim],
                1,
                self.config.vit_dim,
                patch_elements,
            );
            accumulate(
                self.state.patch_emb[patch * self.config.vit_dim .. (patch + 1) * self.config.vit_dim],
                self.weights.v_patch_embedding_linear_b,
            );
        }

        // next up is positional embedding, which is directly just accumulated into the patch embedding but actually broadcasted across the batch (each image gets its own pos emb)!
        // so for each 729, 1152 batch : x = x + pos_embed
        //essentially pos embed dim (b, 729, 1152) + v_pos_embed (b, (1152, 729) ^ T)
        try self.broadcast_pos_embeddings(
            self.state.patch_emb,
            batch_size,
            num_patches,
            self.config.vit_dim,
        );

        // we will now pass our positionally encoded patch embeddings through the ViT blocks.
        // v_x (b, 729, 1152)

        for (0..self.config.n_vit_layers) |l| {
            @memcpy(self.state.v_x, self.state.patch_emb);
            // std.debug.print("Pre-LN1: range [{d}, {d}], ", .{ min_value(self.state.v_x), max_value(self.state.v_x) });
            // print_mean_std("mean/std", self.state.v_x);

            // we will normalize each patch by iterating over each patch, because our layernorm can only do a single patch vector of inputs
            const shape = [_]usize{self.config.vit_dim}; // Each patch normalized across vit_dim

            // Instead of the loop, process all patches at once
            try layer_norm(
                self.state.v_x, // Pass entire tensor of all patches
                &shape, // Normalize in groups of vit_dim
                self.weights.v_norm1_w[l * self.config.vit_dim .. (l + 1) * self.config.vit_dim],
                self.weights.v_norm1_b[l * self.config.vit_dim .. (l + 1) * self.config.vit_dim],
                1e-5,
            );
            // std.debug.print("Post-LN1: range [{d}, {d}], ", .{ min_value(self.state.v_x), max_value(self.state.v_x) });
            // print_mean_std("mean/std", self.state.v_x);

            // now our patch embedding is normalized, we are going to get the attention weights
            // we multiply our patch embedding to get the qkv for all the patches all at once for the specific layer
            // patch_emb (b, 729, 1152) @ v_Wqkv (1152, 3456) = v_qkv (729, 3456)
            try matmul(
                self.allocator,
                self.state.v_x,
                self.weights.v_Wqkv_w[l * self.config.vit_dim * 3 * self.config.vit_dim .. (l + 1) * self.config.vit_dim * 3 * self.config.vit_dim],
                self.state.v_qkv,
                total_patches,
                self.config.vit_dim * 3,
                self.config.vit_dim,
            );

            // next we accumulate the bias for that layer into v_qkv!
            // we need to iterate over all the patches again to do so!
            for (0..total_patches) |patch| {
                // 1 patch from v_qkv (1, 3456) = 1 patch from v_qkv (1, 3456) + Wqkv_b(3456)
                // over all patches it will be v_qkv(729, 3456)
                accumulate(
                    self.state.v_qkv[patch * self.config.vit_dim * 3 .. (patch + 1) * self.config.vit_dim * 3],
                    self.weights.v_Wqkv_b[l * self.config.vit_dim * 3 .. (l + 1) * self.config.vit_dim * 3],
                );
            }

            //----------

            for (0..total_patches) |patch_idx| {
                const row_start = patch_idx * (self.config.vit_dim * 3);
                const q_start = row_start;
                const k_start = row_start + self.config.vit_dim;
                const v_start = row_start + (self.config.vit_dim * 2);

                // // Print values for first patch
                // if (patch_idx == 0) {
                //     std.debug.print("\nQKV split for first patch:\n", .{});
                //     std.debug.print("Q first 3: {any}\n", .{self.state.v_qkv[q_start .. q_start + 3]});
                //     std.debug.print("K first 3: {any}\n", .{self.state.v_qkv[k_start .. k_start + 3]});
                //     std.debug.print("V first 3: {any}\n", .{self.state.v_qkv[v_start .. v_start + 3]});
                // }

                // Copy into separate buffers
                @memcpy(self.state.v_q[patch_idx * self.config.vit_dim .. (patch_idx + 1) * self.config.vit_dim], self.state.v_qkv[q_start .. q_start + self.config.vit_dim]);
                @memcpy(self.state.v_k[patch_idx * self.config.vit_dim .. (patch_idx + 1) * self.config.vit_dim], self.state.v_qkv[k_start .. k_start + self.config.vit_dim]);
                @memcpy(self.state.v_v[patch_idx * self.config.vit_dim .. (patch_idx + 1) * self.config.vit_dim], self.state.v_qkv[v_start .. v_start + self.config.vit_dim]);
            }

            // After splitting, print ranges
            // std.debug.print("\nAfter QKV split - value ranges:\n", .{});
            // std.debug.print("Q range: [{d}, {d}]\n", .{ min_value(self.state.v_q), max_value(self.state.v_q) });
            // std.debug.print("K range: [{d}, {d}]\n", .{ min_value(self.state.v_k), max_value(self.state.v_k) });
            // std.debug.print("V range: [{d}, {d}]\n", .{ min_value(self.state.v_v), max_value(self.state.v_v) });

            const head_dim = self.config.vit_dim / self.config.n_vit_heads;

            for (0..self.config.n_vit_heads) |head| {
                const v_q_head = self.state.v_q[head * head_dim * total_patches .. (head + 1) * head_dim * total_patches];
                const v_k_head = self.state.v_k[head * head_dim * total_patches .. (head + 1) * head_dim * total_patches];
                const v_v_head = self.state.v_v[head * head_dim * total_patches .. (head + 1) * head_dim * total_patches];

                const attn_slice = self.state.v_attn[head * total_patches * total_patches .. (head + 1) * total_patches * total_patches];

                // Scale Q and K before multiplication
                const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(self.config.vit_head_dim)));
                var scaled_q = try self.allocator.alloc(f32, v_q_head.len);
                defer self.allocator.free(scaled_q);
                var scaled_k = try self.allocator.alloc(f32, v_k_head.len);
                defer self.allocator.free(scaled_k);

                // Scale Q and K
                for (v_q_head, 0..) |val, i| scaled_q[i] = val * scale;
                for (v_k_head, 0..) |val, i| scaled_k[i] = val * scale;

                const k_transposed = try transposeSimd(self.allocator, scaled_k, total_patches, head_dim);
                defer self.allocator.free(k_transposed);

                // std.debug.print("Head {}: Pre-matmul Q range: [{any}, {d}], K range: [{any}, {any}]\n", .{ head, min_value(scaled_q), max_value(scaled_q), min_value(k_transposed), max_value(k_transposed) });

                // QK^T multiplication with pre-scaled values
                try matmul(
                    self.allocator,
                    scaled_q,
                    k_transposed,
                    attn_slice,
                    total_patches,
                    total_patches,
                    head_dim,
                );

                // std.debug.print("After QK^T - scores range: [{any}, {any}]\n", .{ min_value(attn_slice), max_value(attn_slice) });

                // Apply row-wise softmax
                // TODO: Replace with function
                var row: usize = 0;
                while (row < total_patches) : (row += 1) {
                    const row_start = row * total_patches;
                    const row_end = row_start + total_patches;
                    const row_slice = attn_slice[row_start..row_end];

                    // Find max for this row only
                    var max: f32 = -std.math.inf(f32);
                    for (row_slice) |val| {
                        max = @max(max, val);
                    }

                    // Compute exp(x - max) and sum for this row
                    var sum: f32 = 0.0;
                    for (row_slice) |*val| {
                        val.* = @exp(val.* - max);
                        sum += val.*;
                    }

                    // Normalize this row
                    if (sum != 0) {
                        for (row_slice) |*val| {
                            val.* /= sum;
                        }
                    } else {
                        // Handle zero sum case - uniform attention
                        const uniform_val = 1.0 / @as(f32, @floatFromInt(total_patches));
                        for (row_slice) |*val| {
                            val.* = uniform_val;
                        }
                    }
                }

                // std.debug.print("After softmax - weights range: [{any}, {any}]\n", .{ min_value(attn_slice), max_value(attn_slice) });

                // attn @ V multiplication
                try matmul(
                    self.allocator,
                    attn_slice,
                    v_v_head,
                    self.state.v_output[head * total_patches * head_dim .. (head + 1) * total_patches * head_dim],
                    total_patches,
                    head_dim,
                    total_patches,
                );

                // const output_slice = self.state.v_output[head * total_patches * head_dim .. (head + 1) * total_patches * head_dim];
                // std.debug.print("After V multiplication - output range: [{any}, {any}]\n", .{ min_value(output_slice), max_value(output_slice) });
            }

            // std.debug.print("Attention output: range [{d}, {d}], ", .{ min_value(self.state.v_output), max_value(self.state.v_output) });
            // print_mean_std("mean/std", self.state.v_output);

            // Next, we will multiply the final output from all the heads with the attention projection layer for this vit block
            // v_output (total_patches, vit_dim) @  v_out_proj_w (vit_dim, vit_dim) = v_proj (total_patches, vit_dim)

            try matmul(
                self.allocator,
                self.state.v_output,
                self.weights.v_out_proj_w[l * self.config.vit_dim * self.config.vit_dim .. (l + 1) * self.config.vit_dim * self.config.vit_dim],
                self.state.v_proj,
                total_patches,
                self.config.vit_dim,
                self.config.vit_dim,
            );

            for (0..total_patches) |patch| {
                accumulate(self.state.v_proj[patch * self.config.vit_dim .. (patch + 1) * self.config.vit_dim], self.weights.v_out_proj_b[l * self.config.vit_dim .. (l + 1) * self.config.vit_dim]);
            }
            // std.debug.print("Before residual: range [{d}, {d}]\n", .{ min_value(self.state.patch_emb), max_value(self.state.patch_emb) });
            accumulate(self.state.patch_emb, self.state.v_proj);

            // std.debug.print("After residual1: range [{d}, {d}], ", .{ min_value(self.state.patch_emb), max_value(self.state.patch_emb) });
            // print_mean_std("mean/std", self.state.patch_emb);

            // reusing v_x now and saving patch embed as a residual carry
            @memcpy(self.state.v_x, self.state.patch_emb);

            // second layernorm

            try layer_norm(
                self.state.v_x,
                &shape,
                self.weights.v_norm2_w[l * self.config.vit_dim .. (l + 1) * self.config.vit_dim],
                self.weights.v_norm2_b[l * self.config.vit_dim .. (l + 1) * self.config.vit_dim],
                1e-5,
            );

            // std.debug.print("Pre-LN2: range [{d}, {d}], ", .{ min_value(self.state.v_x), max_value(self.state.v_x) });
            // print_mean_std("mean/std", self.state.v_x);

            // pass the normalized v_x through the first MLP, upcasting it, and storing it in a buffer
            // v_x(total_patches, vit_dim) @ fc1 (vit_dim, hidden_features) = v_xb(total_patches, hidden_features)
            try matmul(
                self.allocator,
                self.state.v_x,
                self.weights.v_fc1_w[l * self.config.vit_dim * self.config.hidden_features .. (l + 1) * self.config.vit_dim * self.config.hidden_features],
                self.state.v_xb,
                total_patches,
                self.config.hidden_features,
                self.config.vit_dim,
            );

            // then we accumulate the fc1 bias into v_xb by iterating over num patches
            for (0..total_patches) |patch| {
                // iterate over total_patches (xb (1, hidden_features) = xb (1, hidden_features) + fc1_b (hidden_features)
                accumulate(
                    self.state.v_xb[patch * self.config.hidden_features .. (patch + 1) * self.config.hidden_features],
                    self.weights.v_fc1_b[l * self.config.hidden_features .. (l + 1) * self.config.hidden_features],
                );
            }

            // next we will apply GeLU to the fc1 logits (v_xb) to get the activations
            gelu(self.state.v_xb);

            // after this xb contains the activations from fc1!
            // now we will downcast it through fc2.

            // for this we will multiply v_xb with the fc2 weights and store it in v_xb2
            // v_xb(total_patches, hidden_features) @ fc2 (hidden_features, vit_dim) = v_xb2(total_patches, hidden_features)
            try matmul(
                self.allocator,
                self.state.v_xb,
                self.weights.v_fc2_w[l * self.config.hidden_features * self.config.vit_dim .. (l + 1) * self.config.hidden_features * self.config.vit_dim],
                self.state.v_xb2,
                total_patches,
                self.config.vit_dim,
                self.config.hidden_features,
            );
            // then we accumulate the fc2 bias into v_xb2 by iterating over num patches

            for (0..total_patches) |patch| {
                // iterate over total_patches (xb2 (1, vit_dim) = xb (1, vit_dim) + fc1_b (vit_dim)
                accumulate(
                    self.state.v_xb2[patch * self.config.vit_dim .. (patch + 1) * self.config.vit_dim],
                    self.weights.v_fc2_b[l * self.config.vit_dim .. (l + 1) * self.config.vit_dim],
                );
            }
            // std.debug.print("MLP output: range [{d}, {d}], ", .{ min_value(self.state.v_xb2), max_value(self.state.v_xb2) });
            // print_mean_std("mean/std", self.state.v_xb2);

            // now we finally can merge the mlp output into the residual we've been saving all this long
            accumulate(self.state.patch_emb, self.state.v_xb2);

            // std.debug.print("After residual2: range [{d}, {d}], ", .{ min_value(self.state.patch_emb), max_value(self.state.patch_emb) });
            // print_mean_std("mean/std", self.state.patch_emb);
        }
        // now for the final layernorm..
        const shape = [_]usize{self.config.vit_dim};
        try layer_norm(
            self.state.patch_emb,
            &shape,
            self.weights.v_norm_out_w,
            self.weights.v_norm_out_b,
            1e-5,
        );

        // std.debug.print("After Final LN: range [{d}, {d}], ", .{ min_value(self.state.patch_emb), max_value(self.state.patch_emb) });
        // print_mean_std("mean/std", self.state.patch_emb);

        var concat_buffer = try self.allocator.alloc(f32, num_patches * (self.config.vit_dim * 2));
        defer self.allocator.free(concat_buffer);

        // For each patch position
        for (0..num_patches) |patch_idx| {
            // Copy global patch embeddings (first batch)
            const global_start = patch_idx * self.config.vit_dim;
            const global_end = (patch_idx + 1) * self.config.vit_dim;
            const local_start = (patch_idx + num_patches) * self.config.vit_dim;
            const local_end = (patch_idx + num_patches + 1) * self.config.vit_dim;

            // Copy into concatenated form
            const concat_start = patch_idx * (self.config.vit_dim * 2);
            @memcpy(
                concat_buffer[concat_start .. concat_start + self.config.vit_dim],
                self.state.patch_emb[global_start..global_end],
            );
            @memcpy(
                concat_buffer[concat_start + self.config.vit_dim .. concat_start + (self.config.vit_dim * 2)],
                self.state.patch_emb[local_start..local_end],
            );
        }

        // std.debug.print("Final embeddings: range [{d}, {d}], ", .{ min_value(self.state.final_emb), max_value(self.state.final_emb) });
        // print_mean_std("mean/std", self.state.final_emb);

        // now we will pass the final embed through the projection layers:
        // first we will upcast to (num patches, hidden_dim)
        // final_emb (total_patches , vit_dim * 2) @ v_proj_fc1_w (vit_dim * 2, hidden_dim) = v_xb3(total_patches, hidden_dim)
        try matmul(
            self.allocator,
            concat_buffer,
            self.weights.v_proj_fc1_w,
            self.state.v_xb3,
            num_patches,
            self.config.hidden_dim,
            self.config.vit_dim * 2,
        );

        // next up, we apply the bias on v_xb3
        for (0..num_patches) |patch| {
            accumulate(
                self.state.v_xb3[patch * self.config.hidden_dim .. (patch + 1) * self.config.hidden_dim],
                self.weights.v_proj_fc1_b,
            );
        }

        // std.debug.print("After proj1: range [{d}, {d}], ", .{ min_value(self.state.v_xb3), max_value(self.state.v_xb3) });
        // print_mean_std("mean/std", self.state.v_xb3);

        // we will then apply GeLU on the logits from the first projection layer:
        gelu(self.state.v_xb3);

        // std.debug.print("After GeLU Proj: range [{d}, {d}], ", .{ min_value(self.state.v_xb3), max_value(self.state.v_xb3) });
        // print_mean_std("mean/std", self.state.v_xb3);

        // after this we will pass the activated v_xb3:
        // v_xb3(total_patches, hidden_dim) @ v_proj_fc2_w (hidden_dim, dim) = v_xb3(total_patches, dim)

        try matmul(
            self.allocator,
            self.state.v_xb3,
            self.weights.v_proj_fc2_w,
            self.state.projection,
            num_patches,
            self.config.dim,
            self.config.hidden_dim,
        );

        // then we add the final projection bias

        for (0..num_patches) |patch| {
            accumulate(self.state.projection[patch * self.config.dim .. (patch + 1) * self.config.dim], self.weights.v_proj_fc2_b);
        }
    }

    fn min_value(slice: []const f32) f32 {
        var min: f32 = std.math.inf(f32);
        for (slice) |val| {
            min = @min(min, val);
        }
        return min;
    }

    fn max_value(slice: []const f32) f32 {
        var max: f32 = -std.math.inf(f32);
        for (slice) |val| {
            max = @max(max, val);
        }
        return max;
    }

    // Helper function to compute and print mean/std
    fn print_mean_std(prefix: []const u8, values: []const f32) void {
        var sum: f32 = 0;
        var sum_sq: f32 = 0;
        for (values) |v| {
            sum += v;
            sum_sq += v * v;
        }
        const mean = sum / @as(f32, @floatFromInt(values.len));
        const variance = (sum_sq / @as(f32, @floatFromInt(values.len))) - (mean * mean);
        const std_dev = @sqrt(@max(variance, 0));
        std.debug.print("{s}: mean {d}, std {d}\n", .{ prefix, mean, std_dev });
    }

    fn clip_values(slice: []f32, min_val: f32, max_val: f32) void {
        for (slice) |*val| {
            val.* = @min(@max(val.*, min_val), max_val);
        }
    }

    fn rearrangeBCHWtoBTC(allocator: std.mem.Allocator, input: []const f32, batch: usize, channels: usize, height: usize, width: usize, patch_size: usize) ![]f32 {
        const h_patches = height / patch_size;
        const w_patches = width / patch_size;
        const out_size = batch * (h_patches * w_patches) * (channels * patch_size * patch_size);

        var output = try allocator.alloc(f32, out_size);
        errdefer allocator.free(output);

        // Pre-calculate strides
        const in_c_stride = height * width;
        const in_h_stride = width;
        const in_b_stride = channels * height * width;

        const out_t_stride = channels * patch_size * patch_size;
        const out_b_stride = h_patches * w_patches * out_t_stride;

        // Single loop over all elements
        var idx: usize = 0;
        while (idx < out_size) : (idx += 1) {
            // Decode output index into components
            const b = idx / out_b_stride;
            const hw = (idx % out_b_stride) / out_t_stride;
            const cp = idx % out_t_stride;

            const h = hw / w_patches;
            const w = hw % w_patches;
            const ch = cp / (patch_size * patch_size);
            const p = cp % (patch_size * patch_size);
            const ph = p / patch_size;
            const pw = p % patch_size;

            // Calculate input index
            const in_idx = b * in_b_stride +
                ch * in_c_stride +
                (h * patch_size + ph) * in_h_stride +
                (w * patch_size + pw);

            output[idx] = input[in_idx];
        }

        return output;
    }

    fn broadcast_pos_embeddings(
        self: Self,
        patch_emb: []f32,
        batch_size: usize,
        num_patches: usize,
        vit_dim: usize,
    ) !void {
        // First transpose pos embeddings
        const pos_emb_transposed = try transposeSimd(
            self.allocator,
            self.weights.v_pos_embedding,
            vit_dim, // 1152
            num_patches, // 729
        );
        defer self.allocator.free(pos_emb_transposed);

        // Now add to each batch's patches
        for (0..batch_size) |batch| {
            const start_idx = batch * num_patches * vit_dim;
            const end_idx = start_idx + num_patches * vit_dim;
            accumulate(patch_emb[start_idx..end_idx], pos_emb_transposed);
        }
    }
};

const NumericalStats = struct {
    min: f32,
    max: f32,
    mean: f32,
    std: f32,
    has_nan: bool,
    has_inf: bool,
};

fn compute_stats(data: []const f32) NumericalStats {
    if (data.len == 0) {
        return .{
            .min = 0,
            .max = 0,
            .mean = 0,
            .std = 0,
            .has_nan = false,
            .has_inf = false,
        };
    }

    var stats = NumericalStats{
        .min = data[0],
        .max = data[0],
        .mean = 0,
        .std = 0,
        .has_nan = false,
        .has_inf = false,
    };

    // First pass - min, max, mean, and check for nan/inf
    var sum: f64 = 0; // Use f64 for better accuracy in sum
    for (data) |val| {
        if (std.math.isNan(val)) {
            stats.has_nan = true;
        }
        if (std.math.isInf(val)) {
            stats.has_inf = true;
        }
        if (val < stats.min) stats.min = val;
        if (val > stats.max) stats.max = val;
        sum += val;
    }
    stats.mean = @floatCast(sum / @as(f64, @floatFromInt(data.len)));

    // Second pass - standard deviation
    var sum_squared_diff: f64 = 0;
    for (data) |val| {
        const diff = val - stats.mean;
        sum_squared_diff += diff * diff;
    }
    stats.std = @sqrt(@as(f32, @floatCast(sum_squared_diff / @as(f64, @floatFromInt(data.len)))));

    return stats;
}

pub fn debug_tensor(name: []const u8, data: []const f32, layer: ?usize) void {
    const stats = compute_stats(data);

    std.debug.print(
        \\{s} (layer {?d}):
        \\  Range: [{d:0.6}, {d:0.6}]
        \\  Mean: {d:0.6}, Std: {d:0.6}
        \\  Has NaN: {}, Has Inf: {}
        \\
    , .{
        name,
        layer,
        stats.min,
        stats.max,
        stats.mean,
        stats.std,
        stats.has_nan,
        stats.has_inf,
    });
}

// std.debug.print("Pass! \n", .{});
// inference

fn generate(model: *Model, image_path: []const u8, prompt: []const u8, allocator: std.mem.Allocator) ![]const u8 {
    // Clear KV cache at start
    @memset(model.state.k_cache, 0);
    @memset(model.state.v_cache, 0);

    // std.debug.print("====================================================\n", .{});

    // Process image
    const preprocessed = try model.preprocess(image_path, allocator);
    defer allocator.free(preprocessed);
    @memcpy(model.state.img, preprocessed);

    try model.vision_encoder();

    // Format and tokenize prompt
    const formatted_prompt = try std.fmt.allocPrint(allocator, "<image>\nQuestion: {s}\nAnswer:", .{prompt});
    defer allocator.free(formatted_prompt);

    // Initialize tokens
    var context_tokens = std.ArrayList(u32).init(allocator);
    defer context_tokens.deinit();

    try context_tokens.append(model.tokenizer.bos_token);
    var prompt_tokens = try model.tokenizer.encode(formatted_prompt);

    defer prompt_tokens.deinit();
    try context_tokens.appendSlice(prompt_tokens.items);

    // Initialize state tracking
    var full_context: []f32 = undefined;
    var context_length: usize = 0;
    {
        // Get initial embeddings
        const text_embeddings = try model.embed_tokens(context_tokens);
        defer allocator.free(text_embeddings);

        // const init_embedding = try model.merge_embed(text_embeddings, model.state.projection);
        const init_embedding = text_embeddings;
        defer model.allocator.free(init_embedding);

        // Initialize full context
        context_length = init_embedding.len / model.config.dim;
        full_context = try model.allocator.alloc(f32, init_embedding.len);
        @memcpy(full_context, init_embedding);
    }
    defer model.allocator.free(full_context);

    var pos: usize = 0;
    var output = std.ArrayList(u8).init(allocator);
    errdefer output.deinit();

    // Process initial context
    var hidden_states = try model.text_model(full_context, pos);
    defer model.allocator.free(hidden_states);
    pos += context_length;

    const max_tokens: usize = 10;
    var single_token_list = std.ArrayList(u32).init(model.allocator);
    defer single_token_list.deinit();

    generation: for (0..max_tokens) |_| {
        if (pos >= model.config.seq_len) {
            std.debug.print("\nReached maximum sequence length\n", .{});
            break :generation;
        }

        // Get logits for last token
        const state_dim = model.config.dim;

        const last_token_state = hidden_states[(context_length - 1) * state_dim .. context_length * state_dim];

        const logits = try model.lm_head(last_token_state);
        defer model.allocator.free(logits);

        // const top_k = @min(5, logits.len);
        // const top_indices = try get_top_k_indices(allocator, logits, top_k);

        // std.debug.print("Top k indices: {d}\n", .{top_indices});

        // const next_token = @as(u32, @intCast(argmax(logits)));
        const next_token = blk: {
            const top_k = 5;
            const indices = try get_top_k_indices(allocator, logits, top_k);
            std.debug.print("\nTop logits:\n", .{});
            for (indices) |idx| {
                std.debug.print("Token {d} (logit: {d:.3})\n", .{ idx, logits[idx] });
            }
            break :blk @as(u32, @intCast(argmax(logits)));
        };

        if (next_token == model.tokenizer.eos_token) {
            std.debug.print("\nReached EOS token\n", .{});
            break :generation;
        }

        // Update context tokens
        try context_tokens.append(next_token);

        // Decode and append output
        single_token_list.clearRetainingCapacity();
        try single_token_list.append(next_token);
        const token_str = try model.tokenizer.decode(single_token_list);
        defer model.allocator.free(token_str);
        try output.appendSlice(token_str);

        // Embed new token
        const next_embed = try model.embed_tokens(single_token_list);
        defer model.allocator.free(next_embed);

        // Extend context window
        const new_context = try model.allocator.alloc(f32, (context_length + 1) * state_dim);
        @memcpy(new_context[0 .. context_length * state_dim], full_context);
        @memcpy(new_context[context_length * state_dim ..], next_embed);

        // Update full context
        model.allocator.free(full_context);
        full_context = new_context;
        context_length += 1;

        // Process extended context
        const new_hidden = try model.text_model(full_context, pos);
        model.allocator.free(hidden_states);
        hidden_states = new_hidden;
        pos += 1;
    }

    return output.toOwnedSlice();
}

fn get_top_k_indices(allocator: std.mem.Allocator, logits: []const f32, k: usize) ![]usize {
    var indices = try allocator.alloc(usize, logits.len);
    defer allocator.free(indices);

    // Initialize indices
    for (indices, 0..) |*idx, i| {
        idx.* = i;
    }

    // Sort indices by corresponding logit values
    const Context = struct {
        logits: []const f32,
        pub fn lessThan(ctx: @This(), a_idx: usize, b_idx: usize) bool {
            return ctx.logits[b_idx] < ctx.logits[a_idx]; // Descending order
        }
    };

    std.mem.sort(usize, indices, Context{ .logits = logits }, Context.lessThan);

    // Return top k indices
    var result = try allocator.alloc(usize, k);
    @memcpy(result[0..k], indices[0..k]);
    return result;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{
        .enable_memory_limit = true,
        .safety = true,
        .never_unmap = false,
        .retain_metadata = true,
    }){};
    defer {
        const check = gpa.deinit();
        if (check == .leak) {
            std.debug.print("Memory leak detected!\n", .{});
        }
    }

    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();
    const allocator = arena.allocator();

    // Constants //
    const bin_path: []const u8 = "../moondream_f32.bin";
    const config_path: ?[]const u8 = "../model_config.json";
    const image_path: []const u8 = "images/catmonitor.png";

    // Start of loading config file //
    const config_file = try std.fs.cwd().openFile(config_path.?, .{}); // TODO: Add filereader inside the init function for config itself.
    defer config_file.close();

    const config_size = (try config_file.stat()).size; // store the size of the config file so we can allocate the buffer properly

    const buffer = try allocator.alloc(u8, config_size);
    defer allocator.free(buffer);
    _ = try config_file.readAll(buffer);

    var json_tree = try std.json.parseFromSlice(ConfigReader, allocator, buffer, .{});
    defer json_tree.deinit();

    const config = json_tree.value.config();

    // End of loading config file //

    // Start of loading model checkpoint //
    // TODO: Add NULL checking for the bin path

    // loading weights //
    var weights = try Weights.init(config, bin_path, allocator);
    defer weights.deinit();

    // loading tokenizer //
    var tokenizer = try Tokenizer.fromFile("../tokenizer.bin", allocator);
    defer tokenizer.deinit();

    // loading runstate //
    var state = try RunState.init(allocator, config);
    defer state.deinit(allocator);

    // initializing model struct //
    var model = try Model.init(config, weights, tokenizer, state, allocator);

    const prompt = "Describe the image";
    // const max_new_tokens = 5;

    var timer = try std.time.Timer.start();

    const generated_text = try generate(&model, image_path, prompt, allocator);
    defer allocator.free(generated_text);

    const elapsed_ns = timer.read();
    const seconds = @as(f64, @floatFromInt(elapsed_ns)) / 1e9;

    std.debug.print("\nPrompt: {s}\n", .{prompt});
    std.debug.print("Generated text: {s}\n", .{generated_text});
    std.debug.print("Generation took: {d:.3} seconds.\n", .{seconds});
}

// tests
test "softmax" {
    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };

    try softmax(&x);

    var sum: f32 = 0.0;
    for (0..x.len) |value| {
        sum += x[value];
    }

    try std.testing.expect(sum == 1.0);
}

test "tokenizer" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const check = gpa.deinit();
        if (check == .leak) {
            std.debug.print("WARNING: GPA detected memory leaks!\n", .{});
        }
    }
    const allocator = gpa.allocator();

    var tokenizer = try Tokenizer.fromFile("/home/snow/projects/moondream-zig/tokenizer.bin", allocator);
    defer tokenizer.deinit();

    {
        std.debug.print("\n=== Testing Special Character Encoding ===\n", .{});
        const test_str = "a\nb";
        var tokens = try tokenizer.encode(test_str);
        defer tokens.deinit();

        std.debug.print("Input: 'a\\nb'\n", .{});
        std.debug.print("Tokens: ", .{});
        for (tokens.items) |token| {
            std.debug.print("{} ", .{token});
        }
        std.debug.print("\n", .{});

        const decoded = try tokenizer.decode(tokens);
        defer allocator.free(decoded);
        std.debug.print("Decoded: '{s}'\n", .{decoded});
    }

    // First, let's examine special tokens
    {
        std.debug.print("\n=== Special Tokens Analysis ===\n", .{});
        for (tokenizer.special_tokens.items) |special| {
            std.debug.print("Special token ID {}: '{s}' (is_special: {})\n", .{
                special.id,
                special.content,
                special.is_special,
            });
        }
    }

    // Test full prompt tokenization
    {
        std.debug.print("\n=== Testing Full Prompt Tokenization ===\n", .{});
        const prompt = "<image>\nQuestion: what is in this image?\nAnswer:";
        var tokens = try tokenizer.encode(prompt);
        defer tokens.deinit();

        std.debug.print("Input prompt: '{s}'\n", .{prompt});
        std.debug.print("Token sequence ({} tokens):\n", .{tokens.items.len});
        for (tokens.items, 0..) |token, i| {
            // Try to decode each token individually
            var single_token = std.ArrayList(u32).init(allocator);
            defer single_token.deinit();
            try single_token.append(token);
            const decoded = try tokenizer.decode(single_token);
            defer allocator.free(decoded);

            std.debug.print("Token {}: ID {} -> '{s}'\n", .{ i, token, decoded });
        }
    }

    // Test with BOS token prepended
    {
        std.debug.print("\n=== Testing with BOS Token ===\n", .{});
        const prompt = "<image>\nQuestion: what is in this image?\nAnswer:";
        var tokens = std.ArrayList(u32).init(allocator);
        defer tokens.deinit();

        // Add BOS token
        try tokens.append(tokenizer.bos_token);

        // Add encoded prompt
        var prompt_tokens = try tokenizer.encode(prompt);
        defer prompt_tokens.deinit();
        try tokens.appendSlice(prompt_tokens.items);

        std.debug.print("Full token sequence with BOS ({} tokens):\n", .{tokens.items.len});
        for (tokens.items, 0..) |token, i| {
            var single_token = std.ArrayList(u32).init(allocator);
            defer single_token.deinit();
            try single_token.append(token);
            const decoded = try tokenizer.decode(single_token);
            defer allocator.free(decoded);

            std.debug.print("Token {}: ID {} -> '{s}'\n", .{ i, token, decoded });
        }

        // Try decoding the full sequence
        const full_decoded = try tokenizer.decode(tokens);
        defer allocator.free(full_decoded);
        std.debug.print("\nFull sequence decoded: '{s}'\n", .{full_decoded});
    }

    // Test edge cases
    {
        std.debug.print("\n=== Testing Edge Cases ===\n", .{});

        // Test empty string
        {
            var tokens = try tokenizer.encode("");
            defer tokens.deinit();
            std.debug.print("Empty string produces {} tokens\n", .{tokens.items.len});
        }

        // Test whitespace handling
        {
            var tokens = try tokenizer.encode(" \n \t ");
            defer tokens.deinit();
            std.debug.print("Whitespace string produces {} tokens: ", .{tokens.items.len});
            for (tokens.items) |token| {
                std.debug.print("{} ", .{token});
            }
            std.debug.print("\n", .{});
        }
    }
}

test "matrix_multiplies" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const w = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
    const x = [_]f32{ 1.0, 2.0, 3.0 };
    var xout = [_]f32{ 0.0, 0.0, 0.0 };

    try matmul(allocator, &w, &x, &xout, 3, 1, 3);
    try std.testing.expect(xout[0] == 1.0 + 4.0 + 9.0);
    try std.testing.expect(xout[1] == 4.0 + 10.0 + 18.0);
    try std.testing.expect(xout[2] == 7.0 + 16.0 + 27.0);
}

// test "test outer product correctness and performance" {
//     const A = [_]f32{ 1, 2, 3 };
//     const B = [_]f32{ 4, 5, 6 };
//     const M: usize = 3;
//     const N: usize = 3;

//     // Allocate memory for the result matrices
//     const C_simd = try std.heap.page_allocator.alloc(f32, M * N);
//     defer std.heap.page_allocator.free(C_simd);

//     const C_conventional = try std.heap.page_allocator.alloc(f32, M * N);
//     defer std.heap.page_allocator.free(C_conventional);

//     // Compute the outer product using SIMD
//     const start_simd = std.time.nanoTimestamp();
//     try outer(&A, &B, C_simd, M, N);
//     const end_simd = std.time.nanoTimestamp();

//     // Compute the outer product using conventional method
//     const start_conventional = std.time.nanoTimestamp();
//     try outerConventional(&A, &B, C_conventional, M, N);
//     const end_conventional = std.time.nanoTimestamp();

//     // Print the results
//     std.debug.print("SIMD Time: {d} ns\n", .{end_simd - start_simd});
//     std.debug.print("Conventional Time: {d} ns\n", .{end_conventional - start_conventional});

//     // Check correctness
//     for (C_simd, 0..C_simd.len) |elem, idx| {
//         try std.testing.expectEqual(elem, C_conventional[idx]);
//     }
// }

// test "matmul - outer product verification" {
//     const testing = std.testing;
//     const allocator = testing.allocator;

//     // Define input vectors
//     const M: usize = 3; // Size of first vector
//     const K: usize = 1; // Middle dimension (always 1 for outer product)
//     const N: usize = 2; // Size of second vector

//     // Create input vectors
//     const a = [_]f32{ 1, 2, 3 }; // 3x1 vector
//     const b = [_]f32{ 4, 5 }; // 1x2 vector

//     // Create output matrix
//     const C = try allocator.alloc(f32, M * N);
//     defer allocator.free(C);

//     // Zero initialize the output matrix
//     @memset(C, 0);

//     // Perform outer product using matmul
//     try matmul(allocator, &a, &b, C, M, N, K);

//     // Expected result matrix (3x2)
//     const expected = [_]f32{
//         4.0, 5.0, // 1 * [4, 5]
//         8.0, 10.0, // 2 * [4, 5]
//         12.0, 15.0, // 3 * [4, 5]
//     };

//     // Verify results
//     for (C, 0..) |val, i| {
//         try testing.expectApproxEqAbs(expected[i], val, 0.0001);
//     }

//     // Optional: Print the result matrix for visualization
//     std.debug.print("\nOuter product result:\n", .{});
//     for (0..M) |i| {
//         for (0..N) |j| {
//             std.debug.print("{d:.1} ", .{C[i * N + j]});
//         }
//         std.debug.print("\n", .{});
//     }

//     for (0..M) |i| {
//         for (0..N) |j| {
//             std.debug.print("{d:.1} ", .{expected[i * N + j]});
//         }
//         std.debug.print("\n", .{});
//     }
// }

// test "matmul dimension checks" {
//     const allocator = std.testing.allocator;

//     // Correct dimensions for matrix multiplication
//     const M: usize = 3;
//     const N: usize = 4;
//     const K: usize = 2;

//     // Allocate matrices with correct dimensions
//     const A = try allocator.alloc(f32, M * K);
//     defer allocator.free(A);
//     const B = try allocator.alloc(f32, K * N);
//     defer allocator.free(B);
//     const C = try allocator.alloc(f32, M * N);
//     defer allocator.free(C);

//     // Test with correct dimensions (should not trigger assertion)
//     try matmul(allocator, A, B, C, M, N, K);

//     // Allocate matrices with incorrect dimensions
//     const A_wrong = try allocator.alloc(f32, M * (K + 1)); // Incorrect dimension
//     defer allocator.free(A_wrong);
//     const B_wrong = try allocator.alloc(f32, K * N);
//     defer allocator.free(B_wrong);
//     const C_wrong = try allocator.alloc(f32, M * N);
//     defer allocator.free(C_wrong);

//     // Test with incorrect dimensions (should trigger assertion)
//     // This should cause an assertion failure, which we can catch in the test
//     const result = std.testing.expectError(error.Unexpected, matmul(allocator, A_wrong, B_wrong, C_wrong, M, N, K));
//     try std.testing.expect(result == error.Unexpected);
// }
