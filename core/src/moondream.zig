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
    assert(x.len > 0);
    // max of x for numerical stability
    var max: f32 = x[0];
    for (x[1..]) |val| {
        if (val > max) {
            max = val;
        }
    }
    // exp and sum
    var sum: f32 = 0.0;
    for (x) |*val| {
        val.* = std.math.exp(val.* - max); // https://stackoverflow.com/questions/42599498/numerically-stable-softmax
        sum += val.*;
    }
    // normalize
    for (x) |*val| {
        val.* /= sum;
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

fn transposeSimd(allocator: std.mem.Allocator, matrix: []const f32, rows: usize, cols: usize) ![]f32 {
    const transposed = try allocator.alloc(f32, rows * cols);
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

                @as(*VectorType, @alignCast(@ptrCast(transposed.ptr + j * rows + i))).* = vec;
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
    // Slices for specific weights

    ///Text model start///
    word_token_embedding: []f32, // (dim, vocab)

    // Transformer layer start

    // attn layer norm
    t_ln_w: []f32, // (layer, dim)
    t_ln_b: []f32, // (layer, dim)
    // attn qkv
    t_Wqkv_w: []f32, // (layer, dim, n_heads*head_dim*3)
    t_Wqkv_b: []f32, // (layer, n_heads*head_dim*3)
    // output
    t_out_proj_w: []f32, // (layer, seqlen, dim)
    t_out_proj_bias: []f32, // (layer, dim)
    // fully connected
    t_fc1_w: []f32, // (layer, hidden_dim, dim)
    t_fc1_b: []f32, // (layer, hidden_dim)
    t_fc2_w: []f32, // (layer, dim, hidden_dim)
    t_fc2_b: []f32, // (layer, dim)

    //Transformer layer end //

    // lm head
    t_linear_w: []f32, //(vocab, dim)
    t_linear_b: []f32, //(vocab)
    t_ln_out_w: []f32, //(dim)
    t_ln_out_b: []f32, //(dim)
    //Text model end///

    // Vision model start //

    // combining patch embeddngs and pos
    v_patch_embedding_linear_w: []f32, // (vit_dim, patch * patch * channels)
    v_patch_embedding_linear_b: []f32, // (vit_dim)
    v_pos_embedding: []f32, // (1, (img_dim/patch_dim)^2, vit_dim)

    /// Vision Transformer Start
    // attention qkv
    v_Wqkv_w: []f32, // (vit_dim, vit_dim*3)
    v_Wqkv_b: []f32, // (vit_dim * 3)

    //attn out
    v_out_proj_w: []f32, // (vit_dim, vit_dim)
    v_out_proj_b: []f32, // (vit_dim)

    //ViT fc
    v_fc1_w: []f32, // (hidden_features, vit_dim)
    v_fc1_b: []f32, // (hidden_features)
    v_fc2_w: []f32, // (vit_dim, hidden_features)
    v_fc2_b: []f32, // (vit_dim)

    //ViT norm
    v_norm1_w: []f32, // (layer, hidden_features)
    v_norm1_b: []f32, // (layer, hidden_features)
    v_norm2_w: []f32, // (layer, hidden_features)
    v_norm2_b: []f32, // (layer, hidden_features)

    // Vision Transformer End

    //norm
    v_norm_out_w: []f32, // (hidden_features)
    v_norm_out_b: []f32, // (hidden_features)

    // projection
    v_proj_fc1_w: []f32, // (hidden_dim, hidden_features * 2)
    v_proj_fc1_b: []f32, // (hidden_dim)

    v_proj_fc2_w: []f32, // (hidden_features*2, hidden_dim)
    v_proj_fc2_b: []f32, // (hidden_features)

    fn init(config: Config, filename: []const u8, allocator: Allocator) !Weights {

        // Set up slices for each weight

        const sizes = calculateSizes(config);
        const num_weights = calculateTotalSize(sizes);
        const file = try std.fs.cwd().openFile(filename, .{});
        defer file.close();

        const file_size = try file.getEndPos();
        if (file_size != num_weights * @sizeOf(f32)) {
            std.debug.print("Actual file size = {} \n", .{file_size});
            std.debug.print("Estimated file size = {} \n", .{num_weights * @sizeOf(f32)});
            return error.UnexpectedFileSize;
        }
        std.debug.print("{s} read successfully with {} parameters. \n", .{ filename, num_weights });
        var data = try allocator.alloc(f32, num_weights);
        defer allocator.free(data);

        const bytes_read = try file.readAll(std.mem.sliceAsBytes(data));
        if (bytes_read != file_size) {
            return error.IncompleteRead;
        }

        // Memory mapping of slices
        // Set slices for each weight
        var self: Weights = undefined;
        var offset: usize = 0;
        // text model
        self.word_token_embedding = data[offset .. offset + sizes.word_token_embedding];
        offset += sizes.word_token_embedding;

        self.t_ln_w = data[offset .. offset + sizes.t_ln_w];
        offset += sizes.t_ln_w;

        self.t_ln_b = data[offset .. offset + sizes.t_ln_b];
        offset += sizes.t_ln_b;

        self.t_Wqkv_w = data[offset .. offset + sizes.t_Wqkv_w];
        offset += sizes.t_Wqkv_w;

        self.t_Wqkv_b = data[offset .. offset + sizes.t_Wqkv_b];
        offset += sizes.t_Wqkv_b;

        self.t_out_proj_w = data[offset .. offset + sizes.t_out_proj_w];
        offset += sizes.t_out_proj_w;

        self.t_out_proj_bias = data[offset .. offset + sizes.t_out_proj_bias];
        offset += sizes.t_out_proj_bias;

        self.t_fc1_w = data[offset .. offset + sizes.t_fc1_w];
        offset += sizes.t_fc1_w;

        self.t_fc1_b = data[offset .. offset + sizes.t_fc1_b];
        offset += sizes.t_fc1_b;

        self.t_fc2_w = data[offset .. offset + sizes.t_fc2_w];
        offset += sizes.t_fc2_w;

        self.t_fc2_b = data[offset .. offset + sizes.t_fc2_b];
        offset += sizes.t_fc2_b;

        self.t_linear_w = data[offset .. offset + sizes.t_linear_w];
        offset += sizes.t_linear_w;

        self.t_linear_b = data[offset .. offset + sizes.t_linear_b];
        offset += sizes.t_linear_b;

        self.t_ln_out_w = data[offset .. offset + sizes.t_ln_out_w];
        offset += sizes.t_ln_out_w;

        self.t_ln_out_b = data[offset .. offset + sizes.t_ln_out_b];
        offset += sizes.t_ln_out_b;

        // vision model

        self.v_patch_embedding_linear_w = data[offset .. offset + sizes.v_patch_embedding_linear_w];
        offset += sizes.v_patch_embedding_linear_w;

        self.v_patch_embedding_linear_b = data[offset .. offset + sizes.v_patch_embedding_linear_b];
        offset += sizes.v_patch_embedding_linear_b;

        self.v_pos_embedding = data[offset .. offset + sizes.v_pos_embedding];
        offset += sizes.v_pos_embedding;

        self.v_Wqkv_w = data[offset .. offset + sizes.v_Wqkv_w];
        offset += sizes.v_Wqkv_w;

        self.v_Wqkv_b = data[offset .. offset + sizes.v_Wqkv_b];
        offset += sizes.v_Wqkv_b;

        self.v_out_proj_w = data[offset .. offset + sizes.v_out_proj_w];
        offset += sizes.v_out_proj_w;

        self.v_out_proj_b = data[offset .. offset + sizes.v_out_proj_b];
        offset += sizes.v_out_proj_b;

        self.v_fc1_w = data[offset .. offset + sizes.v_fc1_w];
        offset += sizes.v_fc1_w;

        self.v_fc1_b = data[offset .. offset + sizes.v_fc1_b];
        offset += sizes.v_fc1_b;

        self.v_fc2_w = data[offset .. offset + sizes.v_fc2_w];
        offset += sizes.v_fc2_w;

        self.v_fc2_b = data[offset .. offset + sizes.v_fc2_b];
        offset += sizes.v_fc2_b;

        self.v_norm1_w = data[offset .. offset + sizes.v_norm1_w];
        offset += sizes.v_norm1_w;

        self.v_norm1_b = data[offset .. offset + sizes.v_norm1_b];
        offset += sizes.v_norm1_b;

        self.v_norm2_w = data[offset .. offset + sizes.v_norm2_w];
        offset += sizes.v_norm2_w;

        self.v_norm2_b = data[offset .. offset + sizes.v_norm2_b];
        offset += sizes.v_norm2_b;

        self.v_norm_out_w = data[offset .. offset + sizes.v_norm_out_w];
        offset += sizes.v_norm_out_w;

        self.v_norm_out_b = data[offset .. offset + sizes.v_norm_out_b];
        offset += sizes.v_norm_out_b;

        self.v_proj_fc1_w = data[offset .. offset + sizes.v_proj_fc1_w];
        offset += sizes.v_proj_fc1_w;

        self.v_proj_fc1_b = data[offset .. offset + sizes.v_proj_fc1_b];
        offset += sizes.v_proj_fc1_b;

        self.v_proj_fc2_w = data[offset .. offset + sizes.v_proj_fc2_w];
        offset += sizes.v_proj_fc2_w;

        self.v_proj_fc2_b = data[offset .. offset + sizes.v_proj_fc2_b];
        offset += sizes.v_proj_fc2_b;

        std.debug.print("Mapped weights and biases successfully. \n", .{});
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

    fn deinit(self: *Self, allocator: Allocator) void {
        inline for (std.meta.fields(Self)) |f| {
            allocator.free(@field(self, f.name));
        }
        self.* = undefined;
    }
};

// runstate

const RunState = struct {
    const Self = @This();

    img: []align(simd_align) f32,
    patches: []align(simd_align) f32,
    patch_emb: []align(simd_align) f32,
    final_emb: []align(simd_align) f32,
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
        return Self{
            .img = try allocator.alignedAlloc(f32, simd_align, config.img_dim * config.img_dim * config.img_channels),
            .patches = try allocator.alignedAlloc(f32, simd_align, config.img_dim * config.img_dim * config.img_channels),
            .patch_emb = try allocator.alignedAlloc(f32, simd_align, config.num_patches * config.vit_dim),
            .final_emb = try allocator.alignedAlloc(f32, simd_align, config.num_patches * 2 * config.vit_dim),
            .projection = try allocator.alignedAlloc(f32, simd_align, config.num_patches * config.dim),
            .v_x = try allocator.alignedAlloc(f32, simd_align, config.num_patches * config.vit_dim),
            .v_xb = try allocator.alignedAlloc(f32, simd_align, config.num_patches * config.hidden_features),
            .v_xb2 = try allocator.alignedAlloc(f32, simd_align, config.num_patches * config.vit_dim),
            .v_xb3 = try allocator.alignedAlloc(f32, simd_align, config.num_patches * config.hidden_dim),
            .v_qkv = try allocator.alignedAlloc(f32, simd_align, config.num_patches * config.vit_dim * 3),
            .v_q = try allocator.alignedAlloc(f32, simd_align, config.num_patches * config.vit_dim),
            .v_k = try allocator.alignedAlloc(f32, simd_align, config.num_patches * config.vit_dim),
            .v_v = try allocator.alignedAlloc(f32, simd_align, config.num_patches * config.vit_dim),
            .v_attn = try allocator.alignedAlloc(f32, simd_align, config.num_patches * config.num_patches * config.vit_dim),
            .v_output = try allocator.alignedAlloc(f32, simd_align, config.num_patches * config.vit_dim),
            .v_proj = try allocator.alignedAlloc(f32, simd_align, config.num_patches * config.vit_dim),
            .k_cache = try allocator.alignedAlloc(f32, simd_align, config.n_layers * config.seq_len * config.dim),
            .v_cache = try allocator.alignedAlloc(f32, simd_align, config.n_layers * config.seq_len * config.dim),
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
const Tokenizer = struct {
    // TODO : Add handling external tokens to tokenizer
    const Self = @This();
    tokens: std.StringHashMap(u32),
    merges: std.ArrayList([]const u8),
    allocator: Allocator,
    eos_token: u32,

    fn init(allocator: Allocator) Tokenizer {
        return .{
            .tokens = std.StringHashMap(u32).init(allocator),
            .merges = std.ArrayList([]const u8).init(allocator),
            .allocator = allocator,
            .eos_token = 50256,
        };
    }

    fn fromFile(filename: []const u8, allocator: Allocator) !Tokenizer {
        var self = Tokenizer.init(allocator);
        errdefer self.deinit();

        const file = try std.fs.cwd().openFile(filename, .{});
        defer file.close();

        var reader = file.reader();

        // Read number of tokens
        const num_tokens = try reader.readInt(u32, .little);

        // Read tokens
        var i: u32 = 0;
        while (i < num_tokens) : (i += 1) {
            const token_id = try reader.readInt(u32, .little);
            const token_len = try reader.readInt(u32, .little);
            const token_content = try allocator.alloc(u8, token_len);
            errdefer allocator.free(token_content);
            _ = try reader.readAll(token_content);

            try self.tokens.put(token_content, token_id);
        }

        // Read number of merges
        const num_merges = try reader.readInt(u32, .little);

        // Read merges
        i = 0;
        while (i < num_merges) : (i += 1) {
            const first_len = try reader.readInt(u16, .little);
            const first = try allocator.alloc(u8, first_len);
            _ = try reader.readAll(first);
            defer allocator.free(first);

            const second_len = try reader.readInt(u16, .little);
            const second = try allocator.alloc(u8, second_len);
            _ = try reader.readAll(second);
            defer allocator.free(second);

            const merge = try std.fmt.allocPrint(allocator, "{s} {s}", .{ first, second });
            try self.merges.append(merge);
        }

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

        var words = std.mem.split(u8, text, " ");
        while (words.next()) |word| {
            var current_word = word;
            while (current_word.len > 0) {
                var longest_token: ?[]const u8 = null;
                var longest_token_id: ?u32 = null;

                // Find the longest matching token
                var token_it = self.tokens.iterator();
                while (token_it.next()) |entry| {
                    const token = entry.key_ptr.*;
                    if (std.mem.startsWith(u8, current_word, token)) {
                        if (longest_token == null or token.len > longest_token.?.len) {
                            longest_token = token;
                            longest_token_id = entry.value_ptr.*;
                        }
                    }
                }

                if (longest_token) |token| {
                    try tokens.append(longest_token_id.?);
                    current_word = current_word[token.len..];
                } else {
                    // If no token matches, treat the first byte as an unknown token
                    try tokens.append(current_word[0]);
                    current_word = current_word[1..];
                }
            }
        }

        return tokens;
    }

    fn info(self: *const Tokenizer) void {
        std.debug.print("vocab size: {}\n", .{self.tokens.count()});
        std.debug.print("merge list size: {}\n", .{self.merges.items.len});
    }

    // TODO: Write Decode Function.
    fn decode(self: *const Tokenizer, tokens: std.ArrayList(u32)) ![]const u8 {
        var decoded_text = std.ArrayList(u8).init(self.allocator);
        errdefer decoded_text.deinit();

        for (tokens.items) |token_id| {
            var found = false;
            var token_it = self.tokens.iterator();
            while (token_it.next()) |entry| {
                if (entry.value_ptr.* == token_id) {
                    const token = entry.key_ptr.*;
                    if (token.len > 1) {
                        switch (token[1]) {
                            0xA0 => {
                                if (token[1] == 0xA0) {
                                    // This is 'Ġ' (0xC4 0xA0 in UTF-8)
                                    try decoded_text.append(' ');
                                    try decoded_text.appendSlice(token[2..]);
                                } else {
                                    try decoded_text.appendSlice(token);
                                }
                            },
                            0x82 => {
                                if (token.len > 1 and token[1] == 0x82) {
                                    // This is 'Ċ' (0xC4 0x82 in UTF-8)
                                    try decoded_text.append('\n');
                                    try decoded_text.appendSlice(token[2..]);
                                } else {
                                    try decoded_text.appendSlice(token);
                                }
                            },
                            else => try decoded_text.appendSlice(token),
                        }
                    }
                    found = true;
                    break;
                }
            }

            if (!found) {
                std.debug.print("Token not found: {}\n", .{token_id});
                return error.TokenNotFound;
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

    fn init(config: Config, weights: Weights, tokenizer: Tokenizer, state: RunState, allocator: Allocator) !Model {
        return Model{
            .config = config,
            .weights = weights,
            .tokenizer = tokenizer,
            .state = state,
            .allocator = allocator,
        };
    }

    fn text_model(self: Self, embeddings: []f32, pos: usize) !void {
        // each token embedding is of size dim, so we divide by it to see
        // how many tokens are present
        // so we perform this assert before making any further calculations
        assert(embeddings.len % self.config.dim == 0);

        // we define our layernorm epsilon constant:
        const eps = 1e-5;
        const dim = self.config.dim;
        const n_heads = self.config.n_heads;
        const head_dim = self.config.head_dim;

        //now we find the number of tokens, or the query length

        const q_len = embeddings.len / dim;

        // we will add the incoming q_len to the current pos which indicates the sequence length of the key and value vectors
        // this also allows us to keep track of the current position in the kv cache

        std.debug.print("q_len: {any} \n", .{q_len});

        // other constants
        //layernorm input
        const ln_in = try self.allocator.alloc(f32, embeddings.len);
        defer self.allocator.free(ln_in);

        // qkv states tensor
        const qkv = try self.allocator.alloc(f32, q_len * dim * 3);
        defer self.allocator.free(qkv);

        const q = try self.allocator.alloc(f32, q_len * dim);
        const k = try self.allocator.alloc(f32, q_len * dim);
        const v = try self.allocator.alloc(f32, q_len * dim);
        defer self.allocator.free(q);
        defer self.allocator.free(k);
        defer self.allocator.free(v);

        const q_r = try self.allocator.alloc(f32, q_len * dim);
        const k_r = try self.allocator.alloc(f32, q_len * dim);
        const v_r = try self.allocator.alloc(f32, q_len * dim);
        // Allocate memory for query_rot and query_pass
        // Allocate rotary and pass-through parts with proper sizes
        const rotary_dim = head_dim / 2; // Half for rotation
        const query_rot = try self.allocator.alloc(f32, n_heads * q_len * rotary_dim);
        defer self.allocator.free(query_rot);
        const query_pass = try self.allocator.alloc(f32, n_heads * q_len * rotary_dim);
        defer self.allocator.free(query_pass);
        const key_rot = try self.allocator.alloc(f32, n_heads * q_len * rotary_dim);
        defer self.allocator.free(key_rot);
        const key_pass = try self.allocator.alloc(f32, n_heads * q_len * rotary_dim);
        defer self.allocator.free(key_pass);
        const kv_seq_len = q_len + pos;
        std.debug.print("kv seq len : {any} \n", .{kv_seq_len});
        const scores = try self.allocator.alloc(f32, q_len * kv_seq_len);
        defer self.allocator.free(scores);

        for (0..self.config.n_layers) |l| {

            // get the residual before performing layernorm
            // this way the original input embeddings will act as a residual

            // we then perform layernorm

            for (0..q_len) |query| {
                try layer_norm(
                    ln_in[query * dim .. (query + 1) * dim],
                    self.weights.t_ln_w[l * dim .. (l + 1) * dim],
                    self.weights.t_ln_b[l * dim .. (l + 1) * dim],
                    eps,
                );
            }

            // next we will perform sdpa

            // we will first extract the q, k, v states as a combined vector:
            try matmul(
                self.allocator,
                ln_in,
                self.weights.t_Wqkv_w[l * dim * 3 * dim .. (l + 1) * dim * 3 * dim],
                qkv,
                q_len,
                dim * 3,
                dim,
            );

            // printMatrix(q_len, dim * 3, qkv, 2, 5);

            // we split qkv into three state vectors for q, k, v

            // Manual strided copy
            for (0..q_len * dim) |i| {
                q[i] = qkv[i * 3];
                k[i] = qkv[i * 3 + 1];
                v[i] = qkv[i * 3 + 2];
            }

            // we then convert the q, k, v vectors into a different view of dims (n_heads, q_len, head_dim)
            // TODO : Check if they actually are (n_heads, q_len, head_dim)
            for (0..n_heads) |h| {
                for (0..q_len) |i| {
                    for (0..head_dim) |j| {
                        const old_index = i * dim + h * head_dim + j;
                        const new_index = h * q_len * head_dim + i * head_dim + j;
                        q_r[new_index] = q[old_index];
                        k_r[new_index] = k[old_index];
                        v_r[new_index] = v[old_index];
                    }
                }
            }

            // first we need to calculate the inverse frequencies.
            // the length of the sin and cos cache is equal to the max sequence length = 2048
            // we the obtain the position frequencies from the cos sin cache for the current position + the number of tokens we are processing, which would be from 0 till the position of the current token

            // we will then apply RoPE
            // after that, we update the kv cache, and hence the pos counter for the kv cache will also update
            // we update this in the generate function
            // after we finish generation, we reset pos to 0

            const freqs = self.get_rot_emb(kv_seq_len);
            const cos = freqs.@"0";
            const sin = freqs.@"1";

            // add query rot and query pass here:

            // Split the query states into rotary and pass-through parts
            for (0..n_heads) |h| {
                for (0..q_len) |i| {
                    const base_old = h * q_len * head_dim + i * head_dim;
                    const base_new = h * q_len * rotary_dim + i * rotary_dim;

                    // Copy first half to rotary (0 to rotary_dim)
                    @memcpy(query_rot[base_new .. base_new + rotary_dim], q_r[base_old .. base_old + rotary_dim]);
                    @memcpy(key_rot[base_new .. base_new + rotary_dim], k_r[base_old .. base_old + rotary_dim]);

                    // Copy second half to pass (rotary_dim to head_dim)
                    @memcpy(query_pass[base_new .. base_new + rotary_dim], q_r[base_old + rotary_dim .. base_old + head_dim]);
                    @memcpy(key_pass[base_new .. base_new + rotary_dim], k_r[base_old + rotary_dim .. base_old + head_dim]);
                }
            }

            try self.apply_rope(query_rot, key_rot, cos, sin, pos, q_len);

            // Merge rotary and pass parts back into q_r and k_r
            for (0..n_heads) |h| {
                for (0..q_len) |i| {
                    const base_old = h * q_len * rotary_dim + i * rotary_dim;
                    const base_new = h * q_len * head_dim + i * head_dim;

                    // Copy rotary part back (first half)
                    @memcpy(q_r[base_new .. base_new + rotary_dim], query_rot[base_old .. base_old + rotary_dim]);
                    @memcpy(k_r[base_new .. base_new + rotary_dim], key_rot[base_old .. base_old + rotary_dim]);

                    // Copy pass part back (second half)
                    @memcpy(q_r[base_new + rotary_dim .. base_new + head_dim], query_pass[base_old .. base_old + rotary_dim]);
                    @memcpy(k_r[base_new + rotary_dim .. base_new + head_dim], key_pass[base_old .. base_old + rotary_dim]);
                }
            }

            // Update KV cache
            const l_off = self.config.seq_len * dim;
            @memcpy(self.state.k_cache[l * l_off ..], k_r);
            @memcpy(self.state.v_cache[l * l_off ..], v_r);

            // attention

            try self.attention(q_r, k_r, v_r, embeddings, pos, q_len, l);

            // mlp

            try self.mlp(embeddings, embeddings, q_len, l);
        }
    }

    fn lm_head(self: Self, hidden_states: []f32, q_len: usize) ![]f32 {
        const dim = self.config.dim;
        const vocab_size = self.config.vocab;

        // If q_len > 1, we need to process each token to get their logits
        // Allocate memory for all logits (vocab_size for each token in q_len)
        const logits = try self.allocator.alloc(f32, q_len * vocab_size);

        // Process each token in the sequence
        for (0..q_len) |i| {
            // Get the hidden state for the current token
            const token_hidden = hidden_states[i * dim .. (i + 1) * dim];

            // Allocate memory for normalized hidden state
            const normalized = try self.allocator.alloc(f32, dim);
            defer self.allocator.free(normalized);

            // Copy current token's hidden state to normalize it
            @memcpy(normalized, token_hidden);

            // Apply layer normalization
            try layer_norm(
                normalized,
                self.weights.t_ln_out_w,
                self.weights.t_ln_out_b,
                1e-5, // epsilon
            );

            // Project to vocabulary size using lm_head weights
            try matmul(
                self.allocator,
                normalized,
                self.weights.t_linear_w,
                logits[i * vocab_size .. (i + 1) * vocab_size],
                1, // batch size is 1 for each token
                vocab_size,
                dim,
            );

            // Add bias if it exists

            accumulate(logits, self.weights.t_linear_b);
        }

        return logits;
    }

    fn attention_mask(allocator: std.mem.Allocator, pos: usize, seq_len: usize) ![]f32 {
        // Total sequence length includes both past (pos) and current (seq_len) tokens
        const total_seq_len = pos + seq_len;

        // Allocate 2D mask of shape [seq_len, total_seq_len]
        const mask = try allocator.alloc(f32, seq_len * total_seq_len);

        // Fill everything with 1s first - this allows attention to all past tokens
        @memset(mask, 1.0);

        // For the current sequence part, implement causal masking
        for (0..seq_len) |i| {
            for (0..seq_len) |j| {
                // Calculate position in the full sequence
                const col = pos + j;
                const mask_idx = i * total_seq_len + col;

                // Ensure we don't write out of bounds
                if (mask_idx >= mask.len) {
                    return error.IndexOutOfBounds;
                }

                // Create causal mask: token i can only attend to tokens 0..=i
                mask[mask_idx] = if (j <= i) 1.0 else 0.0;
            }
        }

        return mask;
    }

    fn attention(self: Self, q: []f32, k: []f32, v: []f32, embeddings: []f32, pos: usize, q_len: usize, l: usize) !void {
        const n_heads = self.config.n_heads;
        const head_dim = self.config.head_dim;
        const kv_seq_len = q_len + pos;
        const dim = self.config.dim;

        // separate scores are allocated for each head:
        const scores = try self.allocator.alloc(f32, n_heads * q_len * kv_seq_len);
        self.allocator.free(scores);

        // attention output will be the same size as the input embedding
        const attn_output = try self.allocator.alloc(f32, q_len * dim);
        defer self.allocator.free(attn_output);

        @memset(attn_output, 0);

        const mask = try attention_mask(self.allocator, pos, q_len);
        defer self.allocator.free(mask);

        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

        // Compute attention scores for each head
        for (0..n_heads) |h| {
            // Calculate scaled dot product attention
            // Q * K^T / sqrt(head_dim)

            for (0..q_len) |i| {
                for (0..kv_seq_len) |j| {
                    var score: f32 = 0.0;
                    const q_offset = h * q_len * head_dim + i * head_dim;
                    const k_offset = h * kv_seq_len * head_dim + j * head_dim;

                    // Compute dot product
                    for (0..head_dim) |d| {
                        score += q[q_offset + d] * k[k_offset + d];
                    }

                    // Scale and apply mask
                    score *= scale;
                    score *= mask[i * kv_seq_len + j];

                    scores[h * q_len * kv_seq_len + i * kv_seq_len + j] = score;
                }
            }
        }

        // Apply softmax to scores
        for (0..n_heads) |h| {
            for (0..q_len) |i| {
                const start_idx = h * q_len * kv_seq_len + i * kv_seq_len;
                const end_idx = start_idx + kv_seq_len;
                try softmax(scores[start_idx..end_idx]);
            }
        }

        // Compute attention output
        for (0..n_heads) |h| {
            for (0..q_len) |i| {
                for (0..head_dim) |d| {
                    var sum: f32 = 0.0;
                    for (0..kv_seq_len) |j| {
                        const score = scores[h * q_len * kv_seq_len + i * kv_seq_len + j];
                        const v_val = v[h * kv_seq_len * head_dim + j * head_dim + d];
                        sum += score * v_val;
                    }
                    const out_idx = i * dim + h * head_dim + d;
                    attn_output[out_idx] = sum;
                }
            }
        }

        // Allocate temporary buffer for projection output
        const proj_output = try self.allocator.alloc(f32, q_len * dim);
        defer self.allocator.free(proj_output);

        try matmul(
            self.allocator,
            attn_output,
            self.weights.t_out_proj_w[l * self.config.dim * self.config.dim .. (l + 1) * self.config.dim * self.config.dim],
            proj_output,
            q_len,
            dim,
            dim,
        );

        accumulate(proj_output, self.weights.t_out_proj_bias[l * self.config.dim .. (l + 1) * self.config.dim]);
        accumulate(embeddings, proj_output);
    }

    fn mlp(self: Self, input: []f32, embeddings: []f32, q_len: usize, l: usize) !void {
        const dim = self.config.dim;
        const hidden_dim = self.config.hidden_dim;

        const fc1_out = try self.allocator.alloc(f32, q_len * hidden_dim);
        defer self.allocator.free(fc1_out);

        const fc2_out = try self.allocator.alloc(f32, q_len * dim);
        defer self.allocator.free(fc2_out);

        // first we will perform the first linear layer

        try matmul(
            self.allocator,
            input,
            self.weights.t_fc1_w[l * dim * hidden_dim .. (l + 1) * dim * hidden_dim],
            fc1_out,
            q_len,
            hidden_dim,
            dim,
        );

        accumulate(fc1_out, self.weights.t_fc1_b[l * hidden_dim .. (l + 1) * hidden_dim]);

        // we will then apply the gelu activation function

        gelu(fc1_out);

        // we will then perform the second linear layer
        // downcasting the activated vector from hidden_dim to dim

        try matmul(
            self.allocator,
            fc1_out,
            self.weights.t_fc2_w[l * hidden_dim * dim .. (l + 1) * hidden_dim * dim],
            fc2_out,
            q_len,
            dim,
            hidden_dim,
        );

        accumulate(fc2_out, self.weights.t_fc2_b[l * dim .. (l + 1) * dim]);

        // we will then add the residual connection

        accumulate(embeddings, fc2_out);
    }

    fn getvector(self: Self, vector: []f32, head_index: usize, seq_index: usize, head_dim_index: usize) f32 {
        // Calculate the flat index for `q` assuming it's shaped as (num_heads, q_len, head_dim)
        const flat_index = seq_index * self.config.dim + head_index * self.config.head_dim + head_dim_index;
        return vector[flat_index];
    }

    // we will then transform each of q, k, v vectors
    // they will go from (q_len, dim) to (n_heads, qlen, head_dim)
    // dim = n_heads * head_dim

    pub fn layer_norm(
        inputs: []f32,
        weight: []const f32,
        bias: []const f32,
        eps: f32,
    ) !void {
        const len = inputs.len;
        if (len == 0) return error.EmptyInput;
        if (len != weight.len or len != bias.len) return error.DimensionMismatch;

        // Compute the mean
        var mean: f32 = 0.0;
        for (inputs) |x| {
            mean += x;
        }
        const n: f32 = @floatFromInt(len);
        mean /= n;

        // Compute the variance
        var variance: f32 = 0.0;
        for (inputs) |x| {
            const diff = x - mean;
            variance += diff * diff;
        }
        variance /= n;

        // Compute standard deviation
        const std_dev = @sqrt(variance + eps);

        // Normalize the inputs
        for (inputs, 0..inputs.len) |*x, i| {
            const normalized = (x.* - mean) / std_dev;
            x.* = normalized * weight[i] + bias[i];

            // Check for numerical stability
            if (std.math.isNan(x.*) or std.math.isInf(x.*)) {
                std.debug.print("Warning: Output contains NaN or Inf at index {d}. Input: {d}, Normalized: {d}, Weight: {d}, Bias: {d}, Mean: {d}, Std: {d}\n", .{ i, x.*, normalized, weight[i], bias[i], mean, std_dev });
                return error.NumericalInstability;
            }
        }
    }

    fn set_cos_sin_cache(self: Self) !void {
        const max_cached_seq_len = self.config.max_pos_embeddings;
        const head_dim = self.config.head_dim;
        const half_dim = head_dim / 2; // This should be 32 for head_dim=64
        const base = self.config.rope_theta;

        std.debug.print("Cache initialization:\n", .{});
        std.debug.print("max_seq_len: {d}\n", .{max_cached_seq_len});
        std.debug.print("head_dim: {d}, half_dim: {d}\n", .{ head_dim, half_dim });
        std.debug.print("cache size should be: {d}\n", .{max_cached_seq_len * half_dim});

        // Calculate inverse frequencies
        const inv_freq = try self.allocator.alloc(f32, half_dim);
        defer self.allocator.free(inv_freq);

        // Fill inverse frequencies
        for (0..half_dim) |i| {
            const x = @as(f32, @floatFromInt(i * 2)) / @as(f32, @floatFromInt(head_dim));
            inv_freq[i] = 1.0 / std.math.pow(f32, base, x);
        }

        // Generate position indices
        const t = try self.allocator.alloc(f32, max_cached_seq_len);
        defer self.allocator.free(t);

        for (0..max_cached_seq_len) |i| {
            t[i] = @floatFromInt(i);
        }

        // Compute frequencies - this should output max_cached_seq_len * half_dim
        const freqs = try self.allocator.alloc(f32, max_cached_seq_len * half_dim);
        defer self.allocator.free(freqs);

        // Compute outer product
        for (0..max_cached_seq_len) |i| {
            for (0..half_dim) |j| {
                freqs[i * half_dim + j] = t[i] * inv_freq[j];
            }
        }

        // At this point freqs is of size max_cached_seq_len * half_dim
        // We directly apply cos and sin to freqs
        try cos_2d(freqs, self.state.cos_cache);
        try sin_2d(freqs, self.state.sin_cache);

        // Add verification
        std.debug.print("freqs.len: {d}, cos_cache.len: {d}\n", .{ freqs.len, self.state.cos_cache.len });
        if (self.state.cos_cache.len != max_cached_seq_len * half_dim or
            self.state.sin_cache.len != max_cached_seq_len * half_dim)
        {
            return error.CacheSizeMismatch;
        }
    }

    fn get_rot_emb(self: Self, seq_len: usize) struct { []f32, []f32 } {
        const half_dim = self.config.head_dim / 2;
        // Should create arrays of size seq_len * half_dim
        return .{
            .@"0" = self.state.cos_cache[0 .. seq_len * half_dim],
            .@"1" = self.state.sin_cache[0 .. seq_len * half_dim],
        };
    }

    fn apply_rope(self: Self, q: []f32, k: []f32, cos: []f32, sin: []f32, pos: usize, qlen: usize) !void {
        const n_heads = self.config.n_heads;
        const head_dim = self.config.head_dim;
        const half_dim = head_dim / 2;

        std.debug.print("RoPE debug:\n", .{});
        std.debug.print("cos len: {d}\n", .{cos.len});
        std.debug.print("head_dim: {d}\n", .{head_dim});
        std.debug.print("half_dim: {d}\n", .{half_dim});
        std.debug.print("pos: {d}\n", .{pos});
        std.debug.print("qlen: {d}\n", .{qlen});
        std.debug.print("max position access will be: {d}\n", .{(pos + qlen - 1) * half_dim + (half_dim - 1)});

        // Verify dimensions
        if (q.len != n_heads * qlen * head_dim) {
            std.debug.print("Expected q length: {d}, got: {d}\n", .{ n_heads * qlen * head_dim, q.len });
            return error.DimensionMismatch;
        }

        const q_rotated = try self.allocator.alloc(f32, q.len);
        defer self.allocator.free(q_rotated);
        const k_rotated = try self.allocator.alloc(f32, k.len);
        defer self.allocator.free(k_rotated);

        for (0..n_heads) |h| {
            for (0..qlen) |seq_idx| {
                const base_idx = h * qlen * head_dim + seq_idx * head_dim;

                // Debug the indexing
                if (base_idx >= q.len or base_idx + head_dim > q.len) {
                    std.debug.print("Invalid base_idx: {d}, q.len: {d}\n", .{ base_idx, q.len });
                    return error.IndexOutOfBounds;
                }
                const position = pos + seq_idx;

                // First do the rotation
                // For first half: copy negated values from second half
                for (0..half_dim) |dim| {
                    q_rotated[base_idx + dim] = -q[base_idx + dim + half_dim];
                    k_rotated[base_idx + dim] = -k[base_idx + dim + half_dim];
                }
                // For second half: copy values from first half
                for (0..half_dim) |dim| {
                    q_rotated[base_idx + half_dim + dim] = q[base_idx + dim];
                    k_rotated[base_idx + half_dim + dim] = k[base_idx + dim];
                }

                // Then apply the rotation using cos/sin
                for (0..half_dim) |dim| {
                    const cos_val = cos[position * half_dim + dim];
                    const sin_val = sin[position * half_dim + dim];

                    // Apply to first half
                    const idx1 = base_idx + dim;
                    const idx2 = base_idx + dim + half_dim;

                    if (idx2 >= q.len) {
                        std.debug.print("Index out of bounds: {d} >= {d}\n", .{ idx2, q.len });
                        return error.IndexOutOfBounds;
                    }

                    const q1 = q[idx1];
                    const q2 = q[idx2];
                    q[idx1] = q1 * cos_val - q2 * sin_val;
                    q[idx2] = q1 * sin_val + q2 * cos_val;

                    const k1 = k[idx1];
                    const k2 = k[idx2];
                    k[idx1] = k1 * cos_val - k2 * sin_val;
                    k[idx2] = k1 * sin_val + k2 * cos_val;
                }
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

        return text_embed;
    }

    fn merge_embed(self: Self, text_embed: []f32, image_embed: []f32) ![]f32 {
        const embedding = try self.allocator.alloc(f32, text_embed.len + image_embed.len);
        std.debug.print("Merging text embed of size {any} and image embed of size {any} \n", .{ text_embed.len / self.config.dim, image_embed.len / self.config.dim });
        @memcpy(embedding[0..image_embed.len], image_embed);
        @memcpy(embedding[0..text_embed.len], text_embed);
        return embedding;
    }
    /// This function will load the images and then preprocess them into the required format
    pub fn preprocess(self: Self, image_path: []const u8, allocator: Allocator) ![]f32 {
        // Load the image
        const target_height = self.config.img_dim;
        const target_width = self.config.img_dim;

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

        var float_image = try allocator.alloc(f32, target_width * target_height * @as(usize, @intCast(channels)));
        const mean = [3]f32{ 0.5, 0.5, 0.5 };
        const stddev = [3]f32{ 0.5, 0.5, 0.5 };

        // reorganize the data from (H,W,C) to (C, H, W) format that torch uses
        // initially data is stored in [R,G,B,R,G,B,R,G,B...R,G,B] format
        // now we want to store it as [R,R,R,R,R,R,..G,G,G,G,G..B,B,B,B,B] format where the RGB values are contiguous
        for (0..@as(usize, @intCast(channels))) |ch| {
            for (0..target_height) |h| {
                for (0..target_width) |w| {
                    const src_idx = (h * target_width + w) * @as(usize, @intCast(channels)) + ch;
                    const dst_idx = ch * target_height * target_width + h * target_width + w;

                    const pixel_value: u8 = resized_data[src_idx];
                    // scale to 0-1 range
                    const scaled_value = @as(f32, @floatFromInt(pixel_value)) / 255.0;
                    // apply normalization
                    const normalized_value = (scaled_value - mean[ch]) / stddev[ch];
                    float_image[dst_idx] = normalized_value;
                }
            }
        }
        return float_image;
    }
    fn vision_encoder(self: Self) !void {
        std.debug.print("Image len : {any} \n", .{self.state.img.len});
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
        const num_patches_h = img_h / patch_h;
        const num_patches_w = img_h / patch_w;
        const num_patches = self.config.num_patches;

        // we are going to change the format of our image from (C, H, W) to (h * w, C * p1 * p2) or (729, 3 * 14 * 14)
        for (0..num_patches_h) |h_patch| {
            for (0..num_patches_w) |w_patch| {
                for (0..channels) |ch| {
                    for (0..patch_h) |h| {
                        for (0..patch_w) |w| {
                            const src_idx = ch * img_h * img_w + (h_patch * patch_h + h) * img_w + (w_patch * patch_w + w);
                            const dest_idx = (h_patch * num_patches_w + w_patch) * channels * patch_h * patch_w + (ch * patch_h * patch_w + h * patch_w + w);
                            self.state.patches[dest_idx] = self.state.img[src_idx];
                        }
                    }
                }
            }
        }

        // we get the patch embedding by doing matmul with the patch embedding linear layer and adding bias
        // each patch is individually multiplied and then stored into self.state.patches!

        for (0..num_patches) |patch| {
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

        // next up is positional embedding, which is directly just accumulated into the patch embedding!
        // x = x + pos_embed
        // pos embed dim (729, 1152) + v_pos_embed (1152, 729) ^ Transposed
        accumulate(self.state.patch_emb, try transposeSimd(self.allocator, self.weights.v_pos_embedding, 1152, 729));

        // we will now pass our positionally encoded patch embeddings through the ViT blocks.
        // v_x (729, 1152)

        for (0..self.config.n_vit_layers) |l| {
            @memcpy(self.state.v_x, self.state.patch_emb);

            // we will normalize each patch by iterating over each patch, because our layernorm can only do a single patch vector of inputs
            for (0..num_patches) |patch| {
                try layer_norm(
                    self.state.v_x[patch * self.config.vit_dim .. (patch + 1) * self.config.vit_dim],
                    self.weights.v_norm1_w[l * self.config.vit_dim .. (l + 1) * self.config.vit_dim],
                    self.weights.v_norm1_b[l * self.config.vit_dim .. (l + 1) * self.config.vit_dim],
                    1e-5,
                );
            }
            // now our patch embedding is normalized, we are going to get the attention weights
            // we multiply our patch embedding to get the qkv for all the patches all at once for the specific layer
            // patch_emb (729, 1152) @ v_Wqkv (1152, 3456) = v_qkv (729, 3456)
            try matmul(
                self.allocator,
                self.state.v_x,
                self.weights.v_Wqkv_w[l * self.config.vit_dim * 3 * self.config.vit_dim .. (l + 1) * self.config.vit_dim * 3 * self.config.vit_dim],
                self.state.v_qkv,
                num_patches,
                self.config.vit_dim * 3,
                self.config.vit_dim,
            );
            // next we accumulate the bias for that layer into v_qkv!
            // we need to iterate over all the patches again to do so!
            for (0..num_patches) |patch| {
                // 1 patch from v_qkv (1, 3456) = 1 patch from v_qkv (1, 3456) + Wqkv_b(3456)
                // over all patches it will be v_qkv(729, 3456)
                accumulate(
                    self.state.v_qkv[patch * self.config.vit_dim * 3 .. (patch + 1) * self.config.vit_dim * 3],
                    self.weights.v_Wqkv_b[l * self.config.vit_dim * 3 .. (l + 1) * self.config.vit_dim * 3],
                );
            }

            @memcpy(self.state.v_q, self.state.v_qkv[0 .. num_patches * self.config.vit_dim]);
            @memcpy(self.state.v_k, self.state.v_qkv[num_patches * self.config.vit_dim .. num_patches * self.config.vit_dim * 2]);
            @memcpy(self.state.v_v, self.state.v_qkv[num_patches * self.config.vit_dim * 2 .. num_patches * self.config.vit_dim * 3]);

            const head_dim = self.config.vit_dim / self.config.n_vit_heads;

            for (0..self.config.n_vit_heads) |head| {
                const v_q_head = self.state.v_q[head * head_dim * num_patches .. (head + 1) * head_dim * num_patches];
                const v_k_head = self.state.v_k[head * head_dim * num_patches .. (head + 1) * head_dim * num_patches];
                const v_v_head = self.state.v_v[head * head_dim * num_patches .. (head + 1) * head_dim * num_patches];

                // Compute the attention score by taking the dot product of the query and key for this head
                // v_q_head (num_patches, head_dim) @ v_k_head.T (head_dim, num_patches) = v_attn (num_patches, num_patches)
                try matmul(
                    self.allocator,
                    v_q_head,
                    v_k_head, // We would need to transpose v_k_head for this operation
                    self.state.v_attn[head * num_patches * num_patches .. (head + 1) * num_patches * num_patches],
                    num_patches,
                    num_patches,
                    head_dim,
                );

                // Scale the attention scores by sqrt(head_dim) to stabilize gradients as per the scaled dot-product attention mechanism
                const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
                for (0..num_patches * num_patches) |i| {
                    self.state.v_attn[head * num_patches * num_patches + i] *= scale;
                }

                // Apply softmax to get the attention probabilities
                // We will be applying softmax row-wise over the num_patches dimension
                try softmax(self.state.v_attn[head * num_patches * num_patches .. (head + 1) * num_patches * num_patches]);

                // Multiply attention probabilities with value matrix to get the final output for this head
                // v_attn (num_patches, num_patches) @ v_v_head (num_patches, head_dim) = output (num_patches, head_dim)
                try matmul(
                    self.allocator,
                    self.state.v_attn[head * num_patches * num_patches .. (head + 1) * num_patches * num_patches],
                    v_v_head,
                    self.state.v_output[head * num_patches * head_dim .. (head + 1) * num_patches * head_dim],
                    num_patches,
                    head_dim,
                    num_patches,
                );
            }

            // Next, we will multiply the final output from all the heads with the attention projection layer for this vit block
            // v_output (num_patches, vit_dim) @  v_out_proj_w (vit_dim, vit_dim) = v_proj (num_patches, vit_dim)

            try matmul(
                self.allocator,
                self.state.v_output,
                self.weights.v_out_proj_w[l * self.config.vit_dim * self.config.vit_dim .. (l + 1) * self.config.vit_dim * self.config.vit_dim],
                self.state.v_proj,
                num_patches,
                self.config.vit_dim,
                self.config.vit_dim,
            );

            //TODO : investigate if the qkv weights and proj weights use ReLU???

            for (0..num_patches) |patch| {
                accumulate(self.state.v_proj[patch * self.config.vit_dim .. (patch + 1) * self.config.vit_dim], self.weights.v_out_proj_b[l * self.config.vit_dim .. (l + 1) * self.config.vit_dim]);
            }

            accumulate(self.state.patch_emb, self.state.v_proj);

            // reusing v_x now and saving patch embed as a residual carry
            @memcpy(self.state.v_x, self.state.patch_emb);

            // second layernorm
            for (0..num_patches) |patch| {
                try layer_norm(
                    self.state.v_x[patch * self.config.vit_dim .. (patch + 1) * self.config.vit_dim],
                    self.weights.v_norm2_w[l * self.config.vit_dim .. (l + 1) * self.config.vit_dim],
                    self.weights.v_norm2_b[l * self.config.vit_dim .. (l + 1) * self.config.vit_dim],
                    1e-5,
                );
            }

            // pass the normalized v_x through the first MLP, upcasting it, and storing it in a buffer
            // v_x(num_patches, vit_dim) @ fc1 (vit_dim, hidden_features) = v_xb(num_patches, hidden_features)

            try matmul(
                self.allocator,
                self.state.v_x,
                self.weights.v_fc1_w[l * self.config.vit_dim * self.config.hidden_features .. (l + 1) * self.config.vit_dim * self.config.hidden_features],
                self.state.v_xb,
                num_patches,
                self.config.hidden_features,
                self.config.vit_dim,
            );

            // then we accumulate the fc1 bias into v_xb by iterating over num patches
            for (0..num_patches) |patch| {
                // iterate over num_patches (xb (1, hidden_features) = xb (1, hidden_features) + fc1_b (hidden_features)
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
            // v_xb(num_patches, hidden_features) @ fc2 (hidden_features, vit_dim) = v_xb2(num_patches, hidden_features)
            try matmul(
                self.allocator,
                self.state.v_xb,
                self.weights.v_fc2_w[l * self.config.hidden_features * self.config.vit_dim .. (l + 1) * self.config.hidden_features * self.config.vit_dim],
                self.state.v_xb2,
                num_patches,
                self.config.vit_dim,
                self.config.hidden_features,
            );
            // then we accumulate the fc2 bias into v_xb2 by iterating over num patches

            for (0..num_patches) |patch| {
                // iterate over num_patches (xb2 (1, vit_dim) = xb (1, vit_dim) + fc1_b (vit_dim)
                accumulate(
                    self.state.v_xb2[patch * self.config.vit_dim .. (patch + 1) * self.config.vit_dim],
                    self.weights.v_fc1_b[l * self.config.vit_dim .. (l + 1) * self.config.vit_dim],
                );
            }

            // now we finally can merge the mlp output into the residual we've been saving all this long
            accumulate(self.state.patch_emb, self.state.v_xb2);
        }
        // now for the final layernorm..

        for (0..num_patches) |patch| {
            try layer_norm(
                self.state.patch_emb[patch * self.config.vit_dim .. (patch + 1) * self.config.vit_dim],
                self.weights.v_norm_out_w,
                self.weights.v_norm_out_b,
                1e-5,
            );
        }

        for (0..num_patches) |patch| {
            // 0 to 1152
            // 1152 to 2304
            @memcpy(self.state.final_emb[patch * self.config.vit_dim .. (patch + 1) * self.config.vit_dim], self.state.patch_emb[patch * self.config.vit_dim .. (patch + 1) * self.config.vit_dim]);
            @memcpy(self.state.final_emb[(patch + 1) * self.config.vit_dim .. (patch + 2) * self.config.vit_dim], self.state.patch_emb[patch * self.config.vit_dim .. (patch + 1) * self.config.vit_dim]);
        }

        // now we will pass the final embed through the projection layers:
        // first we will upcast to (num patches, hidden_dim)
        // final_emb (num_patches , vit_dim * 2) @ v_proj_fc1_w (vit_dim * 2, hidden_dim) = v_xb3(num_patches, hidden_dim)
        try matmul(
            self.allocator,
            self.state.final_emb,
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

        // we will then apply GeLU on the logits from the first projection layer:
        gelu(self.state.v_xb3);

        // after this we will pass the activated v_xb3:
        // v_xb3(num_patches, hidden_dim) @ v_proj_fc2_w (hidden_dim, dim) = v_xb3(num_patches, dim)
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
};
// std.debug.print("Pass! \n", .{});
// inference
fn generate(model: *Model, image_path: []const u8, prompt: []const u8, allocator: std.mem.Allocator) ![]const u8 {
    const preprocessed = try model.preprocess(image_path, allocator);
    defer allocator.free(preprocessed);

    @memcpy(model.state.img, preprocessed);
    try model.vision_encoder();

    // Format prompt and encode tokens
    const formatted_prompt = try std.fmt.allocPrint(allocator, "\n\nQuestion: {s}\n\nAnswer:", .{prompt});
    defer allocator.free(formatted_prompt);

    var tokens = try model.tokenizer.encode(formatted_prompt);
    defer tokens.deinit();

    var tokens_with_eos = std.ArrayList(u32).init(allocator);
    defer tokens_with_eos.deinit();
    try tokens_with_eos.append(model.tokenizer.eos_token);
    try tokens_with_eos.appendSlice(tokens.items);

    // Get initial embeddings
    var initial_tokens = std.ArrayList(u32).init(model.allocator);
    defer initial_tokens.deinit();
    try initial_tokens.appendSlice(tokens_with_eos.items);
    const text_embed = try model.embed_tokens(initial_tokens);
    defer model.allocator.free(text_embed);

    std.debug.print("Text embeddings len : {any} \n", .{text_embed.len});
    std.debug.print("Image embeddings len : {any} \n", .{model.state.projection.len});

    const init_embedding = try model.merge_embed(text_embed, model.state.projection);
    defer allocator.free(init_embedding);

    // we precompute the cos and sin cache for the positional encoding
    try model.set_cos_sin_cache();

    // // first we will perform multiquery attention on the vision and text embeddings from the image + prompt

    var pos: usize = 0; // this defines the current position in the kv cache, or the number of tokens already processed. This starts at 0.

    // // first we will process the prompt and the image embeddings altogether

    try model.text_model(init_embedding, pos);

    // // we will add the incoming q_len to the current pos which indicates the sequence length of the key and value vectors

    pos += init_embedding.len / model.config.dim;
    std.debug.print("pos : {any} \n", .{pos});

    // Initialize output buffer
    var output = std.ArrayList(u8).init(allocator);
    defer output.deinit();

    var current_embeddings = std.ArrayList(f32).init(allocator);
    defer current_embeddings.deinit();

    // Create ArrayList for tokens to decode
    var token_list = std.ArrayList(u32).init(model.allocator);
    defer token_list.deinit();

    // Single token list for embedding
    var single_token_list = std.ArrayList(u32).init(model.allocator);
    defer single_token_list.deinit();

    const max_tokens: usize = 100;
    generation: for (0..max_tokens) |_| {
        const logits = try model.lm_head(
            current_embeddings.items,
            current_embeddings.items.len / model.config.dim,
        );

        defer model.allocator.free(logits);

        const next_token = try sampleNextToken(logits);

        std.debug.print("Done!\n", .{});

        // Check for EOS
        if (next_token == model.tokenizer.eos_token) {
            std.debug.print("EOS token found!\n", .{});
            break :generation;
        }

        // Clear token lists and add next token
        token_list.clearRetainingCapacity();
        try token_list.append(next_token);

        single_token_list.clearRetainingCapacity();
        try single_token_list.append(next_token);

        // Decode and append to output
        const token_str = try model.tokenizer.decode(token_list);
        defer model.allocator.free(token_str);
        try output.appendSlice(token_str);

        const next_embed = try model.embed_tokens(single_token_list);
        defer model.allocator.free(next_embed);

        // Clear current embeddings and set to next token's embeddings
        try current_embeddings.resize(next_embed.len);
        @memcpy(current_embeddings.items, next_embed);

        // Process next token
        try model.text_model(current_embeddings.items, pos);
        pos += 1;
    }

    return output.toOwnedSlice();
}

fn sampleNextToken(logits: []f32) !u32 {
    // Use argmax to find the index of the maximum logit
    const max_index = argmax(logits);
    return @intCast(max_index);
}

// main

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Constants //
    const bin_path: []const u8 = "../moondream_f32.bin";
    const config_path: ?[]const u8 = "../model_config.json";
    const image_path: []const u8 = "images/frierenburger.png";

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
    defer weights.deinit(allocator);

    // loading tokenizer //
    var tokenizer = try Tokenizer.fromFile("../tokenizer.bin", allocator);
    defer tokenizer.deinit();

    // loading runstate //
    var state = try RunState.init(allocator, config);
    defer state.deinit(allocator);

    // initializing model struct //
    var model = try Model.init(config, weights, tokenizer, state, allocator);

    const prompt = "Hello";
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
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    var tokenizer = try Tokenizer.fromFile("../tokenizer.bin", allocator);
    defer tokenizer.deinit();

    //checking vocab size and merge list size
    try std.testing.expectEqual(50000, tokenizer.merges.items.len);
    try std.testing.expectEqual(50257, tokenizer.tokens.count());
    //checking some token lookups
    try std.testing.expectEqual(29302, tokenizer.lookup("tube"));
    try std.testing.expectEqual(null, tokenizer.lookup("AAAAAAAAAAAAAAAA")); // null value check
    try std.testing.expectEqual(50256, tokenizer.lookup("<|endoftext|>")); // null value check

    // checking encoding
    const input: []const u8 = "Moondream is a small VLM that punches above its weight.";
    const tokenization = try tokenizer.encode(input);
    const tokenization_slice: []u32 = tokenization.items;
    defer tokenization.deinit();
    const expected_tokenization: []const u32 = &[_]u32{ 31640, 25966, 271, 64, 17470, 47468, 44, 5562, 35512, 2052, 29370, 896, 6551, 13 };

    for (tokenization_slice, 0..) |token, i| {
        try std.testing.expect(token == expected_tokenization[i]);
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
