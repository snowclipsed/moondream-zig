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
fn softmax(x: []f32) void {
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
            .img_channels = @intCast(self.img_channels),
            .img_dim = @intCast(self.img_dim),
            .patch_size = @intCast(self.patch_size),
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

    // Vision Model
    img_channels: usize, // number of channels per patch, RGB has 3
    img_dim: usize, // dimension of the the image, 378x378 default
    patch_size: usize, // size of patch, 14x14 default
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
    v_x: []align(simd_align) f32,
    v_q: []align(simd_align) f32,
    v_k: []align(simd_align) f32,
    v_v: []align(simd_align) f32,
    x: []align(simd_align) f32,
    xb: []align(simd_align) f32,
    xb2: []align(simd_align) f32,
    qkv_combined: []align(simd_align) f32, // a buffer that holds the combined kqv
    q: []align(simd_align) f32,
    k: []align(simd_align) f32,
    v: []align(simd_align) f32,
    attn: []align(simd_align) f32,
    k_cache: []align(simd_align) f32,
    v_cache: []align(simd_align) f32,
    cos_cache: []align(simd_align) f32,
    sin_cache: []align(simd_align) f32,
    inv_freq: []align(simd_align) f32,
    hb: []align(simd_align) f32,
    hb2: []align(simd_align) f32,
    logits: []align(simd_align) f32,

    fn init(allocator: Allocator, config: Config) !Self {
        return Self{
            .img = try allocator.alignedAlloc(f32, simd_align, config.img_dim * config.img_dim * config.img_channels),
            .patches = try allocator.alignedAlloc(f32, simd_align, config.img_dim * config.img_dim * config.img_channels),
            .patch_emb = try allocator.alignedAlloc(f32, simd_align, (config.img_dim * config.img_dim * config.vit_dim) / (config.patch_size * config.patch_size)),
            .v_x = try allocator.alignedAlloc(f32, simd_align, (config.img_dim * config.img_dim * config.vit_dim) / (config.patch_size * config.patch_size)),
            .v_q = try allocator.alignedAlloc(f32, simd_align, config.vit_dim * config.vit_dim),
            .v_k = try allocator.alignedAlloc(f32, simd_align, config.vit_dim * config.vit_dim),
            .v_v = try allocator.alignedAlloc(f32, simd_align, config.vit_dim * config.vit_dim),
            .x = try allocator.alignedAlloc(f32, simd_align, config.dim),
            .xb = try allocator.alignedAlloc(f32, simd_align, config.dim),
            .xb2 = try allocator.alignedAlloc(f32, simd_align, config.dim),
            .qkv_combined = try allocator.alignedAlloc(f32, simd_align, config.dim * 3),
            .q = try allocator.alignedAlloc(f32, simd_align, config.n_heads * config.head_dim),
            .k = try allocator.alignedAlloc(f32, simd_align, config.n_heads * config.head_dim),
            .v = try allocator.alignedAlloc(f32, simd_align, config.n_heads * config.head_dim),
            .attn = try allocator.alignedAlloc(f32, simd_align, config.n_heads * config.seq_len),
            .k_cache = try allocator.alignedAlloc(f32, simd_align, config.n_layers * config.seq_len * config.dim), // TODO : REPLACE WITH KV DIM
            .v_cache = try allocator.alignedAlloc(f32, simd_align, config.n_layers * config.seq_len * config.dim),
            .cos_cache = try allocator.alignedAlloc(f32, simd_align, config.max_pos_embeddings * config.dim),
            .sin_cache = try allocator.alignedAlloc(f32, simd_align, config.max_pos_embeddings * config.dim),
            .inv_freq = try allocator.alignedAlloc(f32, simd_align, config.dim / 2),
            .hb = try allocator.alignedAlloc(f32, simd_align, config.hidden_dim),
            .hb2 = try allocator.alignedAlloc(f32, simd_align, config.dim),
            .logits = try allocator.alignedAlloc(f32, simd_align, config.vocab),
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
                    try decoded_text.appendSlice(entry.key_ptr.*);
                    try decoded_text.append(' '); // Add a space between tokens
                    found = true;
                    break;
                }
            }

            if (!found) {
                std.debug.print("Tokens : {any} \n", .{tokens.items[token_id]});
                return error.TokenNotFound;
            }
        }

        // Remove the trailing space if it exists
        if (decoded_text.items.len > 0 and decoded_text.items[decoded_text.items.len - 1] == ' ') {
            _ = decoded_text.pop();
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

    fn text_model(self: Self, tokens: std.ArrayList(u32), pos: usize) !void {
        const token = tokens.items[pos];

        const sqrt: f32 = @floatFromInt(self.config.head_dim);
        // embed
        const embedding = self.embed(token);
        // TODO : clean up these print statements
        // std.debug.print("\n embed len for token {any}: {any} \n", .{ token, embedding.len });
        // std.debug.print("\n embed for {any} \n", .{embedding}); // embedding for a single token
        @memcpy(self.state.x, embedding); // copy embedding to activation for processing

        // pass through layers

        for (0..self.config.n_layers) |l| {
            if (self.config.head_dim * self.config.n_heads != self.config.dim) {
                return error.UnexpectedHiddenSize;
            }

            // First layernorm
            @memcpy(self.state.xb, self.state.x);

            try layer_norm(
                self.state.xb,
                self.weights.t_ln_w[l * self.config.dim .. (l + 1) * self.config.dim],
                self.weights.t_ln_b[l * self.config.dim .. (l + 1) * self.config.dim],
                1e-5,
            );

            // multiply the embedding (self.state.x) by Wkqv and get combined kqv
            // split that into k,q,v
            //  x (1, 2048)  @ W (2048 * 6144) = out (1 * 6144)
            // offset for the weights will be : config.n_layers * config.dim * config.n_heads * config.head_dim * 3,

            // matmuls for combined kqv
            // xb (1, 2048) @ Wqkv (2048, 6144) -> qkv_combined(1, 6144)
            try matmul(
                self.allocator,
                self.state.xb, // (1, dim)
                self.weights.t_Wqkv_w[l * self.config.dim * 3 * self.config.n_heads * self.config.head_dim .. (l + 1) * self.config.dim * 3 * self.config.n_heads * self.config.head_dim],
                self.state.qkv_combined,
                1,
                self.config.dim * 3,
                self.config.dim,
            );

            // add bias
            // qkv_combined (1, 2048) + bias (1, 2048)
            accumulate(self.state.qkv_combined, self.weights.t_Wqkv_b[l * self.config.n_heads * self.config.head_dim * 3 .. (l + 1) * self.config.n_heads * self.config.head_dim * 3]);

            // split into q, k, v
            @memcpy(self.state.q, self.state.qkv_combined[0 .. self.config.n_heads * self.config.head_dim]); // (1, 2048)
            @memcpy(self.state.k, self.state.qkv_combined[self.config.n_heads * self.config.head_dim .. self.config.n_heads * self.config.head_dim * 2]); // (1, 2048)
            @memcpy(self.state.v, self.state.qkv_combined[self.config.n_heads * self.config.head_dim * 2 .. self.config.n_heads * self.config.head_dim * 3]); // (1, 2048)

            // Apply RoPE to Q and K
            // TODO: Check implementation, could be a breakage point
            try self.apply_rope(self.state.q, pos);
            try self.apply_rope(self.state.k, pos);

            // define kv cache
            // the offset for each layer for the key and value vectors will be:
            const l_off = l * self.config.seq_len * self.config.dim; // using dim instead of kv dim

            // we define a pointer to a row of k and v cache
            const k_cache_row = self.state.k_cache[l_off + pos * self.config.dim .. l_off + (pos + 1) * self.config.dim]; // using dim instead of kv dim, size 2048
            const v_cache_row = self.state.v_cache[l_off + pos * self.config.dim .. l_off + (pos + 1) * self.config.dim]; // size 2048

            // we then update the kv cache for that specific row using the pointer
            @memcpy(k_cache_row, self.state.k);
            @memcpy(v_cache_row, self.state.v);

            // attention

            for (0..self.config.n_heads) |head| {
                // extract query vector for the head
                const q = self.state.q[head * self.config.head_dim .. (head + 1) * self.config.head_dim];
                // define slice of attn weights for this head
                const attn = self.state.attn[head * self.config.seq_len .. (head + 1) * self.config.seq_len];

                for (0..pos + 1) |t| {
                    const k = self.state.k_cache[l_off + t * self.config.dim + head * self.config.head_dim .. l_off + t * self.config.dim + (head + 1) * self.config.head_dim];
                    var score: f32 = vector_dot_product(q, k);

                    score /= std.math.sqrt(sqrt);
                    attn[t] = score;
                }

                softmax(attn[0 .. pos + 1]);

                // attn for V
                const xb = self.state.xb[head * self.config.head_dim .. (head + 1) * self.config.head_dim];

                @memset(xb, 0.0);
                for (0..pos + 1) |t| {
                    const v = self.state.v_cache[l_off + t * self.config.head_dim + head * self.config.head_dim .. l_off + t * self.config.head_dim + (head + 1) * self.config.head_dim];
                    const a = attn[t];

                    vector_weighted_sum(xb, v, a);
                }
            }

            try matmul(
                self.allocator,
                self.state.xb, // (1, 2048)
                self.weights.t_out_proj_w[l * self.config.dim * self.config.dim .. (l + 1) * self.config.dim * self.config.dim], // (2048, 2048)
                self.state.xb2, // (1, 2048)
                1,
                self.config.dim,
                self.config.dim,
            );

            accumulate(self.state.xb2, self.weights.t_out_proj_bias[l * self.config.dim .. (l + 1) * self.config.dim]);

            // TODO: Add residual connections

            // residual connection back to x
            accumulate(self.state.x, self.state.xb2);

            // TODO: Add FFN

            // upcasting
            // X (1, 2048) @ fc1 (2048,8192) -> hb (1, 8192)

            try matmul(
                self.allocator,
                self.state.x,
                self.weights.t_fc1_w[l * self.config.dim * self.config.hidden_dim .. (l + 1) * self.config.dim * self.config.hidden_dim],
                self.state.hb,
                1,
                self.config.hidden_dim,
                self.config.dim,
            );

            accumulate(self.state.hb, self.weights.t_fc1_b[l * self.config.hidden_dim .. (l + 1) * self.config.hidden_dim]);
            // hb (1, 8192) + fc1b (1, 8192)

            gelu(self.state.hb);

            // hb (1,8192) @ t_fc2_w(8192, 2048) -> hb2(1, 2048)

            try matmul(
                self.allocator,
                self.state.hb,
                self.weights.t_fc2_w[l * self.config.hidden_dim * self.config.dim .. (l + 1) * self.config.hidden_dim * self.config.dim],
                self.state.hb2,
                1,
                self.config.dim,
                self.config.hidden_dim,
            );
            // hb2 (1, 2048) + self.weights (1, 2048)

            accumulate(self.state.hb2, self.weights.t_fc2_b[l * self.config.dim .. (l + 1) * self.config.dim]);

            // residual connection from mlp back to x

            // x (1, 2048) + hb2(1, 2048)
            accumulate(self.state.x, self.state.hb2);

            // final LN and then linear

            try layer_norm(
                self.state.x,
                self.weights.t_ln_out_w,
                self.weights.t_ln_out_b,
                1e-5,
            );

            try matmul(
                self.allocator,
                self.state.x,
                self.weights.t_linear_w,
                self.state.logits,
                1,
                self.config.vocab,
                self.config.dim,
            );
            accumulate(self.state.logits, self.weights.t_linear_b);
            // std.debug.print("logits : {any} \n", .{self.state.logits[0..10]});
            // std.debug.print("================= \n", .{});
        }
    }

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
        const std_ = @sqrt(variance + eps);

        // Normalize the inputs
        for (inputs, 0..inputs.len) |*x, i| {
            if (std_ > 1e-5) {
                x.* = (x.* - mean) / std_ * weight[i] + bias[i];
            } else {
                // If std_ is very small, apply a small scaling factor
                x.* = (x.* - mean) * 0.01 * weight[i] + bias[i];
            }

            // Clip extreme values
            x.* = std.math.clamp(x.*, eps, 1e5);

            // Check for numerical stability
            if (std.math.isNan(x.*) or std.math.isInf(x.*)) {
                std.debug.print("Warning: Output contains NaN or Inf at index {d}. Input: {d}, Weight: {d}, Bias: {d}, Mean: {d}, Std: {d}\n", .{ i, x.*, weight[i], bias[i], mean, std_ });
                return error.NumericalInstability;
            }
        }
    }
    fn set_cos_sin_cache(self: Self, tokens: std.ArrayList(u32)) !void {
        const half_dim = self.config.dim / 2;

        // Compute inverse frequency
        var i: usize = 0;
        while (i < half_dim) : (i += 1) {
            const x = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(half_dim));
            self.state.inv_freq[i] = 1.0 / std.math.pow(f32, self.config.rope_theta, x);
        }

        // Allocate position frequencies
        var position_freqs = try self.allocator.alloc(f32, self.config.seq_len * half_dim);
        defer self.allocator.free(position_freqs);

        // Calculate position frequencies
        for (0..tokens.items.len) |pos| {
            for (0..half_dim) |j| {
                position_freqs[pos * half_dim + j] = @as(f32, @floatFromInt(pos)) * self.state.inv_freq[j];
            }
        }

        // Cache cosine and sine values
        const cache_len = self.config.seq_len * half_dim;
        for (0..cache_len) |itr| {
            const cos_val = @cos(position_freqs[itr]);
            const sin_val = @sin(position_freqs[itr]);

            // Store values twice to create the full dim-dimensional embedding
            self.state.cos_cache[itr] = cos_val;
            self.state.cos_cache[itr + cache_len] = cos_val;
            self.state.sin_cache[itr] = sin_val;
            self.state.sin_cache[itr + cache_len] = sin_val;
        }
    }

    fn apply_rope(self: Self, vec: []f32, pos: usize) !void {
        assert(vec.len == self.config.dim);
        const half_dim = self.config.dim / 2;
        const cos = self.state.cos_cache[pos * half_dim .. (pos + 1) * half_dim];
        const sin = self.state.sin_cache[pos * half_dim .. (pos + 1) * half_dim];

        var i: usize = 0;
        while (i < half_dim) : (i += 1) {
            const temp1 = vec[2 * i];
            const temp2 = vec[2 * i + 1];
            vec[2 * i] = temp1 * cos[i] - temp2 * sin[i];
            vec[2 * i + 1] = temp1 * sin[i] + temp2 * cos[i];
        }
    }
    fn embed(self: Self, token: usize) []f32 {
        return self.weights.word_token_embedding[token * self.config.dim ..][0..self.config.dim];
    }

    /// This function will load the images and then preprocess them into the required format
    pub fn preprocess(
        self: Self,
        image_path: []const u8,
        allocator: Allocator,
    ) ![]f32 {
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
        const num_patches = num_patches_h * num_patches_w;

        // we are going to change the format of our image from (C, H, W) to (h * w, channels * p1 * p2)
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
            try matmul(self.allocator, self.state.patches[patch * patch_elements .. (patch + 1) * patch_elements], self.weights.v_patch_embedding_linear_w, self.state.patch_emb[patch * self.config.vit_dim .. (patch + 1) * self.config.vit_dim], 1, self.config.vit_dim, patch_elements);
            accumulate(self.state.patch_emb[patch * self.config.vit_dim .. (patch + 1) * self.config.vit_dim], self.weights.v_patch_embedding_linear_b);
        }

        // next up is positional embedding, which is directly just accumulated into the patch embedding!
        // x = x + pos_embed

        accumulate(self.state.patch_emb, self.weights.v_pos_embedding);

        // we will now pass our positionally encoded patch embeddings through the ViT blocks.
        @memcpy(self.state.v_x, self.state.patch_emb);

        for (0..num_patches) |patch| {
            for (0..self.config.n_vit_layers) |l| {
                try layer_norm(self.state.patch_emb[patch * self.config.vit_dim .. (patch + 1) * self.config.vit_dim], self.weights.v_norm1_w[l * self.config.vit_dim .. (l + 1) * self.config.vit_dim], self.weights.v_norm1_b[l * self.config.vit_dim .. (l + 1) * self.config.vit_dim], 1e-5);
            }
        }
        std.debug.print("Pass! \n", .{});
    }
};

// inference
fn generate(model: *Model, image_path: []const u8, prompt: []const u8, max_new_tokens: usize, temperature: f32, top_p: f32, allocator: std.mem.Allocator) ![]const u8 {
    const preprocessed = try model.preprocess(image_path, allocator);
    defer allocator.free(preprocessed);

    // Now you can safely use float_image with SIMD operations
    @memcpy(model.state.img, preprocessed);
    try model.vision_encoder();

    var tokens = try model.tokenizer.encode(prompt);
    defer tokens.deinit();

    try model.set_cos_sin_cache(tokens);

    var generated_tokens = std.ArrayList(u32).init(allocator);
    defer generated_tokens.deinit();

    try generated_tokens.appendSlice(tokens.items);

    var rng = std.rand.DefaultPrng.init(@intCast(std.time.milliTimestamp()));

    while (generated_tokens.items.len < tokens.items.len + max_new_tokens) {
        try model.text_model(generated_tokens, generated_tokens.items.len - 1);

        const next_token = try sampleNextToken(model.state.logits, temperature, top_p, &rng, allocator);
        try generated_tokens.append(next_token);

        if (next_token == model.tokenizer.eos_token) break;
    }
    const decoding = try model.tokenizer.decode(generated_tokens);

    return decoding;
}

fn sampleNextToken(logits: []f32, temperature: f32, top_p: f32, rng: *std.rand.DefaultPrng, allocator: Allocator) !u32 {
    var adjusted_logits = try allocator.alloc(f32, logits.len);
    defer allocator.free(adjusted_logits);

    // Apply temperature
    for (logits, 0..) |logit, i| {
        adjusted_logits[i] = logit / temperature;
    }

    // Convert to probabilities
    softmax(adjusted_logits);
    std.debug.print("Softmax logits : {any}", .{adjusted_logits[0..10]});
    std.debug.print("\n===========================\n\n", .{});

    // Implement top-p sampling
    const cumulative_probs = try allocator.alloc(f32, logits.len);
    defer allocator.free(cumulative_probs);

    @memcpy(cumulative_probs, adjusted_logits);
    std.sort.pdq(f32, cumulative_probs, {}, std.sort.desc(f32));

    var cumsum: f32 = 0;
    var cutoff_index: usize = 0;
    for (cumulative_probs, 0..) |prob, i| {
        cumsum += prob;
        if (cumsum > top_p) {
            cutoff_index = i;
            break;
        }
    }

    // Zero out probabilities below the cutoff
    for (adjusted_logits) |*prob| {
        if (prob.* < cumulative_probs[cutoff_index]) {
            prob.* = 0;
        }
    }

    // Renormalize
    softmax(adjusted_logits);

    // Sample from the distribution
    const random_value = rng.random().float(f32);
    var sum: f32 = 0;
    for (adjusted_logits, 0..) |prob, i| {
        sum += prob;
        if (sum > random_value) {
            return @intCast(i);
        }
    }

    return error.SamplingFailed;
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

    // initializing text model //
    var model = try Model.init(config, weights, tokenizer, state, allocator);

    const prompt = "this is a whiteboard <image> Hello, World!";
    const max_new_tokens = 5;
    const temperature = 0.1;
    const top_p = 0.9;

    var timer = try std.time.Timer.start();

    const generated_text = try generate(&model, image_path, prompt, max_new_tokens, temperature, top_p, allocator);
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

    softmax(&x);

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
