const std = @import("std");
const assert = std.debug.assert;
const mem = std.mem;
const Allocator = mem.Allocator;
const Thread = std.Thread;
const builtin = @import("builtin");
const T: usize = 64; // Tile size (adjust as necessary)
const V: usize = 32; // Vector size (adjust as necessary)
const simd_align = @alignOf(@Vector(V, f32));
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

// REDO this function with vectors.
// This function accumulates an f32 array b into another f32 array a
fn accumulate(a: []f32, b: []f32) void {
    assert(a.len == b.len);
    for (0..a.len) |i| {
        a[i] += b[i];
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

/// Matrix Multiplication
pub fn matmul(allocator: std.mem.Allocator, A: []const f32, B: []const f32, C: []f32, M: usize, N: usize, K: usize) !void {
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

    // vision
    img_channels: i32, // number of channels per patch, RGB has 3
    img_dim: i32, // dimension of the the image, 378x378 default
    patch_size: i32, // size of patch, 14x14 default
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
            .img_channels = @intCast(self.img_channels),
            .img_dim = @intCast(self.img_dim),
            .patch_size = @intCast(self.patch_size),
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

    // Vision Model
    img_channels: usize, // number of channels per patch, RGB has 3
    img_dim: usize, // dimension of the the image, 378x378 default
    patch_size: usize, // size of patch, 14x14 default
    vit_dim: usize, // width of each patch embedding created from linear patch embedding layer, 1152 default
    n_vit_layers: usize, // number of ViT layers, 27 default for the vision model
    n_vit_heads: usize, // number of attn heads for each attn layer, 16 default
    vit_head_dim: usize, // size of each attn head, 72 default
    hidden_features: usize, // size of hidden features in ViT fc layers, 4304 in length
};

/// Struct defines the weights of moondream
/// All weights are transposed
/// Naming convention :
/// "t_" prefix : text_model (phi 1.5)
/// "v_" prefix : vision_model (vision encoder)
/// "_w" suffix : weights
/// "_b" suffix : biases
const Weights = struct {
    const Self = @This();
    // TODO: Add Self reference to this struct
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
            // TODO : Recheck this once
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

    x: []align(simd_align) f32,
    q: []align(simd_align) f32,
    k: []align(simd_align) f32,
    v: []align(simd_align) f32,
    k_cache: []align(simd_align) f32,
    v_cache: []align(simd_align) f32,

    fn init(allocator: Allocator, config: Config) !Self {
        return Self{
            .x = try allocator.alignedAlloc(f32, simd_align, config.dim),
            .q = try allocator.alignedAlloc(f32, simd_align, config.n_heads * config.head_dim),
            .k = try allocator.alignedAlloc(f32, simd_align, config.n_heads * config.head_dim),
            .v = try allocator.alignedAlloc(f32, simd_align, config.n_heads * config.head_dim),
            .k_cache = try allocator.alignedAlloc(f32, simd_align, config.n_layers * config.seq_len * config.dim), // TODO : REPLACE WITH KV DIM
            .v_cache = try allocator.alignedAlloc(f32, simd_align, config.n_layers * config.seq_len * config.dim),
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

    fn init(allocator: Allocator) Tokenizer {
        return .{
            .tokens = std.StringHashMap(u32).init(allocator),
            .merges = std.ArrayList([]const u8).init(allocator),
            .allocator = allocator,
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

// RoPE

// attention
fn attention() !void {}

// vision transformer
const VisionTransformerLayer = struct {};

// text model
const TextModel = struct {
    const Self = @This();
    config: Config,
    weights: Weights,
    tokenizer: Tokenizer,
    state: RunState,
    allocator: Allocator,

    fn init(config: Config, weights: Weights, tokenizer: Tokenizer, state: RunState, allocator: Allocator) !TextModel {
        return TextModel{
            .config = config,
            .weights = weights,
            .tokenizer = tokenizer,
            .state = state,
            .allocator = allocator,
        };
    }

    fn forward(self: Self, token: usize) !void {
        // embed
        const embedding = self.embed(token);
        std.debug.print("\n embed len for {any} \n", .{embedding.len});
        std.debug.print("\n embed for {any} \n", .{embedding});
        // pass through layers

        // layer norm

        // attn
    }

    fn embed(self: Self, token: usize) []f32 {
        const dim = self.config.dim;
        return self.weights.word_token_embedding[token * dim ..][0..dim];
    }
};

// vision model

const VisionModel = struct {};

// forward loop
pub fn forward() !void {}

// inference
pub fn generate() !void {}

// main

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Constants //
    const bin_path: []const u8 = "../moondream_f32.bin";
    const config_path: ?[]const u8 = "../model_config.json";

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

    var weights = try Weights.init(config, bin_path, allocator);
    defer weights.deinit(allocator);

    var tokenizer = try Tokenizer.fromFile("../tokenizer.bin", allocator);
    defer tokenizer.deinit();

    var state = try RunState.init(allocator, config);
    defer state.deinit(allocator);

    var text_model = try TextModel.init(config, weights, tokenizer, state, allocator);
    // defer text_model.deinit;
    // End of loading model checkpoint
    const text = "hello!";
    const token = try tokenizer.encode(text);
    std.debug.print("\n token : {}", .{token.items[0]});
    try text_model.forward(token.items[0]);
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
