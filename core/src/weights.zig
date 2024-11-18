const std = @import("std");
const Allocator = std.mem.Allocator;
const Config = @import("config.zig").Config;

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
