const std = @import("std");
const print = std.debug.print;
const Allocator = std.mem.Allocator;
const Weights = @import("weights.zig").Weights;
const Config = @import("config.zig").Config;
const Tensor = @import("tensor.zig").Tensor;
const Slice = @import("tensor.zig").Slice;
const ops = @import("ops.zig");
const hgemm = @import("hgemm.zig");
const sgemm = @import("sgemmnew.zig");
const c = @cImport({
    @cInclude("stb_image.h");
    @cInclude("stb_image_resize2.h");
    @cInclude("stdio.h");
});

pub const VisionModel = struct {
    const Self = @This();
    config: Config,
    weights: Weights,
    allocator: Allocator,

    pub fn init(config: Config, weights: Weights, allocator: Allocator) !VisionModel {
        return VisionModel{
            .config = config,
            .weights = weights,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.* = undefined;
    }
    pub fn createPatches(self: Self, image_path: []const u8) !Tensor(f16) {
        // Load the image using stb_image
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

        // Create buffer for resized image (global patch)
        const resized_data = try self.allocator.alloc(u8, self.config.img_dim * self.config.img_dim * @as(usize, @intCast(channels)));
        defer self.allocator.free(resized_data);

        // Resize image for global patch
        const result = c.stbir_resize_uint8_srgb(
            img_data,
            width,
            height,
            0,
            resized_data.ptr,
            @as(c_int, @intCast(self.config.img_dim)),
            @as(c_int, @intCast(self.config.img_dim)),
            0,
            if (channels == 3) c.STBIR_RGB else c.STBIR_RGBA,
        );

        if (result == 0) {
            return error.FailedToResizeImage;
        }

        // Check if image is small enough to just duplicate global patch
        const max_dim = @max(width, height);
        if (max_dim < @divTrunc(@as(c_int, @intCast(self.config.img_dim)) * 14, 10)) { // equivalent to 1.4 multiplier
            // Create tensor in BHWC format to match PyTorch's initial format
            var patches = try Tensor(f16).init(self.allocator, &[_]usize{
                2, // batch
                self.config.img_dim, // height
                self.config.img_dim, // width
                @as(usize, @intCast(channels)), // channels
            });
            errdefer patches.deinit();

            // Fill both patches with the same resized data
            var b: usize = 0;
            while (b < 2) : (b += 1) {
                var h: usize = 0;
                while (h < self.config.img_dim) : (h += 1) {
                    var w: usize = 0;
                    while (w < self.config.img_dim) : (w += 1) {
                        var ch: usize = 0;
                        while (ch < channels) : (ch += 1) {
                            // Calculate source index in resized_data (HWC layout)
                            const src_idx = (h * self.config.img_dim + w) * @as(usize, @intCast(channels)) + ch;

                            // Calculate destination index in BHWC layout
                            const dst_idx = b * (self.config.img_dim * self.config.img_dim * @as(usize, @intCast(channels))) +
                                h * (self.config.img_dim * @as(usize, @intCast(channels))) +
                                w * @as(usize, @intCast(channels)) +
                                ch;

                            patches.data[dst_idx] = @floatCast(@as(f16, @floatFromInt(resized_data[src_idx])));
                        }
                    }
                }
            }

            return patches;
        } else {
            // Create tensor in BHWC format
            var patches = try Tensor(f16).init(self.allocator, &[_]usize{
                2, // batch
                self.config.img_dim, // height
                self.config.img_dim, // width
                @as(usize, @intCast(channels)), // channels
            });
            errdefer patches.deinit();

            // Fill both patches with the same resized data using BHWC layout
            var b: usize = 0;
            while (b < 2) : (b += 1) {
                var h: usize = 0;
                while (h < self.config.img_dim) : (h += 1) {
                    var w: usize = 0;
                    while (w < self.config.img_dim) : (w += 1) {
                        var ch: usize = 0;
                        while (ch < channels) : (ch += 1) {
                            // Calculate source index in resized_data (HWC layout)
                            const src_idx = (h * self.config.img_dim + w) * @as(usize, @intCast(channels)) + ch;

                            // Calculate destination index in BHWC layout
                            const dst_idx = b * (self.config.img_dim * self.config.img_dim * @as(usize, @intCast(channels))) +
                                h * (self.config.img_dim * @as(usize, @intCast(channels))) +
                                w * @as(usize, @intCast(channels)) +
                                ch;

                            patches.data[dst_idx] = @floatCast(@as(f16, @floatFromInt(resized_data[src_idx])));
                        }
                    }
                }
            }

            return patches;
        }
    }

    pub fn encode_image(self: Self, image_path: []const u8) !Tensor(f16) {
        // Create patches (currently in BHWC)
        var patches = try self.createPatches(image_path);
        defer patches.deinit();

        // Convert to BCHW format
        var bchw_patches = try ops.convert_bhwc_to_bchw(self.allocator, patches);
        defer bchw_patches.deinit();

        // Scale to 0-1 range (this matches to_dtype with scale=True)
        for (bchw_patches.data) |*val| {
            val.* /= @as(f16, 255.0);
        }

        // Create mean and std tensors
        var mean = try Tensor(f16).init(self.allocator, &[_]usize{3});
        defer mean.deinit();
        mean.data[0] = 0.5;
        mean.data[1] = 0.5;
        mean.data[2] = 0.5;

        var stdev = try Tensor(f16).init(self.allocator, &[_]usize{3});
        defer stdev.deinit();
        stdev.data[0] = 0.5;
        stdev.data[1] = 0.5;
        stdev.data[2] = 0.5;

        // Normalize (now operating on BCHW format)
        var normalized = try ops.normalize_patch(self.allocator, bchw_patches, mean, stdev);
        defer normalized.deinit();

        // Run through vision encoder - returns [B, M, N]
        var encoder_output = try self.vision_encoder(normalized);
        defer encoder_output.deinit();

        // Get slices for each batch
        var output_0 = try encoder_output.getDimensionSlice(0, 0); // [M, N]
        defer output_0.deinit();
        var output_1 = try encoder_output.getDimensionSlice(0, 1); // [M, N]
        defer output_1.deinit();

        // Concatenate along the last dimension (N)
        var concat_output = try ops.concat(f16, output_0, output_1, output_0.shape.len - 1); // [M, 2N]
        defer concat_output.deinit();
        // Run through final MLP
        return try self.encode_mlp(concat_output);
    }

    pub fn vision_encoder(self: Self, input: Tensor(f16)) !Tensor(f16) {
        const batch = input.shape[0];
        const eps = 1e-5;

        // Verify batch size
        if (batch != 2) {
            std.log.err("Expected batch size 2, got {d}\n", .{batch});
            return error.UnexpectedBatchCount;
        }

        // Rearrange input from BCHW to BTC format first
        var x = try ops.rearrangeBCHWtoBTC(self.allocator, input, self.config.patch_size);
        defer x.deinit();

        // Remember original 3D shape for reshaping back
        const B = x.shape[0];
        const M = x.shape[1]; // sequence length
        const N = x.shape[2]; // embedding dimension

        // Reshape from [B, M, N] to [(B*M), N] for 2D operations
        try x.reshape(&[_]usize{ B * M, N });

        // Linear projection - already in 2D format [(B*M), N]
        var projected_f32 = try Tensor(f32).init(self.allocator, &[_]usize{ B * M, self.config.vit_dim });
        defer projected_f32.deinit();

        try hgemm.matmul(self.allocator, x, self.weights.v_patch_embedding_linear_w, projected_f32);

        var x_curr = try projected_f32.castTo(f16);
        defer x_curr.deinit();

        // Bias add needs 2D shape
        try ops.broadcast_add(f16, &x_curr, self.weights.v_patch_embedding_linear_b);

        // Reshape to [B, M, N] for positional embedding addition
        try x_curr.reshape(&[_]usize{ B, M, self.config.vit_dim });

        // Add positional embeddings (which should already be in shape [1, M, N])
        try ops.broadcast_add(f16, &x_curr, self.weights.v_pos_embedding);

        // Process transformer blocks
        for (0..self.config.n_vit_layers) |block| {
            // Reshape for 2D operations
            try x_curr.reshape(&[_]usize{ B * M, self.config.vit_dim });

            // Store original input for residual connection
            var x_orig = try x_curr.copy();
            defer x_orig.deinit();

            // Layer norm
            var ln1_w = try self.weights.v_norm1_w.getDimensionSlice(0, block);
            defer ln1_w.deinit();
            var ln1_b = try self.weights.v_norm1_b.getDimensionSlice(0, block);
            defer ln1_b.deinit();

            var ln1_out = try ops.layerNorm(f16, x_curr, ln1_w, ln1_b, eps);
            defer ln1_out.deinit();

            // Attention block
            var attn_out = try self.attention_block(ln1_out, block);
            defer attn_out.deinit();

            // Add attention output to original input (first residual connection)
            try ops.add(f16, &x_orig, attn_out);
            @memcpy(x_curr.data, x_orig.data);

            // Store pre-MLP state for second residual
            var pre_mlp = try x_curr.copy();
            defer pre_mlp.deinit();

            // Second layer norm
            var ln2_w = try self.weights.v_norm2_w.getDimensionSlice(0, block);
            defer ln2_w.deinit();
            var ln2_b = try self.weights.v_norm2_b.getDimensionSlice(0, block);
            defer ln2_b.deinit();

            var ln2_out = try ops.layerNorm(f16, x_curr, ln2_w, ln2_b, eps);
            defer ln2_out.deinit();

            // MLP block
            var mlp_out = try self.mlp(ln2_out, block);
            defer mlp_out.deinit();

            // Add MLP output to pre-MLP state (second residual connection)
            try ops.add(f16, &pre_mlp, mlp_out);
            @memcpy(x_curr.data, pre_mlp.data);
        }

        // Final layer norm needs 2D input
        try x_curr.reshape(&[_]usize{ B * M, self.config.vit_dim });
        var final_out = try ops.layerNorm(f16, x_curr, self.weights.v_norm_out_w, self.weights.v_norm_out_b, eps);
        errdefer final_out.deinit();

        // Reshape result back to 3D [B, M, N]
        try final_out.reshape(&[_]usize{ B, M, self.config.vit_dim });

        return final_out;
    }

    pub fn attention_block(self: Self, x: Tensor(f16), layer: usize) !Tensor(f16) {
        const q_len = x.shape[0];
        const d_model = x.shape[1];
        const n_heads = self.config.n_vit_heads;
        const head_dim = self.config.vit_head_dim;

        var layer_v_Wqkv_w = try self.weights.v_Wqkv_w.getDimensionSlice(0, layer);
        defer layer_v_Wqkv_w.deinit();

        var layer_v_Wqkv_b = try self.weights.v_Wqkv_b.getDimensionSlice(0, layer);
        defer layer_v_Wqkv_b.deinit();

        var qkv_f32 = try Tensor(f32).init(self.allocator, &[_]usize{ q_len, 3 * d_model });
        defer qkv_f32.deinit();

        try hgemm.matmul(self.allocator, x, layer_v_Wqkv_w, qkv_f32);

        var qkv = try qkv_f32.castTo(f16);
        defer qkv.deinit();

        try ops.broadcast_add(f16, &qkv, layer_v_Wqkv_b);

        const num_chunks = 3;

        var q = try ops.getChunk(f16, qkv, 1, 0, num_chunks);
        defer q.deinit();
        var k = try ops.getChunk(f16, qkv, 1, 1, num_chunks);
        defer k.deinit();
        var v = try ops.getChunk(f16, qkv, 1, 2, num_chunks);
        defer v.deinit();

        // 3. Reshape each tensor from [q_len, n_heads * head_dim] to [q_len, n_heads, head_dim]
        try q.reshape(&[_]usize{ q_len, n_heads, head_dim });
        try k.reshape(&[_]usize{ q_len, n_heads, head_dim });
        try v.reshape(&[_]usize{ q_len, n_heads, head_dim });

        // 4. Transpose from [q_len, n_heads, head_dim] to [n_heads, q_len, head_dim]
        try ops.transposeAxes(f16, &q, 0, 1); // Swap q_len and n_heads
        try ops.transposeAxes(f16, &k, 0, 1);
        try ops.transposeAxes(f16, &v, 0, 1);

        var attn_out = try ops.masklessDotProductAttention(q, k, v, self.allocator);
        defer attn_out.deinit();

        // 6. Transpose from [n_heads, q_len, head_dim] to [q_len, n_heads, head_dim]
        try ops.transposeAxes(f16, &attn_out, 0, 1); // Swap q_len and n_heads

        // 7. Reshape from [seq_len, n_heads, head_dim] to [q_len, d_model]
        try attn_out.reshape(&[_]usize{ q_len, d_model });

        // 8. Linear layer
        var layer_v_out_proj_w = try self.weights.v_out_proj_w.getDimensionSlice(0, layer);
        defer layer_v_out_proj_w.deinit();

        var layer_v_out_proj_b = try self.weights.v_out_proj_b.getDimensionSlice(0, layer);
        defer layer_v_out_proj_b.deinit();

        var out_f32 = try Tensor(f32).init(self.allocator, &[_]usize{ q_len, d_model });
        defer out_f32.deinit();

        try hgemm.matmul(self.allocator, attn_out, layer_v_out_proj_w, out_f32);

        var out = try out_f32.castTo(f16);
        errdefer out.deinit();

        try ops.broadcast_add(f16, &out, layer_v_out_proj_b);
        return out;
    }

    fn mlp(self: Self, input: Tensor(f16), layer: usize) !Tensor(f16) {
        var layer_v_fc1_w = try self.weights.v_fc1_w.getDimensionSlice(0, layer);
        defer layer_v_fc1_w.deinit();
        var layer_v_fc1_b = try self.weights.v_fc1_b.getDimensionSlice(0, layer);
        defer layer_v_fc1_b.deinit();
        var layer_v_fc2_w = try self.weights.v_fc2_w.getDimensionSlice(0, layer);
        defer layer_v_fc2_w.deinit();
        var layer_v_fc2_b = try self.weights.v_fc2_b.getDimensionSlice(0, layer);
        defer layer_v_fc2_b.deinit();

        var fc1_f32 = try Tensor(f32).init(self.allocator, &[_]usize{ input.shape[0], self.config.hidden_features });
        defer fc1_f32.deinit();

        try hgemm.matmul(self.allocator, input, layer_v_fc1_w, fc1_f32);

        var fc1 = try fc1_f32.castTo(f16);
        defer fc1.deinit();

        try ops.broadcast_add(f16, &fc1, layer_v_fc1_b);

        try ops.gelu(f16, &fc1);

        var fc2_f32 = try Tensor(f32).init(self.allocator, &[_]usize{ input.shape[0], self.config.vit_dim });
        defer fc2_f32.deinit();
        try hgemm.matmul(
            self.allocator,
            fc1,
            layer_v_fc2_w,
            fc2_f32,
        );

        var fc2 = try fc2_f32.castTo(f16);
        errdefer fc2.deinit();
        try ops.broadcast_add(f16, &fc2, layer_v_fc2_b);

        return fc2;
    }

    fn encode_mlp(self: Self, input: Tensor(f16)) !Tensor(f16) {
        const layer_v_proj_fc1_w = self.weights.v_proj_fc1_w;
        const layer_v_proj_fc1_b = self.weights.v_proj_fc1_b;
        const layer_v_proj_fc2_w = self.weights.v_proj_fc2_w;
        const layer_v_proj_fc2_b = self.weights.v_proj_fc2_b;

        var fc1_f32 = try Tensor(f32).init(self.allocator, &[_]usize{ input.shape[0], self.config.hidden_dim });
        defer fc1_f32.deinit();
        try hgemm.matmul(self.allocator, input, layer_v_proj_fc1_w, fc1_f32);

        var fc1 = try fc1_f32.castTo(f16);
        defer fc1.deinit();

        try ops.broadcast_add(f16, &fc1, layer_v_proj_fc1_b);

        try ops.gelu(f16, &fc1);

        var fc2_f32 = try Tensor(f32).init(self.allocator, &[_]usize{ input.shape[0], self.config.dim });
        defer fc2_f32.deinit();
        try hgemm.matmul(
            self.allocator,
            fc1,
            layer_v_proj_fc2_w,
            fc2_f32,
        );

        var fc2 = try fc2_f32.castTo(f16);
        errdefer fc2.deinit();
        try ops.broadcast_add(f16, &fc2, layer_v_proj_fc2_b);

        return fc2;
    }
};
