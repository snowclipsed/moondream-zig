const std = @import("std");
const Timer = std.time.Timer;
const print = std.debug.print;
const Allocator = std.mem.Allocator;
const Weights = @import("weights.zig").Weights;
const Config = @import("config.zig").Config;
const Tensor = @import("tensor.zig").Tensor;
const Slice = @import("tensor.zig").Slice;
const ops = @import("ops.zig");
const hgemm = @import("hgemm.zig");
const sgemm = @import("sgemm.zig");
const printTimeDiff = @import("timediff.zig").printTimeDiff;
const getAndPrintTimeDiff = @import("timediff.zig").getAndPrintTimeDiff;
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
        var timer = try Timer.start();
        const total_start = timer.read();

        // Create patches (currently in BHWC)
        const patches_start = timer.read();
        var patches = try self.createPatches(image_path);
        defer patches.deinit();
        try printTimeDiff(&timer, patches_start, "Create Patches");

        // Convert to BCHW format
        const convert_start = timer.read();
        var bchw_patches = try ops.convert_bhwc_to_bchw(self.allocator, patches);
        defer bchw_patches.deinit();
        try printTimeDiff(&timer, convert_start, "Convert BHWC to BCHW");

        // Scale to 0-1 range
        const scale_start = timer.read();
        for (bchw_patches.data) |*val| {
            val.* /= @as(f16, 255.0);
        }
        try printTimeDiff(&timer, scale_start, "Scale to 0-1");

        // Create and initialize mean and std tensors
        const norm_init_start = timer.read();
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
        try printTimeDiff(&timer, norm_init_start, "Initialize Normalization Tensors");

        // Normalize
        const normalize_start = timer.read();
        var normalized = try ops.normalize_patch(self.allocator, bchw_patches, mean, stdev);
        defer normalized.deinit();
        try printTimeDiff(&timer, normalize_start, "Normalize Patches");

        // Run through vision encoder
        const encoder_start = timer.read();
        var encoder_output = try self.vision_encoder(normalized);
        defer encoder_output.deinit();
        try printTimeDiff(&timer, encoder_start, "Vision Encoder");

        // Get slices for each batch
        const slice_start = timer.read();
        var output_0 = try encoder_output.getDimensionSlice(0, 0);
        defer output_0.deinit();
        var output_1 = try encoder_output.getDimensionSlice(0, 1);
        defer output_1.deinit();
        try printTimeDiff(&timer, slice_start, "Get Dimension Slices");

        // Concatenate outputs
        const concat_start = timer.read();
        var concat_output = try ops.concat(f16, output_0, output_1, output_0.shape.len - 1);
        defer concat_output.deinit();
        try printTimeDiff(&timer, concat_start, "Concatenate Outputs");

        // Final MLP
        const mlp_start = timer.read();
        var final_output = try self.encode_mlp(concat_output);
        errdefer final_output.deinit();
        try printTimeDiff(&timer, mlp_start, "Final MLP");

        // Print total execution time
        try printTimeDiff(&timer, total_start, "Total Image Encoding");

        return final_output;
    }

    pub fn vision_encoder(self: Self, input: Tensor(f16)) !Tensor(f16) {
        var timer = try Timer.start();
        const total_start = timer.read();

        var block_label_buf: [64]u8 = undefined;
        const batch = input.shape[0];
        const eps = 1e-5;

        // Verify batch size
        if (batch != 2) {
            std.log.err("Expected batch size 2, got {d}\n", .{batch});
            return error.UnexpectedBatchCount;
        }

        // Rearrange input from BCHW to BTC format
        const rearrange_start = timer.read();
        var x = try ops.rearrangeBCHWtoBTC(self.allocator, input, self.config.patch_size);
        defer x.deinit();
        try printTimeDiff(&timer, rearrange_start, "BCHW to BTC Rearrange");

        const B = x.shape[0];
        const M = x.shape[1];
        const N = x.shape[2];

        // Initial reshape and linear projection
        const proj_start = timer.read();
        try x.reshape(&[_]usize{ B * M, N });
        var projected_f32 = try Tensor(f32).init(self.allocator, &[_]usize{ B * M, self.config.vit_dim });
        defer projected_f32.deinit();
        try hgemm.matmul(self.allocator, x, self.weights.v_patch_embedding_linear_w, projected_f32);
        try printTimeDiff(&timer, proj_start, "Linear Projection");

        // Cast and bias add
        const bias_start = timer.read();
        var x_curr = try projected_f32.castWithSimd(f16);
        defer x_curr.deinit();
        try ops.broadcast_add_simd(&x_curr, self.weights.v_patch_embedding_linear_b);
        try printTimeDiff(&timer, bias_start, "Bias Add");

        // Positional embedding
        const pos_start = timer.read();
        try x_curr.reshape(&[_]usize{ B, M, self.config.vit_dim });
        try ops.broadcast_add_simd(&x_curr, self.weights.v_pos_embedding);
        try printTimeDiff(&timer, pos_start, "Positional Embedding");

        // Process transformer blocks
        var total_ln1_time: i128 = 0;
        var total_attention_time: i128 = 0;
        var total_ln2_time: i128 = 0;
        var total_mlp_time: i128 = 0;
        var total_residual_time: i128 = 0;

        const blocks_start = timer.read();
        for (0..self.config.n_vit_layers) |block| {
            const block_start = timer.read();
            try x_curr.reshape(&[_]usize{ B * M, self.config.vit_dim });

            var x_orig = try x_curr.copy();
            defer x_orig.deinit();

            // First layer norm
            const ln1_start = timer.read();
            var ln1_w = try self.weights.v_norm1_w.getDimensionSlice(0, block);
            defer ln1_w.deinit();
            var ln1_b = try self.weights.v_norm1_b.getDimensionSlice(0, block);
            defer ln1_b.deinit();
            var ln1_out = try ops.layerNorm(f16, x_curr, ln1_w, ln1_b, eps);
            defer ln1_out.deinit();
            total_ln1_time += timer.read() - ln1_start;

            // Attention block
            const attn_start = timer.read();
            var attn_out = try self.attention_block(ln1_out, block);
            defer attn_out.deinit();
            total_attention_time += timer.read() - attn_start;

            // First residual connection
            const res1_start = timer.read();
            try ops.add(f16, &x_orig, attn_out);
            @memcpy(x_curr.data, x_orig.data);
            total_residual_time += timer.read() - res1_start;

            var pre_mlp = try x_curr.copy();
            defer pre_mlp.deinit();

            // Second layer norm
            const ln2_start = timer.read();
            var ln2_w = try self.weights.v_norm2_w.getDimensionSlice(0, block);
            defer ln2_w.deinit();
            var ln2_b = try self.weights.v_norm2_b.getDimensionSlice(0, block);
            defer ln2_b.deinit();
            var ln2_out = try ops.layerNorm(f16, x_curr, ln2_w, ln2_b, eps);
            defer ln2_out.deinit();
            total_ln2_time += timer.read() - ln2_start;

            // MLP block
            const mlp_start = timer.read();
            var mlp_out = try self.mlp(ln2_out, block);
            defer mlp_out.deinit();
            total_mlp_time += timer.read() - mlp_start;

            // Second residual connection
            const res2_start = timer.read();
            try ops.add(f16, &pre_mlp, mlp_out);
            @memcpy(x_curr.data, pre_mlp.data);
            total_residual_time += timer.read() - res2_start;

            // Format the block label and print timing
            const block_label = try std.fmt.bufPrint(&block_label_buf, "Transformer Block {d}", .{block});
            try printTimeDiff(&timer, block_start, block_label);
        }
        try printTimeDiff(&timer, blocks_start, "Total Transformer Blocks");

        // Print average times per block
        const blocks_f64 = @as(f64, @floatFromInt(self.config.n_vit_layers));
        const stdout = std.io.getStdOut().writer();
        try stdout.print("\x1b[93m [VISION PROFILE] Average times per transformer block:\x1b[0m\n", .{});
        try stdout.print("\x1b[93m Layer Norm 1: {d:.2}ms\x1b[0m\n", .{@as(f64, @floatFromInt(total_ln1_time)) / blocks_f64 / 1_000_000.0});
        try stdout.print("\x1b[93m Attention: {d:.2}ms\x1b[0m\n", .{@as(f64, @floatFromInt(total_attention_time)) / blocks_f64 / 1_000_000.0});
        try stdout.print("\x1b[93m Layer Norm 2: {d:.2}ms\x1b[0m\n", .{@as(f64, @floatFromInt(total_ln2_time)) / blocks_f64 / 1_000_000.0});
        try stdout.print("\x1b[93m MLP: {d:.2}ms\x1b[0m\n", .{@as(f64, @floatFromInt(total_mlp_time)) / blocks_f64 / 1_000_000.0});
        try stdout.print("\x1b[93m Residual Connections: {d:.2}ms\x1b[0m\n", .{@as(f64, @floatFromInt(total_residual_time)) / blocks_f64 / 1_000_000.0});

        // Final layer norm
        const final_ln_start = timer.read();
        try x_curr.reshape(&[_]usize{ B * M, self.config.vit_dim });
        var final_out = try ops.layerNorm(f16, x_curr, self.weights.v_norm_out_w, self.weights.v_norm_out_b, eps);
        errdefer final_out.deinit();
        try final_out.reshape(&[_]usize{ B, M, self.config.vit_dim });
        try printTimeDiff(&timer, final_ln_start, "Final Layer Norm");

        // Print total execution time
        try printTimeDiff(&timer, total_start, "Total Vision Encoder");

        return final_out;
    }

    pub fn attention_block(self: Self, x: Tensor(f16), layer: usize) !Tensor(f16) {
        var timer = try Timer.start();
        const total_start = timer.read();

        const q_len = x.shape[0];
        const d_model = x.shape[1];
        const n_heads = self.config.n_vit_heads;
        const head_dim = self.config.vit_head_dim;

        // Get QKV weights and biases
        const weight_start = timer.read();
        var layer_v_Wqkv_w = try self.weights.v_Wqkv_w.getDimensionSlice(0, layer);
        defer layer_v_Wqkv_w.deinit();
        var layer_v_Wqkv_b = try self.weights.v_Wqkv_b.getDimensionSlice(0, layer);
        defer layer_v_Wqkv_b.deinit();
        try printTimeDiff(&timer, weight_start, "QKV Weight Loading");

        // QKV linear projection
        const qkv_proj_start = timer.read();
        var qkv_f32 = try Tensor(f32).init(self.allocator, &[_]usize{ q_len, 3 * d_model });
        defer qkv_f32.deinit();
        try hgemm.matmul(self.allocator, x, layer_v_Wqkv_w, qkv_f32);
        try printTimeDiff(&timer, qkv_proj_start, "QKV Linear Projection");

        // Cast and bias add
        const cast_start = timer.read();
        var qkv = try qkv_f32.castWithSimd(f16);
        defer qkv.deinit();
        try ops.broadcast_add_simd(&qkv, layer_v_Wqkv_b);
        try printTimeDiff(&timer, cast_start, "QKV Cast and Bias Add");

        // Split QKV
        const split_start = timer.read();
        const num_chunks = 3;
        var q = try ops.getChunk(f16, qkv, 1, 0, num_chunks);
        defer q.deinit();
        var k = try ops.getChunk(f16, qkv, 1, 1, num_chunks);
        defer k.deinit();
        var v = try ops.getChunk(f16, qkv, 1, 2, num_chunks);
        defer v.deinit();
        try printTimeDiff(&timer, split_start, "QKV Split");

        // Reshape and transpose operations
        const reshape_start = timer.read();
        try q.reshape(&[_]usize{ q_len, n_heads, head_dim });
        try k.reshape(&[_]usize{ q_len, n_heads, head_dim });
        try v.reshape(&[_]usize{ q_len, n_heads, head_dim });

        try ops.transposeAxes(f16, &q, 0, 1);
        try ops.transposeAxes(f16, &k, 0, 1);
        try ops.transposeAxes(f16, &v, 0, 1);
        try printTimeDiff(&timer, reshape_start, "QKV Reshape and Transpose");

        // Attention computation
        const attention_start = timer.read();
        var attn_out = try ops.multimasklessDotProductAttention(q, k, v, self.allocator);
        defer attn_out.deinit();
        try printTimeDiff(&timer, attention_start, "Dot Product Attention");

        // Post-attention reshape
        const post_attn_start = timer.read();
        try ops.transposeAxes(f16, &attn_out, 0, 1);
        try attn_out.reshape(&[_]usize{ q_len, d_model });
        try printTimeDiff(&timer, post_attn_start, "Post-Attention Reshape");

        // Output projection
        const proj_start = timer.read();
        var layer_v_out_proj_w = try self.weights.v_out_proj_w.getDimensionSlice(0, layer);
        defer layer_v_out_proj_w.deinit();
        var layer_v_out_proj_b = try self.weights.v_out_proj_b.getDimensionSlice(0, layer);
        defer layer_v_out_proj_b.deinit();

        var out_f32 = try Tensor(f32).init(self.allocator, &[_]usize{ q_len, d_model });
        defer out_f32.deinit();
        try hgemm.matmul(self.allocator, attn_out, layer_v_out_proj_w, out_f32);
        try printTimeDiff(&timer, proj_start, "Output Projection");

        // Final cast and bias
        const final_start = timer.read();
        var out = try out_f32.castWithSimd(f16);
        errdefer out.deinit();
        try ops.broadcast_add_simd(&out, layer_v_out_proj_b);
        try printTimeDiff(&timer, final_start, "Final Cast and Bias Add");

        // Print total attention block time
        try printTimeDiff(&timer, total_start, "Total Attention Block");

        return out;
    }

    fn mlp(self: Self, input: Tensor(f16), layer: usize) !Tensor(f16) {
        var timer = try Timer.start();
        const total_start = timer.read();
        const hidden_features = self.config.hidden_features;
        const input_shape_0 = input.shape[0];
        const vit_dim = self.config.vit_dim;

        // Load weights and biases
        const weight_start = timer.read();
        var layer_v_fc1_w = try self.weights.v_fc1_w.getDimensionSlice(0, layer);
        defer layer_v_fc1_w.deinit();
        var layer_v_fc1_b = try self.weights.v_fc1_b.getDimensionSlice(0, layer);
        defer layer_v_fc1_b.deinit();
        var layer_v_fc2_w = try self.weights.v_fc2_w.getDimensionSlice(0, layer);
        defer layer_v_fc2_w.deinit();
        var layer_v_fc2_b = try self.weights.v_fc2_b.getDimensionSlice(0, layer);
        defer layer_v_fc2_b.deinit();
        try printTimeDiff(&timer, weight_start, "MLP Weight Loading");

        // First linear layer
        const fc1_start = timer.read();
        var fc1_f32 = try Tensor(f32).init(self.allocator, &[_]usize{ input_shape_0, hidden_features });
        defer fc1_f32.deinit();
        try hgemm.matmul(self.allocator, input, layer_v_fc1_w, fc1_f32);
        try printTimeDiff(&timer, fc1_start, "FC1 Linear");

        // Cast and bias for first layer
        const cast1_start = timer.read();
        var fc1 = try fc1_f32.castWithSimd(f16);
        defer fc1.deinit();
        try ops.broadcast_add_simd(&fc1, layer_v_fc1_b);
        try printTimeDiff(&timer, cast1_start, "FC1 Cast and Bias");

        // GELU activation
        const gelu_start = timer.read();
        try ops.gelu(f16, &fc1);
        try printTimeDiff(&timer, gelu_start, "GELU Activation");

        // Second linear layer
        const fc2_start = timer.read();
        var fc2_f32 = try Tensor(f32).init(self.allocator, &[_]usize{ input_shape_0, vit_dim });
        defer fc2_f32.deinit();
        try hgemm.matmul(
            self.allocator,
            fc1,
            layer_v_fc2_w,
            fc2_f32,
        );
        try printTimeDiff(&timer, fc2_start, "FC2 Linear");

        // Cast and bias for second layer
        const cast2_start = timer.read();
        var fc2 = try fc2_f32.castWithSimd(f16);
        errdefer fc2.deinit();
        try ops.broadcast_add_simd(&fc2, layer_v_fc2_b);
        try printTimeDiff(&timer, cast2_start, "FC2 Cast and Bias");

        // Print total MLP time
        try printTimeDiff(&timer, total_start, "Total MLP");

        return fc2;
    }
    fn encode_mlp(self: Self, input: Tensor(f16)) !Tensor(f16) {
        var timer = try Timer.start();
        const total_start = timer.read();

        // Get weight references
        const weight_start = timer.read();
        const layer_v_proj_fc1_w = self.weights.v_proj_fc1_w;
        const layer_v_proj_fc1_b = self.weights.v_proj_fc1_b;
        const layer_v_proj_fc2_w = self.weights.v_proj_fc2_w;
        const layer_v_proj_fc2_b = self.weights.v_proj_fc2_b;
        try printTimeDiff(&timer, weight_start, "Projection Weight References");

        // First linear layer
        const fc1_start = timer.read();
        var fc1_f32 = try Tensor(f32).init(self.allocator, &[_]usize{ input.shape[0], self.config.hidden_dim });
        defer fc1_f32.deinit();
        try hgemm.matmul(self.allocator, input, layer_v_proj_fc1_w, fc1_f32);
        try printTimeDiff(&timer, fc1_start, "Projection FC1 Linear");

        // Cast and bias for first layer
        const cast1_start = timer.read();
        var fc1 = try fc1_f32.castWithSimd(f16);
        defer fc1.deinit();
        try ops.broadcast_add_simd(&fc1, layer_v_proj_fc1_b);
        try printTimeDiff(&timer, cast1_start, "Projection FC1 Cast and Bias");

        // GELU activation
        const gelu_start = timer.read();
        try ops.gelu(f16, &fc1);
        try printTimeDiff(&timer, gelu_start, "Projection GELU Activation");

        // Second linear layer
        const fc2_start = timer.read();
        var fc2_f32 = try Tensor(f32).init(self.allocator, &[_]usize{ input.shape[0], self.config.dim });
        defer fc2_f32.deinit();
        try hgemm.matmul(
            self.allocator,
            fc1,
            layer_v_proj_fc2_w,
            fc2_f32,
        );
        try printTimeDiff(&timer, fc2_start, "Projection FC2 Linear");

        // Cast and bias for second layer
        const cast2_start = timer.read();
        var fc2 = try fc2_f32.castWithSimd(f16);
        errdefer fc2.deinit();
        try ops.broadcast_add_simd(&fc2, layer_v_proj_fc2_b);
        try printTimeDiff(&timer, cast2_start, "Projection FC2 Cast and Bias");

        // Print total encode MLP time
        try printTimeDiff(&timer, total_start, "Total Projection MLP");

        return fc2;
    }
};
