const std = @import("std");
const Timer = std.time.Timer;
const print = std.debug.print;
const Allocator = std.mem.Allocator;
const Weights = @import("weights.zig").Weights;
const Config = @import("config.zig").Config;
const Tensor = @import("tensor.zig").Tensor;
const Slice = @import("tensor.zig").Slice;
const ops = @import("ops.zig");
const hgemm = @import("hgemmnew.zig");
const sgemm = @import("sgemm.zig");
const printTimeDiff = @import("timediffvision.zig").printTimeDiff;
const getAndPrintTimeDiff = @import("timediffvision.zig").getAndPrintTimeDiff;
const c = @cImport({
    @cInclude("stb_image.h");
    @cInclude("stb_image_resize2.h");
    @cInclude("stdio.h");
});

const mode = std.builtin.FloatMode.optimized;
comptime {
    @setFloatMode(mode);
}

const PreSlicedWeights = @import("preslice.zig").PreSlicedWeights;

pub fn VisionModel(comptime model_config: Config) type {
    return struct {
        const Self = @This();
        const config = model_config;

        // Common calculations as comptime constants
        const eps: comptime_float = 1e-5;
        const batch_size: comptime_int = 2;
        const patches_per_image = (config.img_dim / config.patch_size) * (config.img_dim / config.patch_size);
        const q_len = batch_size * patches_per_image;
        const head_dim: usize = config.vit_head_dim;
        const d_model: usize = config.vit_dim;

        // Pre-computed attention dimensions
        const attention_dims = struct {
            const q_dim: usize = config.vit_dim;
            const k_dim: usize = config.vit_dim;
            const v_dim: usize = config.vit_dim;
        };

        // Pre-computed shapes for tensors
        const tensor_shapes = struct {
            const qkv: [2]usize = .{ batch_size * patches_per_image, config.vit_dim * 3 };
            const attention: [3]usize = .{ batch_size * patches_per_image, config.n_vit_heads, config.vit_head_dim };
            const patches: [4]usize = .{ batch_size, config.img_dim, config.img_dim, config.img_channels };
            const q_shape: [3]usize = .{ batch_size * patches_per_image, config.vit_dim, config.vit_head_dim };
            const k_shape: [3]usize = .{ batch_size * patches_per_image, config.vit_dim, config.vit_head_dim };
            const v_shape: [3]usize = .{ batch_size * patches_per_image, config.vit_dim, config.vit_head_dim };
        };

        // Pre-computed patch dimensions
        const patch_dims = struct {
            const area: usize = config.patch_size * config.patch_size;
            const volume: usize = area * config.img_channels;
            const total: usize = patches_per_image;
        };

        // Compile-time validation
        comptime {
            if (config.vit_head_dim * config.n_vit_heads != config.vit_dim) {
                @compileError("Invalid head dimensions");
            }
            if (config.img_dim % config.patch_size != 0) {
                @compileError("Image dimension must be divisible by patch size");
            }
        }

        // Instance fields
        weights: Weights,
        presliced_weights: PreSlicedWeights,
        allocator: Allocator,

        pub fn init(weights: Weights, allocator: Allocator) !Self {
            // Create pre-sliced weights
            var presliced_weights = try PreSlicedWeights.init(allocator, weights, config.n_vit_layers);
            errdefer presliced_weights.deinit();

            return Self{
                .weights = weights,
                .presliced_weights = presliced_weights,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.presliced_weights.deinit();
            self.* = undefined;
        }

        pub fn createPatches(self: Self, image_path: []const u8) !Tensor(f16) {
            // Create null-terminated path for stb_image
            var path_buffer = try self.allocator.alloc(u8, image_path.len + 1);
            defer self.allocator.free(path_buffer);
            @memcpy(path_buffer[0..image_path.len], image_path);
            path_buffer[image_path.len] = 0;

            // Load the image using stb_image
            var width: c_int = 0;
            var height: c_int = 0;
            var channels: c_int = 0;

            const img_data = c.stbi_load(path_buffer.ptr, &width, &height, &channels, 0);
            if (img_data == null) {
                const err_str = c.stbi_failure_reason();
                std.debug.print("STB Image loading failed: {s}\n", .{err_str});
                return error.FailedToLoadImage;
            }
            defer c.stbi_image_free(img_data);

            std.debug.print("Loaded image with width: {d}, height: {d}, channels: {d}\n", .{ width, height, channels });

            // Create buffer for resized image (global patch)
            const resized_data = try self.allocator.alloc(u8, Self.config.img_dim * Self.config.img_dim * @as(usize, @intCast(channels)));
            defer self.allocator.free(resized_data);

            // Resize image for global patch
            const result = c.stbir_resize_uint8_srgb(
                img_data,
                width,
                height,
                0,
                resized_data.ptr,
                @as(c_int, @intCast(Self.config.img_dim)),
                @as(c_int, @intCast(Self.config.img_dim)),
                0,
                if (channels == 3) c.STBIR_RGB else c.STBIR_RGBA,
            );

            if (result == 0) {
                return error.FailedToResizeImage;
            }

            // Check if image is small enough to just duplicate global patch
            const max_dim = @max(width, height);
            if (max_dim < @divTrunc(@as(c_int, @intCast(Self.config.img_dim)) * 14, 10)) { // equivalent to 1.4 multiplier
                // Create tensor in BHWC format to match PyTorch's initial format
                var patches = try Tensor(f16).init(self.allocator, &[_]usize{
                    2, // batch
                    Self.config.img_dim, // height
                    Self.config.img_dim, // width
                    @as(usize, @intCast(channels)), // channels
                });
                errdefer patches.deinit();

                // Fill both patches with the same resized data
                var b: usize = 0;
                while (b < 2) : (b += 1) {
                    var h: usize = 0;
                    while (h < Self.config.img_dim) : (h += 1) {
                        var w: usize = 0;
                        while (w < Self.config.img_dim) : (w += 1) {
                            var ch: usize = 0;
                            while (ch < channels) : (ch += 1) {
                                // Calculate source index in resized_data (HWC layout)
                                const src_idx = (h * Self.config.img_dim + w) * @as(usize, @intCast(channels)) + ch;

                                // Calculate destination index in BHWC layout
                                const dst_idx = b * (Self.config.img_dim * Self.config.img_dim * @as(usize, @intCast(channels))) +
                                    h * (Self.config.img_dim * @as(usize, @intCast(channels))) +
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
                    Self.config.img_dim, // height
                    Self.config.img_dim, // width
                    @as(usize, @intCast(channels)), // channels
                });
                errdefer patches.deinit();

                // Fill both patches with the same resized data using BHWC layout
                var b: usize = 0;
                while (b < 2) : (b += 1) {
                    var h: usize = 0;
                    while (h < Self.config.img_dim) : (h += 1) {
                        var w: usize = 0;
                        while (w < Self.config.img_dim) : (w += 1) {
                            var ch: usize = 0;
                            while (ch < channels) : (ch += 1) {
                                // Calculate source index in resized_data (HWC layout)
                                const src_idx = (h * Self.config.img_dim + w) * @as(usize, @intCast(channels)) + ch;

                                // Calculate destination index in BHWC layout
                                const dst_idx = b * (Self.config.img_dim * Self.config.img_dim * @as(usize, @intCast(channels))) +
                                    h * (Self.config.img_dim * @as(usize, @intCast(channels))) +
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

            // Verify batch size
            if (batch != 2) {
                std.log.err("Expected batch size 2, got {d}\n", .{batch});
                return error.UnexpectedBatchCount;
            }

            // Rearrange input from BCHW to BTC format
            const rearrange_start = timer.read();
            var x = try ops.rearrangeBCHWtoBTC(self.allocator, input, Self.config.patch_size);
            defer x.deinit();
            try printTimeDiff(&timer, rearrange_start, "BCHW to BTC Rearrange");

            const B = x.shape[0];
            const M = x.shape[1];
            const N = x.shape[2];

            // Initial reshape and linear projection
            const proj_start = timer.read();
            try x.reshape(&[_]usize{ B * M, N });
            var projected = try hgemm.matmul(x, self.weights.v_patch_embedding_linear_w, self.allocator);
            defer projected.deinit();
            try printTimeDiff(&timer, proj_start, "Linear Projection");

            // Cast and bias add
            const bias_start = timer.read();

            try ops.broadcast_add_simd(&projected, self.weights.v_patch_embedding_linear_b);
            try printTimeDiff(&timer, bias_start, "Bias Add");

            // Positional embedding
            const pos_start = timer.read();
            try projected.reshape(&[_]usize{ B, M, Self.config.vit_dim });
            try ops.broadcast_add_simd(&projected, self.weights.v_pos_embedding);
            try printTimeDiff(&timer, pos_start, "Positional Embedding");

            // Process transformer blocks
            var total_ln1_time: i128 = 0;
            var total_attention_time: i128 = 0;
            var total_ln2_time: i128 = 0;
            var total_mlp_time: i128 = 0;
            var total_residual_time: i128 = 0;

            const blocks_start = timer.read();
            for (0..Self.config.n_vit_layers) |block| {
                const block_start = timer.read();
                try projected.reshape(&[_]usize{ B * M, Self.config.vit_dim });

                var x_orig = try projected.copy();
                defer x_orig.deinit();

                // First layer norm
                const ln1_start = timer.read();
                const ln1_w = self.presliced_weights.v_norm1_w[block];
                const ln1_b = self.presliced_weights.v_norm1_b[block];
                var ln1_out = try ops.layerNorm(f16, projected, ln1_w, ln1_b, eps);
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
                @memcpy(projected.data, x_orig.data);
                total_residual_time += timer.read() - res1_start;

                var pre_mlp = try projected.copy();
                defer pre_mlp.deinit();

                // Second layer norm
                const ln2_start = timer.read();
                const ln2_w = self.presliced_weights.v_norm2_w[block];
                const ln2_b = self.presliced_weights.v_norm2_b[block];
                var ln2_out = try ops.layerNorm(f16, projected, ln2_w, ln2_b, eps);
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
                @memcpy(projected.data, pre_mlp.data);
                total_residual_time += timer.read() - res2_start;

                // Format the block label and print timing
                const block_label = try std.fmt.bufPrint(&block_label_buf, "Transformer Block {d}", .{block});
                try printTimeDiff(&timer, block_start, block_label);
            }
            try printTimeDiff(&timer, blocks_start, "Total Transformer Blocks");

            // Print average times per block
            const blocks_f64 = @as(f64, @floatFromInt(Self.config.n_vit_layers));
            const stdout = std.io.getStdOut().writer();
            try stdout.print("\x1b[93m [VISION PROFILE] Average times per transformer block:\x1b[0m\n", .{});
            try stdout.print("\x1b[93m Layer Norm 1: {d:.2}ms\x1b[0m\n", .{@as(f64, @floatFromInt(total_ln1_time)) / blocks_f64 / 1_000_000.0});
            try stdout.print("\x1b[93m Attention: {d:.2}ms\x1b[0m\n", .{@as(f64, @floatFromInt(total_attention_time)) / blocks_f64 / 1_000_000.0});
            try stdout.print("\x1b[93m Layer Norm 2: {d:.2}ms\x1b[0m\n", .{@as(f64, @floatFromInt(total_ln2_time)) / blocks_f64 / 1_000_000.0});
            try stdout.print("\x1b[93m MLP: {d:.2}ms\x1b[0m\n", .{@as(f64, @floatFromInt(total_mlp_time)) / blocks_f64 / 1_000_000.0});
            try stdout.print("\x1b[93m Residual Connections: {d:.2}ms\x1b[0m\n", .{@as(f64, @floatFromInt(total_residual_time)) / blocks_f64 / 1_000_000.0});

            // Final layer norm
            const final_ln_start = timer.read();
            try projected.reshape(&[_]usize{ B * M, Self.config.vit_dim });
            var final_out = try ops.layerNorm(f16, projected, self.weights.v_norm_out_w, self.weights.v_norm_out_b, eps);
            errdefer final_out.deinit();
            try final_out.reshape(&[_]usize{ B, M, Self.config.vit_dim });
            try printTimeDiff(&timer, final_ln_start, "Final Layer Norm");

            // Print total execution time
            try printTimeDiff(&timer, total_start, "Total Vision Encoder");

            return final_out;
        }

        pub fn attention_block(self: Self, x: Tensor(f16), layer: usize) !Tensor(f16) {
            var timer = try Timer.start();
            const total_start = timer.read();

            const n_heads = Self.config.n_vit_heads;

            // Get QKV weights and biases
            const weight_start = timer.read();
            const layer_v_Wqkv_w = self.presliced_weights.v_Wqkv_w[layer];
            const layer_v_Wqkv_b = self.presliced_weights.v_Wqkv_b[layer];
            try printTimeDiff(&timer, weight_start, "QKV Weight Loading");

            // QKV linear projection
            const qkv_proj_start = timer.read();
            var qkv = try hgemm.matmul(x, layer_v_Wqkv_w, self.allocator);
            defer qkv.deinit();
            try printTimeDiff(&timer, qkv_proj_start, "QKV Linear Projection");

            // Cast and bias add
            const cast_start = timer.read();
            try ops.broadcast_add_simd(&qkv, layer_v_Wqkv_b);
            try printTimeDiff(&timer, cast_start, "QKV Bias Add");

            // Split QKV using views
            const split_start = timer.read();
            const num_chunks = 3;
            var qkv_view = try qkv.asView();
            defer qkv_view.deinit();

            // Get chunks using views and convert to contiguous tensors
            var q = try (try qkv_view.getChunkView(1, 0, num_chunks)).toContiguousTensor();
            defer q.deinit();
            var k = try (try qkv_view.getChunkView(1, 1, num_chunks)).toContiguousTensor();
            defer k.deinit();
            var v = try (try qkv_view.getChunkView(1, 2, num_chunks)).toContiguousTensor();
            defer v.deinit();
            try printTimeDiff(&timer, split_start, "QKV Split");

            // Reshape and transpose operations
            const reshape_start = timer.read();
            try q.reshape(&[_]usize{ q_len, n_heads, head_dim });
            try k.reshape(&[_]usize{ q_len, n_heads, head_dim });
            try v.reshape(&[_]usize{ q_len, n_heads, head_dim });
            try printTimeDiff(&timer, reshape_start, "QKV Reshape ");
            const transpose_start = timer.read();
            try ops.transposeAxes(f16, &q, 0, 1);
            try ops.transposeAxes(f16, &k, 0, 1);
            try ops.transposeAxes(f16, &v, 0, 1);
            try printTimeDiff(&timer, transpose_start, "QKV Transpose");

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

            var out = try hgemm.matmul(attn_out, layer_v_out_proj_w, self.allocator);
            errdefer out.deinit();
            try printTimeDiff(&timer, proj_start, "Output Projection");

            // Final cast and bias
            const final_start = timer.read();
            try ops.broadcast_add_simd(&out, layer_v_out_proj_b);
            try printTimeDiff(&timer, final_start, "Final Cast and Bias Add");

            // Print total attention block time
            try printTimeDiff(&timer, total_start, "Total Attention Block");

            return out;
        }

        fn mlp(self: Self, input: Tensor(f16), layer: usize) !Tensor(f16) {
            var timer = try Timer.start();
            const total_start = timer.read();

            // Load weights and biases
            const weight_start = timer.read();
            const layer_v_fc1_w = self.presliced_weights.v_fc1_w[layer];
            const layer_v_fc1_b = self.presliced_weights.v_fc1_b[layer];
            const layer_v_fc2_w = self.presliced_weights.v_fc2_w[layer];
            const layer_v_fc2_b = self.presliced_weights.v_fc2_b[layer];
            try printTimeDiff(&timer, weight_start, "MLP Weight Loading");

            // First linear layer
            const fc1_start = timer.read();

            var fc1 = try hgemm.matmul(input, layer_v_fc1_w, self.allocator);
            defer fc1.deinit();

            try printTimeDiff(&timer, fc1_start, "FC1 Linear");

            // Cast and bias for first layer
            const cast1_start = timer.read();
            try ops.broadcast_add_simd(&fc1, layer_v_fc1_b);
            try printTimeDiff(&timer, cast1_start, "FC1 Cast and Bias");

            // GELU activation
            const gelu_start = timer.read();
            try ops.gelu(f16, &fc1);
            try printTimeDiff(&timer, gelu_start, "GELU Activation");

            // Second linear layer
            const fc2_start = timer.read();

            var fc2 = try hgemm.matmul(fc1, layer_v_fc2_w, self.allocator);
            try printTimeDiff(&timer, fc2_start, "FC2 Linear");

            // Cast and bias for second layer
            const cast2_start = timer.read();
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
            var fc1 = try hgemm.matmul(input, layer_v_proj_fc1_w, self.allocator);
            defer fc1.deinit();
            try printTimeDiff(&timer, fc1_start, "Projection FC1 Linear");

            // Cast and bias for first layer
            const cast1_start = timer.read();
            try ops.broadcast_add_simd(&fc1, layer_v_proj_fc1_b);
            try printTimeDiff(&timer, cast1_start, "Projection FC1 Cast and Bias");

            // GELU activation
            const gelu_start = timer.read();
            try ops.gelu(f16, &fc1);
            try printTimeDiff(&timer, gelu_start, "Projection GELU Activation");

            // Second linear layer
            const fc2_start = timer.read();

            var fc2 = try hgemm.matmul(fc1, layer_v_proj_fc2_w, self.allocator);
            errdefer fc2.deinit();
            try printTimeDiff(&timer, fc2_start, "Projection FC2 Linear");

            // Cast and bias for second layer
            const cast2_start = timer.read();
            try ops.broadcast_add_simd(&fc2, layer_v_proj_fc2_b);
            try printTimeDiff(&timer, cast2_start, "Projection FC2 Cast and Bias");

            // Print total encode MLP time
            try printTimeDiff(&timer, total_start, "Total Projection MLP");

            return fc2;
        }
    };
}
