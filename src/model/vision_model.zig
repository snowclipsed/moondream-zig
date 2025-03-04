const std = @import("std");
const Allocator = std.mem.Allocator;

const Weights = @import("weights.zig").Weights;
const visionPreSlicedWeights = @import("../preprocessing/preslice_vision.zig").visionPreSlicedWeights;
const Config = @import("config.zig").Config;

const Tensor = @import("../core/tensor.zig").Tensor;
const ops = @import("../core/ops.zig");
const hgemm = @import("../core/hgemm.zig");

const rearrangeBCHWtoBTC = @import("../utils/reshape_handler.zig").rearrangeBCHWtoBTC;
const normalizePatch = @import("../utils/reshape_handler.zig").normalizePatch;
const convertBHWCtoBCHW = @import("../utils/reshape_handler.zig").convertBHWCtoBCHW;
const attention = @import("../core/attention.zig");

const c = @cImport({
    @cInclude("stb_image.h");
    @cInclude("stb_image_resize2.h");
    @cInclude("stdio.h");
});

const mode = std.builtin.FloatMode.optimized;

comptime {
    @setFloatMode(mode);
}

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
        presliced_weights: visionPreSlicedWeights,
        allocator: Allocator,

        pub fn init(weights: Weights, allocator: Allocator) !Self {
            // Create pre-sliced weights
            var presliced_weights = try visionPreSlicedWeights.init(allocator, weights, config.n_vit_layers);
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
            // Normalize path before creating C string
            var normalized_path = try self.allocator.alloc(u8, image_path.len + 1);
            defer self.allocator.free(normalized_path);

            for (image_path, 0..) |ca, i| {
                normalized_path[i] = if (ca == '\\') '/' else ca;
            }
            normalized_path[image_path.len] = 0; // Null terminate

            var width: c_int = 0;
            var height: c_int = 0;
            var channels: c_int = 0;

            // Use normalized path
            const img_data = c.stbi_load(normalized_path.ptr, &width, &height, &channels, 0);
            if (img_data == null) {
                const err_str = c.stbi_failure_reason();
                std.debug.print("STB Image loading failed: {s}\n", .{err_str});
                return error.ImageNotFound; // More specific error type
            }
            defer c.stbi_image_free(img_data);

            const resized_data = try self.allocator.alloc(u8, Self.config.img_dim * Self.config.img_dim * @as(usize, @intCast(channels)));
            defer self.allocator.free(resized_data);

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

            var patches = try Tensor(f16).init(self.allocator, &[_]usize{
                2,
                Self.config.img_dim,
                Self.config.img_dim,
                @as(usize, @intCast(channels)),
            });
            errdefer patches.deinit();

            var b: usize = 0;
            while (b < 2) : (b += 1) {
                var h: usize = 0;
                while (h < Self.config.img_dim) : (h += 1) {
                    var w: usize = 0;
                    while (w < Self.config.img_dim) : (w += 1) {
                        var ch: usize = 0;
                        while (ch < channels) : (ch += 1) {
                            const src_idx = (h * Self.config.img_dim + w) * @as(usize, @intCast(channels)) + ch;
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

        pub fn encode_image(self: Self, image_path: []const u8) !Tensor(f16) {
            var patches = try self.createPatches(image_path);
            defer patches.deinit();

            var bchw_patches = try convertBHWCtoBCHW(self.allocator, patches);
            defer bchw_patches.deinit();

            for (bchw_patches.data) |*val| {
                val.* /= @as(f16, 255.0);
            }

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

            var normalized = try normalizePatch(self.allocator, bchw_patches, mean, stdev);
            defer normalized.deinit();

            var encoder_output = try self.vision_encoder(normalized);
            defer encoder_output.deinit();

            var output_0 = try encoder_output.getDimensionSlice(0, 0);
            defer output_0.deinit();

            var output_1 = try encoder_output.getDimensionSlice(0, 1);
            defer output_1.deinit();

            var concat_output = try ops.concat(f16, output_0, output_1, output_0.shape.len - 1);
            defer concat_output.deinit();

            var final_output = try self.encode_mlp(concat_output);
            errdefer final_output.deinit();

            return final_output;
        }

        pub fn vision_encoder(self: Self, input: Tensor(f16)) !Tensor(f16) {
            const batch = input.shape[0];

            if (batch != 2) {
                std.log.err("Expected batch size 2, got {d}\n", .{batch});
                return error.UnexpectedBatchCount;
            }

            var x = try rearrangeBCHWtoBTC(self.allocator, input, Self.config.patch_size);
            defer x.deinit();

            const B = x.shape[0];
            const M = x.shape[1];
            const N = x.shape[2];

            try x.reshape(&[_]usize{ B * M, N });
            var projected = try hgemm.matmul(x, self.weights.v_patch_embedding_linear_w, self.allocator);
            defer projected.deinit();

            try ops.broadcast_add_simd(&projected, self.weights.v_patch_embedding_linear_b);
            errdefer projected.deinit();

            try projected.reshape(&[_]usize{ B, M, Self.config.vit_dim });
            errdefer projected.deinit();
            try ops.broadcast_add_simd(&projected, self.weights.v_pos_embedding);
            errdefer projected.deinit();

            for (0..Self.config.n_vit_layers) |block| {
                try projected.reshape(&[_]usize{ B * M, Self.config.vit_dim });
                errdefer projected.deinit();

                var x_orig = try projected.copy();
                errdefer projected.deinit();
                defer x_orig.deinit();

                const ln1_w = self.presliced_weights.v_norm1_w[block];
                const ln1_b = self.presliced_weights.v_norm1_b[block];
                var ln1_out = try ops.layerNorm(f16, projected, ln1_w, ln1_b, eps);
                defer ln1_out.deinit();

                var attn_out = try self.attention_block(ln1_out, block);
                defer attn_out.deinit();

                try ops.add(f16, &x_orig, attn_out);
                @memcpy(projected.data, x_orig.data);

                var pre_mlp = try projected.copy();
                defer pre_mlp.deinit();

                const ln2_w = self.presliced_weights.v_norm2_w[block];
                const ln2_b = self.presliced_weights.v_norm2_b[block];
                var ln2_out = try ops.layerNorm(f16, projected, ln2_w, ln2_b, eps);
                defer ln2_out.deinit();

                var mlp_out = try self.mlp(ln2_out, block);
                defer mlp_out.deinit();

                try ops.add(f16, &pre_mlp, mlp_out);
                @memcpy(projected.data, pre_mlp.data);
            }

            try projected.reshape(&[_]usize{ B * M, Self.config.vit_dim });
            var final_out = try ops.layerNorm(f16, projected, self.weights.v_norm_out_w, self.weights.v_norm_out_b, eps);
            errdefer final_out.deinit();
            try final_out.reshape(&[_]usize{ B, M, Self.config.vit_dim });
            errdefer final_out.deinit();

            return final_out;
        }
        pub fn attention_block(self: Self, x: Tensor(f16), layer: usize) !Tensor(f16) {
            const n_heads = Self.config.n_vit_heads;

            const layer_v_Wqkv_w = self.presliced_weights.v_Wqkv_w[layer];
            const layer_v_Wqkv_b = self.presliced_weights.v_Wqkv_b[layer];

            var qkv = try hgemm.matmul(x, layer_v_Wqkv_w, self.allocator);
            defer qkv.deinit();

            try ops.broadcast_add_simd(&qkv, layer_v_Wqkv_b);
            errdefer qkv.deinit();

            const num_chunks = 3;
            var qkv_view = try qkv.asView();
            errdefer qkv.deinit();
            defer qkv_view.deinit();

            var chunk_view_q = try qkv_view.getChunkView(1, 0, num_chunks);
            errdefer chunk_view_q.deinit();
            var q = try chunk_view_q.toContiguousTensor();
            defer chunk_view_q.deinit();
            defer q.deinit();

            var chunk_view_k = try qkv_view.getChunkView(1, 1, num_chunks);
            errdefer chunk_view_k.deinit();
            var k = try chunk_view_k.toContiguousTensor();
            defer chunk_view_k.deinit();
            defer k.deinit();

            var chunk_view_v = try qkv_view.getChunkView(1, 2, num_chunks);
            errdefer chunk_view_v.deinit();
            var v = try chunk_view_v.toContiguousTensor();
            defer chunk_view_v.deinit();
            defer v.deinit();

            try q.reshape(&[_]usize{ q_len, n_heads, head_dim });
            try k.reshape(&[_]usize{ q_len, n_heads, head_dim });
            try v.reshape(&[_]usize{ q_len, n_heads, head_dim });

            try ops.transposeAxes(f16, &q, 0, 1);
            try ops.transposeAxes(f16, &k, 0, 1);
            try ops.transposeAxes(f16, &v, 0, 1);

            var attn_out = try attention.multiMasklessSDPA(
                n_heads,
                head_dim,
                q,
                k,
                v,
                self.allocator,
            );
            defer attn_out.deinit();

            try ops.transposeAxes(f16, &attn_out, 0, 1);
            errdefer attn_out.deinit();

            try attn_out.reshape(&[_]usize{ q_len, d_model });
            errdefer attn_out.deinit();

            var layer_v_out_proj_w = try self.weights.v_out_proj_w.getDimensionSlice(0, layer);
            defer layer_v_out_proj_w.deinit();
            var layer_v_out_proj_b = try self.weights.v_out_proj_b.getDimensionSlice(0, layer);
            defer layer_v_out_proj_b.deinit();

            var out = try hgemm.matmul(attn_out, layer_v_out_proj_w, self.allocator);
            errdefer out.deinit();

            try ops.broadcast_add_simd(&out, layer_v_out_proj_b);

            return out;
        }

        fn mlp(self: Self, input: Tensor(f16), layer: usize) !Tensor(f16) {
            const layer_v_fc1_w = self.presliced_weights.v_fc1_w[layer];
            const layer_v_fc1_b = self.presliced_weights.v_fc1_b[layer];
            const layer_v_fc2_w = self.presliced_weights.v_fc2_w[layer];
            const layer_v_fc2_b = self.presliced_weights.v_fc2_b[layer];

            var fc1 = try hgemm.matmul(input, layer_v_fc1_w, self.allocator);
            defer fc1.deinit();

            try ops.broadcast_add_simd(&fc1, layer_v_fc1_b);

            try ops.gelu(f16, &fc1);

            var fc2 = try hgemm.matmul(fc1, layer_v_fc2_w, self.allocator);
            errdefer fc2.deinit();

            try ops.broadcast_add_simd(&fc2, layer_v_fc2_b);

            return fc2;
        }

        fn encode_mlp(self: Self, input: Tensor(f16)) !Tensor(f16) {
            const layer_v_proj_fc1_w = self.weights.v_proj_fc1_w;
            const layer_v_proj_fc1_b = self.weights.v_proj_fc1_b;
            const layer_v_proj_fc2_w = self.weights.v_proj_fc2_w;
            const layer_v_proj_fc2_b = self.weights.v_proj_fc2_b;

            var fc1 = try hgemm.matmul(input, layer_v_proj_fc1_w, self.allocator);
            defer fc1.deinit();

            try ops.broadcast_add_simd(&fc1, layer_v_proj_fc1_b);
            errdefer fc1.deinit();

            try ops.gelu(f16, &fc1);
            errdefer fc1.deinit();

            var fc2 = try hgemm.matmul(fc1, layer_v_proj_fc2_w, self.allocator);
            errdefer fc2.deinit();

            try ops.broadcast_add_simd(&fc2, layer_v_proj_fc2_b);

            return fc2;
        }
    };
}
