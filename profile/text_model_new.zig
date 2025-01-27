const std = @import("std");
const print = std.debug.print;
const Allocator = std.mem.Allocator;
const Weights = @import("weights.zig").Weights;
const Config = @import("config.zig").Config;
const Tensor = @import("tensor.zig").Tensor;
const TensorView = @import("tensor.zig").TensorView;
const Slice = @import("tensor.zig").Slice;
const ops = @import("ops.zig");
const hgemm = @import("hgemm.zig");
const Timer = std.time.Timer;
const printTimeDiff = @import("timedifftext.zig").printTimeDiff;
const TextPreSlicedWeights = @import("preslice_text.zig").TextPreSlicedWeights;

const mode = std.builtin.FloatMode.optimized;
comptime {
    @setFloatMode(mode);
}

pub fn TextModel(comptime model_config: Config) type {
    return struct {
        const Self = @This();
        const config = model_config;

        // Common calculations as comptime constants
        const eps: comptime_float = 1e-5;
        const head_dim: usize = config.head_dim;
        const d_model: usize = config.dim;
        const vocab_size: usize = config.vocab;
        const max_seq_len: usize = config.seq_len;

        const KVCacheType = KVCache(config);
        pub const LayerCache = KVCacheType.LayerCache;

        // Pre-computed dimensions for attention
        const attention_dims = struct {
            const q_dim: usize = config.dim;
            const k_dim: usize = config.dim;
            const v_dim: usize = config.dim;
            const qkv_dim: usize = config.dim * 3;
            const n_heads: usize = config.n_heads;
        };

        // Pre-computed shapes for common tensors
        const tensor_shapes = struct {
            const qkv: [2]usize = .{ max_seq_len, config.dim * 3 };
            const attention: [3]usize = .{ config.n_heads, max_seq_len, config.head_dim };
            const hidden: [2]usize = .{ max_seq_len, config.dim };
            const mlp_intermediate: [2]usize = .{ max_seq_len, config.hidden_dim };
        };

        // Pre-computed rotary embedding dimensions
        const rotary_dims = struct {
            const rot_dim: usize = config.head_dim / 2;
            const theta: comptime_float = 10000.0;
        };

        // Compile-time validation
        comptime {
            if (config.head_dim * config.n_heads != config.dim) {
                @compileError("Invalid head dimensions");
            }
            if (config.seq_len == 0) {
                @compileError("Sequence length must be greater than 0");
            }
            if (config.vocab == 0) {
                @compileError("Vocabulary size must be greater than 0");
            }
            if (config.hidden_dim < config.dim) {
                @compileError("Hidden dimension must be greater than or equal to model dimension");
            }
        }

        // Instance fields
        weights: Weights,
        presliced_weights: TextPreSlicedWeights,
        allocator: Allocator,
        freqs_cis: Tensor(f32),

        pub fn init(weights: Weights, allocator: Allocator) !Self {
            // Create pre-sliced weights
            var presliced_weights = try TextPreSlicedWeights.init(allocator, weights, config.n_layers);
            errdefer presliced_weights.deinit();

            // Precompute freqs_cis
            var freqs_cis = try ops.precomputeFreqsCis(f32, allocator, config.n_heads, config.dim, rotary_dims.theta);
            errdefer freqs_cis.deinit();

            return Self{
                .weights = weights,
                .presliced_weights = presliced_weights,
                .allocator = allocator,
                .freqs_cis = freqs_cis,
            };
        }

        pub fn deinit(self: *Self) void {
            self.presliced_weights.deinit();
            self.freqs_cis.deinit();
            self.* = undefined;
        }

        // Change the input type to u32
        pub fn text_encoder(self: Self, input_ids: Tensor(u32)) !Tensor(f16) {
            if (input_ids.shape.len != 1) {
                return error.InvalidInputShape;
            }

            const seq_length = input_ids.shape[0];
            if (seq_length > Self.config.seq_len) {
                return error.SequenceTooLong;
            }

            const embedding_dim = self.weights.word_token_embedding.shape[1];

            var output = try Tensor(f16).init(self.allocator, &[_]usize{ seq_length, embedding_dim });
            errdefer output.deinit();

            for (0..seq_length) |s| {
                // Direct use of token_id without float conversion
                const token_id = input_ids.data[s];

                if (token_id >= Self.config.vocab) {
                    output.deinit();
                    return error.TokenIdOutOfBounds;
                }

                var embed_vec = try self.weights.word_token_embedding.getDimensionSlice(0, token_id);
                defer embed_vec.deinit();

                @memcpy(output.data[s * embedding_dim .. (s + 1) * embedding_dim], embed_vec.data);
            }

            return output;
        }

        pub fn text_decoder(self: Self, input_embeds: Tensor(f16), kv_cache: ?*KVCacheType) !struct { output: Tensor(f16), cache: KVCacheType } {
            var timer = try Timer.start();
            const total_start = timer.read();

            // Initialize timing tracking variables
            var total_ln_time: i128 = 0;
            var total_attention_time: i128 = 0;
            var total_mlp_time: i128 = 0;
            var total_residual_time: i128 = 0;

            var block_label_buf: [64]u8 = undefined;

            // Copy input embeddings
            const hidden_start = timer.read();
            var hidden = try input_embeds.copy();
            defer hidden.deinit();
            try printTimeDiff(&timer, hidden_start, "Input Copy");

            // Initialize new KV cache
            const cache_start = timer.read();
            var new_cache = try KVCacheType.init(self.allocator);
            errdefer new_cache.deinit();
            try printTimeDiff(&timer, cache_start, "Cache Initialization");

            // Add shape verification
            if (hidden.shape.len != 2) {
                return error.InvalidInputShape;
            }

            // Process transformer blocks
            const blocks_start = timer.read();
            for (0..Self.config.n_layers) |layer| {
                const block_start = timer.read();

                // Layer normalization
                const ln_start = timer.read();
                const layer_ln_w = self.presliced_weights.t_ln_w[layer];
                const layer_ln_b = self.presliced_weights.t_ln_b[layer];

                var ln_in = try ops.layerNorm(
                    f16,
                    hidden,
                    layer_ln_w,
                    layer_ln_b,
                    eps,
                );
                defer ln_in.deinit();
                total_ln_time += timer.read() - ln_start;

                // Attention block
                const attn_start = timer.read();
                const layer_kv_cache = if (kv_cache) |cache| try cache.getLayerCache(layer) else null;
                var attn_out = try self.attention_block(ln_in, layer, layer_kv_cache);
                defer attn_out.deinit();
                total_attention_time += timer.read() - attn_start;

                // MLP block
                const mlp_start = timer.read();
                var mlp_out = try self.mlp(ln_in, layer);
                defer mlp_out.deinit();
                total_mlp_time += timer.read() - mlp_start;

                // Residual connections
                const residual_start = timer.read();
                try ops.add(f16, &attn_out, mlp_out);
                try ops.add(f16, &hidden, attn_out);

                total_residual_time += timer.read() - residual_start;

                // Print block timing
                const block_label = try std.fmt.bufPrint(&block_label_buf, "Transformer Block {d}", .{layer});
                try printTimeDiff(&timer, block_start, block_label);
            }
            try printTimeDiff(&timer, blocks_start, "Total Transformer Blocks");

            // Print average times per block
            const blocks_f64 = @as(f64, @floatFromInt(Self.config.n_layers));
            const stdout = std.io.getStdOut().writer();
            try stdout.print("\x1b[93m [TEXT PROFILE] Average times per transformer block:\x1b[0m\n", .{});
            try stdout.print("\x1b[93m Layer Norm: {d:.2}ms\x1b[0m\n", .{@as(f64, @floatFromInt(total_ln_time)) / blocks_f64 / 1_000_000.0});
            try stdout.print("\x1b[93m Attention: {d:.2}ms\x1b[0m\n", .{@as(f64, @floatFromInt(total_attention_time)) / blocks_f64 / 1_000_000.0});
            try stdout.print("\x1b[93m MLP: {d:.2}ms\x1b[0m\n", .{@as(f64, @floatFromInt(total_mlp_time)) / blocks_f64 / 1_000_000.0});
            try stdout.print("\x1b[93m Residual Connections: {d:.2}ms\x1b[0m\n", .{@as(f64, @floatFromInt(total_residual_time)) / blocks_f64 / 1_000_000.0});

            // Print total execution time
            try printTimeDiff(&timer, total_start, "Total Text Decoder");

            var final_output = try hidden.copy();
            errdefer final_output.deinit();
            return .{
                .output = final_output,
                .cache = new_cache,
            };
        }
        fn attention_block(self: Self, input: Tensor(f16), layer: usize, layer_cache: ?*LayerCache) !Tensor(f16) {
            var timer = try Timer.start();
            const total_start = timer.read();

            const n_heads = Self.config.n_heads;
            const seq_len = input.shape[0];
            const rot_dim = Self.config.head_dim / 2;
            const pos = if (layer_cache) |cache| cache.getCurrentLen() else 0;

            // QKV projection
            const weight_start = timer.read();
            const layer_t_Wqkv_w = self.presliced_weights.t_Wqkv_w[layer];
            const layer_t_Wqkv_b = self.presliced_weights.t_Wqkv_b[layer];
            const layer_out_proj_w = self.presliced_weights.t_out_proj_w[layer];
            const layer_out_proj_b = self.presliced_weights.t_out_proj_b[layer];
            try printTimeDiff(&timer, weight_start, "QKV and Output Projection Weights");

            const qkv_proj_matmul = timer.read();
            var qkv = try hgemm.matmul(input, layer_t_Wqkv_w, self.allocator);
            defer qkv.deinit();
            try printTimeDiff(&timer, qkv_proj_matmul, "QKV Projection Matmul");

            const qkv_proj_add = timer.read();
            try ops.broadcast_add_simd(&qkv, layer_t_Wqkv_b);
            try printTimeDiff(&timer, qkv_proj_add, "QKV Projection Bias");

            // Split QKV and reshape - optimized for both paths
            const split_start = timer.read();
            const num_chunks = 3;
            var qkv_view = try qkv.asView();
            defer qkv_view.deinit();

            // Get chunks using views and convert to contiguous tensors
            const q_split_time = timer.read();
            var q_chunk_view = try qkv_view.getChunkView(1, 0, num_chunks);
            defer q_chunk_view.deinit();
            try printTimeDiff(&timer, q_split_time, "QKV Split Q");

            const q_contig_time = timer.read();
            var q = try q_chunk_view.toContiguousTensor();
            defer q.deinit();
            try printTimeDiff(&timer, q_contig_time, "Q Contiguous");

            const k_split_time = timer.read();
            var k_chunk_view = try qkv_view.getChunkView(1, 1, num_chunks);
            defer k_chunk_view.deinit();
            try printTimeDiff(&timer, k_split_time, "QKV Split K");

            const k_contig_time = timer.read();
            var k = try k_chunk_view.toContiguousTensor();
            defer k.deinit();
            try printTimeDiff(&timer, k_contig_time, "K Contiguous");

            const v_split_time = timer.read();
            var v_chunk_view = try qkv_view.getChunkView(1, 2, num_chunks);
            defer v_chunk_view.deinit();
            try printTimeDiff(&timer, v_split_time, "QKV Split V");

            const v_contig_time = timer.read();
            var v = try v_chunk_view.toContiguousTensor();
            defer v.deinit();
            try printTimeDiff(&timer, v_contig_time, "V Contiguous");
            try printTimeDiff(&timer, split_start, "TOTAL QKV Split");

            // Reshape and transpose operations
            const reshape_start = timer.read();
            try q.reshape(&[_]usize{ seq_len, n_heads, head_dim });
            try k.reshape(&[_]usize{ seq_len, n_heads, head_dim });
            try v.reshape(&[_]usize{ seq_len, n_heads, head_dim });
            try printTimeDiff(&timer, reshape_start, "QKV Reshape");

            const transpose_start = timer.read();
            try ops.transposeAxes(f16, &q, 0, 1);
            try ops.transposeAxes(f16, &k, 0, 1);
            try ops.transposeAxes(f16, &v, 0, 1);
            try printTimeDiff(&timer, transpose_start, "QKV Transpose");

            // Rotary embeddings
            const rotary_start = timer.read();
            var position_ids = try Tensor(usize).init(self.allocator, &[_]usize{seq_len});
            defer position_ids.deinit();

            for (0..seq_len) |i| {
                position_ids.data[i] = pos + i;
            }

            var qr = try ops.applyRotaryEmb(
                self.allocator,
                q,
                n_heads,
                head_dim,
                self.freqs_cis,
                position_ids,
                rot_dim,
            );
            defer qr.deinit();

            var kr = try ops.applyRotaryEmb(
                self.allocator,
                k,
                n_heads,
                head_dim,
                self.freqs_cis,
                position_ids,
                rot_dim,
            );
            errdefer kr.deinit();
            try printTimeDiff(&timer, rotary_start, "Rotary Embeddings");

            // KV cache handling
            const cache_start = timer.read();
            var k_final: Tensor(f16) = undefined;
            var v_final: Tensor(f16) = undefined;

            if (layer_cache) |cache| {
                try cache.updateCache(kr, v, seq_len);
                const active_cache = try cache.getActiveCache();
                errdefer {
                    active_cache.key.deinit();
                    active_cache.value.deinit();
                }
                k_final = active_cache.key;
                v_final = active_cache.value;
                kr.deinit();
            } else {
                k_final = try kr.copy();
                v_final = try v.copy();
                defer kr.deinit();
                defer v.deinit();
            }
            defer k_final.deinit();
            defer v_final.deinit();
            try printTimeDiff(&timer, cache_start, "KV Cache Operations");

            // Attention mask
            const mask_start = timer.read();
            var attn_mask = try ops.createAttentionMask(self.allocator, pos, seq_len);
            defer attn_mask.deinit();
            try printTimeDiff(&timer, mask_start, "Create Attention Mask");

            // Choose attention path based on sequence length
            const attn_start = timer.read();
            var attn_output: Tensor(f16) = undefined;

            if (seq_len == 1) {
                // Single token fast path
                attn_output = try ops.singleTokenAttentionFast(qr, k_final, v_final, attn_mask, self.allocator);
            } else {
                // Multi-token path
                attn_output = try ops.multiscaledDotProductAttention(qr, k_final, v_final, attn_mask, n_heads, head_dim, self.allocator);
            }
            defer attn_output.deinit();

            try printTimeDiff(&timer, attn_start, "Attention Computation");

            // Attention output processing
            const out_process_start = timer.read();
            try ops.transposeAxes(f16, &attn_output, 0, 1);
            try attn_output.reshape(&[_]usize{ seq_len, n_heads * head_dim });
            try printTimeDiff(&timer, out_process_start, "Output Processing");

            // Output projection
            const proj_start = timer.read();
            var out_proj = try hgemm.matmul(attn_output, layer_out_proj_w, self.allocator);
            errdefer out_proj.deinit();
            try ops.broadcast_add_simd(&out_proj, layer_out_proj_b);
            try printTimeDiff(&timer, proj_start, "Output Projection");

            // Print total time
            try printTimeDiff(&timer, total_start, "Total Attention Block");

            return out_proj;
        }

        fn mlp(self: Self, input: Tensor(f16), layer: usize) !Tensor(f16) {
            var timer = try Timer.start();
            const total_start = timer.read();

            const weight_start = timer.read();
            const layer_t_fc1_w = self.presliced_weights.t_fc1_w[layer];
            const layer_t_fc1_b = self.presliced_weights.t_fc1_b[layer];
            const layer_t_fc2_w = self.presliced_weights.t_fc2_w[layer];
            const layer_t_fc2_b = self.presliced_weights.t_fc2_b[layer];
            try printTimeDiff(&timer, weight_start, "MLP Weight Loading");

            // First linear layer
            const fc1_start = timer.read();

            var fc1 = try hgemm.matmul(input, layer_t_fc1_w, self.allocator);
            defer fc1.deinit();

            try ops.broadcast_add_simd(&fc1, layer_t_fc1_b);
            try printTimeDiff(&timer, fc1_start, "FC1 Layer");

            // GELU activation
            const gelu_start = timer.read();
            try ops.gelu(f16, &fc1);
            try printTimeDiff(&timer, gelu_start, "GELU Activation");

            // Second linear layer - same pattern
            const fc2_start = timer.read();

            var fc2 = try hgemm.matmul(fc1, layer_t_fc2_w, self.allocator);
            errdefer fc2.deinit();

            try ops.broadcast_add_simd(&fc2, layer_t_fc2_b);
            try printTimeDiff(&timer, fc2_start, "FC2 Layer");

            // Print total time
            try printTimeDiff(&timer, total_start, "Total MLP Block");

            return fc2;
        }

        pub fn lm_head(self: Self, hidden: Tensor(f16)) !Tensor(f16) {
            var timer = try Timer.start();
            const total_start = timer.read();

            // Input validation
            if (hidden.shape.len != 2) {
                return error.InvalidInputShape;
            }

            // Get last hidden state
            const slice_start = timer.read();
            const seq_len = hidden.shape[0];
            var last_hidden = try hidden.getDimensionSlice(0, seq_len - 1);
            defer last_hidden.deinit();
            try printTimeDiff(&timer, slice_start, "Hidden State Slice");

            // Layer normalization
            const norm_start = timer.read();
            var normalized = try ops.layerNorm(f16, last_hidden, self.weights.t_ln_out_w, self.weights.t_ln_out_b, 1e-5);
            defer normalized.deinit();
            try printTimeDiff(&timer, norm_start, "Layer Normalization");

            // Reshape
            const reshape_start = timer.read();
            try normalized.reshape(&[_]usize{ 1, Self.config.dim });
            try printTimeDiff(&timer, reshape_start, "Reshape");

            // Linear projection
            const proj_start = timer.read();
            var logits = try hgemm.matmul(normalized, self.weights.t_linear_w, self.allocator);
            errdefer logits.deinit();

            try ops.broadcast_add_simd(&logits, self.weights.t_linear_b);
            try printTimeDiff(&timer, proj_start, "Linear Projection");

            // Print total time
            try printTimeDiff(&timer, total_start, "Total LM Head");

            return logits;
        }
    };
}

pub fn KVCache(comptime model_config: Config) type {
    return struct {
        const Self = @This();
        const config = model_config;
        const max_seq_len: usize = 2048;
        const n_layers: usize = config.n_layers;
        const n_heads: usize = config.n_heads; // 32
        const head_dim: usize = config.head_dim; // 64

        layers: []LayerCache,
        allocator: Allocator,

        pub fn init(allocator: Allocator) !Self {
            var layers = try allocator.alloc(LayerCache, n_layers);
            errdefer allocator.free(layers);

            var initialized: usize = 0;
            errdefer {
                for (layers[0..initialized]) |*layer| {
                    layer.deinit();
                }
            }

            for (0..n_layers) |i| {
                layers[i] = try LayerCache.init(allocator);
                initialized += 1;
            }

            return Self{
                .layers = layers,
                .allocator = allocator,
            };
        }

        pub const LayerCache = struct {
            // Store data as 32-byte aligned arrays for SIMD
            key: []align(32) f16,
            value: []align(32) f16,
            current_len: usize,
            allocator: Allocator,

            pub fn init(allocator: Allocator) !LayerCache {
                // Flat allocation, we'll handle the layout ourselves
                const total_size = n_heads * max_seq_len * head_dim;
                const key = try allocator.alignedAlloc(f16, 32, total_size);
                errdefer allocator.free(key);
                const value = try allocator.alignedAlloc(f16, 32, total_size);
                errdefer allocator.free(value);

                return LayerCache{
                    .key = key,
                    .value = value,
                    .current_len = 0,
                    .allocator = allocator,
                };
            }

            pub fn getCurrentLen(self: *const LayerCache) usize {
                return self.current_len;
            }

            pub fn reset(self: *LayerCache) void {
                self.current_len = 0;
            }

            pub fn deinit(self: *LayerCache) void {
                self.allocator.free(self.key);
                self.allocator.free(self.value);
                self.* = undefined;
            }

            // Ultra-optimized for head_dim=64
            inline fn copyHead(dst: [*]align(32) f16, src: [*]const f16) void {
                // Copy 64 f16s = 128 bytes in 4 AVX chunks of 32 bytes each
                const chunk = @Vector(16, f16);
                const src_arr = src[0..64];
                const dst_arr = dst[0..64];

                // 4 chunks of 16 f16s each = 64 total f16s
                inline for (0..4) |i| {
                    const offset = i * 16;
                    const src_ptr = @as(*const [16]f16, @ptrCast(&src_arr[offset]));
                    const vec = @as(chunk, @bitCast(src_ptr.*));
                    @as(*align(32) chunk, @alignCast(@ptrCast(&dst_arr[offset]))).* = vec;
                }
            }

            pub fn getActiveCache(self: *LayerCache) !struct { key: Tensor(f16), value: Tensor(f16) } {
                var key_active = try Tensor(f16).init(self.allocator, &[_]usize{ n_heads, self.current_len, head_dim });
                errdefer key_active.deinit();
                var value_active = try Tensor(f16).init(self.allocator, &[_]usize{ n_heads, self.current_len, head_dim });
                errdefer value_active.deinit();

                if (self.current_len == 1) {
                    // Single token fast path - each head is exactly 64 f16s
                    for (0..n_heads) |h| {
                        const src_offset = h * max_seq_len * head_dim;
                        const dst_offset = h * head_dim;
                        @memcpy(
                            key_active.data[dst_offset..][0..head_dim],
                            self.key[src_offset..][0..head_dim],
                        );
                        @memcpy(
                            value_active.data[dst_offset..][0..head_dim],
                            self.value[src_offset..][0..head_dim],
                        );
                    }
                } else {
                    // Multi-token path
                    const elements_per_head = self.current_len * head_dim;
                    for (0..n_heads) |h| {
                        const src_offset = h * max_seq_len * head_dim;
                        const dst_offset = h * elements_per_head;
                        @memcpy(
                            key_active.data[dst_offset..][0..elements_per_head],
                            self.key[src_offset..][0..elements_per_head],
                        );
                        @memcpy(
                            value_active.data[dst_offset..][0..elements_per_head],
                            self.value[src_offset..][0..elements_per_head],
                        );
                    }
                }

                return .{ .key = key_active, .value = value_active };
            }

            pub fn updateCache(self: *LayerCache, kr: Tensor(f16), v: Tensor(f16), seq_len: usize) !void {
                const new_len = self.current_len + seq_len;
                if (new_len > max_seq_len) return error.SequenceTooLong;

                // Ultra-optimized single token path
                if (seq_len == 1) {
                    // We know each head is exactly 64 f16s = 128 bytes
                    inline for (0..n_heads) |h| {
                        const k_src_ptr: [*]const f16 = @ptrCast(&kr.data[h * head_dim]);
                        const k_dst_ptr: [*]align(32) f16 = @alignCast(@ptrCast(&self.key[h * max_seq_len * head_dim + self.current_len * head_dim]));
                        copyHead(k_dst_ptr, k_src_ptr);

                        const v_src_ptr: [*]const f16 = @ptrCast(&v.data[h * head_dim]);
                        const v_dst_ptr: [*]align(32) f16 = @alignCast(@ptrCast(&self.value[h * max_seq_len * head_dim + self.current_len * head_dim]));
                        copyHead(v_dst_ptr, v_src_ptr);
                    }
                } else {
                    // Standard multi-token path (not optimized since not used in inference)
                    for (0..n_heads) |h| {
                        const src_base = h * seq_len * head_dim;
                        const dst_base = h * max_seq_len * head_dim + self.current_len * head_dim;

                        var token: usize = 0;
                        while (token < seq_len) : (token += 1) {
                            const src_offset = src_base + token * head_dim;
                            const dst_offset = dst_base + token * head_dim;
                            @memcpy(
                                self.key[dst_offset..][0..head_dim],
                                kr.data[src_offset..][0..head_dim],
                            );
                            @memcpy(
                                self.value[dst_offset..][0..head_dim],
                                v.data[src_offset..][0..head_dim],
                            );
                        }
                    }
                }

                self.current_len = new_len;
            }
        };

        pub fn deinit(self: *Self) void {
            for (self.layers) |*layer| {
                layer.deinit();
            }
            self.allocator.free(self.layers);
            self.* = undefined;
        }

        pub fn reset(self: *Self) void {
            for (self.layers) |*layer| {
                layer.reset();
            }
        }

        pub fn getLayerCache(self: *Self, layer: usize) !*LayerCache {
            if (layer >= n_layers) {
                return error.LayerIndexOutOfBounds;
            }
            return &self.layers[layer];
        }
    };
}
// pub fn KVCache(comptime model_config: Config) type {
//     return struct {
//         const Self = @This();
//         const config = model_config;
//         const max_seq_len: usize = 2048;
//         const n_layers: usize = config.n_layers;
//         const n_heads: usize = config.n_heads;
//         const head_dim: usize = config.head_dim;

//         // SIMD optimization constants
//         const SimdVec = @Vector(8, f16);
//         const simd_width = 8;

//         layers: []LayerCache,
//         allocator: Allocator,

//         pub fn init(allocator: Allocator) !Self {
//             var layers = try allocator.alloc(LayerCache, n_layers);
//             errdefer allocator.free(layers);

//             var initialized: usize = 0;
//             errdefer {
//                 for (layers[0..initialized]) |*layer| {
//                     layer.deinit();
//                 }
//             }

//             for (0..n_layers) |i| {
//                 layers[i] = try LayerCache.init(allocator);
//                 initialized += 1;
//             }

//             return Self{
//                 .layers = layers,
//                 .allocator = allocator,
//             };
//         }

//         pub const LayerCache = struct {
//             key: Tensor(f16),
//             value: Tensor(f16),
//             current_len: usize,

//             fn simdCopySlice(dst: []f16, src: []const f16, length: usize) void {
//                 const aligned_len = length - (length % simd_width);
//                 var i: usize = 0;

//                 // SIMD copy for aligned chunks
//                 while (i < aligned_len) : (i += simd_width) {
//                     const src_aligned = @as(*align(2) const [simd_width]f16, @ptrCast(src[i..].ptr));
//                     const vec = @as(SimdVec, @bitCast(src_aligned.*));
//                     @as(*SimdVec, @alignCast(@ptrCast(dst[i..].ptr))).* = vec;
//                 }

//                 // Copy remaining elements
//                 while (i < length) : (i += 1) {
//                     dst[i] = src[i];
//                 }
//             }

//             pub fn init(allocator: Allocator) !LayerCache {
//                 return LayerCache{
//                     .key = try Tensor(f16).init(allocator, &[_]usize{ n_heads, max_seq_len, head_dim }),
//                     .value = try Tensor(f16).init(allocator, &[_]usize{ n_heads, max_seq_len, head_dim }),
//                     .current_len = 0,
//                 };
//             }

//             pub fn reset(self: *LayerCache) void {
//                 self.current_len = 0;
//             }

//             pub fn deinit(self: *LayerCache) void {
//                 self.key.deinit();
//                 self.value.deinit();
//                 self.* = undefined;
//             }

//             pub fn getActiveCache(self: *LayerCache) !struct { key: Tensor(f16), value: Tensor(f16) } {
//                 // Create new tensors with active dimensions
//                 var key_active = try Tensor(f16).init(self.key.allocator, &[_]usize{ n_heads, self.current_len, head_dim });
//                 errdefer key_active.deinit();

//                 var value_active = try Tensor(f16).init(self.value.allocator, &[_]usize{ n_heads, self.current_len, head_dim });
//                 errdefer value_active.deinit();

//                 // Calculate strides for proper memory layout
//                 const src_head_stride = max_seq_len * head_dim;
//                 const dst_head_stride = self.current_len * head_dim;

//                 // Copy data maintaining exact memory layout
//                 for (0..n_heads) |h| {
//                     // Process each head
//                     const src_head_offset = h * src_head_stride;
//                     const dst_head_offset = h * dst_head_stride;

//                     // Copy data for each sequence position
//                     for (0..self.current_len) |s| {
//                         const src_offset = src_head_offset + s * head_dim;
//                         const dst_offset = dst_head_offset + s * head_dim;

//                         // Use SIMD copy for the head dimension
//                         simdCopySlice(key_active.data[dst_offset..][0..head_dim], self.key.data[src_offset..][0..head_dim], head_dim);
//                         simdCopySlice(value_active.data[dst_offset..][0..head_dim], self.value.data[src_offset..][0..head_dim], head_dim);
//                     }
//                 }

//                 return .{ .key = key_active, .value = value_active };
//             }
//         };

//         pub fn deinit(self: *Self) void {
//             for (self.layers) |*layer| {
//                 layer.deinit();
//             }
//             self.allocator.free(self.layers);
//             self.* = undefined;
//         }

//         pub fn reset(self: *Self) void {
//             for (self.layers) |*layer| {
//                 layer.reset();
//             }
//         }

//         pub fn getLayerCache(self: *Self, layer: usize) !*LayerCache {
//             if (layer >= n_layers) {
//                 return error.LayerIndexOutOfBounds;
//             }
//             return &self.layers[layer];
//         }
//     };
// }
