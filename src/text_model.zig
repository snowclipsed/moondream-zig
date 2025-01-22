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

pub const TextModel = struct {
    const Self = @This();
    config: Config,
    weights: Weights,
    allocator: Allocator,
    freqs_cis: Tensor(f32),

    pub fn init(config: Config, weights: Weights, allocator: Allocator) !TextModel {
        var textmodel = TextModel{
            .config = config,
            .weights = weights,
            .allocator = allocator,
            .freqs_cis = undefined,
        };

        const theta = 10000.0;
        textmodel.freqs_cis = try ops.precomputeFreqsCis(f32, allocator, config.n_heads, config.dim, theta);

        return textmodel;
    }

    pub fn deinit(self: *Self) void {
        self.weights.deinit();
        self.freqs_cis.deinit();

        self.* = undefined;
    }

    // Change the input type to u32
    pub fn text_encoder(self: Self, input_ids: Tensor(u32)) !Tensor(f16) {
        if (input_ids.shape.len != 1) {
            return error.InvalidInputShape;
        }

        const seq_length = input_ids.shape[0];
        if (seq_length > self.config.seq_len) {
            return error.SequenceTooLong;
        }

        const embedding_dim = self.weights.word_token_embedding.shape[1];

        var output = try ops.zeros(f16, self.allocator, &[_]usize{ seq_length, embedding_dim });
        errdefer output.deinit();

        for (0..seq_length) |s| {
            // Direct use of token_id without float conversion
            const token_id = input_ids.data[s];

            if (token_id >= self.config.vocab) {
                output.deinit();
                return error.TokenIdOutOfBounds;
            }

            var embed_vec = try self.weights.word_token_embedding.getDimensionSlice(0, token_id);
            defer embed_vec.deinit();

            @memcpy(output.data[s * embedding_dim .. (s + 1) * embedding_dim], embed_vec.data);
        }

        return output;
    }

    pub fn text_decoder(self: Self, input_embeds: Tensor(f16), kv_cache: ?*KVCache) !struct { output: Tensor(f16), cache: KVCache } {
        const eps = 1e-5; // TODO move to config
        var hidden = try input_embeds.copy();
        defer hidden.deinit();

        // Initialize new KV cache
        var new_cache = try KVCache.init(self.allocator, self.config.n_layers, self.config.n_heads, self.config.head_dim);
        errdefer new_cache.deinit();

        // Add shape verification before the loop
        if (hidden.shape.len != 2) {
            return error.InvalidInputShape;
        }

        for (0..self.config.n_layers) |layer| {
            var layer_ln_w = try self.weights.t_ln_w.getDimensionSlice(0, layer);
            defer layer_ln_w.deinit();
            var layer_ln_b = try self.weights.t_ln_b.getDimensionSlice(0, layer);
            defer layer_ln_b.deinit();

            var ln_in = try ops.layerNorm(
                f16,
                hidden,
                layer_ln_w,
                layer_ln_b,
                eps,
            );
            defer ln_in.deinit();
            //layernorm is probably okay!

            // Get layer cache if it exists
            const layer_kv_cache = if (kv_cache) |cache| try cache.getLayerCache(layer) else null;

            var attn_out = try self.attention_block(ln_in, layer, layer_kv_cache);
            defer attn_out.deinit();

            var mlp_out = try self.mlp(ln_in, layer);
            defer mlp_out.deinit();

            // Residual connection

            try ops.add(f16, &attn_out, mlp_out);

            try ops.add(f16, &hidden, attn_out);
        }

        return .{
            .output = try hidden.copy(),
            .cache = new_cache,
        };
    }
    fn attention_block(self: Self, input: Tensor(f16), layer: usize, layer_cache: ?*LayerCache) !Tensor(f16) {
        const n_heads = self.config.n_heads;
        const head_dim = self.config.head_dim;
        const seq_len = input.shape[0];
        const rot_dim = self.config.head_dim / 2;

        const pos = if (layer_cache) |cache| cache.current_len else 0;

        var layer_t_Wqkv_w = try self.weights.t_Wqkv_w.getDimensionSlice(0, layer);
        defer layer_t_Wqkv_w.deinit();
        var layer_t_Wqkv_b = try self.weights.t_Wqkv_b.getDimensionSlice(0, layer);
        defer layer_t_Wqkv_b.deinit();

        var qkv_f32 = try Tensor(f32).init(self.allocator, &[_]usize{ seq_len, 3 * n_heads * head_dim });
        defer qkv_f32.deinit();
        try hgemm.matmul(self.allocator, input, layer_t_Wqkv_w, qkv_f32);

        var qkv = try qkv_f32.castTo(f16);
        defer qkv.deinit();

        try ops.broadcast_add(f16, &qkv, layer_t_Wqkv_b); // change to broadcastadd

        // 2. Now we need to split this into Q, K, V
        // The qkv tensor has shape [seq_len, 3 * n_heads * head_dim]
        // We want to split into 3 tensors of shape [seq_len, n_heads * head_dim]

        const num_chunks = 3;

        var q = try ops.getChunk(f16, qkv, 1, 0, num_chunks);
        defer q.deinit();
        var k = try ops.getChunk(f16, qkv, 1, 1, num_chunks);
        defer k.deinit();
        var v = try ops.getChunk(f16, qkv, 1, 2, num_chunks);

        // 3. Reshape each tensor from [seq_len, n_heads * head_dim] to [seq_len, n_heads, head_dim]
        try q.reshape(&[_]usize{ seq_len, n_heads, head_dim });
        try k.reshape(&[_]usize{ seq_len, n_heads, head_dim });
        try v.reshape(&[_]usize{ seq_len, n_heads, head_dim });

        // // 4. Transpose from [seq_len, n_heads, head_dim] to [n_heads, seq_len, head_dim]
        try ops.transposeAxes(f16, &q, 0, 1); // Swap seq_len and n_heads
        try ops.transposeAxes(f16, &k, 0, 1);
        try ops.transposeAxes(f16, &v, 0, 1);

        // 5. Apply rotary embeddings
        var position_ids = try Tensor(usize).init(self.allocator, &[_]usize{seq_len});
        defer position_ids.deinit();

        for (0..seq_len) |i| {
            position_ids.data[i] = pos + i;
        }

        var qr = try ops.applyRotaryEmb(
            self.allocator,
            q,
            self.freqs_cis,
            position_ids,
            rot_dim,
            false,
        );
        defer qr.deinit();

        var kr = try ops.applyRotaryEmb(
            self.allocator,
            k,
            self.freqs_cis,
            position_ids,
            rot_dim,
            false,
        );

        // Handle KV cache
        var k_final: Tensor(f16) = undefined;
        var v_final: Tensor(f16) = undefined;

        if (layer_cache) |cache| {
            const new_len = cache.current_len + seq_len;
            if (new_len > 2048) {
                return error.SequenceTooLong;
            }

            // Copy new values to the cache at the correct position
            for (0..n_heads) |h| {
                const k_src_start = h * seq_len * head_dim;
                const k_dst_start = h * 2048 * head_dim + cache.current_len * head_dim;
                const k_len = seq_len * head_dim;

                @memcpy(
                    cache.key.data[k_dst_start .. k_dst_start + k_len],
                    kr.data[k_src_start .. k_src_start + k_len],
                );

                const v_src_start = h * seq_len * head_dim;
                const v_dst_start = h * 2048 * head_dim + cache.current_len * head_dim;
                const v_len = seq_len * head_dim;

                @memcpy(
                    cache.value.data[v_dst_start .. v_dst_start + v_len],
                    v.data[v_src_start .. v_src_start + v_len],
                );
            }

            // Update cache length
            cache.current_len = new_len;

            // Get active cache views
            const active_cache = try cache.getActiveCache();
            k_final = active_cache.key;
            v_final = active_cache.value;

            // Clean up original rotated tensors
            kr.deinit();
            v.deinit();
        } else {
            k_final = try kr.copy();
            v_final = try v.copy();
            defer kr.deinit();
            defer v.deinit();
        }

        defer k_final.deinit();
        defer v_final.deinit();

        var attn_mask = try ops.createAttentionMask(self.allocator, pos, seq_len);
        defer attn_mask.deinit();

        // Perform masked attention
        var attn_output = try ops.scaledDotProductAttention(qr, k_final, v_final, attn_mask, self.allocator);
        defer attn_output.deinit();
        // Reshape output back
        try ops.transposeAxes(f16, &attn_output, 0, 1);
        try attn_output.reshape(&[_]usize{ seq_len, n_heads * head_dim });

        // Linear layer

        var layer_out_proj_w = try self.weights.t_out_proj_w.getDimensionSlice(0, layer);
        defer layer_out_proj_w.deinit();
        var layer_out_proj_b = try self.weights.t_out_proj_bias.getDimensionSlice(0, layer);
        defer layer_out_proj_b.deinit();

        var out_proj_f32 = try Tensor(f32).init(self.allocator, &[_]usize{ seq_len, self.config.dim });
        defer out_proj_f32.deinit();
        try hgemm.matmul(self.allocator, attn_output, layer_out_proj_w, out_proj_f32);

        var out_proj = try out_proj_f32.castTo(f16);
        errdefer out_proj.deinit();

        try ops.broadcast_add(f16, &out_proj, layer_out_proj_b);
        return out_proj;
    }

    fn mlp(self: Self, input: Tensor(f16), layer: usize) !Tensor(f16) {
        var layer_t_fc1_w = try self.weights.t_fc1_w.getDimensionSlice(0, layer);
        defer layer_t_fc1_w.deinit();
        var layer_t_fc1_b = try self.weights.t_fc1_b.getDimensionSlice(0, layer);
        defer layer_t_fc1_b.deinit();
        var layer_t_fc2_w = try self.weights.t_fc2_w.getDimensionSlice(0, layer);
        defer layer_t_fc2_w.deinit();
        var layer_t_fc2_b = try self.weights.t_fc2_b.getDimensionSlice(0, layer);
        defer layer_t_fc2_b.deinit();

        var fc1_f32 = try Tensor(f32).init(self.allocator, &[_]usize{ input.shape[0], self.config.hidden_dim });
        defer fc1_f32.deinit();
        try hgemm.matmul(self.allocator, input, layer_t_fc1_w, fc1_f32);

        var fc1 = try fc1_f32.castTo(f16);
        defer fc1.deinit();

        try ops.broadcast_add(f16, &fc1, layer_t_fc1_b);

        try ops.gelu(f16, &fc1);

        var fc2_f32 = try Tensor(f32).init(self.allocator, &[_]usize{ input.shape[0], self.config.dim });
        defer fc2_f32.deinit();
        try hgemm.matmul(
            self.allocator,
            fc1,
            layer_t_fc2_w,
            fc2_f32,
        );

        var fc2 = try fc2_f32.castTo(f16);
        errdefer fc2.deinit();

        try ops.broadcast_add(f16, &fc2, layer_t_fc2_b);

        return fc2;
    }

    pub fn lm_head(self: Self, hidden: Tensor(f16)) !Tensor(f16) {
        if (hidden.shape.len != 2) {
            return error.InvalidInputShape;
        }

        const seq_len = hidden.shape[0];
        // Get last hidden state using getDimensionSlice
        var last_hidden = try hidden.getDimensionSlice(0, seq_len - 1);
        defer last_hidden.deinit();

        var normalized = try ops.layerNorm(f16, last_hidden, self.weights.t_ln_out_w, self.weights.t_ln_out_b, 1e-5);
        defer normalized.deinit();

        try normalized.reshape(&[_]usize{ 1, self.config.dim }); // TODO: Check correctness of this

        var logits_f32 = try Tensor(f32).init(self.allocator, &[_]usize{ 1, self.config.vocab });
        defer logits_f32.deinit();

        try hgemm.matmul(self.allocator, normalized, self.weights.t_linear_w, logits_f32);

        var logits = try logits_f32.castTo(f16);
        errdefer logits.deinit();

        try ops.broadcast_add(f16, &logits, self.weights.t_linear_b);

        return logits;
    }
};

pub const LayerCache = struct {
    key: Tensor(f16),
    value: Tensor(f16),
    current_len: usize,

    pub fn init(allocator: Allocator, n_heads: usize, head_dim: usize) !LayerCache {
        const max_seq_len = 2048; // Match PyTorch's size
        return LayerCache{
            .key = try Tensor(f16).init(allocator, &[_]usize{ n_heads, max_seq_len, head_dim }),
            .value = try Tensor(f16).init(allocator, &[_]usize{ n_heads, max_seq_len, head_dim }),
            .current_len = 0,
        };
    }

    pub fn deinit(self: *LayerCache) void {
        self.key.deinit();
        self.value.deinit();
    }

    pub fn getActiveCache(self: *LayerCache) !struct { key: Tensor(f16), value: Tensor(f16) } {
        var key_slice = try self.key.getSliceRange(&[_]Slice{
            Slice.full(),
            Slice.from(0, self.current_len),
            Slice.full(),
        });
        errdefer key_slice.deinit();

        var value_slice = try self.value.getSliceRange(&[_]Slice{
            Slice.full(),
            Slice.from(0, self.current_len),
            Slice.full(),
        });
        errdefer value_slice.deinit();

        return .{ .key = key_slice, .value = value_slice };
    }
};

pub const KVCache = struct {
    layers: []LayerCache,
    allocator: Allocator,

    pub fn init(allocator: Allocator, n_layers: usize, n_heads: usize, head_dim: usize) !KVCache {
        var layers = try allocator.alloc(LayerCache, n_layers);
        errdefer allocator.free(layers);

        for (0..n_layers) |i| {
            layers[i] = try LayerCache.init(allocator, n_heads, head_dim);
        }

        return KVCache{
            .layers = layers,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *KVCache) void {
        for (self.layers) |*layer| {
            layer.deinit();
        }
        self.allocator.free(self.layers);
    }

    pub fn getLayerCache(self: *KVCache, layer: usize) !*LayerCache {
        if (layer >= self.layers.len) {
            return error.LayerIndexOutOfBounds;
        }
        return &self.layers[layer];
    }
};
