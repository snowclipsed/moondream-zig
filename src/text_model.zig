const std = @import("std");
const print = std.debug.print;
const Allocator = std.mem.Allocator;
const Weights = @import("weights.zig").Weights;
const Config = @import("config.zig").Config;
const Tensor = @import("tensor.zig").Tensor;
const Slice = @import("tensor.zig").Slice;
const ops = @import("ops.zig");
const matmul = @import("matmul.zig");

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

    pub fn text_encoder(self: Self, input_ids: Tensor(f32)) !Tensor(f32) {
        if (input_ids.shape.len != 1) {
            return error.InvalidInputShape;
        }

        const seq_length = input_ids.shape[0];
        if (seq_length > self.config.seq_len) {
            return error.SequenceTooLong;
        }

        const embedding_dim = self.weights.word_token_embedding.shape[1];

        // Using ops.zeros instead of Tensor.zeros
        var output = try ops.zeros(f32, self.allocator, &[_]usize{ seq_length, embedding_dim });
        errdefer output.deinit();

        // Rest of the function remains the same
        for (0..seq_length) |s| {
            const token_id = @as(usize, @intFromFloat(input_ids.data[s]));

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

    pub fn text_decoder(self: Self, input_embeds: Tensor(f32), kv_cache: ?*KVCache) !struct { output: Tensor(f32), cache: KVCache } {
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
                f32,
                hidden,
                layer_ln_w,
                layer_ln_b,
                eps,
            );

            //layernorm is probably okay!

            // Get layer cache if it exists
            const layer_kv_cache = if (kv_cache) |cache| try cache.getLayerCache(layer) else null;

            var attn_out = try self.attention_block(ln_in, layer, layer_kv_cache);
            defer attn_out.deinit();

            // MLP

            var mlp_out = try self.mlp(ln_in, layer);
            defer mlp_out.deinit();

            // Residual connection

            try ops.add(f32, &attn_out, mlp_out);

            try ops.add(f32, &hidden, attn_out);

            // frees

            defer ln_in.deinit(); // this needs to be here.
        }

        return .{
            .output = try hidden.copy(),
            .cache = new_cache,
        };
    }

    fn attention_block(self: Self, input: Tensor(f32), layer: usize, layer_cache: ?*LayerCache) !Tensor(f32) {
        const n_heads = self.config.n_heads;
        const head_dim = self.config.head_dim;
        const seq_len = input.shape[0];
        const rot_dim = self.config.head_dim / 2;

        const pos = if (layer_cache) |cache| cache.key.shape[1] else 0;

        // print("Attention block: layer {d}, pos {d}\n", .{ layer, pos });

        var layer_t_Wqkv_w = try self.weights.t_Wqkv_w.getDimensionSlice(0, layer);
        defer layer_t_Wqkv_w.deinit();
        var layer_t_Wqkv_b = try self.weights.t_Wqkv_b.getDimensionSlice(0, layer);
        defer layer_t_Wqkv_b.deinit();

        var qkv = try matmul.matmul(f32, input, layer_t_Wqkv_w, self.allocator);
        defer qkv.deinit();

        try ops.broadcast_add(f32, &qkv, layer_t_Wqkv_b); // change to broadcastadd

        // 2. Now we need to split this into Q, K, V
        // The qkv tensor has shape [seq_len, 3 * n_heads * head_dim]
        // We want to split into 3 tensors of shape [seq_len, n_heads * head_dim]

        const num_chunks = 3;

        var q = try ops.getChunk(f32, qkv, 1, 0, num_chunks);
        defer q.deinit();
        var k = try ops.getChunk(f32, qkv, 1, 1, num_chunks);
        defer k.deinit();
        var v = try ops.getChunk(f32, qkv, 1, 2, num_chunks);

        // 3. Reshape each tensor from [seq_len, n_heads * head_dim] to [seq_len, n_heads, head_dim]
        try q.reshape(&[_]usize{ seq_len, n_heads, head_dim });
        try k.reshape(&[_]usize{ seq_len, n_heads, head_dim });
        try v.reshape(&[_]usize{ seq_len, n_heads, head_dim });

        // // 4. Transpose from [seq_len, n_heads, head_dim] to [n_heads, seq_len, head_dim]
        try ops.transposeAxes(f32, &q, 0, 1); // Swap seq_len and n_heads
        try ops.transposeAxes(f32, &k, 0, 1);
        try ops.transposeAxes(f32, &v, 0, 1);

        // 5. Apply rotary embeddings
        var position_ids = try Tensor(usize).init(self.allocator, &[_]usize{seq_len});
        defer position_ids.deinit();

        for (0..seq_len) |i| {
            position_ids.data[i] = pos + i;
        }

        var qr = try ops.applyRotaryEmb(
            f32,
            self.allocator,
            q,
            self.freqs_cis,
            position_ids,
            rot_dim,
            false,
        );
        defer qr.deinit();
        var kr = try ops.applyRotaryEmb(
            f32,
            self.allocator,
            k,
            self.freqs_cis,
            position_ids,
            rot_dim,
            false,
        );
        // qr and kr are correct! and rotary embeddings are correct!
        // Handle KV cache
        var k_final: Tensor(f32) = undefined;
        var v_final: Tensor(f32) = undefined;

        if (layer_cache) |cache| {
            // Concatenate with existing cache
            k_final = try ops.concat(f32, cache.key, kr, 1);
            v_final = try ops.concat(f32, cache.value, v, 1);

            // Update cache with new K and V
            cache.key.deinit();
            cache.value.deinit();

            // Transfer ownership
            cache.key = kr; // kr ownership moved to cache
            cache.value = v; // v ownership moved to cache

            // Remove the defers for kr and v since ownership is transferred
            // (make sure no defer kr.deinit() or defer v.deinit() exist)
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
        var attn_output = try ops.scaledDotProductAttention(f32, qr, k_final, v_final, attn_mask, self.allocator);

        // Reshape output back
        try ops.transposeAxes(f32, &attn_output, 0, 1);
        try attn_output.reshape(&[_]usize{ seq_len, n_heads * head_dim });
        defer attn_output.deinit();

        // Linear layer

        var layer_out_proj_w = try self.weights.t_out_proj_w.getDimensionSlice(0, layer);
        defer layer_out_proj_w.deinit();
        var layer_out_proj_b = try self.weights.t_out_proj_bias.getDimensionSlice(0, layer);
        defer layer_out_proj_b.deinit();
        var out_proj = try matmul.matmul(f32, attn_output, layer_out_proj_w, self.allocator);

        try ops.broadcast_add(f32, &out_proj, layer_out_proj_b);

        return out_proj;
    }

    fn mlp(self: Self, input: Tensor(f32), layer: usize) !Tensor(f32) {
        var layer_t_fc1_w = try self.weights.t_fc1_w.getDimensionSlice(0, layer);
        defer layer_t_fc1_w.deinit();
        var layer_t_fc1_b = try self.weights.t_fc1_b.getDimensionSlice(0, layer);
        defer layer_t_fc1_b.deinit();
        var layer_t_fc2_w = try self.weights.t_fc2_w.getDimensionSlice(0, layer);
        defer layer_t_fc2_w.deinit();
        var layer_t_fc2_b = try self.weights.t_fc2_b.getDimensionSlice(0, layer);
        defer layer_t_fc2_b.deinit();

        var fc1 = try matmul.matmul(f32, input, layer_t_fc1_w, self.allocator);
        try ops.broadcast_add(f32, &fc1, layer_t_fc1_b);

        try ops.gelu(f32, &fc1);

        var fc2 = try matmul.matmul(f32, fc1, layer_t_fc2_w, self.allocator);
        try ops.broadcast_add(f32, &fc2, layer_t_fc2_b);
        fc1.deinit();

        return fc2;
    }

    pub fn lm_head(self: Self, hidden: Tensor(f32)) !Tensor(f32) {
        if (hidden.shape.len != 2) {
            return error.InvalidInputShape;
        }

        const seq_len = hidden.shape[0];
        // const hidden_dim = hidden.shape[1];

        // Get last hidden state using getDimensionSlice
        var last_hidden = try hidden.getDimensionSlice(0, seq_len - 1);
        defer last_hidden.deinit();

        var normalized = try ops.layerNorm(f32, last_hidden, self.weights.t_ln_out_w, self.weights.t_ln_out_b, 1e-5);
        defer normalized.deinit();

        try normalized.reshape(&[_]usize{ 1, self.config.dim }); // TODO: Check correctness of this

        var logits = try matmul.matmul(f32, normalized, self.weights.t_linear_w, self.allocator);

        try ops.broadcast_add(f32, &logits, self.weights.t_linear_b);

        return logits;
    }
};

pub const LayerCache = struct {
    key: Tensor(f32),
    value: Tensor(f32),

    pub fn init(allocator: Allocator, n_heads: usize, head_dim: usize) !LayerCache {
        return LayerCache{
            .key = try Tensor(f32).init(allocator, &[_]usize{ n_heads, 0, head_dim }),
            .value = try Tensor(f32).init(allocator, &[_]usize{ n_heads, 0, head_dim }),
        };
    }

    pub fn deinit(self: *LayerCache) void {
        self.key.deinit();
        self.value.deinit();
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
