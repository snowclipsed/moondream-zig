const std = @import("std");
const print = std.debug.print;
const Allocator = std.mem.Allocator;
const Weights = @import("weights.zig").Weights;
const Config = @import("config.zig").Config;
const Tensor = @import("tensor.zig").Tensor;
const TensorView = @import("tensor.zig").TensorView;
const Slice = @import("tensor.zig").Slice;
const ops = @import("ops.zig");
const hgemm = @import("hgemmnew.zig");
const sgemm = @import("sgemm.zig");
const Timer = std.time.Timer;
const printTimeDiff = @import("timedifftext.zig").printTimeDiff;
const TextPreSlicedWeights = @import("preslice_text.zig").TextPreSlicedWeights;

pub const TextModel = struct {
    const Self = @This();
    config: Config,
    weights: Weights,
    presliced_weights: TextPreSlicedWeights,
    allocator: Allocator,
    freqs_cis: Tensor(f32),

    pub fn init(config: Config, weights: Weights, allocator: Allocator) !TextModel {
        // Create pre-sliced weights
        var presliced_weights = try TextPreSlicedWeights.init(allocator, weights, config.n_layers);
        errdefer presliced_weights.deinit();

        // Precompute freqs_cis
        const theta = 10000.0;
        var freqs_cis = try ops.precomputeFreqsCis(f32, allocator, config.n_heads, config.dim, theta);
        errdefer freqs_cis.deinit();

        return TextModel{
            .config = config,
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
        if (seq_length > self.config.seq_len) {
            return error.SequenceTooLong;
        }

        const embedding_dim = self.weights.word_token_embedding.shape[1];

        var output = try Tensor(f16).init(self.allocator, &[_]usize{ seq_length, embedding_dim });
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
        // var timer = try Timer.start();
        // const total_start = timer.read();

        const eps = 1e-5;

        // Initialize timing tracking variables
        // var total_ln_time: i128 = 0;
        // var total_attention_time: i128 = 0;
        // var total_mlp_time: i128 = 0;
        // var total_residual_time: i128 = 0;

        // var block_label_buf: [64]u8 = undefined;

        // Copy input embeddings
        // const hidden_start = timer.read();
        var hidden = try input_embeds.copy();
        defer hidden.deinit();
        // try printTimeDiff(&timer, hidden_start, "Input Copy");

        // Initialize new KV cache
        // const cache_start = timer.read();
        var new_cache = try KVCache.init(self.allocator, self.config.n_layers, self.config.n_heads, self.config.head_dim);
        errdefer new_cache.deinit();
        // try printTimeDiff(&timer, cache_start, "Cache Initialization");

        // Add shape verification
        if (hidden.shape.len != 2) {
            return error.InvalidInputShape;
        }

        // Process transformer blocks
        // const blocks_start = timer.read();
        for (0..self.config.n_layers) |layer| {
            // const block_start = timer.read();

            // Layer normalization
            // const ln_start = timer.read();
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
            // total_ln_time += timer.read() - ln_start;

            // Attention block
            // const attn_start = timer.read();
            const layer_kv_cache = if (kv_cache) |cache| try cache.getLayerCache(layer) else null;
            var attn_out = try self.attention_block(ln_in, layer, layer_kv_cache);
            defer attn_out.deinit();
            // total_attention_time += timer.read() - attn_start;

            // MLP block
            // const mlp_start = timer.read();
            var mlp_out = try self.mlp(ln_in, layer);
            defer mlp_out.deinit();
            // total_mlp_time += timer.read() - mlp_start;

            // Residual connections
            // const residual_start = timer.read();
            try ops.add(f16, &attn_out, mlp_out);
            try ops.add(f16, &hidden, attn_out);

            // total_residual_time += timer.read() - residual_start;

            // Print block timing
            // const block_label = try std.fmt.bufPrint(&block_label_buf, "Transformer Block {d}", .{layer});
            // try printTimeDiff(&timer, block_start, block_label);
        }
        // try printTimeDiff(&timer, blocks_start, "Total Transformer Blocks");

        // Print average times per block
        // const blocks_f64 = @as(f64, @floatFromInt(self.config.n_layers));
        // const stdout = std.io.getStdOut().writer();
        // try stdout.print("\x1b[93m [TEXT PROFILE] Average times per transformer block:\x1b[0m\n", .{});
        // try stdout.print("\x1b[93m Layer Norm: {d:.2}ms\x1b[0m\n", .{@as(f64, @floatFromInt(total_ln_time)) / blocks_f64 / 1_000_000.0});
        // try stdout.print("\x1b[93m Attention: {d:.2}ms\x1b[0m\n", .{@as(f64, @floatFromInt(total_attention_time)) / blocks_f64 / 1_000_000.0});
        // try stdout.print("\x1b[93m MLP: {d:.2}ms\x1b[0m\n", .{@as(f64, @floatFromInt(total_mlp_time)) / blocks_f64 / 1_000_000.0});
        // try stdout.print("\x1b[93m Residual Connections: {d:.2}ms\x1b[0m\n", .{@as(f64, @floatFromInt(total_residual_time)) / blocks_f64 / 1_000_000.0});

        // Print total execution time
        // try printTimeDiff(&timer, total_start, "Total Text Decoder");

        return .{
            .output = try hidden.copy(),
            .cache = new_cache,
        };
    }
    fn attention_block(self: Self, input: Tensor(f16), layer: usize, layer_cache: ?*LayerCache) !Tensor(f16) {
        var timer = try Timer.start();
        const total_start = timer.read();

        const n_heads = self.config.n_heads;
        const head_dim = self.config.head_dim;
        const seq_len = input.shape[0];
        const rot_dim = self.config.head_dim / 2;

        const pos = if (layer_cache) |cache| cache.current_len else 0;

        // QKV projection
        const qkv_proj_start = timer.read();
        const layer_t_Wqkv_w = self.presliced_weights.t_Wqkv_w[layer];
        const layer_t_Wqkv_b = self.presliced_weights.t_Wqkv_b[layer];
        const layer_out_proj_w = self.presliced_weights.t_out_proj_w[layer];
        const layer_out_proj_b = self.presliced_weights.t_out_proj_b[layer];

        var qkv = try hgemm.matmul(input, layer_t_Wqkv_w, self.allocator);
        defer qkv.deinit();

        try ops.broadcast_add_simd(&qkv, layer_t_Wqkv_b);
        try printTimeDiff(&timer, qkv_proj_start, "QKV Projection");

        // Split QKV and reshape
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
        try printTimeDiff(&timer, split_start, "QKV Split");

        // Reshape and transpose operations
        const reshape_start = timer.read();
        try q.reshape(&[_]usize{ seq_len, n_heads, head_dim });
        try k.reshape(&[_]usize{ seq_len, n_heads, head_dim });
        try v.reshape(&[_]usize{ seq_len, n_heads, head_dim });
        try printTimeDiff(&timer, reshape_start, "QKV Reshape ");
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
            self.freqs_cis,
            position_ids,
            rot_dim,
        );
        defer qr.deinit();

        var kr = try ops.applyRotaryEmb(
            self.allocator,
            k,
            self.freqs_cis,
            position_ids,
            rot_dim,
        );
        try printTimeDiff(&timer, rotary_start, "Rotary Embeddings");

        // KV cache handling
        const cache_start = timer.read();
        var k_final: Tensor(f16) = undefined;
        var v_final: Tensor(f16) = undefined;

        if (layer_cache) |cache| {
            const new_len = cache.current_len + seq_len;
            if (new_len > 2048) {
                return error.SequenceTooLong;
            }

            // Copy new values to the cache
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

            cache.current_len = new_len;

            const active_cache = try cache.getActiveCache();
            k_final = active_cache.key;
            v_final = active_cache.value;

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
        try printTimeDiff(&timer, cache_start, "KV Cache Operations");

        // Attention mask and computation
        const attn_start = timer.read();
        var attn_mask = try ops.createAttentionMask(self.allocator, pos, seq_len);
        defer attn_mask.deinit();

        var attn_output = try ops.multiscaledDotProductAttention(qr, k_final, v_final, attn_mask, self.allocator);
        defer attn_output.deinit();

        try ops.transposeAxes(f16, &attn_output, 0, 1);
        try attn_output.reshape(&[_]usize{ seq_len, n_heads * head_dim });
        try printTimeDiff(&timer, attn_start, "Attention Computation");

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
        // var timer = try Timer.start();
        // const total_start = timer.read();

        // Weight loading
        // const weight_start = timer.read();
        const layer_t_fc1_w = self.presliced_weights.t_fc1_w[layer];
        const layer_t_fc1_b = self.presliced_weights.t_fc1_b[layer];
        const layer_t_fc2_w = self.presliced_weights.t_fc2_w[layer];
        const layer_t_fc2_b = self.presliced_weights.t_fc2_b[layer];
        // try printTimeDiff(&timer, weight_start, "MLP Weight Loading");

        // First linear layer
        // const fc1_start = timer.read();

        var fc1 = try hgemm.matmul(input, layer_t_fc1_w, self.allocator);
        defer fc1.deinit();

        try ops.broadcast_add_simd(&fc1, layer_t_fc1_b);
        // try printTimeDiff(&timer, fc1_start, "FC1 Layer");

        // GELU activation
        // const gelu_start = timer.read();
        try ops.gelu(f16, &fc1);
        // try printTimeDiff(&timer, gelu_start, "GELU Activation");

        // Second linear layer
        // const fc2_start = timer.read();

        var fc2 = try hgemm.matmul(fc1, layer_t_fc2_w, self.allocator);
        errdefer fc2.deinit();

        try ops.broadcast_add_simd(&fc2, layer_t_fc2_b);
        // try printTimeDiff(&timer, fc2_start, "FC2 Layer");

        // Print total time
        // try printTimeDiff(&timer, total_start, "Total MLP Block");

        return fc2;
    }

    pub fn lm_head(self: Self, hidden: Tensor(f16)) !Tensor(f16) {
        // var timer = try Timer.start();
        // const total_start = timer.read();

        // Input validation
        if (hidden.shape.len != 2) {
            return error.InvalidInputShape;
        }

        // Get last hidden state
        // const slice_start = timer.read();
        const seq_len = hidden.shape[0];
        var last_hidden = try hidden.getDimensionSlice(0, seq_len - 1);
        defer last_hidden.deinit();
        // try printTimeDiff(&timer, slice_start, "Hidden State Slice");

        // Layer normalization
        // const norm_start = timer.read();
        var normalized = try ops.layerNorm(f16, last_hidden, self.weights.t_ln_out_w, self.weights.t_ln_out_b, 1e-5);
        defer normalized.deinit();
        // try printTimeDiff(&timer, norm_start, "Layer Normalization");

        // Reshape
        // const reshape_start = timer.read();
        try normalized.reshape(&[_]usize{ 1, self.config.dim });
        // try printTimeDiff(&timer, reshape_start, "Reshape");

        // Linear projection
        // const proj_start = timer.read();
        var logits = try hgemm.matmul(normalized, self.weights.t_linear_w, self.allocator);
        errdefer logits.deinit();

        try ops.broadcast_add_simd(&logits, self.weights.t_linear_b);
        // try printTimeDiff(&timer, proj_start, "Linear Projection");

        // Print total time
        // try printTimeDiff(&timer, total_start, "Total LM Head");

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
    pub fn reset(self: *LayerCache) void {
        self.current_len = 0;
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
    pub fn reset(self: *KVCache) void {
        for (self.layers) |*layer| {
            layer.reset();
        }
    }

    pub fn getLayerCache(self: *KVCache, layer: usize) !*LayerCache {
        if (layer >= self.layers.len) {
            return error.LayerIndexOutOfBounds;
        }
        return &self.layers[layer];
    }
};
