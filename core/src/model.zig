const std = @import("std");
const assert = std.debug.assert;
const mem = std.mem;
const Allocator = mem.Allocator;
const Thread = std.Thread;
const builtin = @import("builtin");
const T: usize = 64; // Tile size (adjust as necessary)
const V: usize = 32; // Vector size (adjust as necessary)
const simd_align = @alignOf(@Vector(V, f32));
const c = @cImport({
    @cInclude("stb_image.h");
    @cInclude("stb_image_resize2.h");
    @cInclude("stdio.h");
});

const Config = @import("config.zig").Config;
const Weights = @import("weights.zig").Weights;
const Tokenizer = @import("tokenizer.zig").Tokenizer;
const RunState = @import("runstate.zig").RunState;

const Model = struct {
    const Self = @This();
    config: Config,
    weights: Weights,
    tokenizer: Tokenizer,
    state: RunState,
    allocator: Allocator,

    fn init(config: Config, weights: Weights, tokenizer: Tokenizer, state: RunState, allocator: Allocator) !Model {
        return Model{
            .config = config,
            .weights = weights,
            .tokenizer = tokenizer,
            .state = state,
            .allocator = allocator,
        };
    }

    fn text_model(self: Self, embeddings: []f32, pos: usize) !void {
        // each token embedding is of size dim, so we divide by it to see
        // how many tokens are present
        // so we perform this assert before making any further calculations
        assert(embeddings.len % self.config.dim == 0);

        // we define our layernorm epsilon constant:
        const eps = 1e-5;
        const dim = self.config.dim;
        const n_heads = self.config.n_heads;
        const head_dim = self.config.head_dim;
        const rotary_emb_dim = head_dim / 2;
        const pass_dim = head_dim - rotary_emb_dim;

        //now we find the number of tokens, or the query length

        const q_len = embeddings.len / dim;

        // we will add the incoming q_len to the current pos which indicates the sequence length of the key and value vectors
        // this also allows us to keep track of the current position in the kv cache

        std.debug.print("q_len: {any} \n", .{q_len});

        // other constants
        //layernorm input
        const ln_in = try self.allocator.alloc(f32, embeddings.len);
        defer self.allocator.free(ln_in);

        // qkv states tensor
        const qkv = try self.allocator.alloc(f32, q_len * dim * 3);
        defer self.allocator.free(qkv);

        const q = try self.allocator.alloc(f32, q_len * dim);
        const k = try self.allocator.alloc(f32, q_len * dim);
        const v = try self.allocator.alloc(f32, q_len * dim);
        defer self.allocator.free(q);
        defer self.allocator.free(k);
        defer self.allocator.free(v);

        const q_r = try self.allocator.alloc(f32, q_len * dim);
        const k_r = try self.allocator.alloc(f32, q_len * dim);
        const v_r = try self.allocator.alloc(f32, q_len * dim);
        // Allocate memory for query_rot and query_pass
        const query_rot = try self.allocator.alloc(f32, n_heads * q_len * (rotary_emb_dim));
        defer self.allocator.free(query_rot);
        const query_pass = try self.allocator.alloc(f32, n_heads * q_len * (rotary_emb_dim));
        defer self.allocator.free(query_pass);
        const key_rot = try self.allocator.alloc(f32, n_heads * q_len * (rotary_emb_dim));
        defer self.allocator.free(key_rot);
        const key_pass = try self.allocator.alloc(f32, n_heads * q_len * (rotary_emb_dim));
        defer self.allocator.free(key_pass);
        const kv_seq_len = q_len + pos;
        std.debug.print("kv seq len : {any} \n", .{kv_seq_len});
        const scores = try self.allocator.alloc(f32, q_len * kv_seq_len);
        defer self.allocator.free(scores);

        for (0..self.config.n_layers) |l| {

            // get the residual before performing layernorm
            // this way the original input embeddings will act as a residual

            // we then perform layernorm

            for (0..q_len) |query| {
                try layer_norm(
                    ln_in[query * dim .. (query + 1) * dim],
                    self.weights.t_ln_w[l * dim .. (l + 1) * dim],
                    self.weights.t_ln_b[l * dim .. (l + 1) * dim],
                    eps,
                );
            }

            // next we will perform sdpa

            // we will first extract the q, k, v states as a combined vector:
            try matmul(
                self.allocator,
                ln_in,
                self.weights.t_Wqkv_w[l * dim * 3 * dim .. (l + 1) * dim * 3 * dim],
                qkv,
                q_len,
                dim * 3,
                dim,
            );

            // printMatrix(q_len, dim * 3, qkv, 2, 5);

            // we split qkv into three state vectors for q, k, v

            // Manual strided copy
            for (0..q_len * dim) |i| {
                q[i] = qkv[i * 3];
                k[i] = qkv[i * 3 + 1];
                v[i] = qkv[i * 3 + 2];
            }

            // we then convert the q, k, v vectors into a different view of dims (n_heads, q_len, head_dim)
            // TODO : Check if they actually are (n_heads, q_len, head_dim)
            for (0..n_heads) |h| {
                for (0..q_len) |i| {
                    for (0..head_dim) |j| {
                        const old_index = i * dim + h * head_dim + j;
                        const new_index = h * q_len * head_dim + i * head_dim + j;
                        q_r[new_index] = q[old_index];
                        k_r[new_index] = k[old_index];
                        v_r[new_index] = v[old_index];
                    }
                }
            }

            // first we need to calculate the inverse frequencies.
            // the length of the sin and cos cache is equal to the max sequence length = 2048
            // we the obtain the position frequencies from the cos sin cache for the current position + the number of tokens we are processing, which would be from 0 till the position of the current token

            // we will then apply RoPE
            // after that, we update the kv cache, and hence the pos counter for the kv cache will also update
            // we update this in the generate function
            // after we finish generation, we reset pos to 0

            const freqs = self.get_rot_emb(kv_seq_len);
            const cos = freqs.@"0";
            const sin = freqs.@"1";

            // add query rot and query pass here:

            // Split the query states into rotary and pass-through parts
            for (0..n_heads) |h| {
                for (0..q_len) |i| {
                    // Rotary part
                    for (0..rotary_emb_dim) |j| {
                        const old_index = h * q_len * head_dim + i * head_dim + j;
                        const new_rot_index = h * q_len * rotary_emb_dim + i * rotary_emb_dim + j;
                        query_rot[new_rot_index] = q_r[old_index];
                        key_rot[new_rot_index] = k_r[old_index];
                    }

                    // Pass-through part
                    for (0..head_dim - rotary_emb_dim) |j| {
                        const old_index = h * q_len * head_dim + i * head_dim + rotary_emb_dim + j;
                        const new_pass_index = h * q_len * (head_dim - rotary_emb_dim) +
                            i * (head_dim - rotary_emb_dim) + j;
                        query_pass[new_pass_index] = q_r[old_index];
                        key_pass[new_pass_index] = k_r[old_index];
                    }
                }
            }

            try self.apply_rope(query_rot, key_rot, cos, sin, pos, q_len);

            // here we will concatenate the query_rot and query_pass vectors back together into the q_r vector
            // the same will be done for key_rot and key_pass vectors into the k_r vector
            for (0..n_heads) |h| {
                for (0..q_len) |seq_idx| {
                    const out_base = h * q_len * head_dim + seq_idx * head_dim;
                    const rot_base = h * q_len * rotary_emb_dim + seq_idx * rotary_emb_dim;
                    const pass_base = h * q_len * pass_dim + seq_idx * pass_dim;

                    @memcpy(q_r[out_base .. out_base + rotary_emb_dim], query_rot[rot_base .. rot_base + rotary_emb_dim]);
                    @memcpy(q_r[out_base + rotary_emb_dim .. out_base + head_dim], query_pass[pass_base .. pass_base + pass_dim]);
                    @memcpy(k_r[out_base .. out_base + rotary_emb_dim], key_rot[rot_base .. rot_base + rotary_emb_dim]);
                    @memcpy(k_r[out_base + rotary_emb_dim .. out_base + head_dim], key_pass[pass_base .. pass_base + pass_dim]);
                }
            }

            // we will now update the kv cache
            // the size of the kv cache is (n_layers, seq_len, dim)

            const l_off = self.config.seq_len * self.config.dim;
            @memcpy(self.state.k_cache[l * l_off .. (l + 1) * l_off], k_r);
            @memcpy(self.state.v_cache[l * l_off .. (l + 1) * l_off], v_r);

            // attention

            try self.attention(q_r, k_r, v_r, embeddings, pos, q_len, l);

            // mlp

            try self.mlp(embeddings, embeddings, q_len, l);
        }
    }

    fn lm_head(self: Self, hidden_states: []f32, q_len: usize) ![]f32 {
        const dim = self.config.dim;
        const vocab_size = self.config.vocab;

        // If q_len > 1, we need to process each token to get their logits
        // Allocate memory for all logits (vocab_size for each token in q_len)
        const logits = try self.allocator.alloc(f32, q_len * vocab_size);

        // Process each token in the sequence
        for (0..q_len) |i| {
            // Get the hidden state for the current token
            const token_hidden = hidden_states[i * dim .. (i + 1) * dim];

            // Allocate memory for normalized hidden state
            const normalized = try self.allocator.alloc(f32, dim);
            defer self.allocator.free(normalized);

            // Copy current token's hidden state to normalize it
            @memcpy(normalized, token_hidden);

            // Apply layer normalization
            try layer_norm(
                normalized,
                self.weights.t_ln_out_w,
                self.weights.t_ln_out_b,
                1e-5, // epsilon
            );

            // Project to vocabulary size using lm_head weights
            try matmul(
                self.allocator,
                normalized,
                self.weights.t_linear_w,
                logits[i * vocab_size .. (i + 1) * vocab_size],
                1, // batch size is 1 for each token
                vocab_size,
                dim,
            );

            // Add bias if it exists

            accumulate(logits, self.weights.t_linear_b);
        }

        return logits;
    }

    fn attention_mask(allocator: std.mem.Allocator, pos: usize, seq_len: usize) ![]f32 {
        // Total sequence length includes both past (pos) and current (seq_len) tokens
        const total_seq_len = pos + seq_len;

        // Allocate 2D mask of shape [seq_len, total_seq_len]
        const mask = try allocator.alloc(f32, seq_len * total_seq_len);

        // Fill everything with 1s first - this allows attention to all past tokens
        @memset(mask, 1.0);

        // For the current sequence part, implement causal masking
        for (0..seq_len) |i| {
            for (0..seq_len) |j| {
                // Calculate position in the full sequence
                const col = pos + j;
                const mask_idx = i * total_seq_len + col;

                // Ensure we don't write out of bounds
                if (mask_idx >= mask.len) {
                    return error.IndexOutOfBounds;
                }

                // Create causal mask: token i can only attend to tokens 0..=i
                mask[mask_idx] = if (j <= i) 1.0 else 0.0;
            }
        }

        return mask;
    }

    fn attention(self: Self, q: []f32, k: []f32, v: []f32, embeddings: []f32, pos: usize, q_len: usize, l: usize) !void {
        const n_heads = self.config.n_heads;
        const head_dim = self.config.head_dim;
        const kv_seq_len = q_len + pos;
        const dim = self.config.dim;

        // separate scores are allocated for each head:
        const scores = try self.allocator.alloc(f32, n_heads * q_len * kv_seq_len);
        self.allocator.free(scores);

        // attention output will be the same size as the input embedding
        const attn_output = try self.allocator.alloc(f32, q_len * dim);
        defer self.allocator.free(attn_output);

        @memset(attn_output, 0);

        const mask = try attention_mask(self.allocator, pos, q_len);
        defer self.allocator.free(mask);

        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

        // Compute attention scores for each head
        for (0..n_heads) |h| {
            // Calculate scaled dot product attention
            // Q * K^T / sqrt(head_dim)

            for (0..q_len) |i| {
                for (0..kv_seq_len) |j| {
                    var score: f32 = 0.0;
                    const q_offset = h * q_len * head_dim + i * head_dim;
                    const k_offset = h * kv_seq_len * head_dim + j * head_dim;

                    // Compute dot product
                    for (0..head_dim) |d| {
                        score += q[q_offset + d] * k[k_offset + d];
                    }

                    // Scale and apply mask
                    score *= scale;
                    score *= mask[i * kv_seq_len + j];

                    scores[h * q_len * kv_seq_len + i * kv_seq_len + j] = score;
                }
            }
        }

        // Apply softmax to scores
        for (0..n_heads) |h| {
            for (0..q_len) |i| {
                const start_idx = h * q_len * kv_seq_len + i * kv_seq_len;
                const end_idx = start_idx + kv_seq_len;
                try softmax(scores[start_idx..end_idx]);
            }
        }

        // Compute attention output
        for (0..n_heads) |h| {
            for (0..q_len) |i| {
                for (0..head_dim) |d| {
                    var sum: f32 = 0.0;
                    for (0..kv_seq_len) |j| {
                        const score = scores[h * q_len * kv_seq_len + i * kv_seq_len + j];
                        const v_val = v[h * kv_seq_len * head_dim + j * head_dim + d];
                        sum += score * v_val;
                    }
                    const out_idx = i * dim + h * head_dim + d;
                    attn_output[out_idx] = sum;
                }
            }
        }

        // Allocate temporary buffer for projection output
        const proj_output = try self.allocator.alloc(f32, q_len * dim);
        defer self.allocator.free(proj_output);

        try matmul(
            self.allocator,
            attn_output,
            self.weights.t_out_proj_w[l * self.config.dim * self.config.dim .. (l + 1) * self.config.dim * self.config.dim],
            proj_output,
            q_len,
            dim,
            dim,
        );

        accumulate(proj_output, self.weights.t_out_proj_bias[l * self.config.dim .. (l + 1) * self.config.dim]);
        accumulate(embeddings, proj_output);
    }

    fn mlp(self: Self, input: []f32, embeddings: []f32, q_len: usize, l: usize) !void {
        const dim = self.config.dim;
        const hidden_dim = self.config.hidden_dim;

        const fc1_out = try self.allocator.alloc(f32, q_len * hidden_dim);
        defer self.allocator.free(fc1_out);

        const fc2_out = try self.allocator.alloc(f32, q_len * dim);
        defer self.allocator.free(fc2_out);

        // first we will perform the first linear layer

        try matmul(
            self.allocator,
            input,
            self.weights.t_fc1_w[l * dim * hidden_dim .. (l + 1) * dim * hidden_dim],
            fc1_out,
            q_len,
            hidden_dim,
            dim,
        );

        accumulate(fc1_out, self.weights.t_fc1_b[l * hidden_dim .. (l + 1) * hidden_dim]);

        // we will then apply the gelu activation function

        gelu(fc1_out);

        // we will then perform the second linear layer
        // downcasting the activated vector from hidden_dim to dim

        try matmul(
            self.allocator,
            fc1_out,
            self.weights.t_fc2_w[l * hidden_dim * dim .. (l + 1) * hidden_dim * dim],
            fc2_out,
            q_len,
            dim,
            hidden_dim,
        );

        accumulate(fc2_out, self.weights.t_fc2_b[l * dim .. (l + 1) * dim]);

        // we will then add the residual connection

        accumulate(embeddings, fc2_out);
    }

    fn getvector(self: Self, vector: []f32, head_index: usize, seq_index: usize, head_dim_index: usize) f32 {
        // Calculate the flat index for `q` assuming it's shaped as (num_heads, q_len, head_dim)
        const flat_index = seq_index * self.config.dim + head_index * self.config.head_dim + head_dim_index;
        return vector[flat_index];
    }

    // we will then transform each of q, k, v vectors
    // they will go from (q_len, dim) to (n_heads, qlen, head_dim)
    // dim = n_heads * head_dim

    pub fn layer_norm(
        inputs: []f32,
        weight: []const f32,
        bias: []const f32,
        eps: f32,
    ) !void {
        const len = inputs.len;
        if (len == 0) return error.EmptyInput;
        if (len != weight.len or len != bias.len) return error.DimensionMismatch;

        // Compute the mean
        var mean: f32 = 0.0;
        for (inputs) |x| {
            mean += x;
        }
        const n: f32 = @floatFromInt(len);
        mean /= n;

        // Compute the variance
        var variance: f32 = 0.0;
        for (inputs) |x| {
            const diff = x - mean;
            variance += diff * diff;
        }
        variance /= n;

        // Compute standard deviation
        const std_dev = @sqrt(variance + eps);

        // Normalize the inputs
        for (inputs, 0..inputs.len) |*x, i| {
            const normalized = (x.* - mean) / std_dev;
            x.* = normalized * weight[i] + bias[i];

            // Check for numerical stability
            if (std.math.isNan(x.*) or std.math.isInf(x.*)) {
                std.debug.print("Warning: Output contains NaN or Inf at index {d}. Input: {d}, Normalized: {d}, Weight: {d}, Bias: {d}, Mean: {d}, Std: {d}\n", .{ i, x.*, normalized, weight[i], bias[i], mean, std_dev });
                return error.NumericalInstability;
            }
        }
    }

    fn set_cos_sin_cache(self: Self) !void {
        const max_cached_seq_len = self.config.max_pos_embeddings;
        const dim = self.config.head_dim / 2; // equivalent to head dim * 0.5 = 32
        const base = self.config.rope_theta;

        std.debug.print("Cache initialization:\n", .{});
        std.debug.print("max_seq_len: {d}\n", .{max_cached_seq_len});
        std.debug.print("cache size should be: {d}\n", .{max_cached_seq_len * dim});

        const inv_freq = try self.allocator.alloc(f32, dim / 2);
        defer self.allocator.free(inv_freq);

        var i: usize = 0;
        var index: usize = 0;

        while (i < dim) : (i += 2) {
            const x = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(dim));
            inv_freq[index] = 1.0 / std.math.pow(f32, base, x);
            index += 1;
        }

        const t = try self.allocator.alloc(f32, max_cached_seq_len);
        defer self.allocator.free(t);

        const freqs = try self.allocator.alloc(f32, max_cached_seq_len * dim / 2);
        for (0..max_cached_seq_len) |itx| {
            t[itx] = @floatFromInt(itx);
        }

        // performing equivalent of torch.outer
        outer(t, inv_freq, freqs);

        // concatenate two freq tensors together so we have a tensor of size (max_cached_seq_len, dim) back again
        const emb = try cat(self.allocator, freqs, freqs, max_cached_seq_len, dim / 2, max_cached_seq_len, dim / 2, 1);
        defer self.allocator.free(emb);

        // now we just have to apply sin and cost to the entire emb tensor and that becomes the sin and cos cache matrix!
        // size of both sin and cos cache is (max_cached_seq_len, dim)
        try cos_2d(emb, self.state.cos_cache);
        try sin_2d(emb, self.state.sin_cache);
        // printMatrix(max_cached_seq_len, dim, self.state.cos_cache, 3, 5);
    }

    fn get_rot_emb(self: Self, seq_len: usize) struct { []f32, []f32 } {
        const half_dim = self.config.head_dim / 2;
        // Should create arrays of size seq_len * half_dim
        return .{
            .@"0" = self.state.cos_cache[0 .. seq_len * half_dim],
            .@"1" = self.state.sin_cache[0 .. seq_len * half_dim],
        };
    }

    fn apply_rope(self: Self, q: []f32, k: []f32, cos: []f32, sin: []f32, pos: usize, qlen: usize) !void {
        const n_heads = self.config.n_heads;
        const head_dim = self.config.head_dim;
        const half_dim = head_dim / 2;

        std.debug.print("RoPE debug:\n", .{});
        std.debug.print("cos len: {d}\n", .{cos.len});
        std.debug.print("head_dim: {d}\n", .{head_dim});
        std.debug.print("half_dim: {d}\n", .{half_dim});
        std.debug.print("pos: {d}\n", .{pos});
        std.debug.print("qlen: {d}\n", .{qlen});
        std.debug.print("max position access will be: {d}\n", .{(pos + qlen - 1) * half_dim + (half_dim - 1)});

        const q_rotated = try self.allocator.alloc(f32, q.len);
        defer self.allocator.free(q_rotated);
        const k_rotated = try self.allocator.alloc(f32, k.len);
        defer self.allocator.free(k_rotated);

        for (0..n_heads) |h| {
            for (0..qlen) |seq_idx| {
                const base_idx = h * qlen * head_dim + seq_idx * head_dim;
                const position = pos + seq_idx;

                // First do the rotation
                // For first half: copy negated values from second half
                for (0..half_dim) |dim| {
                    q_rotated[base_idx + dim] = -q[base_idx + dim + half_dim];
                    k_rotated[base_idx + dim] = -k[base_idx + dim + half_dim];
                }
                // For second half: copy values from first half
                for (0..half_dim) |dim| {
                    q_rotated[base_idx + half_dim + dim] = q[base_idx + dim];
                    k_rotated[base_idx + half_dim + dim] = k[base_idx + dim];
                }

                // Then apply the rotation using cos/sin
                for (0..half_dim) |dim| {
                    const cos_val = cos[position * half_dim + dim];
                    const sin_val = sin[position * half_dim + dim];

                    // Apply to first half
                    const idx1 = base_idx + dim;
                    const idx2 = base_idx + dim + half_dim;

                    const q1 = q[idx1];
                    const q2 = q[idx2];
                    q[idx1] = q1 * cos_val - q2 * sin_val;
                    q[idx2] = q1 * sin_val + q2 * cos_val;

                    const k1 = k[idx1];
                    const k2 = k[idx2];
                    k[idx1] = k1 * cos_val - k2 * sin_val;
                    k[idx2] = k1 * sin_val + k2 * cos_val;
                }
            }
        }
    }

    fn embed_tokens(self: Self, tokens: std.ArrayList(u32)) ![]f32 {
        var text_embed = try self.allocator.alloc(f32, tokens.items.len * self.config.dim);
        errdefer self.allocator.free(text_embed);

        for (tokens.items, 0..) |token, i| {
            if (token >= self.weights.word_token_embedding.len / self.config.dim) {
                return error.TokenOutOfBounds;
            }
            const src_start = token * self.config.dim;
            const src_end = src_start + self.config.dim;
            const dst_start = i * self.config.dim;
            const dst_end = dst_start + self.config.dim;
            @memcpy(text_embed[dst_start..dst_end], self.weights.word_token_embedding[src_start..src_end]);
        }

        return text_embed;
    }

    fn merge_embed(self: Self, text_embed: []f32, image_embed: []f32) ![]f32 {
        const embedding = try self.allocator.alloc(f32, text_embed.len + image_embed.len);
        std.debug.print("Merging text embed of size {any} and image embed of size {any} \n", .{ text_embed.len / self.config.dim, image_embed.len / self.config.dim });
        @memcpy(embedding[0..image_embed.len], image_embed);
        @memcpy(embedding[0..text_embed.len], text_embed);
        return embedding;
    }
    /// This function will load the images and then preprocess them into the required format
    pub fn preprocess(self: Self, image_path: []const u8, allocator: Allocator) ![]f32 {
        // Load the image
        const target_height = self.config.img_dim;
        const target_width = self.config.img_dim;

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

        const resized_data = try allocator.alloc(u8, target_height * target_width * @as(usize, @intCast(channels)));

        const result = c.stbir_resize_uint8_srgb(
            img_data,
            width,
            height,
            0,
            resized_data.ptr,
            @as(c_int, @intCast(target_width)),
            @as(c_int, @intCast(target_height)),
            0,
            if (channels == 3) c.STBIR_RGB else c.STBIR_RGBA,
        );

        if (result == 0) {
            return error.FailedToResizeImage;
        }

        var float_image = try allocator.alloc(f32, target_width * target_height * @as(usize, @intCast(channels)));
        const mean = [3]f32{ 0.5, 0.5, 0.5 };
        const stddev = [3]f32{ 0.5, 0.5, 0.5 };

        // reorganize the data from (H,W,C) to (C, H, W) format that torch uses
        // initially data is stored in [R,G,B,R,G,B,R,G,B...R,G,B] format
        // now we want to store it as [R,R,R,R,R,R,..G,G,G,G,G..B,B,B,B,B] format where the RGB values are contiguous
        for (0..@as(usize, @intCast(channels))) |ch| {
            for (0..target_height) |h| {
                for (0..target_width) |w| {
                    const src_idx = (h * target_width + w) * @as(usize, @intCast(channels)) + ch;
                    const dst_idx = ch * target_height * target_width + h * target_width + w;

                    const pixel_value: u8 = resized_data[src_idx];
                    // scale to 0-1 range
                    const scaled_value = @as(f32, @floatFromInt(pixel_value)) / 255.0;
                    // apply normalization
                    const normalized_value = (scaled_value - mean[ch]) / stddev[ch];
                    float_image[dst_idx] = normalized_value;
                }
            }
        }
        return float_image;
    }
    fn vision_encoder(self: Self) !void {
        std.debug.print("Image len : {any} \n", .{self.state.img.len});
        // now the vision encoder will take in the float image and divide it into
        // patches of self.patch_size x self.patch_size (14 x 14)
        // we have to rearrange the values of the patches

        // calculate the number of patches along height and width, these are the same for now
        // because we're using tensors which images resized to 378 x 378 through preprocess()

        // just defining constants from the config here
        const channels = self.config.img_channels; // 3 channels
        const img_h = self.config.img_dim;
        const img_w = self.config.img_dim;
        const patch_h = self.config.patch_size;
        const patch_w = self.config.patch_size;
        const patch_elements = patch_h * patch_w * channels;
        const num_patches_h = img_h / patch_h;
        const num_patches_w = img_h / patch_w;
        const num_patches = self.config.num_patches;

        // we are going to change the format of our image from (C, H, W) to (h * w, C * p1 * p2) or (729, 3 * 14 * 14)
        for (0..num_patches_h) |h_patch| {
            for (0..num_patches_w) |w_patch| {
                for (0..channels) |ch| {
                    for (0..patch_h) |h| {
                        for (0..patch_w) |w| {
                            const src_idx = ch * img_h * img_w + (h_patch * patch_h + h) * img_w + (w_patch * patch_w + w);
                            const dest_idx = (h_patch * num_patches_w + w_patch) * channels * patch_h * patch_w + (ch * patch_h * patch_w + h * patch_w + w);
                            self.state.patches[dest_idx] = self.state.img[src_idx];
                        }
                    }
                }
            }
        }

        // we get the patch embedding by doing matmul with the patch embedding linear layer and adding bias
        // each patch is individually multiplied and then stored into self.state.patches!

        for (0..num_patches) |patch| {
            // for each patch, we do matrix multiplication
            // (1, 3 * 14 * 14)  @ (3 * 14 * 14, 1152)
            // all this is stored in patch_emb which is (729, 1152)
            try matmul(
                self.allocator,
                self.state.patches[patch * patch_elements .. (patch + 1) * patch_elements],
                self.weights.v_patch_embedding_linear_w,
                self.state.patch_emb[patch * self.config.vit_dim .. (patch + 1) * self.config.vit_dim],
                1,
                self.config.vit_dim,
                patch_elements,
            );
            accumulate(
                self.state.patch_emb[patch * self.config.vit_dim .. (patch + 1) * self.config.vit_dim],
                self.weights.v_patch_embedding_linear_b,
            );
        }

        // next up is positional embedding, which is directly just accumulated into the patch embedding!
        // x = x + pos_embed
        // pos embed dim (729, 1152) + v_pos_embed (1152, 729) ^ Transposed
        accumulate(self.state.patch_emb, try transposeSimd(self.allocator, self.weights.v_pos_embedding, 1152, 729));

        // we will now pass our positionally encoded patch embeddings through the ViT blocks.
        // v_x (729, 1152)

        for (0..self.config.n_vit_layers) |l| {
            @memcpy(self.state.v_x, self.state.patch_emb);

            // we will normalize each patch by iterating over each patch, because our layernorm can only do a single patch vector of inputs
            for (0..num_patches) |patch| {
                try layer_norm(
                    self.state.v_x[patch * self.config.vit_dim .. (patch + 1) * self.config.vit_dim],
                    self.weights.v_norm1_w[l * self.config.vit_dim .. (l + 1) * self.config.vit_dim],
                    self.weights.v_norm1_b[l * self.config.vit_dim .. (l + 1) * self.config.vit_dim],
                    1e-5,
                );
            }
            // now our patch embedding is normalized, we are going to get the attention weights
            // we multiply our patch embedding to get the qkv for all the patches all at once for the specific layer
            // patch_emb (729, 1152) @ v_Wqkv (1152, 3456) = v_qkv (729, 3456)
            try matmul(
                self.allocator,
                self.state.v_x,
                self.weights.v_Wqkv_w[l * self.config.vit_dim * 3 * self.config.vit_dim .. (l + 1) * self.config.vit_dim * 3 * self.config.vit_dim],
                self.state.v_qkv,
                num_patches,
                self.config.vit_dim * 3,
                self.config.vit_dim,
            );
            // next we accumulate the bias for that layer into v_qkv!
            // we need to iterate over all the patches again to do so!
            for (0..num_patches) |patch| {
                // 1 patch from v_qkv (1, 3456) = 1 patch from v_qkv (1, 3456) + Wqkv_b(3456)
                // over all patches it will be v_qkv(729, 3456)
                accumulate(
                    self.state.v_qkv[patch * self.config.vit_dim * 3 .. (patch + 1) * self.config.vit_dim * 3],
                    self.weights.v_Wqkv_b[l * self.config.vit_dim * 3 .. (l + 1) * self.config.vit_dim * 3],
                );
            }

            @memcpy(self.state.v_q, self.state.v_qkv[0 .. num_patches * self.config.vit_dim]);
            @memcpy(self.state.v_k, self.state.v_qkv[num_patches * self.config.vit_dim .. num_patches * self.config.vit_dim * 2]);
            @memcpy(self.state.v_v, self.state.v_qkv[num_patches * self.config.vit_dim * 2 .. num_patches * self.config.vit_dim * 3]);

            const head_dim = self.config.vit_dim / self.config.n_vit_heads;

            for (0..self.config.n_vit_heads) |head| {
                const v_q_head = self.state.v_q[head * head_dim * num_patches .. (head + 1) * head_dim * num_patches];
                const v_k_head = self.state.v_k[head * head_dim * num_patches .. (head + 1) * head_dim * num_patches];
                const v_v_head = self.state.v_v[head * head_dim * num_patches .. (head + 1) * head_dim * num_patches];

                // Compute the attention score by taking the dot product of the query and key for this head
                // v_q_head (num_patches, head_dim) @ v_k_head.T (head_dim, num_patches) = v_attn (num_patches, num_patches)
                try matmul(
                    self.allocator,
                    v_q_head,
                    v_k_head, // We would need to transpose v_k_head for this operation
                    self.state.v_attn[head * num_patches * num_patches .. (head + 1) * num_patches * num_patches],
                    num_patches,
                    num_patches,
                    head_dim,
                );

                // Scale the attention scores by sqrt(head_dim) to stabilize gradients as per the scaled dot-product attention mechanism
                const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
                for (0..num_patches * num_patches) |i| {
                    self.state.v_attn[head * num_patches * num_patches + i] *= scale;
                }

                // Apply softmax to get the attention probabilities
                // We will be applying softmax row-wise over the num_patches dimension
                try softmax(self.state.v_attn[head * num_patches * num_patches .. (head + 1) * num_patches * num_patches]);

                // Multiply attention probabilities with value matrix to get the final output for this head
                // v_attn (num_patches, num_patches) @ v_v_head (num_patches, head_dim) = output (num_patches, head_dim)
                try matmul(
                    self.allocator,
                    self.state.v_attn[head * num_patches * num_patches .. (head + 1) * num_patches * num_patches],
                    v_v_head,
                    self.state.v_output[head * num_patches * head_dim .. (head + 1) * num_patches * head_dim],
                    num_patches,
                    head_dim,
                    num_patches,
                );
            }

            // Next, we will multiply the final output from all the heads with the attention projection layer for this vit block
            // v_output (num_patches, vit_dim) @  v_out_proj_w (vit_dim, vit_dim) = v_proj (num_patches, vit_dim)

            try matmul(
                self.allocator,
                self.state.v_output,
                self.weights.v_out_proj_w[l * self.config.vit_dim * self.config.vit_dim .. (l + 1) * self.config.vit_dim * self.config.vit_dim],
                self.state.v_proj,
                num_patches,
                self.config.vit_dim,
                self.config.vit_dim,
            );

            //TODO : investigate if the qkv weights and proj weights use ReLU???

            for (0..num_patches) |patch| {
                accumulate(self.state.v_proj[patch * self.config.vit_dim .. (patch + 1) * self.config.vit_dim], self.weights.v_out_proj_b[l * self.config.vit_dim .. (l + 1) * self.config.vit_dim]);
            }

            accumulate(self.state.patch_emb, self.state.v_proj);

            // reusing v_x now and saving patch embed as a residual carry
            @memcpy(self.state.v_x, self.state.patch_emb);

            // second layernorm
            for (0..num_patches) |patch| {
                try layer_norm(
                    self.state.v_x[patch * self.config.vit_dim .. (patch + 1) * self.config.vit_dim],
                    self.weights.v_norm2_w[l * self.config.vit_dim .. (l + 1) * self.config.vit_dim],
                    self.weights.v_norm2_b[l * self.config.vit_dim .. (l + 1) * self.config.vit_dim],
                    1e-5,
                );
            }

            // pass the normalized v_x through the first MLP, upcasting it, and storing it in a buffer
            // v_x(num_patches, vit_dim) @ fc1 (vit_dim, hidden_features) = v_xb(num_patches, hidden_features)

            try matmul(
                self.allocator,
                self.state.v_x,
                self.weights.v_fc1_w[l * self.config.vit_dim * self.config.hidden_features .. (l + 1) * self.config.vit_dim * self.config.hidden_features],
                self.state.v_xb,
                num_patches,
                self.config.hidden_features,
                self.config.vit_dim,
            );

            // then we accumulate the fc1 bias into v_xb by iterating over num patches
            for (0..num_patches) |patch| {
                // iterate over num_patches (xb (1, hidden_features) = xb (1, hidden_features) + fc1_b (hidden_features)
                accumulate(
                    self.state.v_xb[patch * self.config.hidden_features .. (patch + 1) * self.config.hidden_features],
                    self.weights.v_fc1_b[l * self.config.hidden_features .. (l + 1) * self.config.hidden_features],
                );
            }

            // next we will apply GeLU to the fc1 logits (v_xb) to get the activations
            gelu(self.state.v_xb);

            // after this xb contains the activations from fc1!
            // now we will downcast it through fc2.

            // for this we will multiply v_xb with the fc2 weights and store it in v_xb2
            // v_xb(num_patches, hidden_features) @ fc2 (hidden_features, vit_dim) = v_xb2(num_patches, hidden_features)
            try matmul(
                self.allocator,
                self.state.v_xb,
                self.weights.v_fc2_w[l * self.config.hidden_features * self.config.vit_dim .. (l + 1) * self.config.hidden_features * self.config.vit_dim],
                self.state.v_xb2,
                num_patches,
                self.config.vit_dim,
                self.config.hidden_features,
            );
            // then we accumulate the fc2 bias into v_xb2 by iterating over num patches

            for (0..num_patches) |patch| {
                // iterate over num_patches (xb2 (1, vit_dim) = xb (1, vit_dim) + fc1_b (vit_dim)
                accumulate(
                    self.state.v_xb2[patch * self.config.vit_dim .. (patch + 1) * self.config.vit_dim],
                    self.weights.v_fc1_b[l * self.config.vit_dim .. (l + 1) * self.config.vit_dim],
                );
            }

            // now we finally can merge the mlp output into the residual we've been saving all this long
            accumulate(self.state.patch_emb, self.state.v_xb2);
        }
        // now for the final layernorm..

        for (0..num_patches) |patch| {
            try layer_norm(
                self.state.patch_emb[patch * self.config.vit_dim .. (patch + 1) * self.config.vit_dim],
                self.weights.v_norm_out_w,
                self.weights.v_norm_out_b,
                1e-5,
            );
        }

        for (0..num_patches) |patch| {
            // 0 to 1152
            // 1152 to 2304
            @memcpy(self.state.final_emb[patch * self.config.vit_dim .. (patch + 1) * self.config.vit_dim], self.state.patch_emb[patch * self.config.vit_dim .. (patch + 1) * self.config.vit_dim]);
            @memcpy(self.state.final_emb[(patch + 1) * self.config.vit_dim .. (patch + 2) * self.config.vit_dim], self.state.patch_emb[patch * self.config.vit_dim .. (patch + 1) * self.config.vit_dim]);
        }

        // now we will pass the final embed through the projection layers:
        // first we will upcast to (num patches, hidden_dim)
        // final_emb (num_patches , vit_dim * 2) @ v_proj_fc1_w (vit_dim * 2, hidden_dim) = v_xb3(num_patches, hidden_dim)
        try matmul(
            self.allocator,
            self.state.final_emb,
            self.weights.v_proj_fc1_w,
            self.state.v_xb3,
            num_patches,
            self.config.hidden_dim,
            self.config.vit_dim * 2,
        );

        // next up, we apply the bias on v_xb3
        for (0..num_patches) |patch| {
            accumulate(
                self.state.v_xb3[patch * self.config.hidden_dim .. (patch + 1) * self.config.hidden_dim],
                self.weights.v_proj_fc1_b,
            );
        }

        // we will then apply GeLU on the logits from the first projection layer:
        gelu(self.state.v_xb3);

        // after this we will pass the activated v_xb3:
        // v_xb3(num_patches, hidden_dim) @ v_proj_fc2_w (hidden_dim, dim) = v_xb3(num_patches, dim)
        try matmul(
            self.allocator,
            self.state.v_xb3,
            self.weights.v_proj_fc2_w,
            self.state.projection,
            num_patches,
            self.config.dim,
            self.config.hidden_dim,
        );

        // then we add the final projection bias

        for (0..num_patches) |patch| {
            accumulate(self.state.projection[patch * self.config.dim .. (patch + 1) * self.config.dim], self.weights.v_proj_fc2_b);
        }
    }
};
