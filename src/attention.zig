const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
const Allocator = std.mem.Allocator;
const Thread = std.Thread;
const sgemminplace = @import("sgemminplace.zig").matmul;
const softmax = @import("ops.zig").softmax;

// ----- Unmasked Attention ----- //

const ThreadWorkspace = struct {
    allocator: Allocator,
    query_head: Tensor(f32),
    key_head: Tensor(f32),
    value_head: Tensor(f32),
    attn_weights: Tensor(f32),
    out_head: Tensor(f32),

    pub fn init(allocator: Allocator, q_len: usize, kv_len: usize, head_dim: usize) !ThreadWorkspace {
        return ThreadWorkspace{
            .allocator = allocator,
            .query_head = try Tensor(f32).init(allocator, &[_]usize{ q_len, head_dim }),
            .key_head = try Tensor(f32).init(allocator, &[_]usize{ head_dim, kv_len }),
            .value_head = try Tensor(f32).init(allocator, &[_]usize{ kv_len, head_dim }),
            .attn_weights = try Tensor(f32).init(allocator, &[_]usize{ q_len, kv_len }),
            .out_head = try Tensor(f32).init(allocator, &[_]usize{ q_len, head_dim }),
        };
    }

    pub fn deinit(self: *ThreadWorkspace) void {
        self.query_head.deinit();
        self.key_head.deinit();
        self.value_head.deinit();
        self.attn_weights.deinit();
        self.out_head.deinit();
    }
};

// Persistent thread pool context
const ThreadPoolContext = struct {
    workspaces: []ThreadWorkspace,
    allocator: Allocator,

    pub fn init(allocator: Allocator, thread_count: usize, q_len: usize, kv_len: usize, head_dim: usize) !ThreadPoolContext {
        var workspaces = try allocator.alloc(ThreadWorkspace, thread_count);
        errdefer allocator.free(workspaces);

        for (0..thread_count) |i| {
            workspaces[i] = try ThreadWorkspace.init(allocator, q_len, kv_len, head_dim);
        }

        return ThreadPoolContext{
            .workspaces = workspaces,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ThreadPoolContext) void {
        for (self.workspaces) |*workspace| {
            workspace.deinit();
        }
        self.allocator.free(self.workspaces);
    }
};

// Updated thread context with workspace
const AttnThreadContext = struct {
    start_head: usize,
    end_head: usize,
    query: Tensor(f32),
    key: Tensor(f32),
    value: Tensor(f32),
    out: *Tensor(f16),
    scale: f32,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    workspace: *ThreadWorkspace,
};

pub fn multiMasklessScaledDotProductAttention(
    n_heads: comptime_int,
    head_dim: comptime_int,
    query: Tensor(f16),
    key: Tensor(f16),
    value: Tensor(f16),
    allocator: Allocator,
) !Tensor(f16) {
    const q_len = query.shape[1];
    const kv_len = key.shape[1];

    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    var out = try Tensor(f16).init(allocator, &[_]usize{ n_heads, q_len, head_dim });
    errdefer out.deinit();

    var query_f32 = try query.castWithSimd(f32);
    defer query_f32.deinit();
    var key_f32 = try key.castWithSimd(f32);
    defer key_f32.deinit();
    var value_f32 = try value.castWithSimd(f32);
    defer value_f32.deinit();

    var key_transpose = try Tensor(f32).init(allocator, &[_]usize{ n_heads, head_dim, kv_len });
    defer key_transpose.deinit();

    for (0..n_heads) |h| {
        for (0..kv_len) |k| {
            for (0..head_dim) |d| {
                const src_idx = h * kv_len * head_dim + k * head_dim + d;
                const dst_idx = h * head_dim * kv_len + d * kv_len + k;
                key_transpose.data[dst_idx] = key_f32.data[src_idx];
            }
        }
    }

    const thread_count: usize = @intCast(@min(n_heads / 2, try Thread.getCpuCount() / 2));

    var thread_pool = try ThreadPoolContext.init(allocator, thread_count, q_len, kv_len, head_dim);
    defer thread_pool.deinit();

    const heads_per_thread = n_heads / thread_count;
    const remaining_heads = n_heads % thread_count;
    var threads = try allocator.alloc(Thread, thread_count);
    defer allocator.free(threads);

    var current_head: usize = 0;
    var thread_contexts = try allocator.alloc(AttnThreadContext, thread_count);
    defer allocator.free(thread_contexts);

    const worker = struct {
        fn process(ctx: AttnThreadContext) !void {
            const workspace = ctx.workspace;
            const cpu_count = try Thread.getCpuCount();
            const n_threads: usize = @intCast(@min(8, try Thread.getCpuCount() / 2));
            const matmul_threads = @as(usize, @max(cpu_count / 2, cpu_count - n_threads));
            for (ctx.start_head..ctx.end_head) |h| {
                const query_slice = h * ctx.q_len * ctx.head_dim;
                const key_slice = h * ctx.head_dim * ctx.kv_len;
                const value_slice = h * ctx.kv_len * ctx.head_dim;

                @memcpy(workspace.query_head.data, ctx.query.data[query_slice..][0 .. ctx.q_len * ctx.head_dim]);
                @memcpy(workspace.key_head.data, ctx.key.data[key_slice..][0 .. ctx.head_dim * ctx.kv_len]);
                @memcpy(workspace.value_head.data, ctx.value.data[value_slice..][0 .. ctx.kv_len * ctx.head_dim]);

                try sgemminplace(f32, workspace.query_head, workspace.key_head, &workspace.attn_weights, workspace.allocator, matmul_threads);

                for (workspace.attn_weights.data) |*w| {
                    w.* *= ctx.scale;
                }

                try softmax(&workspace.attn_weights, 1, workspace.allocator);
                try sgemminplace(f32, workspace.attn_weights, workspace.value_head, &workspace.out_head, workspace.allocator, matmul_threads);

                const out_slice = h * ctx.q_len * ctx.head_dim;
                for (0..ctx.q_len * ctx.head_dim) |i| {
                    ctx.out.data[out_slice + i] = @floatCast(workspace.out_head.data[i]);
                }
            }
        }
    }.process;

    for (0..thread_count) |t| {
        const start_head = current_head;
        const extra_head: usize = if (t < remaining_heads) 1 else 0;
        current_head += heads_per_thread + extra_head;

        thread_contexts[t] = AttnThreadContext{
            .start_head = start_head,
            .end_head = current_head,
            .query = query_f32,
            .key = key_transpose,
            .value = value_f32,
            .out = &out,
            .scale = scale,
            .q_len = q_len,
            .kv_len = kv_len,
            .head_dim = head_dim,
            .workspace = &thread_pool.workspaces[t],
        };

        threads[t] = try Thread.spawn(.{}, worker, .{thread_contexts[t]});
    }

    for (threads) |thread| {
        thread.join();
    }

    return out;
}
// Helper function to copy head data
fn copyHeadData(comptime T: type, dst: *Tensor(T), src: Tensor(T), head_idx: usize) !void {
    const slice_size = src.shape[1] * src.shape[2];
    const start_idx = head_idx * slice_size;
    @memcpy(dst.data[0..slice_size], src.data[start_idx..][0..slice_size]);
}

// ----- Masked Attention ----- //

const SimdF16x8 = @Vector(8, f16);
const SimdF32x8 = @Vector(8, f32);

// Model constants
const MODEL_HEAD_DIM: usize = 64;
const MODEL_N_HEADS: usize = 32;
const SIMD_WIDTH: usize = 8;
const HEAD_DIM_CHUNKS: usize = MODEL_HEAD_DIM / SIMD_WIDTH;
const MAX_KV_LEN: usize = 2048; // Maximum sequence length
const MAX_THREADS: usize = 32; // Maximum number of threads (same as MODEL_N_HEADS)

// Create attention mask for proper causal attention alignment
pub fn createAttentionMask(allocator: Allocator, pos: usize, seq_len: usize) !Tensor(bool) {
    // First create the base mask of shape [seq_len, pos + seq_len]
    var mask = try Tensor(bool).init(allocator, &[_]usize{ seq_len, pos + seq_len });
    errdefer mask.deinit();

    // Fill the first part (before pos) with true
    for (0..seq_len) |i| {
        for (0..pos) |j| {
            const idx = i * (pos + seq_len) + j;
            mask.data[idx] = true;
        }
    }

    // Fill the second part (pos onwards) with lower triangular matrix
    for (0..seq_len) |i| {
        for (0..seq_len) |j| {
            const idx = i * (pos + seq_len) + (j + pos);
            mask.data[idx] = j <= i; // Lower triangular
        }
    }

    // Reshape to add head dimension [1, seq_len, pos + seq_len]
    try mask.reshape(&[_]usize{ 1, seq_len, pos + seq_len });

    return mask;
}

/// Thread context for masked attention
const MaskedAttnThreadContext = struct {
    start_head: usize,
    end_head: usize,
    query: Tensor(f32),
    key: Tensor(f32),
    value: Tensor(f32),
    mask: Tensor(bool),
    out: *Tensor(f16),
    scale: f32,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    workspace: *MaskedThreadWorkspace,
};

/// Thread pool context for masked attention
const MaskedThreadPoolContext = struct {
    workspaces: []MaskedThreadWorkspace,
    allocator: Allocator,

    pub fn init(allocator: Allocator, thread_count: usize, q_len: usize, kv_len: usize, head_dim: usize) !MaskedThreadPoolContext {
        var workspaces = try allocator.alloc(MaskedThreadWorkspace, thread_count);
        errdefer allocator.free(workspaces);

        for (0..thread_count) |i| {
            workspaces[i] = try MaskedThreadWorkspace.init(allocator, q_len, kv_len, head_dim);
        }

        return MaskedThreadPoolContext{
            .workspaces = workspaces,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *MaskedThreadPoolContext) void {
        for (self.workspaces) |*workspace| {
            workspace.deinit();
        }
        self.allocator.free(self.workspaces);
    }
};

/// Thread-local workspace for masked attention
const MaskedThreadWorkspace = struct {
    query_head: Tensor(f32),
    key_head: Tensor(f32),
    value_head: Tensor(f32),
    attn_weights: Tensor(f32),
    out_head: Tensor(f32),
    allocator: Allocator,

    pub fn init(allocator: Allocator, q_len: usize, kv_len: usize, head_dim: usize) !MaskedThreadWorkspace {
        return MaskedThreadWorkspace{
            .query_head = try Tensor(f32).init(allocator, &[_]usize{ q_len, head_dim }),
            .key_head = try Tensor(f32).init(allocator, &[_]usize{ head_dim, kv_len }),
            .value_head = try Tensor(f32).init(allocator, &[_]usize{ kv_len, head_dim }),
            .attn_weights = try Tensor(f32).init(allocator, &[_]usize{ q_len, kv_len }),
            .out_head = try Tensor(f32).init(allocator, &[_]usize{ q_len, head_dim }),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *MaskedThreadWorkspace) void {
        self.query_head.deinit();
        self.key_head.deinit();
        self.value_head.deinit();
        self.attn_weights.deinit();
        self.out_head.deinit();
    }
};

pub fn multiMaskedScaledDotProductAttention(
    query: Tensor(f16),
    key: Tensor(f16),
    value: Tensor(f16),
    mask: Tensor(bool),
    n_heads: usize,
    head_dim: usize,
    allocator: Allocator,
) !Tensor(f16) {
    const q_len = query.shape[1];
    const kv_len = key.shape[1];

    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    var out = try Tensor(f16).init(allocator, &[_]usize{ n_heads, q_len, head_dim });
    errdefer out.deinit();

    var query_f32 = try query.castWithSimd(f32);
    defer query_f32.deinit();
    var value_f32 = try value.castWithSimd(f32);
    defer value_f32.deinit();

    var key_f32 = try key.castWithSimd(f32);
    defer key_f32.deinit();
    var key_transpose = try Tensor(f32).init(allocator, &[_]usize{ n_heads, head_dim, kv_len });
    defer key_transpose.deinit();

    for (0..n_heads) |h| {
        for (0..kv_len) |k| {
            for (0..head_dim) |d| {
                const src_idx = h * kv_len * head_dim + k * head_dim + d;
                const dst_idx = h * head_dim * kv_len + d * kv_len + k;
                key_transpose.data[dst_idx] = key_f32.data[src_idx];
            }
        }
    }

    const thread_count: usize = @intCast(@min(n_heads / 2, try Thread.getCpuCount() / 2));

    var thread_pool = try MaskedThreadPoolContext.init(allocator, thread_count, q_len, kv_len, head_dim);
    defer thread_pool.deinit();

    var threads = try allocator.alloc(Thread, thread_count);
    defer allocator.free(threads);

    const heads_per_thread = n_heads / thread_count;
    const remaining_heads = n_heads % thread_count;

    var current_head: usize = 0;
    var thread_contexts = try allocator.alloc(MaskedAttnThreadContext, thread_count);
    defer allocator.free(thread_contexts);

    const worker = struct {
        fn process(ctx: MaskedAttnThreadContext) !void {
            const workspace = ctx.workspace;
            const cpu_count = try Thread.getCpuCount();
            const n_threads: usize = @intCast(@min(16, try Thread.getCpuCount() / 2));
            const matmul_threads = @as(usize, @max(cpu_count / 2, cpu_count - n_threads));

            for (ctx.start_head..ctx.end_head) |h| {
                const query_slice = h * ctx.q_len * ctx.head_dim;
                @memcpy(workspace.query_head.data, ctx.query.data[query_slice..][0 .. ctx.q_len * ctx.head_dim]);

                const key_slice = h * ctx.head_dim * ctx.kv_len;
                @memcpy(workspace.key_head.data, ctx.key.data[key_slice..][0 .. ctx.head_dim * ctx.kv_len]);

                const value_slice = h * ctx.kv_len * ctx.head_dim;
                @memcpy(workspace.value_head.data, ctx.value.data[value_slice..][0 .. ctx.kv_len * ctx.head_dim]);

                try sgemminplace(f32, workspace.query_head, workspace.key_head, &workspace.attn_weights, ctx.workspace.allocator, matmul_threads);

                for (workspace.attn_weights.data) |*w| {
                    w.* *= ctx.scale;
                }

                for (0..ctx.q_len) |i| {
                    for (0..ctx.kv_len) |j| {
                        const mask_idx = i * ctx.mask.shape[2] + j;
                        const weights_idx = i * ctx.kv_len + j;
                        if (!ctx.mask.data[mask_idx]) {
                            workspace.attn_weights.data[weights_idx] = -std.math.inf(f32);
                        }
                    }
                }

                try softmax(&workspace.attn_weights, 1, ctx.workspace.allocator);

                try sgemminplace(f32, workspace.attn_weights, workspace.value_head, &workspace.out_head, ctx.workspace.allocator, matmul_threads);

                const out_slice = h * ctx.q_len * ctx.head_dim;
                for (0..ctx.q_len * ctx.head_dim) |i| {
                    ctx.out.data[out_slice + i] = @floatCast(workspace.out_head.data[i]);
                }
            }
        }
    }.process;

    for (0..thread_count) |t| {
        const start_head = current_head;
        const extra_head: usize = if (t < remaining_heads) 1 else 0;
        current_head += heads_per_thread + extra_head;

        thread_contexts[t] = MaskedAttnThreadContext{
            .start_head = start_head,
            .end_head = current_head,
            .query = query_f32,
            .key = key_transpose,
            .value = value_f32,
            .mask = mask,
            .out = &out,
            .scale = scale,
            .q_len = q_len,
            .kv_len = kv_len,
            .head_dim = head_dim,
            .workspace = &thread_pool.workspaces[t],
        };

        threads[t] = try Thread.spawn(.{}, worker, .{thread_contexts[t]});
    }

    for (threads) |thread| {
        thread.join();
    }

    return out;
}

// Static workspace for each thread
const SingleHeadWorkspace = struct {
    scores: [MAX_KV_LEN]f32 align(32),
    out_head: [MODEL_HEAD_DIM]f32 align(32),
};

// Static global workspaces
var global_workspaces: [MAX_THREADS]SingleHeadWorkspace = undefined;

const HeadContext = struct {
    start_head: usize,
    end_head: usize,
    query: Tensor(f16),
    key: Tensor(f16),
    value: Tensor(f16),
    mask: Tensor(bool),
    out: *Tensor(f16),
    kv_len: usize,
    workspace: *SingleHeadWorkspace,
};

// Worker function
fn processHeads(ctx: HeadContext) void {
    const scores = &ctx.workspace.scores;
    const out_head = &ctx.workspace.out_head;
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(MODEL_HEAD_DIM)));

    // Process each head in this thread's range
    for (ctx.start_head..ctx.end_head) |h| {
        const query_offset = h * MODEL_HEAD_DIM;
        const key_offset = h * ctx.kv_len * MODEL_HEAD_DIM;
        const value_offset = key_offset;

        // Compute attention scores with SIMD
        for (0..ctx.kv_len) |j| {
            var sum: f32 = 0;
            const k_base = key_offset + j * MODEL_HEAD_DIM;

            // Process in chunks of 8 elements
            inline for (0..HEAD_DIM_CHUNKS) |chunk| {
                const q_ptr = @as(*const SimdF16x8, @ptrCast(@alignCast(&ctx.query.data[query_offset + chunk * SIMD_WIDTH])));
                const k_ptr = @as(*const SimdF16x8, @ptrCast(@alignCast(&ctx.key.data[k_base + chunk * SIMD_WIDTH])));

                const q_f32: SimdF32x8 = @as(SimdF32x8, @floatCast(q_ptr.*));
                const k_f32: SimdF32x8 = @as(SimdF32x8, @floatCast(k_ptr.*));

                const mul = q_f32 * k_f32;
                sum += @reduce(.Add, mul);
            }

            scores[j] = sum * scale;
            if (!ctx.mask.data[j]) {
                scores[j] = -std.math.inf(f32);
            }
        }

        // Optimized softmax with branchless operations
        var max_val: f32 = scores[0];
        for (scores[1..ctx.kv_len]) |s| {
            max_val = @max(max_val, s);
        }

        var sum: f32 = 0;
        for (scores[0..ctx.kv_len]) |*s| {
            s.* = @exp(s.* - max_val);
            sum += s.*;
        }

        const inv_sum = 1.0 / sum;
        for (scores[0..ctx.kv_len]) |*s| {
            s.* *= inv_sum;
        }

        // Clear output accumulator
        @memset(out_head, 0);

        // Weighted sum with SIMD
        for (0..ctx.kv_len) |j| {
            const weight = scores[j];
            const v_base = value_offset + j * MODEL_HEAD_DIM;

            inline for (0..HEAD_DIM_CHUNKS) |chunk| {
                const v_ptr = @as(*const SimdF16x8, @ptrCast(@alignCast(&ctx.value.data[v_base + chunk * SIMD_WIDTH])));
                const out_ptr = @as(*SimdF32x8, @ptrCast(@alignCast(&out_head[chunk * SIMD_WIDTH])));

                const v_f32: SimdF32x8 = @as(SimdF32x8, @floatCast(v_ptr.*));
                out_ptr.* += v_f32 * @as(SimdF32x8, @splat(weight));
            }
        }

        // Copy to output with casting
        inline for (0..HEAD_DIM_CHUNKS) |chunk| {
            const src = @as(*align(32) const SimdF32x8, @alignCast(@ptrCast(&out_head[chunk * SIMD_WIDTH])));
            const dst_ptr = ctx.out.data.ptr + h * MODEL_HEAD_DIM + chunk * SIMD_WIDTH;
            const dst_vec: SimdF16x8 = @floatCast(src.*);
            @as(*SimdF16x8, @alignCast(@ptrCast(dst_ptr))).* = dst_vec;
        }
    }
}

pub fn singleMaskedScaledDotProductAttention(
    query: Tensor(f16),
    key: Tensor(f16),
    value: Tensor(f16),
    mask: Tensor(bool),
    allocator: Allocator,
) !Tensor(f16) {
    const kv_len = key.shape[1];
    if (kv_len > MAX_KV_LEN) return error.SequenceTooLong;

    // Initialize output tensor (only allocation we need)
    var out = try Tensor(f16).init(allocator, &[_]usize{ MODEL_N_HEADS, 1, MODEL_HEAD_DIM });
    errdefer out.deinit();

    // Set up threads
    const thread_count = @min(MODEL_N_HEADS, try Thread.getCpuCount());
    var threads: [MAX_THREADS]Thread = undefined;
    var contexts: [MAX_THREADS]HeadContext = undefined;

    // Distribute work
    const heads_per_thread = MODEL_N_HEADS / thread_count;
    const remaining_heads = MODEL_N_HEADS % thread_count;
    var current_head: usize = 0;

    // Launch threads
    for (0..thread_count) |t| {
        const start_head = current_head;
        const extra_head: usize = if (t < remaining_heads) 1 else 0;
        current_head += heads_per_thread + extra_head;

        contexts[t] = HeadContext{
            .start_head = start_head,
            .end_head = current_head,
            .query = query,
            .key = key,
            .value = value,
            .mask = mask,
            .out = &out,
            .kv_len = kv_len,
            .workspace = &global_workspaces[t],
        };

        threads[t] = try Thread.spawn(.{}, processHeads, .{contexts[t]});
    }

    // Wait for completion
    for (threads[0..thread_count]) |thread| {
        thread.join();
    }

    return out;
}
