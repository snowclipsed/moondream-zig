const std = @import("std");
const Tensor = @import("../core/tensor.zig").Tensor;
const Allocator = std.mem.Allocator;
const Thread = std.Thread;
const sgemminplace = @import("../core/sgemm_inplace.zig").matmul;
const softmax = @import("ops.zig").softmax;
const thread_pool_mod = @import("../core/thread_pool.zig");

// Model constants
const MODEL_HEAD_DIM: usize = 64;
const MODEL_N_HEADS: usize = 32;
const SIMD_WIDTH: usize = 8;
const HEAD_DIM_CHUNKS: usize = MODEL_HEAD_DIM / SIMD_WIDTH;
const MAX_KV_LEN: usize = 2048; // Maximum sequence length
const MAX_THREADS: usize = 32; // Maximum number of threads (same as MODEL_N_HEADS)

const SimdF16x8 = @Vector(8, f16);
const SimdF32x8 = @Vector(8, f32);

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

pub fn multiMaskedSDPA(
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

    // Transpose the key matrix (this could be parallelized but is typically not the bottleneck)
    for (0..n_heads) |h| {
        for (0..kv_len) |k| {
            for (0..head_dim) |d| {
                const src_idx = h * kv_len * head_dim + k * head_dim + d;
                const dst_idx = h * head_dim * kv_len + d * kv_len + k;
                key_transpose.data[dst_idx] = key_f32.data[src_idx];
            }
        }
    }

    // Use a single level of parallelism by directly processing each head in the thread pool
    const HeadProcessTask = struct {
        head_idx: usize,
        query_f32: Tensor(f32),
        key_transpose: Tensor(f32),
        value_f32: Tensor(f32),
        mask: Tensor(bool),
        out: *Tensor(f16),
        scale: f32,
        q_len: usize,
        kv_len: usize,
        head_dim: usize,
        workspace: MaskedThreadWorkspace,

        pub fn process(self: *@This()) void {
            const h = self.head_idx;

            // Process a single head completely
            const query_slice = h * self.q_len * self.head_dim;
            @memcpy(self.workspace.query_head.data, self.query_f32.data[query_slice..][0 .. self.q_len * self.head_dim]);

            const key_slice = h * self.head_dim * self.kv_len;
            @memcpy(self.workspace.key_head.data, self.key_transpose.data[key_slice..][0 .. self.head_dim * self.kv_len]);

            const value_slice = h * self.kv_len * self.head_dim;
            @memcpy(self.workspace.value_head.data, self.value_f32.data[value_slice..][0 .. self.kv_len * self.head_dim]);

            // Use a fixed smaller thread count for matrix multiplications to avoid oversubscription
            // This is the key optimization - limit nested parallelism
            const matmul_threads = 1; // No nested parallelism, rely on task-level parallelism

            // Q * K^T
            sgemminplace(f32, self.workspace.query_head, self.workspace.key_head, &self.workspace.attn_weights, self.workspace.allocator, matmul_threads) catch return;

            // Apply scaling
            for (self.workspace.attn_weights.data) |*w| {
                w.* *= self.scale;
            }

            // Apply mask
            for (0..self.q_len) |i| {
                for (0..self.kv_len) |j| {
                    const mask_idx = i * self.mask.shape[2] + j;
                    const weights_idx = i * self.kv_len + j;
                    if (!self.mask.data[mask_idx]) {
                        self.workspace.attn_weights.data[weights_idx] = -std.math.inf(f32);
                    }
                }
            }

            // Softmax
            softmax(&self.workspace.attn_weights, 1, self.workspace.allocator) catch return;

            // weights * V
            sgemminplace(f32, self.workspace.attn_weights, self.workspace.value_head, &self.workspace.out_head, self.workspace.allocator, matmul_threads) catch return;

            // Copy results back to output tensor
            const out_slice = h * self.q_len * self.head_dim;
            for (0..self.q_len * self.head_dim) |i| {
                self.out.data[out_slice + i] = @floatCast(self.workspace.out_head.data[i]);
            }
        }
    };

    var tasks = try allocator.alloc(HeadProcessTask, n_heads);
    defer allocator.free(tasks);

    // Initialize a workspace for each head
    for (0..n_heads) |h| {
        tasks[h] = HeadProcessTask{
            .head_idx = h,
            .query_f32 = query_f32,
            .key_transpose = key_transpose,
            .value_f32 = value_f32,
            .mask = mask,
            .out = &out,
            .scale = scale,
            .q_len = q_len,
            .kv_len = kv_len,
            .head_dim = head_dim,
            .workspace = try MaskedThreadWorkspace.init(allocator, q_len, kv_len, head_dim),
        };
    }

    // Submit tasks to the thread pool
    for (0..n_heads) |h| {
        try thread_pool_mod.submitTask(HeadProcessTask, HeadProcessTask.process, &tasks[h]);
    }

    // Wait for all tasks to complete
    try thread_pool_mod.waitForAll();

    // Clean up workspaces
    for (tasks) |*task| {
        task.workspace.deinit();
    }

    return out;
}

/// Fallback implementation using direct threading when global thread pool is unavailable
fn fallbackMultiMaskedSDPA(
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

// Keep the static workspace structure
const SingleHeadWorkspace = struct {
    scores: [MAX_KV_LEN]f32 align(32),
    out_head: [MODEL_HEAD_DIM]f32 align(32),
};

// Static global workspaces
var global_workspaces: [MAX_THREADS]SingleHeadWorkspace = undefined;

// Task structure for global thread pool
const HeadTask = struct {
    start_head: usize,
    end_head: usize,
    query: Tensor(f16),
    key: Tensor(f16),
    value: Tensor(f16),
    mask: Tensor(bool),
    out: *Tensor(f16),
    kv_len: usize,
    workspace_idx: usize,

    pub fn process(self: *@This()) void {
        const workspace = &global_workspaces[self.workspace_idx];

        // Get the optimized scoring function with the workspace
        processHeadRange(self.start_head, self.end_head, self.query, self.key, self.value, self.mask, self.out, self.kv_len, workspace);
    }
};

// Extracted processing function (same logic as original)
fn processHeadRange(
    start_head: usize,
    end_head: usize,
    query: Tensor(f16),
    key: Tensor(f16),
    value: Tensor(f16),
    mask: Tensor(bool),
    out: *Tensor(f16),
    kv_len: usize,
    workspace: *SingleHeadWorkspace,
) void {
    @setFloatMode(.optimized);

    const scores = &workspace.scores;
    const out_head = &workspace.out_head;
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(MODEL_HEAD_DIM)));

    // Process each head in this thread's range
    for (start_head..end_head) |h| {
        const query_offset = h * MODEL_HEAD_DIM;
        const key_offset = h * kv_len * MODEL_HEAD_DIM;
        const value_offset = key_offset;

        // Compute attention scores with SIMD
        for (0..kv_len) |j| {
            var sum: f32 = 0;
            const k_base = key_offset + j * MODEL_HEAD_DIM;

            // Process in chunks of 8 elements
            inline for (0..HEAD_DIM_CHUNKS) |chunk| {
                const q_ptr = @as(*const SimdF16x8, @ptrCast(@alignCast(&query.data[query_offset + chunk * SIMD_WIDTH])));
                const k_ptr = @as(*const SimdF16x8, @ptrCast(@alignCast(&key.data[k_base + chunk * SIMD_WIDTH])));

                const q_f32: SimdF32x8 = @as(SimdF32x8, @floatCast(q_ptr.*));
                const k_f32: SimdF32x8 = @as(SimdF32x8, @floatCast(k_ptr.*));

                const mul = q_f32 * k_f32;
                sum += @reduce(.Add, mul);
            }

            scores[j] = sum * scale;
            if (!mask.data[j]) {
                scores[j] = -std.math.inf(f32);
            }
        }

        // Optimized softmax with branchless operations
        var max_val: f32 = scores[0];
        for (scores[1..kv_len]) |s| {
            max_val = @max(max_val, s);
        }

        var sum: f32 = 0;
        for (scores[0..kv_len]) |*s| {
            s.* = @exp(s.* - max_val);
            sum += s.*;
        }

        const inv_sum = 1.0 / sum;
        for (scores[0..kv_len]) |*s| {
            s.* *= inv_sum;
        }

        // Clear output accumulator
        @memset(out_head, 0);

        // Weighted sum with SIMD
        for (0..kv_len) |j| {
            const weight = scores[j];
            const v_base = value_offset + j * MODEL_HEAD_DIM;

            inline for (0..HEAD_DIM_CHUNKS) |chunk| {
                const v_ptr = @as(*const SimdF16x8, @ptrCast(@alignCast(&value.data[v_base + chunk * SIMD_WIDTH])));
                const out_ptr = @as(*SimdF32x8, @ptrCast(@alignCast(&out_head[chunk * SIMD_WIDTH])));

                const v_f32: SimdF32x8 = @as(SimdF32x8, @floatCast(v_ptr.*));
                out_ptr.* += v_f32 * @as(SimdF32x8, @splat(weight));
            }
        }

        // Copy to output with casting
        inline for (0..HEAD_DIM_CHUNKS) |chunk| {
            const src = @as(*align(32) const SimdF32x8, @alignCast(@ptrCast(&out_head[chunk * SIMD_WIDTH])));
            const dst_ptr = out.data.ptr + h * MODEL_HEAD_DIM + chunk * SIMD_WIDTH;
            const dst_vec: SimdF16x8 = @floatCast(src.*);
            @as(*SimdF16x8, @alignCast(@ptrCast(dst_ptr))).* = dst_vec;
        }
    }
}

/// Drop-in replacement for singleMaskedSDPA that uses the global thread pool
pub fn singleMaskedSDPA(
    query: Tensor(f16),
    key: Tensor(f16),
    value: Tensor(f16),
    mask: Tensor(bool),
    allocator: Allocator,
) !Tensor(f16) {
    @setFloatMode(.optimized);

    const kv_len = key.shape[1];
    if (kv_len > MAX_KV_LEN) return error.SequenceTooLong;

    // Initialize output tensor (only allocation we need)
    var out = try Tensor(f16).init(allocator, &[_]usize{ MODEL_N_HEADS, 1, MODEL_HEAD_DIM });
    errdefer out.deinit();

    // Get the global thread pool
    const pool = thread_pool_mod.getInstance() catch |err| {
        // Fall back to local implementation if global pool not available
        std.log.warn("Global thread pool not initialized, falling back to direct implementation: {}", .{err});
        return fallbackSingleMaskedSDPA(query, key, value, mask, allocator);
    };

    // Get optimal task count based on CPU count and model parameters
    const cpu_count = try Thread.getCpuCount();
    const min_heads_per_task = 4; // Ensure each task has enough work to justify overhead
    const max_tasks = @min(MODEL_N_HEADS / min_heads_per_task, cpu_count);

    // Adjust task count based on thread pool load
    const active_tasks = pool.active_tasks.load(.acquire);
    const effective_tasks = if (active_tasks > 1)
        @max(1, @min(max_tasks, cpu_count / active_tasks))
    else
        max_tasks;

    const task_count = @min(MODEL_N_HEADS, effective_tasks);

    // Prepare tasks with evenly distributed heads
    var tasks = try allocator.alloc(HeadTask, task_count);
    defer allocator.free(tasks);

    // Distribute heads among tasks (similar to original)
    const heads_per_task = MODEL_N_HEADS / task_count;
    const remaining_heads = MODEL_N_HEADS % task_count;
    var current_head: usize = 0;

    for (0..task_count) |t| {
        const start_head = current_head;
        const extra_head: usize = if (t < remaining_heads) 1 else 0;
        current_head += heads_per_task + extra_head;

        tasks[t] = HeadTask{
            .start_head = start_head,
            .end_head = current_head,
            .query = query,
            .key = key,
            .value = value,
            .mask = mask,
            .out = &out,
            .kv_len = kv_len,
            .workspace_idx = t,
        };

        // Submit to global thread pool
        try thread_pool_mod.submitTask(HeadTask, HeadTask.process, &tasks[t]);
    }

    // Wait for all tasks to complete
    try thread_pool_mod.waitForAll();

    return out;
}

// Fallback implementation using direct execution
fn fallbackSingleMaskedSDPA(
    query: Tensor(f16),
    key: Tensor(f16),
    value: Tensor(f16),
    mask: Tensor(bool),
    allocator: Allocator,
) !Tensor(f16) {
    @setFloatMode(.optimized);

    const kv_len = key.shape[1];

    // Initialize output tensor
    var out = try Tensor(f16).init(allocator, &[_]usize{ MODEL_N_HEADS, 1, MODEL_HEAD_DIM });
    errdefer out.deinit();

    // Process all heads directly in the current thread
    const workspace = &global_workspaces[0];
    processHeadRange(0, MODEL_N_HEADS, query, key, value, mask, &out, kv_len, workspace);

    return out;
}
