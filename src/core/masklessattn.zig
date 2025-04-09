const std = @import("std");
const Tensor = @import("../core/tensor.zig").Tensor;
const Allocator = std.mem.Allocator;
const Thread = std.Thread;
const Mutex = Thread.Mutex;
const Condition = Thread.Condition;
const sgemminplace = @import("../core/sgemm_inplace.zig").matmul;
const softmax = @import("ops.zig").softmax;
const thread_pool = @import("../core/thread_pool.zig").ThreadPool;

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
/// Task context for parallel processing of attention heads
const MasklessHeadTaskContext = struct {
    head_id: usize,
    query: Tensor(f32),
    key: Tensor(f32),
    value: Tensor(f32),
    out: *Tensor(f16),
    scale: f32,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    workspace: *ThreadWorkspace,

    /// Process a single attention head
    pub fn process(self: *MasklessHeadTaskContext) void {
        const h = self.head_id;
        const workspace = self.workspace;

        // Extract head data
        const query_slice = h * self.q_len * self.head_dim;
        const key_slice = h * self.head_dim * self.kv_len;
        const value_slice = h * self.kv_len * self.head_dim;

        @memcpy(workspace.query_head.data, self.query.data[query_slice..][0 .. self.q_len * self.head_dim]);
        @memcpy(workspace.key_head.data, self.key.data[key_slice..][0 .. self.head_dim * self.kv_len]);
        @memcpy(workspace.value_head.data, self.value.data[value_slice..][0 .. self.kv_len * self.head_dim]);

        // Compute attention scores (Q*K^T)
        // Use single-threaded SGEMM to avoid nested parallelism overhead
        sgemminplace(f32, workspace.query_head, workspace.key_head, &workspace.attn_weights, workspace.allocator, 1) catch return;

        // Apply scaling
        for (workspace.attn_weights.data) |*w| {
            w.* *= self.scale;
        }

        // Apply softmax
        softmax(&workspace.attn_weights, 1, workspace.allocator) catch return;

        // Compute output (Attention_weights * V)
        // Again use single-threaded SGEMM
        sgemminplace(f32, workspace.attn_weights, workspace.value_head, &workspace.out_head, workspace.allocator, 1) catch return;

        // Copy result to output tensor with f32->f16 conversion
        const out_slice = h * self.q_len * self.head_dim;
        for (0..self.q_len * self.head_dim) |i| {
            self.out.data[out_slice + i] = @floatCast(workspace.out_head.data[i]);
        }
    }
};

/// Performs multi-headed scaled dot-product attention (without masking) using the global thread pool.
pub fn multiMasklessSDPA(
    n_heads: comptime_int,
    head_dim: comptime_int,
    query: Tensor(f16),
    key: Tensor(f16),
    value: Tensor(f16),
    allocator: Allocator,
) !Tensor(f16) {
    // Check dimensions and initialize output tensor
    const q_len = query.shape[1];
    const kv_len = key.shape[1];
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    var out = try Tensor(f16).init(allocator, &[_]usize{ n_heads, q_len, head_dim });
    errdefer out.deinit();

    // Cast tensors to f32 for computation
    var query_f32 = try query.castWithSimd(f32);
    defer query_f32.deinit();
    var key_f32 = try key.castWithSimd(f32);
    defer key_f32.deinit();
    var value_f32 = try value.castWithSimd(f32);
    defer value_f32.deinit();

    // Transpose key matrices for more efficient memory access
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

    // Check if the thread pool is initialized (will return error if not)
    _ = try @import("../core/thread_pool.zig").getInstance();

    // Determine optimal number of workspaces based on CPU cores and head count
    const cpu_count = try std.Thread.getCpuCount();
    const max_workers = @min(n_heads, cpu_count);

    // Create per-task workspaces
    var workspaces = try allocator.alloc(ThreadWorkspace, max_workers);
    defer {
        for (workspaces) |*workspace| {
            workspace.deinit();
        }
        allocator.free(workspaces);
    }

    for (0..max_workers) |i| {
        workspaces[i] = try ThreadWorkspace.init(allocator, q_len, kv_len, head_dim);
    }

    // Submit tasks to thread pool
    var task_contexts = try allocator.alloc(MasklessHeadTaskContext, n_heads);
    defer allocator.free(task_contexts);

    // Assign a worker workspace to each head - cycling through available workspaces
    for (0..n_heads) |h| {
        const workspace_idx = h % max_workers;

        task_contexts[h] = MasklessHeadTaskContext{
            .head_id = h,
            .query = query_f32,
            .key = key_transpose,
            .value = value_f32,
            .out = &out,
            .scale = scale,
            .q_len = q_len,
            .kv_len = kv_len,
            .head_dim = head_dim,
            .workspace = &workspaces[workspace_idx],
        };

        // Use the module-level submitTask function
        try @import("../core/thread_pool.zig").submitTask(MasklessHeadTaskContext, MasklessHeadTaskContext.process, &task_contexts[h]);
    }

    // Wait for all tasks to complete
    try @import("../core/thread_pool.zig").waitForAll();

    return out;
}
