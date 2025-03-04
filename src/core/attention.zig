const std = @import("std");
const Tensor = @import("../core/tensor.zig").Tensor;
const Allocator = std.mem.Allocator;
const Thread = std.Thread;
const sgemminplace = @import("../core/sgemm_inplace.zig").matmul;
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

/// Performs multi-headed scaled dot-product attention (without masking) optimized for CPU performance.
///
/// This function implements the standard attention formula:
///     Attention(Q, K, V) = softmax(Q*K^T / sqrt(d_k)) * V
/// across multiple heads in parallel, utilizing a thread pool for efficient execution.
///
/// Performance optimizations:
/// - Uses f16 for storage and f32 for computation (mixed precision)
/// - Thread-level parallelism with each thread processing multiple heads
/// - Pre-allocated thread workspaces to minimize allocations during computation
/// - Efficient memory layout with transposed key matrix for better cache utilization
/// - Adaptive threading that scales based on available CPU cores
/// - Nested parallelism within matrix multiplications
/// - Balanced workload distribution across threads
///
/// The implementation follows a zero-copy design pattern where possible and leverages
/// Zig's explicit memory management to avoid unnecessary allocations/deallocations
/// during the computation-intensive attention process.
///
/// Parameters:
///     n_heads: comptime_int - Number of attention heads
///     head_dim: comptime_int - Dimension of each attention head
///     query: Tensor(f16) - Query tensor with shape [n_heads, q_len, head_dim]
///     key: Tensor(f16) - Key tensor with shape [n_heads, kv_len, head_dim]
///     value: Tensor(f16) - Value tensor with shape [n_heads, kv_len, head_dim]
///     allocator: Allocator - Memory allocator for intermediate tensors
///
/// Returns:
///     Tensor(f16) - Output tensor with shape [n_heads, q_len, head_dim]
///
/// Errors:
///     - OutOfMemory if allocator fails to allocate memory
///     - ThreadCreationFailure if thread spawning fails
///     - Other errors from tensor operations or matrix multiplication
///
/// Thread safety:
///     This function is thread-safe and creates a dedicated thread pool for attention
///     computation. The input tensors should not be modified during execution.
///
/// Example:
///     ```
///     var output = try multiMasklessSDPA(12, 64, query_tensor, key_tensor, value_tensor, allocator);
///     defer output.deinit();
///    ```
pub fn multiMasklessSDPA(
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

/// Performs multi-token masked scaled dot-product attention optimized for CPU parallelism.
///
/// This function implements the standard attention formula with masking:
///     Attention(Q, K, V) = softmax(mask(Q*K^T / sqrt(d_k))) * V
/// processing multiple query tokens against key-value pairs simultaneously across
/// attention heads in parallel.
///
/// Key performance optimizations:
/// - Mixed precision: Uses f16 for storage and f32 for computations
/// - Memory layout transformation: Transposes key matrices for optimal cache access
/// - Multi-level parallelism: Distributes heads across threads with nested parallelism
///   for matrix multiplications
/// - Thread-local workspaces: Pre-allocates per-thread memory to avoid allocation
///   during computation
/// - Controlled precision flow: Maintains numerical stability with careful
///   precision management
/// - Balanced workload distribution: Ensures even distribution of heads across threads
///
/// This implementation is designed for inference scenarios where multiple tokens
/// need to be processed simultaneously with arbitrary attention masking patterns.
/// Unlike the single-token path, this handles arbitrary sequence lengths and
/// supports complex masking patterns beyond simple causal masking.
///
/// Parameters:
///     query: Tensor(f16) - Query tensor with shape [n_heads, q_len, head_dim]
///     key: Tensor(f16) - Key tensor with shape [n_heads, kv_len, head_dim]
///     value: Tensor(f16) - Value tensor with shape [n_heads, kv_len, head_dim]
///     mask: Tensor(bool) - Attention mask tensor with shape [q_len, 1, kv_len] or [q_len, kv_len]
///                          where false indicates positions to mask out
///     n_heads: usize - Number of attention heads
///     head_dim: usize - Dimension of each attention head
///     allocator: Allocator - Memory allocator for intermediate tensors
///
/// Returns:
///     Tensor(f16) - Output tensor with shape [n_heads, q_len, head_dim]
///
/// Errors:
///     - OutOfMemory if allocator fails to allocate memory
///     - ThreadCreationFailure if thread spawning fails
///     - Other errors from tensor operations or matrix multiplication
///
/// Thread safety:
///     This function is thread-safe and creates a dedicated thread pool for attention
///     computation. The input tensors should not be modified during execution.
///
/// Performance considerations:
///     - Performance scales well with increasing numbers of CPU cores
///     - Memory bandwidth can become a bottleneck with very large tensors
///     - Optimal performance is achieved when head_dim is a multiple of SIMD width
///
/// Example:
///     ```
///     var mask = try createCausalMask(seq_len, allocator);
///     defer mask.deinit();
///
///     var output = try multiMaskedSDPA(
///         query_tensor, key_tensor, value_tensor, mask,
///         16, 64, allocator
///     );
///     defer output.deinit();
///     ```
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

/// Worker function for parallel processing of attention heads during single-token generation.
///
/// This function is specifically optimized for the critical path of autoregressive inference
/// where a single new token query attends to all previous key-value pairs. The implementation
/// contains numerous performance optimizations that make it especially efficient for this case.
///
/// Performance optimizations:
/// - SIMD vectorization: Uses explicit SIMD types (SimdF16x8, SimdF32x8) to process 8 elements
///   simultaneously, maximizing CPU vector unit utilization
/// - Mixed precision: Computes in f32 for numerical stability but stores in f16 for memory efficiency
/// - Compile-time constants: Uses MODEL_HEAD_DIM, HEAD_DIM_CHUNKS, and SIMD_WIDTH as compile-time
///   constants to enable aggressive compiler optimizations
/// - Branchless softmax: Implements softmax using branchless operations to avoid pipeline stalls
/// - Memory alignment: Ensures proper alignment for SIMD operations with explicit pointer casting
/// - Loop unrolling: Uses inline for loops to force unrolling of the inner loops across SIMD chunks
/// - Direct memory access: Avoids unnecessary copies through direct, aligned pointer manipulation
///
/// Why it works well for single-token generation:
/// 1. Predictable access patterns: With exactly one query token, memory access patterns are
///    highly predictable, allowing for optimal prefetching and cache utilization
/// 2. No matrix multiplication: Replaces general matrix multiplication with direct dot products,
///    eliminating overhead for this specific case
/// 3. SIMD-friendly dimensions: Model dimensions are typically multiples of SIMD width,
///    enabling perfect vectorization without remainder handling
/// 4. Minimal branching: Critical loops contain minimal conditional logic, reducing branch
///    mispredictions in the processor pipeline
/// 5. Workspace reuse: Uses pre-allocated thread-local workspaces, eliminating allocation
///    overhead during the latency-sensitive generation phase
///
/// Parameters:
///     ctx: HeadContext - Contains pointers to input/output tensors and thread-specific workspaces
///
/// Thread safety:
///     Designed to be called from multiple threads simultaneously, with each thread processing
///     different attention heads using thread-local workspaces.
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

/// Performs single-token masked scaled dot-product attention optimized for the token generation phase.
///
/// This function implements a specialized version of the attention mechanism specifically designed
/// for the "hot path" of autoregressive token generation, where a single new token query attends
/// to all previous key-value pairs with masking.
///
/// Unlike the multi-headed general attention, this function:
/// - Processes exactly one token query per head (middle dimension is always 1)
/// - Uses masked attention to prevent attending to future positions
/// - Employs global pre-allocated workspaces to eliminate allocation overhead
/// - Enforces a maximum sequence length (MAX_KV_LEN) for safety and optimization
/// - Uses model-specific constants (MODEL_N_HEADS, MODEL_HEAD_DIM) for specialized execution
///
/// Performance optimizations:
/// - Zero dynamic allocations during computation (only one for the output tensor)
/// - Global workspaces eliminate memory allocation during the critical inference path
/// - Thread-level parallelism across attention heads
/// - Bounds-checking to prevent sequence length overflow
/// - Fixed thread count upper limit to prevent excessive threading overhead
/// - Balanced head distribution across available CPU cores
///
/// This implementation is specifically designed for inference-time token generation where
/// latency is critical and the pattern of computation is highly predictable.
///
/// Parameters:
///     query: Tensor(f16) - Query tensor with shape [MODEL_N_HEADS, 1, MODEL_HEAD_DIM]
///     key: Tensor(f16) - Key tensor with shape [MODEL_N_HEADS, kv_len, MODEL_HEAD_DIM]
///     value: Tensor(f16) - Value tensor with shape [MODEL_N_HEADS, kv_len, MODEL_HEAD_DIM]
///     mask: Tensor(bool) - Attention mask tensor with shape [1, kv_len]
///     allocator: Allocator - Memory allocator for output tensor
///
/// Returns:
///     Tensor(f16) - Output tensor with shape [MODEL_N_HEADS, 1, MODEL_HEAD_DIM]
///
/// Errors:
///     - SequenceTooLong if kv_len exceeds MAX_KV_LEN
///     - OutOfMemory if allocator fails to allocate output tensor
///     - ThreadCreationFailure if thread spawning fails
///
/// Thread safety:
///     This function is thread-safe for inference but assumes global_workspaces are not
///     concurrently accessed by other functions. Not safe for concurrent model inference.
///
/// Example:
///     ```
///     var next_token_embedding = try singleMaskedSDPA(
///         query_tensor, cached_keys, cached_values, attention_mask, allocator
///     );
///     defer next_token_embedding.deinit();
///
pub fn singleMaskedSDPA(
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
