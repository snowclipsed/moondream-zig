const std = @import("std");
const atomic = std.atomic;
const math = std.math;
const Allocator = std.mem.Allocator;
const Tensor = @import("tensor.zig").Tensor;
const Slice = @import("tensor.zig").Slice;

const T: usize = 160;
const V: usize = 8;
const CACHE_LINE_SIZE: usize = atomic.cache_line;
const CHUNK_SIZE: usize = 1;
const AVX2_ALIGNMENT = 32;
const MICRO_KERNEL_SIZE: usize = std.simd.suggestVectorLength(f32) orelse 8;
const Vec8f = @Vector(8, f32);

const ThreadLocalData = struct {
    current_index: atomic.Value(usize) align(CACHE_LINE_SIZE),
    _padding: [CACHE_LINE_SIZE - @sizeOf(atomic.Value(usize))]u8 = undefined,
};

const ThreadContext = struct {
    A: []const f16,
    B: []const f16,
    C: []f32,
    M: usize,
    N: usize,
    K: usize,
    tiles_M: usize,
    tiles_N: usize,
    total_tiles: usize,
    shared_counter: *ThreadLocalData,
};

fn workerThread(ctx: ThreadContext) void {
    var local_C: [T][T]f32 align(AVX2_ALIGNMENT) = undefined;

    while (true) {
        const start_idx = ctx.shared_counter.current_index.fetchAdd(CHUNK_SIZE, .seq_cst);
        if (start_idx >= ctx.total_tiles) break;

        const end_idx = @min(start_idx + CHUNK_SIZE, ctx.total_tiles);
        var idx = start_idx;

        while (idx < end_idx) : (idx += 1) {
            const i = idx / ctx.tiles_N;
            const j = idx % ctx.tiles_N;

            const i_start = i * T;
            const j_start = j * T;
            const i_end = @min(i_start + T, ctx.M);
            const j_end = @min(j_start + T, ctx.N);

            for (0..T) |x| {
                @memset(&local_C[x], 0);
            }

            var k: usize = 0;
            while (k < ctx.K) : (k += T) {
                const k_end = @min(k + T, ctx.K);
                computeTile(ctx, &local_C, i_start, j_start, k, i_end, j_end, k_end);
            }

            for (i_start..i_end) |ii| {
                const row_offset = ii * ctx.N;
                const local_row = ii - i_start;
                for (j_start..j_end) |jj| {
                    const local_col = jj - j_start;
                    ctx.C[row_offset + jj] += local_C[local_row][local_col];
                }
            }
        }
    }
}

fn computeTile(
    ctx: ThreadContext,
    local_C: *[T][T]f32,
    i_start: usize,
    j_start: usize,
    k_start: usize,
    i_end: usize,
    j_end: usize,
    k_end: usize,
) void {
    var A_local: [T][T]f32 align(32) = undefined;
    var B_local: [T][T]f32 align(32) = undefined;

    const k_size = k_end - k_start;
    const i_size = i_end - i_start;
    const j_size = j_end - j_start;

    for (0..i_size) |i| {
        const row_offset = (i_start + i) * ctx.K;
        for (0..k_size) |k| {
            const a_idx = row_offset + k_start + k;
            A_local[i][k] = @floatCast(ctx.A[a_idx]);
        }
    }

    for (0..k_size) |k| {
        const row_offset = (k_start + k) * ctx.N;
        for (0..j_size) |j| {
            const b_idx = row_offset + j_start + j;
            B_local[k][j] = @floatCast(ctx.B[b_idx]);
        }
    }

    var i: usize = 0;
    while (i + 8 <= i_size) : (i += 8) {
        var j: usize = 0;
        while (j + 8 <= j_size) : (j += 8) {
            var acc: [8][8]f32 align(32) = [_][8]f32{[_]f32{0} ** 8} ** 8;

            for (0..k_size) |k| {
                const a_vec = Vec8f{
                    A_local[i][k],     A_local[i + 1][k],
                    A_local[i + 2][k], A_local[i + 3][k],
                    A_local[i + 4][k], A_local[i + 5][k],
                    A_local[i + 6][k], A_local[i + 7][k],
                };

                const b_vec = Vec8f{
                    B_local[k][j],     B_local[k][j + 1],
                    B_local[k][j + 2], B_local[k][j + 3],
                    B_local[k][j + 4], B_local[k][j + 5],
                    B_local[k][j + 6], B_local[k][j + 7],
                };

                inline for (0..8) |bi| {
                    const a_broadcast: Vec8f = @splat(a_vec[bi]);
                    const c_vec = Vec8f{
                        acc[bi][0], acc[bi][1], acc[bi][2], acc[bi][3],
                        acc[bi][4], acc[bi][5], acc[bi][6], acc[bi][7],
                    };
                    const prod = @mulAdd(Vec8f, a_broadcast, b_vec, c_vec);
                    inline for (0..8) |bj| {
                        acc[bi][bj] = prod[bj];
                    }
                }
            }

            for (0..8) |bi| {
                for (0..8) |bj| {
                    local_C[i + bi][j + bj] += acc[bi][bj];
                }
            }
        }

        while (j < j_size) : (j += 1) {
            for (0..8) |bi| {
                var sum: f32 = 0;
                for (0..k_size) |k| {
                    sum = @mulAdd(f32, A_local[i + bi][k], B_local[k][j], sum);
                }
                local_C[i + bi][j] += sum;
            }
        }
    }

    while (i < i_size) : (i += 1) {
        for (0..j_size) |j| {
            var sum: f32 = 0;
            for (0..k_size) |k| {
                sum = @mulAdd(f32, A_local[i][k], B_local[k][j], sum);
            }
            local_C[i][j] += sum;
        }
    }
}

pub fn tiledMatMul(
    allocator: Allocator,
    A: Tensor(f16),
    B: Tensor(f16),
    C: Tensor(f32),
) !void {
    const A_shape = A.shape;
    const B_shape = B.shape;
    const C_shape = C.shape;

    if (A_shape.len != 2 or B_shape.len != 2 or C_shape.len != 2) {
        return error.InvalidTensorDimension;
    }

    const M = A_shape[0];
    const K = A_shape[1];
    const N = B_shape[1];

    if (B_shape[0] != K or C_shape[0] != M or C_shape[1] != N) {
        return error.IncompatibleTensorShapes;
    }

    const A_data = A.getSlice();
    const B_data = B.getSlice();
    const C_data = C.getSlice();

    const tiles_M = (M + T - 1) / T;
    const tiles_N = (N + T - 1) / T;
    const total_tiles = tiles_M * tiles_N;

    var shared_data = ThreadLocalData{ .current_index = atomic.Value(usize).init(0) };

    const num_threads = try std.Thread.getCpuCount();
    var thread_pool = try std.ArrayList(std.Thread).initCapacity(allocator, num_threads);
    defer thread_pool.deinit();

    const context = ThreadContext{
        .A = A_data,
        .B = B_data,
        .C = C_data,
        .M = M,
        .N = N,
        .K = K,
        .tiles_M = tiles_M,
        .tiles_N = tiles_N,
        .total_tiles = total_tiles,
        .shared_counter = &shared_data,
    };

    for (0..num_threads) |_| {
        try thread_pool.append(try std.Thread.spawn(.{}, workerThread, .{context}));
    }

    for (thread_pool.items) |thread| thread.join();
}

fn createTensor(allocator: Allocator, shape: []const usize, data: []const f16) !Tensor(f16) {
    var tensor = try Tensor(f16).init(allocator, shape);
    @memcpy(tensor.getSlice(), data);
    return tensor;
}

fn createFloatTensor(allocator: Allocator, shape: []const usize, data: []const f32) !Tensor(f32) {
    var tensor = try Tensor(f32).init(allocator, shape);
    @memcpy(tensor.getSlice(), data);
    return tensor;
}

test "tiledMatMul simple case" {
    const allocator = std.testing.allocator;
    const A_shape = [_]usize{ 2, 2 };
    const B_shape = [_]usize{ 2, 2 };
    const C_shape = [_]usize{ 2, 2 };

    const A_data = [_]f16{ 1, 2, 3, 4 };
    const B_data = [_]f16{ 5, 6, 7, 8 };
    const expected_C_data = [_]f32{ 19, 22, 43, 50 };

    var A = try createTensor(allocator, &A_shape, &A_data);
    defer A.deinit();
    var B = try createTensor(allocator, &B_shape, &B_data);
    defer B.deinit();
    var C = try createFloatTensor(allocator, &C_shape, &[_]f32{0} ** 4);
    defer C.deinit();

    try tiledMatMul(allocator, A, B, C);

    const C_data = C.getSlice();
    for (C_data, 0..) |val, i| {
        try std.testing.expect(math.approxEqAbs(f32, val, expected_C_data[i], 1e-4));
    }
}

test "tiledMatMul - larger case" {
    const allocator = std.testing.allocator;
    const A_shape = [_]usize{ 4, 4 };
    const B_shape = [_]usize{ 4, 4 };
    const C_shape = [_]usize{ 4, 4 };

    const A_data = [_]f16{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    const B_data = [_]f16{ 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32 };
    const expected_C_data = [_]f32{ 250, 260, 270, 280, 618, 644, 670, 696, 986, 1028, 1070, 1112, 1354, 1412, 1470, 1528 };

    var A = try createTensor(allocator, &A_shape, &A_data);
    defer A.deinit();
    var B = try createTensor(allocator, &B_shape, &B_data);
    defer B.deinit();
    var C = try createFloatTensor(allocator, &C_shape, &[_]f32{0} ** 16);
    defer C.deinit();

    try tiledMatMul(allocator, A, B, C);

    const C_data = C.getSlice();
    for (C_data, 0..) |val, i| {
        try std.testing.expect(math.approxEqAbs(f32, val, expected_C_data[i], 1e-4));
    }
}

test "tiledMatMul - mismatched dimensions" {
    const allocator = std.testing.allocator;
    const A_shape = [_]usize{ 2, 3 };
    const B_shape = [_]usize{ 2, 2 };
    const C_shape = [_]usize{ 2, 2 };

    const A_data = [_]f16{ 1, 2, 3, 4, 5, 6 };
    const B_data = [_]f16{ 7, 8, 9, 10 };

    var A = try createTensor(allocator, &A_shape, &A_data);
    defer A.deinit();
    var B = try createTensor(allocator, &B_shape, &B_data);
    defer B.deinit();
    var C = try createFloatTensor(allocator, &C_shape, &[_]f32{0} ** 4);
    defer C.deinit();

    const result = tiledMatMul(allocator, A, B, C);
    try std.testing.expect(result == error.IncompatibleTensorShapes);
}
