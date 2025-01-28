const std = @import("std");
const atomic = std.atomic;
const Allocator = std.mem.Allocator;
const Tensor = @import("tensor.zig").Tensor;

comptime {
    @setFloatMode(.optimized);
}

// Tuning parameters
const T: usize = blk: {
    const ideal = 16;
    break :blk (ideal + 7) & ~@as(usize, 7);
};

const V: usize = 8;
const CACHE_LINE = atomic.cache_line;
const Vec8f = @Vector(8, f32);

const ThreadData = struct {
    idx: atomic.Value(usize) align(CACHE_LINE),
    _pad: [CACHE_LINE - @sizeOf(atomic.Value(usize))]u8 = undefined,
};

const Ctx = struct {
    A: []const f16,
    B: []const f16,
    C: []f16,
    M: usize,
    N: usize,
    K: usize,
    tm: usize,
    tn: usize,
    tt: usize,
    cnt: *ThreadData,
};

fn tile(ctx: Ctx, local: *[T][T]f32, i_start: usize, j_start: usize, k: usize, i_end: usize, j_end: usize, k_end: usize) void {
    @setRuntimeSafety(false);
    var A_local: [T][T]f32 align(32) = undefined;
    var B_local: [T][T]f32 align(32) = undefined;

    const k_size = k_end - k;
    const i_size = i_end - i_start;
    const j_size = j_end - j_start;

    // Load A [M x K]
    for (0..i_size) |i| {
        const row_offset = (i_start + i) * ctx.K;
        for (0..k_size) |kk| {
            const a_idx = row_offset + k + kk;
            A_local[i][kk] = @floatCast(ctx.A[a_idx]);
        }
    }

    // Load pre-transposed B [N x K]
    for (0..j_size) |j| {
        const row_offset = (j_start + j) * ctx.K;
        for (0..k_size) |kk| {
            const b_idx = row_offset + k + kk;
            B_local[j][kk] = @floatCast(ctx.B[b_idx]);
        }
    }

    var i: usize = 0;
    while (i + V <= i_size) : (i += V) {
        var j: usize = 0;
        while (j + V <= j_size) : (j += V) {
            var acc: [V][V]f32 align(32) = [_][V]f32{[_]f32{0} ** V} ** V;

            for (0..k_size) |kk| {
                const a_vec = Vec8f{
                    A_local[i][kk],     A_local[i + 1][kk],
                    A_local[i + 2][kk], A_local[i + 3][kk],
                    A_local[i + 4][kk], A_local[i + 5][kk],
                    A_local[i + 6][kk], A_local[i + 7][kk],
                };

                const b_vec = Vec8f{
                    B_local[j][kk],     B_local[j + 1][kk],
                    B_local[j + 2][kk], B_local[j + 3][kk],
                    B_local[j + 4][kk], B_local[j + 5][kk],
                    B_local[j + 6][kk], B_local[j + 7][kk],
                };

                inline for (0..V) |bi| {
                    const a_broadcast: Vec8f = @splat(a_vec[bi]);
                    const c_vec = Vec8f{
                        acc[bi][0], acc[bi][1], acc[bi][2], acc[bi][3],
                        acc[bi][4], acc[bi][5], acc[bi][6], acc[bi][7],
                    };
                    const prod = @mulAdd(Vec8f, a_broadcast, b_vec, c_vec);
                    inline for (0..V) |bj| {
                        acc[bi][bj] = prod[bj];
                    }
                }
            }

            for (0..V) |bi| {
                for (0..V) |bj| {
                    local[i + bi][j + bj] += acc[bi][bj];
                }
            }
        }

        while (j < j_size) : (j += 1) {
            for (0..V) |bi| {
                var sum: f32 = 0;
                for (0..k_size) |kk| {
                    sum = @mulAdd(f32, A_local[i + bi][kk], B_local[j][kk], sum);
                }
                local[i + bi][j] += sum;
            }
        }
    }

    while (i < i_size) : (i += 1) {
        for (0..j_size) |j| {
            var sum: f32 = 0;
            for (0..k_size) |kk| {
                sum = @mulAdd(f32, A_local[i][kk], B_local[j][kk], sum);
            }
            local[i][j] += sum;
        }
    }
}

fn work(ctx: Ctx) void {
    @setRuntimeSafety(false);
    var local: [T][T]f32 align(32) = undefined;

    while (true) {
        const idx = ctx.cnt.idx.fetchAdd(1, .seq_cst);
        if (idx >= ctx.tt) break;

        const i = idx / ctx.tn;
        const j = idx % ctx.tn;

        const i_start = i * T;
        const j_start = j * T;
        const i_end = @min(i_start + T, ctx.M);
        const j_end = @min(j_start + T, ctx.N);

        for (0..T) |x| {
            @memset(&local[x], 0);
        }

        var k: usize = 0;
        while (k < ctx.K) : (k += T) {
            const k_end = @min(k + T, ctx.K);
            tile(ctx, &local, i_start, j_start, k, i_end, j_end, k_end);
        }

        // Store in transposed order [N x M]
        for (j_start..j_end) |jj| {
            const row_offset = jj * ctx.M;
            const local_col = jj - j_start;
            for (i_start..i_end) |ii| {
                const local_row = ii - i_start;
                ctx.C[row_offset + ii] = @floatCast(local[local_row][local_col]);
            }
        }
    }
}

pub fn mul(a: Tensor(f16), bt: Tensor(f16), alloc: Allocator) !Tensor(f16) {
    @setRuntimeSafety(false);
    const A_shape = a.shape;
    const BT_shape = bt.shape;

    if (A_shape.len != 2 or BT_shape.len != 2) {
        return error.BadShape;
    }

    const M = A_shape[0];
    const K = A_shape[1];
    const N = BT_shape[0]; // Note: Using first dim since B is pre-transposed!

    if (BT_shape[1] != K) {
        return error.BadShape;
    }

    // Result will be [N x M] (transposed!)
    var result = try Tensor(f16).init(alloc, &[_]usize{ N, M });
    errdefer result.deinit();

    const tm = (M + T - 1) / T;
    const tn = (N + T - 1) / T;
    const tt = tm * tn;

    var counter = ThreadData{ .idx = atomic.Value(usize).init(0) };
    const ctx = Ctx{
        .A = a.getSlice(),
        .B = bt.getSlice(),
        .C = result.getSlice(),
        .M = M,
        .N = N,
        .K = K,
        .tm = tm,
        .tn = tn,
        .tt = tt,
        .cnt = &counter,
    };

    const n_threads = try std.Thread.getCpuCount();
    var threads = try std.ArrayList(std.Thread).initCapacity(alloc, n_threads);
    defer threads.deinit();

    for (0..n_threads) |_| {
        try threads.append(try std.Thread.spawn(.{}, work, .{ctx}));
    }

    for (threads.items) |thread| thread.join();

    return result;
}

// Helper for transposing matrices
pub fn trans(src: Tensor(f16), alloc: Allocator) !Tensor(f16) {
    const M = src.shape[0];
    const N = src.shape[1];
    var dst = try Tensor(f16).init(alloc, &[_]usize{ N, M });
    errdefer dst.deinit();

    const src_data = src.getSlice();
    const dst_data = dst.getSlice();

    var i: usize = 0;
    while (i < M) : (i += T) {
        var j: usize = 0;
        while (j < N) : (j += T) {
            const i_end = @min(i + T, M);
            const j_end = @min(j + T, N);

            for (i..i_end) |ii| {
                for (j..j_end) |jj| {
                    dst_data[jj * M + ii] = src_data[ii * N + jj];
                }
            }
        }
    }
    return dst;
}
///// BENCHMARKS /////
const time = std.time;

fn bench(alloc: Allocator, M: usize, N: usize, K: usize, runs: usize) !f64 {
    var A = try Tensor(f16).init(alloc, &[_]usize{ M, K });
    defer A.deinit();
    var B = try Tensor(f16).init(alloc, &[_]usize{ K, N });
    defer B.deinit();

    // Random init
    var rng = std.rand.DefaultPrng.init(42);
    const r = rng.random();
    for (A.data) |*v| v.* = @floatCast(r.float(f32) * 0.1 - 0.05);
    for (B.data) |*v| v.* = @floatCast(r.float(f32) * 0.1 - 0.05);

    // Pre-transpose B
    var BT = try trans(B, alloc);
    defer BT.deinit();

    // Warmup
    var C = try mul(A, BT, alloc);
    C.deinit();

    // Timed runs
    var t: u64 = 0;
    var timer = try time.Timer.start();

    for (0..runs) |_| {
        timer.reset();
        C = try mul(A, BT, alloc);
        t += timer.read();
        C.deinit();
    }

    const avg_ns = @as(f64, @floatFromInt(t)) / @as(f64, @floatFromInt(runs));
    const avg_s = avg_ns / 1e9;
    const ops = 2 * M * N * K;
    return (@as(f64, @floatFromInt(ops)) / avg_s) / 1e9;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const runs = 5;

    // Test sizes
    const sizes = [_][3]usize{
        .{ 1, 2048, 51200 },
        .{ 800, 2048, 51200 },
        .{ 1, 2048, 6144 }, // Vector
        .{ 800, 2048, 6144 }, // Vector
        .{ 512, 512, 512 }, // Square
        .{ 1024, 1024, 1024 }, // Big square
        .{ 768, 512, 256 }, // Rect
        .{ 2048, 1024, 512 }, // Big rect
        .{ T, T, T }, // Tile
        .{ 2048, 2048, 2048 }, // Huge
    };

    std.debug.print("\nMatrix Multiplication Benchmark\n", .{});
    std.debug.print("============================\n\n", .{});

    for (sizes) |s| {
        const M = s[0];
        const N = s[1];
        const K = s[2];

        const gflops = try bench(alloc, M, N, K, runs);
        std.debug.print("{d}x{d}x{d}: {d:.2} GFLOPS\n", .{ M, N, K, gflops });
    }
}

///// TESTS /////
const testing = std.testing;

const Pat = enum { Id, Seq, Rnd, One };

fn fill(t: anytype, p: Pat, seed: ?u64) void {
    const data = t.getSlice();
    const rows = t.shape[0];
    const cols = t.shape[1];

    switch (p) {
        .Id => for (0..rows) |i| {
            for (0..cols) |j| {
                data[i * cols + j] = if (i == j) 1 else 0;
            }
        },
        .Seq => for (data, 0..) |*v, i| {
            v.* = @floatCast(@as(f32, @floatFromInt(i)) * 0.01);
        },
        .Rnd => {
            var rng = std.rand.DefaultPrng.init(seed orelse 42);
            const r = rng.random();
            for (data) |*v| v.* = @floatCast(r.float(f32) * 0.1 - 0.05);
        },
        .One => {
            for (data) |*v| {
                v.* = 1.0;
            }
        },
    }
}

fn naive(a: Tensor(f16), b: Tensor(f16), alloc: Allocator) !Tensor(f16) {
    const M = a.shape[0];
    const K = a.shape[1];
    const N = b.shape[1];

    if (b.shape[0] != K) return error.BadShape;

    var c = try Tensor(f16).init(alloc, &[_]usize{ M, N });
    errdefer c.deinit();

    const ad = a.getSlice();
    const bd = b.getSlice();
    const cd = c.getSlice();

    for (0..M) |i| {
        for (0..N) |j| {
            var sum: f32 = 0;
            for (0..K) |k| {
                const av: f32 = @floatCast(ad[i * K + k]);
                const bv: f32 = @floatCast(bd[k * N + j]);
                sum += av * bv;
            }
            cd[i * N + j] = @floatCast(sum);
        }
    }
    return c;
}

test "Basic" {
    const alloc = testing.allocator;

    const sizes = [_][3]usize{
        .{ 1, 4, 3 },
        .{ 2, 3, 4 },
        .{ 8, 8, 8 },
        .{ 3, 7, 5 },
        .{ T, T, T },
        .{ T + 1, T, T - 1 },
    };

    const pats = [_]Pat{ .Id, .Seq, .Rnd, .One };

    for (sizes) |s| {
        const M = s[0];
        const K = s[1];
        const N = s[2];

        for (pats) |p| {
            // Make matrices
            var A = try Tensor(f16).init(alloc, &[_]usize{ M, K });
            defer A.deinit();
            var B = try Tensor(f16).init(alloc, &[_]usize{ K, N });
            defer B.deinit();

            fill(&A, p, null);
            fill(&B, p, 12345);

            // Get reference result
            var C_ref = try naive(A, B, alloc);
            defer C_ref.deinit();

            // Get our result
            var BT = try trans(B, alloc);
            defer BT.deinit();
            var C = try mul(A, BT, alloc);
            defer C.deinit();

            // Compare
            try testing.expectEqual(C.shape[0], M);
            try testing.expectEqual(C.shape[1], N);

            const cd = C.getSlice();
            const rd = C_ref.getSlice();

            for (cd, rd, 0..) |c, r, i| {
                const diff = @abs(@as(f32, @floatCast(c)) - @as(f32, @floatCast(r)));
                if (diff >= 1e-3) {
                    std.debug.print("Mismatch at {}: got {d}, want {d}\n", .{ i, c, r });
                    return error.TestFailed;
                }
            }
        }
    }
}

test "Edge" {
    const alloc = testing.allocator;

    // Bad shapes
    {
        var A = try Tensor(f16).init(alloc, &[_]usize{ 2, 3 });
        defer A.deinit();
        var BT = try Tensor(f16).init(alloc, &[_]usize{ 4, 2 });
        defer BT.deinit();

        try testing.expectError(error.BadShape, mul(A, BT, alloc));
    }

    // Stability
    {
        var A = try Tensor(f16).init(alloc, &[_]usize{ 2, 2 });
        defer A.deinit();
        var B = try Tensor(f16).init(alloc, &[_]usize{ 2, 2 });
        defer B.deinit();

        var ad = A.getSlice();
        var bd = B.getSlice();

        // Mixed sizes
        ad[0] = 100;
        ad[1] = 0.01;
        ad[2] = 0.01;
        ad[3] = 100;

        bd[0] = 0.01;
        bd[1] = 100;
        bd[2] = 100;
        bd[3] = 0.01;

        var BT = try trans(B, alloc);
        defer BT.deinit();

        var C = try mul(A, BT, alloc);
        defer C.deinit();

        var C_ref = try naive(A, B, alloc);
        defer C_ref.deinit();

        const cd = C.getSlice();
        const rd = C_ref.getSlice();

        for (cd, rd) |c, r| {
            const diff = @abs(@as(f32, @floatCast(c)) - @as(f32, @floatCast(r)));
            try testing.expect(diff < 1e-2);
        }
    }
}
