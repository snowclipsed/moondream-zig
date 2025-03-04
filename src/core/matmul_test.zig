const std = @import("std");
const Allocator = std.mem.Allocator;

const atomic = std.atomic;
const math = std.math;
const Tensor = @import("tensor.zig").Tensor;
const hgemm = @import("hgemm.zig");
const sgemm = @import("sgemm.zig");
const sgemm_inplace = @import("sgemm_inplace.zig");
const ops = @import("ops.zig");
const Tile = sgemm.Tile;
const time = std.time;
const testing = std.testing;

fn naiveMatMulHGEMM(A: *const Tensor(f16), B: *const Tensor(f16), C: *Tensor(f32)) !void {
    const M = A.shape[0];
    const K = A.shape[1];
    const N = B.shape[1];

    if (C.shape[0] != M or C.shape[1] != N) {
        return error.IncompatibleTensorShapes;
    }

    @memset(C.data, 0);

    for (0..M) |i| {
        for (0..N) |j| {
            var sum: f32 = 0;
            for (0..K) |k| {
                const a_val: f32 = @floatCast(A.data[i * K + k]);
                const b_val: f32 = @floatCast(B.data[k * N + j]);
                sum = @mulAdd(f32, a_val, b_val, sum);
            }
            C.data[i * N + j] = sum;
        }
    }
}

// Helper to fill tensor with test pattern
fn fillTestPattern(tensor: anytype, pattern: enum { Identity, Sequential, Random }) void {
    const shape = tensor.shape;
    const rows = shape[0];
    const cols = shape[1];

    switch (pattern) {
        .Identity => {
            for (0..rows) |i| {
                for (0..cols) |j| {
                    tensor.data[i * cols + j] = if (i == j) 1.0 else 0.0;
                }
            }
        },
        .Sequential => {
            for (tensor.data, 0..) |*val, i| {
                val.* = @floatCast(@as(f32, @floatFromInt(i)) * 0.01);
            }
        },
        .Random => {
            var rng = std.rand.DefaultPrng.init(42);
            const random = rng.random();
            for (tensor.data) |*val| {
                val.* = @floatCast(random.float(f32) * 2.0 - 1.0);
            }
        },
    }
}

// Helper to compare tensors with epsilon
fn compareResults(expected: *const Tensor(f32), actual: *const Tensor(f32), epsilon: f32) !void {
    try testing.expectEqual(expected.shape.len, actual.shape.len);
    for (expected.shape, actual.shape) |s1, s2| {
        try testing.expectEqual(s1, s2);
    }

    var max_diff: f32 = 0;
    for (expected.data, actual.data) |e, a| {
        const diff = @abs(e - a);
        max_diff = @max(max_diff, diff);
        try testing.expect(diff <= epsilon);
    }
}
// Helper functions
fn naiveMatMulF16(A: *const Tensor(f16), B: *const Tensor(f16), C: *Tensor(f16)) !void {
    const M = A.shape[0];
    const K = A.shape[1];
    const N = B.shape[1];

    if (B.shape[0] != K) return error.IncompatibleTensorShapes;
    if (C.shape[0] != M or C.shape[1] != N) return error.IncompatibleTensorShapes;

    const A_data = A.getSlice();
    const B_data = B.getSlice();
    const C_data = C.getSlice();

    for (0..M) |i| {
        for (0..N) |j| {
            var sum: f32 = 0; // Use f32 for intermediate calculations
            for (0..K) |k| {
                const a_val: f32 = @floatCast(A_data[i * K + k]);
                const b_val: f32 = @floatCast(B_data[k * N + j]);
                sum += a_val * b_val;
            }
            C_data[i * N + j] = @floatCast(sum);
        }
    }
}

fn compareResultsF16(expected: *const Tensor(f16), actual: *const Tensor(f16), epsilon: f32) !void {
    if (!std.mem.eql(usize, expected.shape, actual.shape)) {
        return error.ShapeMismatch;
    }

    var total_elements: usize = 1;
    for (expected.shape) |dim| {
        total_elements *= dim;
    }

    const expected_data = expected.getSlice();
    const actual_data = actual.getSlice();

    var max_diff: f32 = 0;
    var max_diff_idx: usize = 0;
    var num_diffs: usize = 0;

    for (0..total_elements) |i| {
        const exp_val: f32 = @floatCast(expected_data[i]);
        const act_val: f32 = @floatCast(actual_data[i]);
        const diff = @abs(exp_val - act_val);

        // Track maximum difference
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
        }

        // Calculate relative error for larger values
        const rel_error = if (@abs(exp_val) > 1.0) diff / @abs(exp_val) else diff;
        const threshold = if (@abs(exp_val) > 1.0) epsilon else epsilon * 0.1;

        if (rel_error > threshold) {
            num_diffs += 1;
            if (num_diffs <= 5) { // Only log first 5 differences
                std.log.warn("Difference at index {}: expected {}, got {} (diff: {d}, rel_error: {d})", .{ i, exp_val, act_val, diff, rel_error });
            }
        }
    }

    if (num_diffs > 0) {
        std.log.err("Total differences: {}, Max diff: {d} at index {}", .{ num_diffs, max_diff, max_diff_idx });
        return error.TestExpectedEqual;
    }
}

test "HGEMM MatMul - Basic functionality" {
    const allocator = testing.allocator;

    // Small matrix test
    {
        const shape_a = [_]usize{ 4, 3 };
        const shape_b = [_]usize{ 3, 4 };

        var A = try Tensor(f16).init(allocator, &shape_a);
        defer A.deinit();
        var B = try Tensor(f16).init(allocator, &shape_b);
        defer B.deinit();

        fillTestPattern(&A, .Sequential);
        fillTestPattern(&B, .Sequential);

        var expected = try Tensor(f16).init(allocator, &[_]usize{ shape_a[0], shape_b[1] });
        defer expected.deinit();

        try naiveMatMulF16(&A, &B, &expected);

        var C = try hgemm.matmul(A, B, allocator);
        defer C.deinit();

        try compareResultsF16(&expected, &C, 0.005);
    }
}

test "HGEMM MatMul - Large matrices" {
    const allocator = testing.allocator;

    // Test with matrices larger than tile size
    {
        const shape_a = [_]usize{ 2000, 180 };
        const shape_b = [_]usize{ 180, 1600 };

        var A = try Tensor(f16).init(allocator, &shape_a);
        defer A.deinit();
        var B = try Tensor(f16).init(allocator, &shape_b);
        defer B.deinit();

        fillTestPattern(&A, .Random);
        fillTestPattern(&B, .Random);

        // Normalize random values to avoid accumulation errors
        for (A.data) |*val| val.* = @floatCast(@as(f32, @floatCast(val.*)) * 0.01);
        for (B.data) |*val| val.* = @floatCast(@as(f32, @floatCast(val.*)) * 0.01);

        var expected = try Tensor(f16).init(allocator, &[_]usize{ shape_a[0], shape_b[1] });
        defer expected.deinit();

        try naiveMatMulF16(&A, &B, &expected);

        var C = try hgemm.matmul(A, B, allocator);
        defer C.deinit();

        try compareResultsF16(&expected, &C, 0.05); // Increased tolerance for large matrices
    }
}

test "HGEMM MatMul - Non-square matrices" {
    const allocator = testing.allocator;

    const test_shapes = [_][2][2]usize{
        .{ .{ 50, 30 }, .{ 30, 70 } },
        .{ .{ 25, 80 }, .{ 80, 35 } },
        .{ .{ 100, 20 }, .{ 20, 100 } },
    };

    for (test_shapes) |shapes| {
        const shape_a = shapes[0];
        const shape_b = shapes[1];

        var A = try Tensor(f16).init(allocator, &shape_a);
        defer A.deinit();
        var B = try Tensor(f16).init(allocator, &shape_b);
        defer B.deinit();

        fillTestPattern(&A, .Random);
        fillTestPattern(&B, .Random);

        // Normalize random values
        for (A.data) |*val| val.* = @floatCast(@as(f32, @floatCast(val.*)) * 0.01);
        for (B.data) |*val| val.* = @floatCast(@as(f32, @floatCast(val.*)) * 0.01);

        var expected = try Tensor(f16).init(allocator, &[_]usize{ shape_a[0], shape_b[1] });
        defer expected.deinit();

        try naiveMatMulF16(&A, &B, &expected);

        var C = try hgemm.matmul(A, B, allocator);
        defer C.deinit();

        try compareResultsF16(&expected, &C, 0.05);
    }
}

test "HGEMM MatMul - Identity matrix" {
    const allocator = testing.allocator;

    // Test multiplication with identity matrix
    {
        const size = 32;
        const shape = [_]usize{ size, size };

        var A = try Tensor(f16).init(allocator, &shape);
        defer A.deinit();
        var I = try Tensor(f16).init(allocator, &shape);
        defer I.deinit();

        fillTestPattern(&A, .Random);
        fillTestPattern(&I, .Identity);

        // Scale random values for A
        for (A.data) |*val| val.* = @floatCast(@as(f32, @floatCast(val.*)) * 0.01);

        // A * I should equal A
        var C = try hgemm.matmul(A, I, allocator);
        defer C.deinit();

        try compareResultsF16(&A, &C, 0.005);
    }
}

test "HGEMM MatMul - Error cases" {
    const allocator = testing.allocator;

    // Test incompatible shapes
    {
        var A = try Tensor(f16).init(allocator, &[_]usize{ 3, 4 });
        defer A.deinit();
        var B = try Tensor(f16).init(allocator, &[_]usize{ 5, 6 }); // Wrong inner dimension
        defer B.deinit();

        try testing.expectError(error.IncompatibleTensorShapes, hgemm.matmul(A, B, allocator));
    }
}

test "HGEMM MatMul - Edge cases" {
    const allocator = testing.allocator;

    // Test matrices with dimensions near tile boundaries
    const test_sizes = [_]usize{ 158, 159, 160, 161, 162 };

    for (test_sizes) |size| {
        var A = try Tensor(f16).init(allocator, &[_]usize{ size, size });
        defer A.deinit();
        var B = try Tensor(f16).init(allocator, &[_]usize{ size, size });
        defer B.deinit();

        fillTestPattern(&A, .Random);
        fillTestPattern(&B, .Random);

        // Scale random values
        for (A.data) |*val| val.* = @floatCast(@as(f32, @floatCast(val.*)) * 0.01);
        for (B.data) |*val| val.* = @floatCast(@as(f32, @floatCast(val.*)) * 0.01);

        var expected = try Tensor(f16).init(allocator, &[_]usize{ size, size });
        defer expected.deinit();

        try naiveMatMulF16(&A, &B, &expected);

        var C = try hgemm.matmul(A, B, allocator);
        defer C.deinit();

        // Use larger epsilon for larger matrices
        try compareResultsF16(&expected, &C, 0.05);
    }
}

test "HGEMM MatMul - Numerical stability" {
    const allocator = testing.allocator;

    // Test with very small and very large numbers
    {
        const shape = [_]usize{ 32, 32 };
        var A = try Tensor(f16).init(allocator, &shape);
        defer A.deinit();
        var B = try Tensor(f16).init(allocator, &shape);
        defer B.deinit();

        // Test with very small numbers
        fillTestPattern(&A, .Random);
        fillTestPattern(&B, .Random);

        // Use very small numbers but not too small for f16
        for (A.data) |*val| val.* = 1e-2;
        for (B.data) |*val| val.* = 1e-2;

        var expected = try Tensor(f16).init(allocator, &shape);
        defer expected.deinit();

        try naiveMatMulF16(&A, &B, &expected);

        var C = try hgemm.matmul(A, B, allocator);
        defer C.deinit();

        try compareResultsF16(&expected, &C, 0.01);

        // Test with moderate numbers (staying well within f16 range)
        for (A.data) |*val| val.* = 1.0;
        for (B.data) |*val| val.* = 1.0;

        var expected2 = try Tensor(f16).init(allocator, &shape);
        defer expected2.deinit();

        try naiveMatMulF16(&A, &B, &expected2);

        var C2 = try hgemm.matmul(A, B, allocator);
        defer C2.deinit();

        try compareResultsF16(&expected2, &C2, 0.01);
    }
}

//////////////////////////

test "SGEMM basic functionality" {
    const allocator = testing.allocator;

    // Test case 1: Square matrices
    {
        var a = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        defer a.deinit();
        var b = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        defer b.deinit();

        a.data[0] = 1.0;
        a.data[1] = 2.0;
        a.data[2] = 3.0;
        a.data[3] = 4.0;

        b.data[0] = 5.0;
        b.data[1] = 6.0;
        b.data[2] = 7.0;
        b.data[3] = 8.0;

        var result = try sgemm.matmul(f32, a, b, allocator);
        defer result.deinit();

        // Compare with known result
        try testing.expectApproxEqAbs(result.data[0], 19.0, 1e-6);
        try testing.expectApproxEqAbs(result.data[1], 22.0, 1e-6);
        try testing.expectApproxEqAbs(result.data[2], 43.0, 1e-6);
        try testing.expectApproxEqAbs(result.data[3], 50.0, 1e-6);
    }
}

test "SGEMM edge cases" {
    const allocator = testing.allocator;

    // Test case 1: 1x1 matrices
    {
        var a = try Tensor(f32).init(allocator, &[_]usize{ 1, 1 });
        defer a.deinit();
        var b = try Tensor(f32).init(allocator, &[_]usize{ 1, 1 });
        defer b.deinit();

        a.data[0] = 3.0;
        b.data[0] = 4.0;

        var result = try sgemm.matmul(f32, a, b, allocator);
        defer result.deinit();

        try testing.expectApproxEqAbs(result.data[0], 12.0, 1e-6);
    }

    // Test case 2: Tall matrix × Wide matrix
    {
        var a = try Tensor(f32).init(allocator, &[_]usize{ 3, 1 });
        defer a.deinit();
        var b = try Tensor(f32).init(allocator, &[_]usize{ 1, 3 });
        defer b.deinit();

        a.data[0] = 1.0;
        a.data[1] = 2.0;
        a.data[2] = 3.0;

        b.data[0] = 4.0;
        b.data[1] = 5.0;
        b.data[2] = 6.0;

        var result = try sgemm.matmul(f32, a, b, allocator);
        defer result.deinit();

        try testing.expectEqual(result.shape[0], @as(usize, 3));
        try testing.expectEqual(result.shape[1], @as(usize, 3));
    }

    // Test case 3: Zero matrices
    {
        var a = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        defer a.deinit();
        var b = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        defer b.deinit();

        @memset(a.data, 0);
        @memset(b.data, 0);

        var result = try sgemm.matmul(f32, a, b, allocator);
        defer result.deinit();

        for (result.data) |val| {
            try testing.expectEqual(val, 0);
        }
    }
}

test "SGEMM error cases" {
    const allocator = testing.allocator;

    // Test case 1: Mismatched dimensions
    {
        var a = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
        defer a.deinit();
        var b = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        defer b.deinit();

        try testing.expectError(error.ShapeMismatch, sgemm.matmul(f32, a, b, allocator));
    }

    // Test case 2: Invalid dimensions
    {
        var a = try Tensor(f32).init(allocator, &[_]usize{2});
        defer a.deinit();
        var b = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        defer b.deinit();

        try testing.expectError(error.InvalidDimensions, sgemm.matmul(f32, a, b, allocator));
    }
}

test "SGEMM correctness against reference" {
    const allocator = testing.allocator;
    const test_sizes = [_][3]usize{
        .{ 32, 32, 32 }, // Small square
        .{ 47, 35, 23 }, // Odd sizes
        .{ 128, 64, 96 }, // Rectangular
        .{ 1, 64, 128 }, // Single row × wide
        .{ 128, 1, 64 }, // Tall × single column
        .{ Tile - 1, Tile + 1, Tile }, // Around tile size
        .{ Tile, Tile, Tile }, // Exactly tile size
        .{ Tile + 1, Tile - 1, Tile }, // Around tile size
    };

    for (test_sizes) |size| {
        const M = size[0];
        const N = size[1];
        const K = size[2];

        // Create random input tensors
        var a = try ops.createRandomTensor(f32, allocator, &[_]usize{ M, K }, 42);
        defer a.deinit();
        var b = try ops.createRandomTensor(f32, allocator, &[_]usize{ K, N }, 43);
        defer b.deinit();

        // Compute using tiled matmul
        var result = try sgemm.matmul(f32, a, b, allocator);
        defer result.deinit();

        // Compute using reference matmul
        var expected = try ops.matmul(f32, &a, b);
        defer expected.deinit();

        // Compare results
        const eps: f32 = 1e-4; // Allow for some floating-point error
        for (result.data, expected.data) |val, exp| {
            try testing.expectApproxEqAbs(val, exp, eps);
        }

        std.debug.print("Test passed for size: M={}, N={}, K={}\n", .{ M, N, K });
    }
}

test "SGEMM numerical stability" {
    const allocator = testing.allocator;

    // Test case 1: Moderately large numbers
    {
        var a = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        defer a.deinit();
        var b = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        defer b.deinit();

        // Using smaller values to avoid overflow
        const large: f32 = 1e3;
        @memset(a.data, large);
        @memset(b.data, large);

        var result = try sgemm.matmul(f32, a, b, allocator);
        defer result.deinit();

        // Check results
        for (result.data) |val| {
            // Verify no infinity and reasonable magnitude
            try testing.expect(!std.math.isInf(val));
            try testing.expect(!std.math.isNan(val));
            // For 2x2 matrices filled with 1e3, each element should be 2 * (1e3 * 1e3) = 2e6
            try testing.expectApproxEqAbs(val, 2e6, 1e-6);
        }
    }

    // Test case 2: Small but non-zero numbers
    {
        var a = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        defer a.deinit();
        var b = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        defer b.deinit();

        const small: f32 = 1e-3;
        @memset(a.data, small);
        @memset(b.data, small);

        var result = try sgemm.matmul(f32, a, b, allocator);
        defer result.deinit();

        // Check results
        for (result.data) |val| {
            try testing.expect(!std.math.isNan(val));
            try testing.expect(val > 0); // Should be positive
            // For 2x2 matrices filled with 1e-3, each element should be 2 * (1e-3 * 1e-3) = 2e-6
            try testing.expectApproxEqAbs(val, 2e-6, 1e-9);
        }
    }

    // Test case 3: Mixed magnitudes
    {
        var a = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        defer a.deinit();
        var b = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        defer b.deinit();

        // First row large, second row small
        a.data[0] = 1e3;
        a.data[1] = 1e3;
        a.data[2] = 1e-3;
        a.data[3] = 1e-3;

        b.data[0] = 1e-3;
        b.data[1] = 1e3;
        b.data[2] = 1e-3;
        b.data[3] = 1e3;

        var result = try sgemm.matmul(f32, a, b, allocator);
        defer result.deinit();

        // Check results
        for (result.data) |val| {
            try testing.expect(!std.math.isInf(val));
            try testing.expect(!std.math.isNan(val));
        }
    }
}

test "SGEMM Inplace basic functionality" {
    const allocator = testing.allocator;

    // Test case 1: Square matrices
    {
        var a = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        defer a.deinit();
        var b = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        defer b.deinit();

        a.data[0] = 1.0;
        a.data[1] = 2.0;
        a.data[2] = 3.0;
        a.data[3] = 4.0;

        b.data[0] = 5.0;
        b.data[1] = 6.0;
        b.data[2] = 7.0;
        b.data[3] = 8.0;

        var result = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        defer result.deinit();
        try sgemm_inplace.matmul(f32, a, b, &result, allocator, null);

        // Compare with known result
        try testing.expectApproxEqAbs(result.data[0], 19.0, 1e-6);
        try testing.expectApproxEqAbs(result.data[1], 22.0, 1e-6);
        try testing.expectApproxEqAbs(result.data[2], 43.0, 1e-6);
        try testing.expectApproxEqAbs(result.data[3], 50.0, 1e-6);
    }
}

test "SGEMM Inplace edge cases" {
    const allocator = testing.allocator;

    // Test case 1: 1x1 matrices
    {
        var a = try Tensor(f32).init(allocator, &[_]usize{ 1, 1 });
        defer a.deinit();
        var b = try Tensor(f32).init(allocator, &[_]usize{ 1, 1 });
        defer b.deinit();

        a.data[0] = 3.0;
        b.data[0] = 4.0;

        var result = try Tensor(f32).init(allocator, &[_]usize{ 1, 1 });
        defer result.deinit();
        try sgemm_inplace.matmul(f32, a, b, &result, allocator, null);

        try testing.expectApproxEqAbs(result.data[0], 12.0, 1e-6);
    }

    // Test case 2: Tall matrix × Wide matrix
    {
        var a = try Tensor(f32).init(allocator, &[_]usize{ 3, 1 });
        defer a.deinit();
        var b = try Tensor(f32).init(allocator, &[_]usize{ 1, 3 });
        defer b.deinit();

        a.data[0] = 1.0;
        a.data[1] = 2.0;
        a.data[2] = 3.0;

        b.data[0] = 4.0;
        b.data[1] = 5.0;
        b.data[2] = 6.0;

        var result = try Tensor(f32).init(allocator, &[_]usize{ 3, 3 });
        defer result.deinit();
        try sgemm_inplace.matmul(f32, a, b, &result, allocator, null);

        try testing.expectEqual(result.shape[0], @as(usize, 3));
        try testing.expectEqual(result.shape[1], @as(usize, 3));
    }

    // Test case 3: Zero matrices
    {
        var a = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        defer a.deinit();
        var b = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        defer b.deinit();

        @memset(a.data, 0);
        @memset(b.data, 0);

        var result = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        defer result.deinit();
        try sgemm_inplace.matmul(f32, a, b, &result, allocator, null);

        for (result.data) |val| {
            try testing.expectEqual(val, 0);
        }
    }
}

test "SGEMM Inplace error cases" {
    const allocator = testing.allocator;

    // Test case 1: Mismatched dimensions
    {
        var a = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
        defer a.deinit();
        var b = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        defer b.deinit();

        var result = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        defer result.deinit();

        try testing.expectError(error.ShapeMismatch, sgemm_inplace.matmul(f32, a, b, &result, allocator, null));
    }

    // Test case 2: Invalid dimensions
    {
        var a = try Tensor(f32).init(allocator, &[_]usize{2});
        defer a.deinit();
        var b = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        defer b.deinit();

        var result = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        defer result.deinit();

        try testing.expectError(error.InvalidDimensions, sgemm_inplace.matmul(f32, a, b, &result, allocator, null));
    }
}
