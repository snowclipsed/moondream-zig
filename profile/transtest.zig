const std = @import("std");
const testing = std.testing;
const Tensor = @import("tensor.zig").Tensor;
const matmul = @import("hgemmtrans.zig").matmul;
const assert = std.debug.assert;

// Helper function to transpose a matrix
fn transpose(allocator: std.mem.Allocator, tensor: Tensor(f16)) !Tensor(f16) {
    const shape = tensor.shape;
    var result = try Tensor(f16).init(allocator, &[_]usize{ shape[1], shape[0] });
    errdefer result.deinit();

    const src = tensor.getSlice();
    const dst = result.getSlice();

    for (0..shape[0]) |i| {
        for (0..shape[1]) |j| {
            dst[j * shape[0] + i] = src[i * shape[1] + j];
        }
    }
    return result;
}

// Helper function to compare f16 values with tolerance
fn approxEqf16(a: f16, b: f16, tolerance: f32) bool {
    const a_f32: f32 = @floatCast(a);
    const b_f32: f32 = @floatCast(b);
    return @abs(a_f32 - b_f32) <= tolerance;
}

// Helper function to compare tensors with tolerance
fn tensorApproxEq(a: Tensor(f16), b: Tensor(f16), tolerance: f32) !bool {
    const a_data = a.getSlice();
    const b_data = b.getSlice();

    if (a_data.len != b_data.len) return false;
    if (!std.mem.eql(usize, a.shape, b.shape)) return false;

    for (a_data, b_data) |a_val, b_val| {
        if (!approxEqf16(a_val, b_val, tolerance)) {
            std.debug.print("Mismatch: {d} vs {d}\n", .{ @as(f32, @floatCast(a_val)), @as(f32, @floatCast(b_val)) });
            return false;
        }
    }
    return true;
}

// Helper function to create and fill a tensor with test data
fn createTestTensor(allocator: std.mem.Allocator, shape: []const usize, pattern: enum { Sequential, Random }) !Tensor(f16) {
    var result = try Tensor(f16).init(allocator, shape);
    errdefer result.deinit();

    const data = result.getSlice();
    var rng = std.rand.DefaultPrng.init(0);

    switch (pattern) {
        .Sequential => {
            for (data, 0..) |*val, i| {
                val.* = @floatCast(@as(f32, @floatFromInt(i)) * 0.1);
            }
        },
        .Random => {
            for (data) |*val| {
                val.* = @floatCast(rng.random().float(f32) * 2.0 - 1.0);
            }
        },
    }
    return result;
}

// Reference implementation for validation
fn naiveMatmul(allocator: std.mem.Allocator, a: Tensor(f16), b: Tensor(f16)) !Tensor(f16) {
    const M = a.shape[0];
    const K = a.shape[1];
    const N = b.shape[1];

    var result = try Tensor(f16).init(allocator, &[_]usize{ M, N });
    errdefer result.deinit();

    const a_data = a.getSlice();
    const b_data = b.getSlice();
    const c_data = result.getSlice();

    for (0..M) |i| {
        for (0..N) |j| {
            var sum: f32 = 0;
            for (0..K) |k| {
                const a_val: f32 = @floatCast(a_data[i * K + k]);
                const b_val: f32 = @floatCast(b_data[k * N + j]);
                sum += a_val * b_val;
            }
            c_data[i * N + j] = @floatCast(sum);
        }
    }
    return result;
}

test "vector-matrix multiplication 1x4 * 4x3" {
    const allocator = testing.allocator;

    var a = try Tensor(f16).init(allocator, &[_]usize{ 1, 4 });
    defer a.deinit();
    var b = try Tensor(f16).init(allocator, &[_]usize{ 4, 3 });
    defer b.deinit();

    // Set test values
    const a_data = a.getSlice();
    const b_data = b.getSlice();

    a_data[0] = @floatCast(@as(f32, 1.0));
    a_data[1] = @floatCast(@as(f32, 2.0));
    a_data[2] = @floatCast(@as(f32, 3.0));
    a_data[3] = @floatCast(@as(f32, 4.0));

    for (b_data, 0..) |*val, i| {
        val.* = @floatCast(@as(f32, @floatFromInt(i)) + 1.0);
    }

    // Transpose B
    var b_t = try transpose(allocator, b);
    defer b_t.deinit();

    // Compute result using our optimized implementation
    var result = try matmul(a, b_t, allocator);
    defer result.deinit();

    // Compute reference result
    var expected = try naiveMatmul(allocator, a, b);
    defer expected.deinit();

    // Compare results
    try testing.expect(try tensorApproxEq(result, expected, 0.01));
}

test "matrix multiplication with different sizes" {
    const test_sizes = [_]struct { m: usize, k: usize, n: usize }{
        .{ .m = 1, .k = 160, .n = 160 }, // Vector-matrix case
        .{ .m = 16, .k = 16, .n = 16 }, // Square matrices
        .{ .m = 32, .k = 16, .n = 8 }, // Rectangular matrices
        .{ .m = 7, .k = 13, .n = 11 }, // Prime-sized matrices
        .{ .m = 160, .k = 160, .n = 160 }, // Exactly one tile
        .{ .m = 200, .k = 160, .n = 120 }, // Multiple tiles
    };

    const allocator = testing.allocator;

    for (test_sizes) |size| {
        // Create test matrices
        var a = try createTestTensor(allocator, &[_]usize{ size.m, size.k }, .Random);
        defer a.deinit();
        var b = try createTestTensor(allocator, &[_]usize{ size.k, size.n }, .Random);
        defer b.deinit();

        // Transpose B
        var b_t = try transpose(allocator, b);
        defer b_t.deinit();

        // Compute optimized result
        var result = try matmul(a, b_t, allocator);
        defer result.deinit();

        // Compute reference result
        var expected = try naiveMatmul(allocator, a, b);
        defer expected.deinit();

        // Compare results with tolerance
        try testing.expect(try tensorApproxEq(result, expected, 0.01));
    }
}

test "edge cases" {
    const allocator = testing.allocator;

    // Test 1x1 matrices
    {
        var a = try createTestTensor(allocator, &[_]usize{ 1, 1 }, .Sequential);
        defer a.deinit();
        var b = try createTestTensor(allocator, &[_]usize{ 1, 1 }, .Sequential);
        defer b.deinit();

        var b_t = try transpose(allocator, b);
        defer b_t.deinit();

        var result = try matmul(a, b_t, allocator);
        defer result.deinit();

        var expected = try naiveMatmul(allocator, a, b);
        defer expected.deinit();

        try testing.expect(try tensorApproxEq(result, expected, 0.01));
    }

    // Test matrices with one dimension exactly equal to SIMD width (8)
    {
        var a = try createTestTensor(allocator, &[_]usize{ 8, 16 }, .Random);
        defer a.deinit();
        var b = try createTestTensor(allocator, &[_]usize{ 16, 24 }, .Random);
        defer b.deinit();

        var b_t = try transpose(allocator, b);
        defer b_t.deinit();

        var result = try matmul(a, b_t, allocator);
        defer result.deinit();

        var expected = try naiveMatmul(allocator, a, b);
        defer expected.deinit();

        try testing.expect(try tensorApproxEq(result, expected, 0.01));
    }
}

test "numerical stability with large values" {
    const allocator = testing.allocator;

    var a = try Tensor(f16).init(allocator, &[_]usize{ 4, 4 });
    defer a.deinit();
    var b = try Tensor(f16).init(allocator, &[_]usize{ 4, 4 });
    defer b.deinit();

    // Fill with large values that could cause overflow in naive implementation
    const a_data = a.getSlice();
    const b_data = b.getSlice();

    for (a_data) |*val| {
        val.* = @floatCast(@as(f32, 100.0));
    }
    for (b_data) |*val| {
        val.* = @floatCast(@as(f32, 100.0));
    }

    var b_t = try transpose(allocator, b);
    defer b_t.deinit();

    var result = try matmul(a, b_t, allocator);
    defer result.deinit();

    // Each element should be 4 * 100 * 100 = 40000
    const expected_val: f16 = @floatCast(@as(f32, 40000.0));
    const result_data = result.getSlice();

    for (result_data) |val| {
        try testing.expect(approxEqf16(val, expected_val, 100.0)); // Larger tolerance for f16 precision
    }
}

test "error cases" {
    const allocator = testing.allocator;

    // Test incompatible dimensions
    {
        var a = try createTestTensor(allocator, &[_]usize{ 2, 3 }, .Sequential);
        defer a.deinit();
        var b = try createTestTensor(allocator, &[_]usize{ 4, 2 }, .Sequential);
        defer b.deinit();

        var b_t = try transpose(allocator, b);
        defer b_t.deinit();

        try testing.expectError(error.IncompatibleTensorShapes, matmul(a, b_t, allocator));
    }

    // Test invalid dimensions (1D tensor)
    {
        var a = try Tensor(f16).init(allocator, &[_]usize{3});
        defer a.deinit();
        var b = try createTestTensor(allocator, &[_]usize{ 3, 2 }, .Sequential);
        defer b.deinit();

        var b_t = try transpose(allocator, b);
        defer b_t.deinit();

        try testing.expectError(error.InvalidTensorDimension, matmul(a, b_t, allocator));
    }
}
