const std = @import("std");
const testing = std.testing;
const allocator = testing.allocator;
const Tensor = @import("tensor.zig").Tensor;
const transposeF16SIMD = @import("ops.zig").transposeF16SIMD;

// Helper function to create a test tensor with known values
fn createTestTensor(batch: usize, rows: usize, cols: usize) !Tensor(f16) {
    var tensor = try Tensor(f16).init(allocator, &[_]usize{ batch, rows, cols });
    errdefer tensor.deinit();

    // Fill with ascending values converted to f16
    for (0..batch) |b| {
        for (0..rows) |r| {
            for (0..cols) |c| {
                const idx = b * rows * cols + r * cols + c;
                tensor.data[idx] = @floatCast(@as(f32, @floatFromInt(idx)) + 0.5);
            }
        }
    }
    return tensor;
}

fn verifyTranspose(original: *const Tensor(f16), transposed: []const f16, batch: usize, rows: usize, cols: usize) !void {
    for (0..rows) |r| {
        for (0..batch) |b| {
            for (0..cols) |c| {
                const orig_idx = b * rows * cols + r * cols + c;
                const trans_idx = r * batch * cols + b * cols + c;
                const orig_val = original.data[orig_idx];
                const trans_val = transposed[trans_idx];

                // Handle special values separately
                if (std.math.isNan(orig_val)) {
                    try testing.expect(std.math.isNan(trans_val));
                    continue;
                }
                if (std.math.isPositiveInf(orig_val)) {
                    try testing.expect(std.math.isPositiveInf(trans_val));
                    continue;
                }
                if (std.math.isNegativeInf(orig_val)) {
                    try testing.expect(std.math.isNegativeInf(trans_val));
                    continue;
                }
                if (orig_val == 0.0) {
                    try testing.expect(trans_val == 0.0);
                    continue;
                }

                // For normal values, use relative error
                const epsilon: f16 = 0.001;
                const abs_diff = @abs(orig_val - trans_val);
                const rel_diff = abs_diff / @abs(orig_val);
                try testing.expect(rel_diff < epsilon);
            }
        }
    }
}

// Basic test with small dimensions
test "transposeF16SIMD - small dimensions" {
    const batch = 2;
    const rows = 3;
    const cols = 4;

    var tensor = try createTestTensor(batch, rows, cols);
    defer tensor.deinit();

    const new_data = try allocator.alignedAlloc(f16, 32, batch * rows * cols);
    defer allocator.free(new_data);

    transposeF16SIMD(&tensor, batch, rows, cols, new_data);
    try verifyTranspose(&tensor, new_data, batch, rows, cols);
}

// Test with dimensions multiple of vector size
test "transposeF16SIMD - aligned dimensions" {
    const batch = 4;
    const rows = 8;
    const cols = 16; // Multiple of vector size (8)

    var tensor = try createTestTensor(batch, rows, cols);
    defer tensor.deinit();

    const new_data = try allocator.alignedAlloc(f16, 32, batch * rows * cols);
    defer allocator.free(new_data);

    transposeF16SIMD(&tensor, batch, rows, cols, new_data);
    try verifyTranspose(&tensor, new_data, batch, rows, cols);
}

// Test with dimensions not multiple of vector size
test "transposeF16SIMD - unaligned dimensions" {
    const batch = 3;
    const rows = 5;
    const cols = 11; // Not multiple of vector size

    var tensor = try createTestTensor(batch, rows, cols);
    defer tensor.deinit();

    const new_data = try allocator.alignedAlloc(f16, 32, batch * rows * cols);
    defer allocator.free(new_data);

    transposeF16SIMD(&tensor, batch, rows, cols, new_data);
    try verifyTranspose(&tensor, new_data, batch, rows, cols);
}

// Test with large dimensions
test "transposeF16SIMD - large dimensions" {
    const batch = 16;
    const rows = 32;
    const cols = 64;

    var tensor = try createTestTensor(batch, rows, cols);
    defer tensor.deinit();

    const new_data = try allocator.alignedAlloc(f16, 32, batch * rows * cols);
    defer allocator.free(new_data);

    transposeF16SIMD(&tensor, batch, rows, cols, new_data);
    try verifyTranspose(&tensor, new_data, batch, rows, cols);
}

// Test edge case with single row/column
test "transposeF16SIMD - edge cases" {
    // Test 1: Single row
    {
        const batch = 2;
        const rows = 1;
        const cols = 8;

        var tensor = try createTestTensor(batch, rows, cols);
        defer tensor.deinit();

        const new_data = try allocator.alignedAlloc(f16, 32, batch * rows * cols);
        defer allocator.free(new_data);

        transposeF16SIMD(&tensor, batch, rows, cols, new_data);
        try verifyTranspose(&tensor, new_data, batch, rows, cols);
    }

    // Test 2: Single column
    {
        const batch = 2;
        const rows = 8;
        const cols = 1;

        var tensor = try createTestTensor(batch, rows, cols);
        defer tensor.deinit();

        const new_data = try allocator.alignedAlloc(f16, 32, batch * rows * cols);
        defer allocator.free(new_data);

        transposeF16SIMD(&tensor, batch, rows, cols, new_data);
        try verifyTranspose(&tensor, new_data, batch, rows, cols);
    }
}

// Performance comparison test
test "transposeF16SIMD - performance" {
    const batch = 32;
    const rows = 64;
    const cols = 128;

    var tensor = try createTestTensor(batch, rows, cols);
    defer tensor.deinit();

    const new_data = try allocator.alignedAlloc(f16, 32, batch * rows * cols);
    defer allocator.free(new_data);

    // Measure SIMD version time
    var timer = try std.time.Timer.start();
    const iterations = 100;

    var i: usize = 0;
    while (i < iterations) : (i += 1) {
        transposeF16SIMD(&tensor, batch, rows, cols, new_data);
    }

    const elapsed_ns = timer.lap();
    const avg_time_ms = @as(f64, @floatFromInt(elapsed_ns)) / (@as(f64, @floatFromInt(iterations)) * 1_000_000.0);

    // Log performance metrics
    std.debug.print("\nPerformance test results:\n", .{});
    std.debug.print("Average time per transpose: {d:.3} ms\n", .{avg_time_ms});
    std.debug.print("Tensor size: {d}x{d}x{d} ({d} elements)\n", .{ batch, rows, cols, batch * rows * cols });
}

// Helper function to implement naive transpose for comparison
fn naiveTranspose(tensor: *const Tensor(f16), batch: usize, rows: usize, cols: usize, new_data: []align(32) f16) void {
    for (0..rows) |r| {
        for (0..batch) |b| {
            for (0..cols) |c| {
                const src_idx = b * rows * cols + r * cols + c;
                const dst_idx = r * batch * cols + b * cols + c;
                new_data[dst_idx] = tensor.data[src_idx];
            }
        }
    }
}

// Correctness comparison test
test "transposeF16SIMD - correctness against naive" {
    const batch = 4;
    const rows = 8;
    const cols = 16;

    var tensor = try createTestTensor(batch, rows, cols);
    defer tensor.deinit();

    const simd_result = try allocator.alignedAlloc(f16, 32, batch * rows * cols);
    defer allocator.free(simd_result);

    const naive_result = try allocator.alignedAlloc(f16, 32, batch * rows * cols);
    defer allocator.free(naive_result);

    // Perform both transposes
    transposeF16SIMD(&tensor, batch, rows, cols, simd_result);
    naiveTranspose(&tensor, batch, rows, cols, naive_result);

    // Compare results
    for (simd_result, naive_result) |simd_val, naive_val| {
        const epsilon: f16 = 0.001;
        try testing.expect(@abs(simd_val - naive_val) < epsilon);
    }
}
