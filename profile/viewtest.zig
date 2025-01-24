const std = @import("std");
const testing = std.testing;
const expectEqual = testing.expectEqual;
const expectError = testing.expectError;
const Tensor = @import("tensor.zig").Tensor;
const TensorView = @import("tensor.zig").TensorView;

// Test utilities
fn expectEqualSlices(comptime T: type, expected: []const T, actual: []const T) !void {
    try testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |e, a| {
        try testing.expectEqual(e, a);
    }
}

test "TensorView - Basic Creation" {
    const allocator = testing.allocator;

    // Create a simple 2x3 tensor
    var tensor = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
    defer tensor.deinit();

    // Fill with test data
    for (tensor.data, 0..) |*val, i| {
        val.* = @floatFromInt(i);
    }

    // Create view
    var view = try tensor.asView();
    defer view.deinit();

    // Verify view properties
    try expectEqual(@as(usize, 2), view.shape[0]);
    try expectEqual(@as(usize, 3), view.shape[1]);
    try expectEqual(@as(usize, 3), view.strides[0]);
    try expectEqual(@as(usize, 1), view.strides[1]);
    try expectEqual(@as(usize, 0), view.offset);

    // Verify data access
    try expectEqual(@as(f32, 0.0), view.data[view.getIndex(&[_]usize{ 0, 0 })]);
    try expectEqual(@as(f32, 1.0), view.data[view.getIndex(&[_]usize{ 0, 1 })]);
    try expectEqual(@as(f32, 3.0), view.data[view.getIndex(&[_]usize{ 1, 0 })]);
}

test "TensorView - Different Data Types" {
    const allocator = testing.allocator;

    // Test with integers
    {
        var tensor_i32 = try Tensor(i32).init(allocator, &[_]usize{ 2, 2 });
        defer tensor_i32.deinit();
        tensor_i32.fill(42);

        var view = try tensor_i32.asView();
        defer view.deinit();

        try expectEqual(@as(i32, 42), view.data[view.getIndex(&[_]usize{ 0, 0 })]);
    }

    // Test with booleans
    {
        var tensor_bool = try Tensor(bool).init(allocator, &[_]usize{ 2, 2 });
        defer tensor_bool.deinit();
        tensor_bool.fill(true);

        var view = try tensor_bool.asView();
        defer view.deinit();

        try expectEqual(true, view.data[view.getIndex(&[_]usize{ 0, 0 })]);
    }
}

test "TensorView - Alignment Check" {
    const allocator = testing.allocator;

    var tensor = try Tensor(f64).init(allocator, &[_]usize{ 4, 4 });
    defer tensor.deinit();

    var view = try tensor.asView();
    defer view.deinit();

    // Check slice alignment
    try expectEqual(@as(u29, 32), @typeInfo(@TypeOf(view.data)).Pointer.alignment);

    // Check that the slice start address is actually aligned
    try expectEqual(@as(usize, 0), @intFromPtr(&view.data[0]) & 31);
}

test "TensorView - Chunk Operations" {
    const allocator = testing.allocator;

    // Create a 4x4 tensor
    var tensor = try Tensor(f32).init(allocator, &[_]usize{ 4, 4 });
    defer tensor.deinit();

    // Fill with sequential values
    for (tensor.data, 0..) |*val, i| {
        val.* = @floatFromInt(i);
    }

    // Get a view of the first two rows
    var view = try tensor.asView();
    defer view.deinit();

    var chunk_view = try view.getChunkView(0, 0, 2);
    defer chunk_view.deinit();

    // Verify chunk view properties
    try expectEqual(@as(usize, 2), chunk_view.shape[0]);
    try expectEqual(@as(usize, 4), chunk_view.shape[1]);
    try expectEqual(@as(usize, 0), chunk_view.offset);

    // Convert back to tensor and verify data
    var chunk_tensor = try chunk_view.toTensor();
    defer chunk_tensor.deinit();

    try expectEqual(@as(usize, 8), chunk_tensor.data.len);
    try expectEqual(@as(f32, 0.0), chunk_tensor.data[0]);
    try expectEqual(@as(f32, 7.0), chunk_tensor.data[7]);
}

test "TensorView - Error Cases" {
    const allocator = testing.allocator;

    var tensor = try Tensor(f32).init(allocator, &[_]usize{ 4, 4 });
    defer tensor.deinit();

    var view = try tensor.asView();
    defer view.deinit();

    // Invalid dimension
    try expectError(error.InvalidDimension, view.getChunkView(2, 0, 2));

    // Invalid number of chunks
    try expectError(error.InvalidNumChunks, view.getChunkView(0, 0, 0));
    try expectError(error.InvalidNumChunks, view.getChunkView(0, 0, 5));

    // Invalid chunk index
    try expectError(error.InvalidChunkIndex, view.getChunkView(0, 2, 2));

    // Uneven chunk size
    try expectError(error.UnevenChunkSize, view.getChunkView(0, 0, 3));
}

test "TensorView - Multiple Dimensions" {
    const allocator = testing.allocator;

    // Create a 2x3x4 tensor
    var tensor = try Tensor(f32).init(allocator, &[_]usize{ 2, 3, 4 });
    defer tensor.deinit();

    for (tensor.data, 0..) |*val, i| {
        val.* = @floatFromInt(i);
    }

    var view = try tensor.asView();
    defer view.deinit();

    // Get middle slice along dimension 1
    var chunk_view = try view.getChunkView(1, 1, 3);
    defer chunk_view.deinit();

    try expectEqual(@as(usize, 2), chunk_view.shape[0]);
    try expectEqual(@as(usize, 1), chunk_view.shape[1]);
    try expectEqual(@as(usize, 4), chunk_view.shape[2]);
}

test "TensorView - Fast Chunk Wrapper" {
    const allocator = testing.allocator;

    var tensor = try Tensor(f32).init(allocator, &[_]usize{ 4, 4 });
    defer tensor.deinit();

    for (tensor.data, 0..) |*val, i| {
        val.* = @floatFromInt(i);
    }

    // Use convenience wrapper
    var chunk = try tensor.getChunkFast(0, 1, 2);
    defer chunk.deinit();

    try expectEqual(@as(usize, 2), chunk.shape[0]);
    try expectEqual(@as(usize, 4), chunk.shape[1]);
    try expectEqual(@as(f32, 8.0), chunk.data[0]);
}

test "TensorView - View of View" {
    const allocator = testing.allocator;

    var tensor = try Tensor(f32).init(allocator, &[_]usize{ 8, 8 });
    defer tensor.deinit();

    for (tensor.data, 0..) |*val, i| {
        val.* = @floatFromInt(i);
    }

    // Create first view
    var view1 = try tensor.asView();
    defer view1.deinit();

    // Get first half
    var view2 = try view1.getChunkView(0, 0, 2);
    defer view2.deinit();

    // Get first quarter of first half
    var view3 = try view2.getChunkView(0, 0, 2);
    defer view3.deinit();

    try expectEqual(@as(usize, 2), view3.shape[0]);
    try expectEqual(@as(usize, 8), view3.shape[1]);

    // Convert back to tensor and verify
    var result = try view3.toTensor();
    defer result.deinit();

    try expectEqual(@as(f32, 0.0), result.data[0]);
    try expectEqual(@as(f32, 15.0), result.data[15]);
}

test "TensorView - Large Data Access Patterns" {
    const allocator = testing.allocator;

    // Create a large tensor to test memory access patterns
    var tensor = try Tensor(f32).init(allocator, &[_]usize{ 64, 64 });
    defer tensor.deinit();

    for (tensor.data, 0..) |*val, i| {
        val.* = @floatFromInt(i);
    }

    var view = try tensor.asView();
    defer view.deinit();

    // Test various access patterns
    const patterns = [_][2]usize{
        .{ 0, 0 }, // First element
        .{ 63, 63 }, // Last element
        .{ 32, 32 }, // Middle element
        .{ 0, 63 }, // Corner elements
        .{ 63, 0 },
    };

    for (patterns) |pattern| {
        const idx = view.getIndex(&pattern);
        const expected: f32 = @floatFromInt(pattern[0] * 64 + pattern[1]);
        try expectEqual(expected, view.data[idx]);
    }
}

test "TensorView - Memory Management" {
    const allocator = testing.allocator;

    // Track allocations
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    // Create and manipulate views in a nested scope
    {
        var tensor = try Tensor(f32).init(arena_allocator, &[_]usize{ 4, 4 });
        defer tensor.deinit();

        var view1 = try tensor.asView();
        defer view1.deinit();

        {
            var view2 = try view1.getChunkView(0, 0, 2);
            defer view2.deinit();

            _ = try view2.toTensor();
        }
    }
}

test "TensorView - Zero Size Dimensions" {
    const allocator = testing.allocator;

    // Test with empty dimensions
    var tensor = try Tensor(f32).init(allocator, &[_]usize{0});
    defer tensor.deinit();

    var view = try tensor.asView();
    defer view.deinit();

    try expectEqual(@as(usize, 0), view.shape[0]);
    try expectEqual(@as(usize, 0), view.data.len);
}

test "TensorView - Single Element Operations" {
    const allocator = testing.allocator;

    // Create 1x1 tensor
    var tensor = try Tensor(f32).init(allocator, &[_]usize{ 1, 1 });
    defer tensor.deinit();
    tensor.data[0] = 42.0;

    var view = try tensor.asView();
    defer view.deinit();

    try expectEqual(@as(f32, 42.0), view.data[view.getIndex(&[_]usize{ 0, 0 })]);

    // Try to create chunk (should fail)
    try expectError(error.InvalidNumChunks, view.getChunkView(0, 0, 2));
}

test "TensorView - Data Consistency" {
    const allocator = testing.allocator;

    var tensor = try Tensor(f32).init(allocator, &[_]usize{ 4, 4 });
    defer tensor.deinit();

    for (tensor.data, 0..) |*val, i| {
        val.* = @floatFromInt(i);
    }

    // Create multiple views and verify they all see the same data
    var view1 = try tensor.asView();
    defer view1.deinit();

    var view2 = try tensor.asView();
    defer view2.deinit();

    // Modify through one view
    const idx = view1.getIndex(&[_]usize{ 2, 2 });
    view1.data[idx] = 99.0;

    // Verify through other view
    try expectEqual(@as(f32, 99.0), view2.data[view2.getIndex(&[_]usize{ 2, 2 })]);
    // Verify through original tensor
    try expectEqual(@as(f32, 99.0), tensor.data[10]); // 2*4 + 2 = 10
}
