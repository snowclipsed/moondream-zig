const std = @import("std");
const testing = std.testing;
const expectEqual = testing.expectEqual;
const expectError = testing.expectError;
const Tensor = @import("tensor.zig").Tensor;

test "TensorView - Basic Reshape Same Dimensions" {
    const allocator = testing.allocator;

    // Create a 2x3 tensor
    var tensor = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
    defer tensor.deinit();

    // Fill with sequential data
    for (tensor.data, 0..) |*val, i| {
        val.* = @floatFromInt(i);
    }

    // Create view and reshape to 3x2
    var view = try tensor.asView();
    defer view.deinit();

    try view.reshape(&[_]usize{ 3, 2 });

    // Verify shape
    try expectEqual(@as(usize, 3), view.shape[0]);
    try expectEqual(@as(usize, 2), view.shape[1]);

    // Verify strides
    try expectEqual(@as(usize, 2), view.strides[0]);
    try expectEqual(@as(usize, 1), view.strides[1]);

    // Verify data access through new shape
    try expectEqual(@as(f32, 0.0), view.data[view.getDataIndex(&[_]usize{ 0, 0 })]);
    try expectEqual(@as(f32, 1.0), view.data[view.getDataIndex(&[_]usize{ 0, 1 })]);
    try expectEqual(@as(f32, 2.0), view.data[view.getDataIndex(&[_]usize{ 1, 0 })]);
}

test "TensorView - Reshape Different Number of Dimensions" {
    const allocator = testing.allocator;

    // Create a 2x3 tensor
    var tensor = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
    defer tensor.deinit();

    for (tensor.data, 0..) |*val, i| {
        val.* = @floatFromInt(i);
    }

    var view = try tensor.asView();
    defer view.deinit();

    // Reshape to 1D
    try view.reshape(&[_]usize{6});

    try expectEqual(@as(usize, 1), view.shape.len);
    try expectEqual(@as(usize, 6), view.shape[0]);
    try expectEqual(@as(usize, 1), view.strides[0]);

    // Verify sequential access
    for (0..6) |i| {
        const val = view.data[view.getDataIndex(&[_]usize{i})];
        try expectEqual(@as(f32, @floatFromInt(i)), val);
    }

    // Reshape to 3D
    try view.reshape(&[_]usize{ 1, 2, 3 });

    try expectEqual(@as(usize, 3), view.shape.len);
    try expectEqual(@as(usize, 1), view.shape[0]);
    try expectEqual(@as(usize, 2), view.shape[1]);
    try expectEqual(@as(usize, 3), view.shape[2]);
}

test "TensorView - Reshape Error Cases" {
    const allocator = testing.allocator;

    var tensor = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
    defer tensor.deinit();

    var view = try tensor.asView();
    defer view.deinit();

    // Try invalid shapes
    try expectError(error.IncompatibleShape, view.reshape(&[_]usize{ 2, 4 })); // Wrong total size
    try expectError(error.IncompatibleShape, view.reshape(&[_]usize{7})); // Wrong total size
    try expectError(error.IncompatibleShape, view.reshape(&[_]usize{ 2, 2, 2 })); // Wrong total size
}

test "TensorView - Reshape Zero Dimensions" {
    const allocator = testing.allocator;

    // Create a scalar tensor (0D)
    var tensor = try Tensor(f32).init(allocator, &[_]usize{});
    defer tensor.deinit();
    tensor.data[0] = 42.0;

    var view = try tensor.asView();
    defer view.deinit();

    // Reshape to 1x1
    try view.reshape(&[_]usize{ 1, 1 });

    try expectEqual(@as(usize, 2), view.shape.len);
    try expectEqual(@as(usize, 1), view.shape[0]);
    try expectEqual(@as(usize, 1), view.shape[1]);
    try expectEqual(@as(f32, 42.0), view.data[view.getDataIndex(&[_]usize{ 0, 0 })]);
}

test "TensorView - Reshape Memory Management" {
    const allocator = testing.allocator;

    var tensor = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
    defer tensor.deinit();

    var view = try tensor.asView();
    defer view.deinit();

    // Multiple reshapes to test memory handling
    try view.reshape(&[_]usize{6});
    try view.reshape(&[_]usize{ 2, 3 });
    try view.reshape(&[_]usize{ 1, 2, 3 });
    try view.reshape(&[_]usize{ 3, 2 });

    // Verify final state
    try expectEqual(@as(usize, 2), view.shape.len);
    try expectEqual(@as(usize, 3), view.shape[0]);
    try expectEqual(@as(usize, 2), view.shape[1]);
}

test "TensorView - Reshape Large Dimensions" {
    const allocator = testing.allocator;

    // Create a larger tensor to test with more data
    var tensor = try Tensor(f32).init(allocator, &[_]usize{ 100, 100 });
    defer tensor.deinit();

    for (tensor.data, 0..) |*val, i| {
        val.* = @floatFromInt(i);
    }

    var view = try tensor.asView();
    defer view.deinit();

    // Reshape to various sizes
    try view.reshape(&[_]usize{10000});
    try view.reshape(&[_]usize{ 50, 200 });
    try view.reshape(&[_]usize{ 25, 40, 10 });

    // Verify data integrity after multiple reshapes
    var sum: f32 = 0;
    var expected_sum: f32 = 0;
    const total_elements = 10000;

    // Sum through view
    var coords = try allocator.alloc(usize, view.shape.len);
    defer allocator.free(coords);
    @memset(coords, 0);

    var i: usize = 0;
    while (i < total_elements) : (i += 1) {
        sum += view.data[view.getDataIndex(coords)];

        // Update coordinates
        var dim = coords.len;
        while (dim > 0) {
            dim -= 1;
            coords[dim] += 1;
            if (coords[dim] < view.shape[dim]) break;
            coords[dim] = 0;
        }
    }

    // Calculate expected sum
    for (0..total_elements) |j| {
        expected_sum += @floatFromInt(j);
    }

    try expectEqual(expected_sum, sum);
}

test "TensorView - Reshape with Offset Views" {
    const allocator = testing.allocator;

    // Create a 4x4 tensor
    var tensor = try Tensor(f32).init(allocator, &[_]usize{ 4, 4 });
    defer tensor.deinit();

    for (tensor.data, 0..) |*val, i| {
        val.* = @floatFromInt(i);
    }

    // Create a view and get a chunk
    var main_view = try tensor.asView();
    defer main_view.deinit();

    var chunk_view = try main_view.getChunkView(0, 1, 2); // Get second quarter
    defer chunk_view.deinit();

    // Reshape the chunk view
    try chunk_view.reshape(&[_]usize{ 1, 8 });

    // Verify that offset is preserved and data access is correct
    try expectEqual(@as(usize, 8), chunk_view.offset);

    for (0..8) |i| {
        const val = chunk_view.data[chunk_view.getDataIndex(&[_]usize{ 0, i })];
        try expectEqual(@as(f32, @floatFromInt(i + 8)), val);
    }
}

test "TensorView - Reshape Data Access Patterns" {
    const allocator = testing.allocator;

    // Create a 3x4 tensor
    var tensor = try Tensor(f32).init(allocator, &[_]usize{ 3, 4 });
    defer tensor.deinit();

    for (tensor.data, 0..) |*val, i| {
        val.* = @floatFromInt(i);
    }

    var view = try tensor.asView();
    defer view.deinit();

    // Test different access patterns after reshape
    try view.reshape(&[_]usize{ 4, 3 });

    // Test row-major access
    var row_sum: f32 = 0;
    for (0..4) |i| {
        for (0..3) |j| {
            row_sum += view.data[view.getDataIndex(&[_]usize{ i, j })];
        }
    }

    // Test column-major access
    var col_sum: f32 = 0;
    for (0..3) |j| {
        for (0..4) |i| {
            col_sum += view.data[view.getDataIndex(&[_]usize{ i, j })];
        }
    }

    // Both access patterns should yield the same sum
    try expectEqual(row_sum, col_sum);

    // Calculate expected sum
    var expected_sum: f32 = 0;
    for (0..12) |i| {
        expected_sum += @floatFromInt(i);
    }

    try expectEqual(expected_sum, row_sum);
}
