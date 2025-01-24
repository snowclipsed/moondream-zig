const std = @import("std");
const testing = std.testing;
const expectEqual = testing.expectEqual;
const transposeAxes = @import("ops.zig").transposeAxes;
const Tensor = @import("tensor.zig").Tensor;
const transposeF16SIMD = @import("ops.zig").transposeF16SIMD;
const Allocator = std.mem.Allocator;
// Keep a copy of the original implementation for comparison
fn transposeAxesOriginal(comptime T: type, tensor: *Tensor(T), dim0: usize, dim1: usize) !void {
    if (dim0 >= tensor.shape.len or dim1 >= tensor.shape.len) {
        return error.InvalidDimension;
    }

    // Calculate strides for the current shape
    var strides = try tensor.allocator.alloc(usize, tensor.shape.len);
    defer tensor.allocator.free(strides);

    strides[tensor.shape.len - 1] = 1;
    var i: usize = tensor.shape.len - 1;
    while (i > 0) : (i -= 1) {
        strides[i - 1] = strides[i] * tensor.shape[i];
    }

    // Create new shape with swapped dimensions
    var new_shape = try tensor.allocator.alloc(usize, tensor.shape.len);
    errdefer tensor.allocator.free(new_shape);

    for (tensor.shape, 0..) |dim, idx| {
        if (idx == dim0) {
            new_shape[idx] = tensor.shape[dim1];
        } else if (idx == dim1) {
            new_shape[idx] = tensor.shape[dim0];
        } else {
            new_shape[idx] = dim;
        }
    }

    // Allocate memory for transposed data
    var new_data = try tensor.allocator.alignedAlloc(T, 32, tensor.data.len);
    errdefer tensor.allocator.free(new_data);

    // Calculate new strides
    var new_strides = try tensor.allocator.alloc(usize, tensor.shape.len);
    defer tensor.allocator.free(new_strides);

    new_strides[tensor.shape.len - 1] = 1;
    i = tensor.shape.len - 1;
    while (i > 0) : (i -= 1) {
        new_strides[i - 1] = new_strides[i] * new_shape[i];
    }

    // Create coordinate arrays
    var coords = try tensor.allocator.alloc(usize, tensor.shape.len);
    defer tensor.allocator.free(coords);
    @memset(coords, 0);

    // Perform the transpose operation
    const total_elements = tensor.data.len;
    var idx: usize = 0;
    while (idx < total_elements) : (idx += 1) {
        // Calculate source coordinates
        var remaining = idx;
        for (0..tensor.shape.len) |dim| {
            coords[dim] = remaining / new_strides[dim];
            remaining = remaining % new_strides[dim];
        }

        // Swap coordinates for the transposed dimensions
        const temp = coords[dim0];
        coords[dim0] = coords[dim1];
        coords[dim1] = temp;

        // Calculate source index using original strides
        var src_idx: usize = 0;
        for (0..tensor.shape.len) |dim| {
            src_idx += coords[dim] * strides[dim];
        }

        new_data[idx] = tensor.data[src_idx];
    }

    // Update tensor with new data and shape
    tensor.allocator.free(tensor.data);
    tensor.data = new_data;
    tensor.allocator.free(tensor.shape);
    tensor.shape = new_shape;
}

fn verifyTranspose(allocator: std.mem.Allocator, shape: []const usize, dim0: usize, dim1: usize) !void {
    // Create two identical tensors
    var tensor1 = try Tensor(f16).init(allocator, shape);
    defer tensor1.deinit();
    var tensor2 = try Tensor(f16).init(allocator, shape);
    defer tensor2.deinit();

    // Fill with sequential values for easy verification
    for (tensor1.data, 0..) |*val, i| {
        val.* = @floatFromInt(i);
        tensor2.data[i] = @floatFromInt(i);
    }

    // Run both implementations
    try transposeAxesOriginal(f16, &tensor1, dim0, dim1);
    try transposeAxes(f16, &tensor2, dim0, dim1); // Our new SIMD version

    // Verify shapes match
    try expectEqual(tensor1.shape.len, tensor2.shape.len);
    for (tensor1.shape, tensor2.shape) |s1, s2| {
        try expectEqual(s1, s2);
    }

    // Verify all data matches
    var mismatch_found = false;
    for (tensor1.data, tensor2.data, 0..) |v1, v2, i| {
        if (@abs(v1 - v2) > 0.0001) { // Small epsilon for f16 comparison
            if (!mismatch_found) {
                std.debug.print("\nMismatch found at index {}!\n", .{i});
                mismatch_found = true;
            }
            std.debug.print("Value at {}: Original={} New={}\n", .{ i, v1, v2 });
        }
    }

    if (mismatch_found) {
        std.debug.print("\nOriginal tensor:\n", .{});
        printTensorDebug(&tensor1);
        std.debug.print("\nNew tensor:\n", .{});
        printTensorDebug(&tensor2);
        return error.TransposeMismatch;
    }
}

fn printTensorDebug(tensor: *const Tensor(f16)) void {
    std.debug.print("Shape: ", .{});
    for (tensor.shape) |s| {
        std.debug.print("{} ", .{s});
    }
    std.debug.print("\nData: \n", .{});

    if (tensor.shape.len == 3) {
        const batch = tensor.shape[0];
        const rows = tensor.shape[1];
        const cols = tensor.shape[2];

        for (0..batch) |b| {
            std.debug.print("Batch {}:\n", .{b});
            for (0..rows) |i| {
                for (0..cols) |j| {
                    const idx = b * rows * cols + i * cols + j;
                    std.debug.print("{d:6.2} ", .{tensor.data[idx]});
                }
                std.debug.print("\n", .{});
            }
            std.debug.print("\n", .{});
        }
    }
}

test "Transpose Verification - Simple Case" {
    const allocator = testing.allocator;
    try verifyTranspose(allocator, &[_]usize{ 2, 3, 4 }, 0, 1);
}

test "Transpose Verification - Square Matrices" {
    const allocator = testing.allocator;
    try verifyTranspose(allocator, &[_]usize{ 2, 8, 8 }, 0, 1);
}

test "Transpose Verification - Irregular Shapes" {
    const allocator = testing.allocator;
    try verifyTranspose(allocator, &[_]usize{ 3, 5, 7 }, 0, 1);
}

test "Transpose Verification - Edge Cases" {
    const allocator = testing.allocator;
    // Test various edge cases
    try verifyTranspose(allocator, &[_]usize{ 1, 8, 8 }, 0, 1); // Single batch
    try verifyTranspose(allocator, &[_]usize{ 2, 1, 8 }, 0, 1); // Single row
    try verifyTranspose(allocator, &[_]usize{ 2, 8, 1 }, 0, 1); // Single column
    try verifyTranspose(allocator, &[_]usize{ 2, 3, 16 }, 0, 1); // Larger than block size
}

const print = std.debug.print;

pub fn benchmark() !void {
    const allocator = std.heap.page_allocator;
    const iterations = 100;

    // Test shapes (batch, rows, cols)
    const shapes = [_][3]usize{
        .{ 2, 768, 64 }, // Small typical attention shape
        .{ 8, 1024, 128 }, // Medium size
        .{ 16, 2048, 256 }, // Large size
        .{ 1, 782, 2048 },
        .{ 1, 2048, 782 },
        .{ 1, 1024, 1024 },
        .{ 1, 1024, 512 },
        .{ 1, 512, 1024 },
        .{ 1, 2048, 1 },
        .{ 1, 1, 2048 },
        .{ 1, 1, 1 },
        .{ 1, 2048, 2048 },
    };

    print("\n=== Transpose Benchmarks ===\n\n", .{});

    // FP16 benchmarks with new SIMD implementation
    print("FP16 Transpose Benchmarks (including new SIMD):\n", .{});
    for (shapes) |shape| {
        try benchmarkShapeWithNewSIMD(shape, iterations, allocator);
    }

    // Original benchmarks
    print("\nOriginal FP16 Transpose Benchmarks:\n", .{});
    for (shapes) |shape| {
        try benchmarkShape(f16, shape, iterations, allocator);
    }

    print("\nFP32 Transpose Benchmarks:\n", .{});
    for (shapes) |shape| {
        try benchmarkShape(f32, shape, iterations, allocator);
    }

    print("\nFP16->FP32->FP16 Transpose Benchmarks:\n", .{});
    for (shapes) |shape| {
        try benchmarkCastShape(shape, iterations, allocator);
    }
}
fn benchmarkShape(comptime T: type, shape: [3]usize, iterations: usize, allocator: std.mem.Allocator) !void {
    var total_time_orig: u64 = 0;
    var total_time_simd: u64 = 0;
    var timer = try std.time.Timer.start();

    print("\nShape: {}x{}x{}\n", .{ shape[0], shape[1], shape[2] });

    // Run original implementation
    {
        // Warmup
        var warmup = try Tensor(T).init(allocator, &shape);
        defer warmup.deinit();
        try fillSequential(T, &warmup);
        try transposeAxesOriginal(T, &warmup, 0, 1);

        // Actual benchmark
        for (0..iterations) |_| {
            var tensor = try Tensor(T).init(allocator, &shape);
            defer tensor.deinit();
            try fillSequential(T, &tensor);

            timer.reset();
            try transposeAxesOriginal(T, &tensor, 0, 1);
            total_time_orig += timer.read();
        }
    }

    // Run SIMD implementation
    {
        // Warmup
        var warmup = try Tensor(T).init(allocator, &shape);
        defer warmup.deinit();
        try fillSequential(T, &warmup);
        try transposeAxes(T, &warmup, 0, 1);

        // Actual benchmark
        for (0..iterations) |_| {
            var tensor = try Tensor(T).init(allocator, &shape);
            defer tensor.deinit();
            try fillSequential(T, &tensor);

            timer.reset();
            try transposeAxes(T, &tensor, 0, 1);
            total_time_simd += timer.read();
        }
    }

    const avg_time_orig = @as(f64, @floatFromInt(total_time_orig)) / @as(f64, @floatFromInt(iterations));
    const avg_time_simd = @as(f64, @floatFromInt(total_time_simd)) / @as(f64, @floatFromInt(iterations));
    const speedup = avg_time_orig / avg_time_simd;

    print("Original Implementation: {d:.2}ms\n", .{avg_time_orig / 1_000_000.0});
    print("SIMD Implementation:     {d:.2}ms\n", .{avg_time_simd / 1_000_000.0});
    print("Speedup:                {d:.2}x\n", .{speedup});

    // Calculate throughput
    const elements = shape[0] * shape[1] * shape[2];
    const size_bytes = elements * @sizeOf(T);
    const throughput_orig = @as(f64, @floatFromInt(size_bytes)) / (avg_time_orig / 1_000_000_000.0);
    const throughput_simd = @as(f64, @floatFromInt(size_bytes)) / (avg_time_simd / 1_000_000_000.0);

    print("Original Throughput:    {d:.2} GB/s\n", .{throughput_orig / 1_000_000_000.0});
    print("SIMD Throughput:        {d:.2} GB/s\n", .{throughput_simd / 1_000_000_000.0});
}

fn benchmarkShapeWithNewSIMD(shape: [3]usize, iterations: usize, allocator: Allocator) !void {
    var total_time_orig: u64 = 0;
    var total_time_simd: u64 = 0;
    var total_time_new_simd: u64 = 0;
    var timer = try std.time.Timer.start();

    print("\nShape: {}x{}x{}\n", .{ shape[0], shape[1], shape[2] });

    // Run original implementation
    {
        // Warmup
        var warmup = try Tensor(f16).init(allocator, &shape);
        defer warmup.deinit();
        try fillSequential(f16, &warmup);
        try transposeAxesOriginal(f16, &warmup, 0, 1);

        for (0..iterations) |_| {
            var tensor = try Tensor(f16).init(allocator, &shape);
            defer tensor.deinit();
            try fillSequential(f16, &tensor);

            timer.reset();
            try transposeAxesOriginal(f16, &tensor, 0, 1);
            total_time_orig += timer.read();
        }
    }

    // Run current SIMD implementation
    {
        // Warmup
        var warmup = try Tensor(f16).init(allocator, &shape);
        defer warmup.deinit();
        try fillSequential(f16, &warmup);
        try transposeAxes(f16, &warmup, 0, 1);

        for (0..iterations) |_| {
            var tensor = try Tensor(f16).init(allocator, &shape);
            defer tensor.deinit();
            try fillSequential(f16, &tensor);

            timer.reset();
            try transposeAxes(f16, &tensor, 0, 1);
            total_time_simd += timer.read();
        }
    }

    // Run new SIMD implementation
    {
        // Warmup
        var warmup = try Tensor(f16).init(allocator, &shape);
        defer warmup.deinit();
        try fillSequential(f16, &warmup);
        const new_data = try allocator.alignedAlloc(f16, 32, shape[0] * shape[1] * shape[2]);
        defer allocator.free(new_data);
        transposeF16SIMD(&warmup, shape[0], shape[1], shape[2], new_data);

        for (0..iterations) |_| {
            var tensor = try Tensor(f16).init(allocator, &shape);
            defer tensor.deinit();
            try fillSequential(f16, &tensor);
            const transposed_data = try allocator.alignedAlloc(f16, 32, shape[0] * shape[1] * shape[2]);
            defer allocator.free(transposed_data);

            timer.reset();
            transposeF16SIMD(&tensor, shape[0], shape[1], shape[2], transposed_data);
            total_time_new_simd += timer.read();
        }
    }

    const avg_time_orig = @as(f64, @floatFromInt(total_time_orig)) / @as(f64, @floatFromInt(iterations));
    const avg_time_simd = @as(f64, @floatFromInt(total_time_simd)) / @as(f64, @floatFromInt(iterations));
    const avg_time_new_simd = @as(f64, @floatFromInt(total_time_new_simd)) / @as(f64, @floatFromInt(iterations));

    const speedup_orig = avg_time_orig / avg_time_new_simd;
    const speedup_simd = avg_time_simd / avg_time_new_simd;

    print("Original Implementation:    {d:.2}ms\n", .{avg_time_orig / 1_000_000.0});
    print("Current SIMD:              {d:.2}ms\n", .{avg_time_simd / 1_000_000.0});
    print("New SIMD Implementation:   {d:.2}ms\n", .{avg_time_new_simd / 1_000_000.0});
    print("Speedup vs Original:       {d:.2}x\n", .{speedup_orig});
    print("Speedup vs Current SIMD:   {d:.2}x\n", .{speedup_simd});

    // Calculate throughput
    const elements = shape[0] * shape[1] * shape[2];
    const size_bytes = elements * @sizeOf(f16);
    const throughput_orig = @as(f64, @floatFromInt(size_bytes)) / (avg_time_orig / 1_000_000_000.0);
    const throughput_simd = @as(f64, @floatFromInt(size_bytes)) / (avg_time_simd / 1_000_000_000.0);
    const throughput_new_simd = @as(f64, @floatFromInt(size_bytes)) / (avg_time_new_simd / 1_000_000_000.0);

    print("Original Throughput:       {d:.2} GB/s\n", .{throughput_orig / 1_000_000_000.0});
    print("Current SIMD Throughput:   {d:.2} GB/s\n", .{throughput_simd / 1_000_000_000.0});
    print("New SIMD Throughput:       {d:.2} GB/s\n", .{throughput_new_simd / 1_000_000_000.0});
}

fn benchmarkCastShape(shape: [3]usize, iterations: usize, allocator: std.mem.Allocator) !void {
    var total_time_orig: u64 = 0;
    var total_time_simd: u64 = 0;
    var timer = try std.time.Timer.start();

    print("\nShape: {}x{}x{}\n", .{ shape[0], shape[1], shape[2] });

    // Original implementation with casting
    {
        // Warmup
        var warmup = try Tensor(f16).init(allocator, &shape);
        defer warmup.deinit();
        try fillSequential(f16, &warmup);

        var warmup_f32 = try warmup.castTo(f32);
        defer warmup_f32.deinit();
        try transposeAxesOriginal(f32, &warmup_f32, 0, 1);
        var warmup_back = try warmup_f32.castTo(f16);
        defer warmup_back.deinit();

        // Actual benchmark
        for (0..iterations) |_| {
            var tensor = try Tensor(f16).init(allocator, &shape);
            defer tensor.deinit();
            try fillSequential(f16, &tensor);

            timer.reset();
            var tensor_f32 = try tensor.castTo(f32);
            defer tensor_f32.deinit();
            try transposeAxesOriginal(f32, &tensor_f32, 0, 1);
            var tensor_back = try tensor_f32.castTo(f16);
            defer tensor_back.deinit();
            total_time_orig += timer.read();
        }
    }

    // SIMD implementation with casting
    {
        // Warmup
        var warmup = try Tensor(f16).init(allocator, &shape);
        defer warmup.deinit();
        try fillSequential(f16, &warmup);

        var warmup_f32 = try warmup.castWithSimd(f32);
        defer warmup_f32.deinit();
        try transposeAxes(f32, &warmup_f32, 0, 1);
        var warmup_back = try warmup_f32.castWithSimd(f16);
        defer warmup_back.deinit();

        // Actual benchmark
        for (0..iterations) |_| {
            var tensor = try Tensor(f16).init(allocator, &shape);
            defer tensor.deinit();
            try fillSequential(f16, &tensor);

            timer.reset();
            var tensor_f32 = try tensor.castWithSimd(f32);
            defer tensor_f32.deinit();
            try transposeAxes(f32, &tensor_f32, 0, 1);
            var tensor_back = try tensor_f32.castWithSimd(f16);
            defer tensor_back.deinit();
            total_time_simd += timer.read();
        }
    }

    const avg_time_orig = @as(f64, @floatFromInt(total_time_orig)) / @as(f64, @floatFromInt(iterations));
    const avg_time_simd = @as(f64, @floatFromInt(total_time_simd)) / @as(f64, @floatFromInt(iterations));
    const speedup = avg_time_orig / avg_time_simd;

    print("Original + Cast: {d:.2}ms\n", .{avg_time_orig / 1_000_000.0});
    print("SIMD + Cast:     {d:.2}ms\n", .{avg_time_simd / 1_000_000.0});
    print("Speedup:         {d:.2}x\n", .{speedup});

    // Calculate throughput
    const elements = shape[0] * shape[1] * shape[2];
    const size_bytes = elements * (@sizeOf(f16) + @sizeOf(f32)); // Account for both f16 and f32 data
    const throughput_orig = @as(f64, @floatFromInt(size_bytes)) / (avg_time_orig / 1_000_000_000.0);
    const throughput_simd = @as(f64, @floatFromInt(size_bytes)) / (avg_time_simd / 1_000_000_000.0);

    print("Original Throughput: {d:.2} GB/s\n", .{throughput_orig / 1_000_000_000.0});
    print("SIMD Throughput:     {d:.2} GB/s\n", .{throughput_simd / 1_000_000_000.0});
}

fn fillSequential(comptime T: type, tensor: *Tensor(T)) !void {
    for (tensor.data, 0..) |*val, i| {
        val.* = @floatFromInt(i % 100); // Modulo to keep numbers reasonable
    }
}

// Helper function to run benchmarks
pub fn main() !void {
    try benchmark();
}
