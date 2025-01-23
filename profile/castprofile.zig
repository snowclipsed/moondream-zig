const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
const time = std.time;
const print = std.debug.print;

// Number of iterations for each benchmark
const ITERATIONS = 100;

// Test cases with different shapes and total sizes
const TestCase = struct {
    shape: []const usize,
    description: []const u8,
};

const TEST_CASES = [_]TestCase{
    // Small sizes
    .{ .shape = &[_]usize{ 16, 16 }, .description = "Small Square" },
    .{ .shape = &[_]usize{ 8, 32 }, .description = "Small Rectangle" },
    .{ .shape = &[_]usize{ 4, 4, 4 }, .description = "Small Cube" },

    // Medium sizes
    .{ .shape = &[_]usize{ 64, 64 }, .description = "Medium Square" },
    .{ .shape = &[_]usize{ 32, 128 }, .description = "Medium Rectangle" },
    .{ .shape = &[_]usize{ 16, 16, 16 }, .description = "Medium Cube" },

    // Large sizes
    .{ .shape = &[_]usize{ 256, 256 }, .description = "Large Square" },
    .{ .shape = &[_]usize{ 128, 512 }, .description = "Large Rectangle" },
    .{ .shape = &[_]usize{ 32, 32, 32 }, .description = "Large Cube" },

    // Very large sizes
    .{ .shape = &[_]usize{ 1024, 1024 }, .description = "Very Large Square" },
    .{ .shape = &[_]usize{ 2048, 512 }, .description = "Very Large Rectangle" },
    .{ .shape = &[_]usize{ 64, 64, 64 }, .description = "Very Large Cube" },

    // Extreme sizes
    .{ .shape = &[_]usize{ 2048, 2048 }, .description = "Extreme Square" },
    .{ .shape = &[_]usize{ 4096, 1024 }, .description = "Extreme Rectangle" },
    .{ .shape = &[_]usize{ 128, 128, 64 }, .description = "Extreme Cube" },

    // Unusual shapes
    .{ .shape = &[_]usize{ 7, 11, 13 }, .description = "Prime Dimensions" },
    .{ .shape = &[_]usize{ 3, 3, 3, 3, 3 }, .description = "5D Hypercube" },
    .{ .shape = &[_]usize{ 1, 1024, 1 }, .description = "Thin Slice" },
    .{ .shape = &[_]usize{ 2, 3, 4, 5 }, .description = "Growing Dimensions" },

    // Mixed sizes
    .{ .shape = &[_]usize{ 512, 8, 64 }, .description = "Mixed 3D" },
    .{ .shape = &[_]usize{ 256, 4, 32, 8 }, .description = "Mixed 4D" },
};

fn calculateTotalSize(shape: []const usize) usize {
    var size: usize = 1;
    for (shape) |dim| {
        size *= dim;
    }
    return size;
}

fn formatSize(size: usize) []const u8 {
    if (size >= 1_000_000_000) {
        return std.fmt.allocPrint(std.heap.page_allocator, "{d:.2}G", .{@as(f64, @floatFromInt(size)) / 1_000_000_000}) catch "??G";
    } else if (size >= 1_000_000) {
        return std.fmt.allocPrint(std.heap.page_allocator, "{d:.2}M", .{@as(f64, @floatFromInt(size)) / 1_000_000}) catch "??M";
    } else if (size >= 1_000) {
        return std.fmt.allocPrint(std.heap.page_allocator, "{d:.2}K", .{@as(f64, @floatFromInt(size)) / 1_000}) catch "??K";
    } else {
        return std.fmt.allocPrint(std.heap.page_allocator, "{d}", .{size}) catch "??";
    }
}

pub fn main() !void {
    // Initialize allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    print("\nBenchmarking FP16 ↔ FP32 Casting Operations\n", .{});
    print("===============================================\n", .{});
    print("Running {} iterations for each shape\n\n", .{ITERATIONS});
    print("{s:>15} | {s:>12} | {s:>15} | {s:>15} | {s:>10} | {s:>20}\n", .{ "Total Size", "Shape", "Regular (μs)", "SIMD (μs)", "Speedup", "Description" });
    print("----------------------------------------------------------------------------------------\n", .{});

    // Test each shape configuration
    for (TEST_CASES) |test_case| {
        const total_size = calculateTotalSize(test_case.shape);
        const size_str = formatSize(total_size);

        // Skip very large tensors if they would exceed memory limits
        if (total_size * @sizeOf(f16) > 2 * 1024 * 1024 * 1024) { // 2GB limit
            print("{s:>15} | {s:>12} | {s:>15} | {s:>15} | {s:>10} | {s:>20}\n", .{ size_str, "SKIPPED", "-", "-", "-", test_case.description });
            continue;
        }

        // Create tensors for testing
        var tensor_f16 = try Tensor(f16).init(allocator, test_case.shape);
        defer tensor_f16.deinit();

        // Fill with some test data
        for (tensor_f16.data, 0..) |_, i| {
            tensor_f16.data[i] = @floatCast(1.5 + @as(f32, @floatFromInt(i % 1000)) * 0.1);
        }

        // Benchmark regular casting (F16 → F32)
        var regular_time: u64 = 0;
        var result_verify: ?*Tensor(f32) = null;
        for (0..ITERATIONS) |i| {
            const start = time.nanoTimestamp();
            var result = try tensor_f16.castTo(f32);
            const end = time.nanoTimestamp();
            regular_time += @intCast(end - start);
            if (i == 0) {
                // Allocate space for the verification tensor
                const verify_tensor = try allocator.create(Tensor(f32));
                verify_tensor.* = result;
                result_verify = verify_tensor;
            } else {
                result.deinit();
            }
        }
        const avg_regular = @divFloor(regular_time, ITERATIONS) / 1000; // Convert to microseconds

        // Benchmark SIMD casting (F16 → F32)
        var simd_time: u64 = 0;
        for (0..ITERATIONS) |i| {
            const start = time.nanoTimestamp();
            var result = try tensor_f16.castWithSimd(f32);
            const end = time.nanoTimestamp();
            simd_time += @intCast(end - start);

            // Verify results match on first iteration
            if (i == 0) {
                for (result.data, result_verify.?.*.data) |simd_val, reg_val| {
                    if (@abs(simd_val - reg_val) > 0.0001) {
                        print("Warning: SIMD results differ from regular casting!\n", .{});
                        break;
                    }
                }
            }
            result.deinit();
        }
        if (result_verify) |rv| {
            rv.*.deinit();
            allocator.destroy(rv);
        }
        const avg_simd = @divFloor(simd_time, ITERATIONS) / 1000;

        // Calculate speedup
        const speedup = if (avg_simd > 0) @as(f32, @floatFromInt(avg_regular)) / @as(f32, @floatFromInt(avg_simd)) else 0;

        // Format shape string
        var shape_str = std.ArrayList(u8).init(allocator);
        defer shape_str.deinit();
        try shape_str.writer().print("{any}", .{test_case.shape});

        print("{s:>15} | {s:>12} | {:>15} | {:>15} | {:>9.2} | {s:>20}\n", .{
            size_str,
            shape_str.items,
            avg_regular,
            avg_simd,
            speedup,
            test_case.description,
        });
    }

    print("\nNow testing F32 → F16 conversion\n", .{});
    print("===============================================\n", .{});
    print("{s:>15} | {s:>12} | {s:>15} | {s:>15} | {s:>10} | {s:>20}\n", .{ "Total Size", "Shape", "Regular (μs)", "SIMD (μs)", "Speedup", "Description" });
    print("----------------------------------------------------------------------------------------\n", .{});

    // Test F32 → F16 conversion
    for (TEST_CASES) |test_case| {
        const total_size = calculateTotalSize(test_case.shape);
        const size_str = formatSize(total_size);

        // Skip very large tensors
        if (total_size * @sizeOf(f32) > 2 * 1024 * 1024 * 1024) { // 2GB limit
            print("{s:>15} | {s:>12} | {s:>15} | {s:>15} | {s:>10} | {s:>20}\n", .{ size_str, "SKIPPED", "-", "-", "-", test_case.description });
            continue;
        }

        var tensor_f32 = try Tensor(f32).init(allocator, test_case.shape);
        defer tensor_f32.deinit();

        // Fill with test data
        for (tensor_f32.data, 0..) |_, i| {
            tensor_f32.data[i] = 1.5 + @as(f32, @floatFromInt(i % 1000)) * 0.1;
        }

        // Benchmark regular casting
        var regular_time: u64 = 0;
        var result_verify: ?*Tensor(f16) = null;
        for (0..ITERATIONS) |i| {
            const start = time.nanoTimestamp();
            var result = try tensor_f32.castTo(f16);
            const end = time.nanoTimestamp();
            regular_time += @intCast(end - start);
            if (i == 0) {
                // Allocate space for the verification tensor
                const verify_tensor = try allocator.create(Tensor(f16));

                verify_tensor.* = result;
                result_verify = verify_tensor;
            } else {
                result.deinit();
            }
        }
        const avg_regular = @divFloor(regular_time, ITERATIONS) / 1000;

        // Benchmark SIMD casting
        var simd_time: u64 = 0;
        for (0..ITERATIONS) |i| {
            const start = time.nanoTimestamp();
            var result = try tensor_f32.castWithSimd(f16);
            const end = time.nanoTimestamp();
            simd_time += @intCast(end - start);

            // Verify results match on first iteration
            if (i == 0) {
                for (result.data, result_verify.?.*.data) |simd_val, reg_val| {
                    if (@abs(@as(f32, @floatCast(simd_val - reg_val))) > 0.01) {
                        print("Warning: SIMD results differ from regular casting!\n", .{});
                        break;
                    }
                }
            }
            result.deinit();
        }
        if (result_verify) |rv| {
            rv.*.deinit();
            allocator.destroy(rv);
        }
        const avg_simd = @divFloor(simd_time, ITERATIONS) / 1000;

        // Calculate speedup
        const speedup = if (avg_simd > 0) @as(f32, @floatFromInt(avg_regular)) / @as(f32, @floatFromInt(avg_simd)) else 0;

        // Format shape string
        var shape_str = std.ArrayList(u8).init(allocator);
        defer shape_str.deinit();
        try shape_str.writer().print("{any}", .{test_case.shape});

        print("{s:>15} | {s:>12} | {:>15} | {:>15} | {:>9.2} | {s:>20}\n", .{
            size_str,
            shape_str.items,
            avg_regular,
            avg_simd,
            speedup,
            test_case.description,
        });
    }
}
