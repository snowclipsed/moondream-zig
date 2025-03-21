const std = @import("std");
const SlabReusingAllocator = @import("../core/slab_reusing_allocator.zig").SlabReusingAllocator;

const c = @cImport({
    @cInclude("stb_image.h");
    @cInclude("stb_image_resize2.h");
    @cInclude("stdio.h");
});

const Pixel = struct {
    r: u8,
    g: u8,
    b: u8,
};

pub fn displayImage(allocator: std.mem.Allocator, image_path: []const u8, scale: f32) !void {
    const stdout = std.io.getStdOut().writer();

    // Check if file exists first
    const file = std.fs.cwd().openFile(image_path, .{}) catch |err| {
        if (err == error.FileNotFound) {
            std.debug.print("[DEBUG] Image file not found: {s}\n", .{image_path});
            return error.FileNotFound;
        }
        return err;
    };
    file.close();

    // Create null-terminated path for C
    var path_buffer = try allocator.alloc(u8, image_path.len + 1);
    defer allocator.free(path_buffer);

    @memcpy(path_buffer[0..image_path.len], image_path);
    path_buffer[image_path.len] = 0; // Null terminator

    // Load image using stb_image
    var width: c_int = undefined;
    var height: c_int = undefined;
    var channels: c_int = undefined;
    const desired_channels = 3; // RGB

    const image_data = c.stbi_load(
        path_buffer.ptr,
        &width,
        &height,
        &channels,
        desired_channels,
    );

    if (image_data == null) {
        const err_str = c.stbi_failure_reason();
        std.debug.print("[DEBUG] STB Error: {s}\n", .{err_str});
        return error.ImageNotFound; // More specific error type that matches our vision model
    }
    defer c.stbi_image_free(image_data);

    // Validate image dimensions
    if (width <= 0 or height <= 0) {
        return error.InvalidImageDimensions;
    }

    // Calculate terminal dimensions with scale factor
    const base_term_width: usize = 80;
    const term_width = @as(usize, @intFromFloat(@as(f32, @floatFromInt(base_term_width)) * scale));
    const term_height = @as(usize, @intFromFloat(@as(f32, @floatFromInt(term_width)) * 0.5 *
        @as(f32, @floatFromInt(height)) / @as(f32, @floatFromInt(width))));

    // Validate calculated dimensions
    if (term_width == 0 or term_height == 0) {
        return error.InvalidResizeDimensions;
    }

    // Allocate memory for resized image
    const resized = allocator.alloc(u8, term_width * term_height * 2 * 3) catch |err| {
        std.debug.print("[DEBUG] Memory allocation failed: {s}\n", .{@errorName(err)});
        return error.MemoryAllocationFailed;
    };
    defer allocator.free(resized);

    // Resize image using stbir
    const resize_result = c.stbir_resize_uint8_linear(
        image_data,
        @intCast(width),
        @intCast(height),
        0,
        resized.ptr,
        @intCast(term_width),
        @intCast(term_height * 2),
        0,
        3,
    );

    if (resize_result == 0) {
        return error.FailedToResizeImage; // Match the error name with our vision model
    }

    // Print the image
    var y: usize = 0;
    while (y < term_height) : (y += 1) {
        // Clear line before printing
        try stdout.print("\x1b[K", .{});

        var x: usize = 0;
        while (x < term_width) : (x += 1) {
            const upper_idx = (y * 2 * term_width + x) * 3;
            const lower_idx = ((y * 2 + 1) * term_width + x) * 3;

            const upper = Pixel{
                .r = resized[upper_idx],
                .g = resized[upper_idx + 1],
                .b = resized[upper_idx + 2],
            };
            const lower = Pixel{
                .r = resized[lower_idx],
                .g = resized[lower_idx + 1],
                .b = resized[lower_idx + 2],
            };

            printColorBlock(upper, lower, x, term_width) catch |err| {
                std.debug.print("[DEBUG] Error printing color block: {s}\n", .{@errorName(err)});
                continue; // Try to continue with the next block
            };
        }

        // Clear to end of line and print newline
        try stdout.print("\x1b[K\n", .{});
    }

    // Reset colors at the end
    try stdout.print("\x1b[0m", .{});
}

fn printColorBlock(upper: Pixel, lower: Pixel, current_x: usize, term_width: usize) !void {
    if (current_x >= term_width) return;

    const writer = std.io.getStdOut().writer();
    try writer.print("\x1b[38;2;{};{};{}m\x1b[48;2;{};{};{}m▀\x1b[0m", .{
        upper.r, upper.g, upper.b,
        lower.r, lower.g, lower.b,
    });
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const gpa_allocator = gpa.allocator();
    var slab_reusing_allocator = SlabReusingAllocator(100).init(gpa_allocator);
    defer slab_reusing_allocator.deinit();
    const allocator = slab_reusing_allocator.allocator();

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();

    // Skip program name
    _ = args.next();

    // Get image path
    const image_path = args.next() orelse {
        std.debug.print("Usage: {s} <image_path> [scale]\n", .{"image-display"});
        return error.InvalidArguments;
    };

    // Get optional scale factor
    const scale_str = args.next();
    const scale: f32 = if (scale_str) |s|
        try std.fmt.parseFloat(f32, s)
    else
        1.0;

    try displayImage(allocator, image_path, scale);
}
