const std = @import("std");

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
        return error.ImageLoadFailed;
    }
    defer c.stbi_image_free(image_data);

    // Calculate terminal dimensions with scale factor
    const base_term_width: usize = 80;
    const term_width = @as(usize, @intFromFloat(@as(f32, @floatFromInt(base_term_width)) * scale));
    const term_height = @as(usize, @intFromFloat(@as(f32, @floatFromInt(term_width)) * 0.5 *
        @as(f32, @floatFromInt(height)) / @as(f32, @floatFromInt(width))));

    // Allocate memory for resized image
    const resized = try allocator.alloc(u8, term_width * term_height * 2 * 3);
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
        return error.ResizeFailed;
    }

    // Print the image
    var y: usize = 0;
    while (y < term_height) : (y += 1) {
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

            try printColorBlock(upper, lower);
        }
        try std.io.getStdOut().writer().print("\n", .{});
    }

    // Reset terminal colors
    try std.io.getStdOut().writer().print("\x1b[0m", .{});
    std.debug.print("Displaying image at scale {d:3}x.\n", .{scale});
}

fn printColorBlock(upper: Pixel, lower: Pixel) !void {
    const writer = std.io.getStdOut().writer();
    try writer.print("\x1b[38;2;{};{};{}m\x1b[48;2;{};{};{}m▀", .{
        upper.r, upper.g, upper.b,
        lower.r, lower.g, lower.b,
    });
}

// pub fn main() !void {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     defer _ = gpa.deinit();
//     const allocator = gpa.allocator();

//     // Example usage with different scales:
//     // 1.0 = original size (80 columns)
//     // 0.5 = half size (40 columns)
//     // 2.0 = double size (160 columns)
//     const image_path: []const u8 = "../images/demo-1.jpg";
//     try displayImage(allocator, image_path, 1.0);
// }
