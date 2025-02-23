const std = @import("std");

// Color constants
const Colors = struct {
    // Regular colors
    const black = "\x1b[30m";
    const red = "\x1b[31m";
    const green = "\x1b[32m";
    const yellow = "\x1b[33m";
    const blue = "\x1b[34m";
    const magenta = "\x1b[35m";
    const cyan = "\x1b[36m";
    const white = "\x1b[37m";

    // Bright colors
    const bright_black = "\x1b[90m";
    const bright_red = "\x1b[91m";
    const bright_green = "\x1b[92m";
    const bright_yellow = "\x1b[93m";
    const bright_blue = "\x1b[94m";
    const bright_magenta = "\x1b[95m";
    const bright_cyan = "\x1b[96m";
    const bright_white = "\x1b[97m";

    // Background colors
    const bg_black = "\x1b[40m";
    const bg_red = "\x1b[41m";
    const bg_green = "\x1b[42m";
    const bg_yellow = "\x1b[43m";
    const bg_blue = "\x1b[44m";
    const bg_magenta = "\x1b[45m";
    const bg_cyan = "\x1b[46m";
    const bg_white = "\x1b[47m";

    // Bright background colors
    const bg_bright_black = "\x1b[100m";
    const bg_bright_red = "\x1b[101m";
    const bg_bright_green = "\x1b[102m";
    const bg_bright_yellow = "\x1b[103m";
    const bg_bright_blue = "\x1b[104m";
    const bg_bright_magenta = "\x1b[105m";
    const bg_bright_cyan = "\x1b[106m";
    const bg_bright_white = "\x1b[107m";

    // Special formatting
    const reset = "\x1b[0m";
    const bold = "\x1b[1m";
    const dim = "\x1b[2m";
    const italic = "\x1b[3m";
    const underline = "\x1b[4m";
    const blink = "\x1b[5m";
    const invert = "\x1b[7m";
    const hidden = "\x1b[8m";
};

fn print8BitColor(writer: anytype, color_code: u8) !void {
    try writer.print("\x1b[38;5;{d}m███ ", .{color_code});
}

fn printColorSection(writer: anytype, comptime title: []const u8, colors: []const []const u8, names: []const []const u8) !void {
    try writer.print("\n{s}:\n", .{title});
    for (colors, names) |color, name| {
        try writer.print("{s}███ {s}{s}", .{ color, Colors.reset, name });
        try writer.writeAll("   ");
    }
    try writer.writeAll("\n");
}

pub fn main() !void {
    const stdout = std.io.getStdOut().writer();

    // Regular colors
    const regular_colors = [_][]const u8{
        Colors.black, Colors.red,     Colors.green, Colors.yellow,
        Colors.blue,  Colors.magenta, Colors.cyan,  Colors.white,
    };
    const regular_names = [_][]const u8{
        "black", "red", "green", "yellow", "blue", "magenta", "cyan", "white",
    };

    // Bright colors
    const bright_colors = [_][]const u8{
        Colors.bright_black, Colors.bright_red,     Colors.bright_green, Colors.bright_yellow,
        Colors.bright_blue,  Colors.bright_magenta, Colors.bright_cyan,  Colors.bright_white,
    };
    const bright_names = [_][]const u8{
        "bright_black", "bright_red",     "bright_green", "bright_yellow",
        "bright_blue",  "bright_magenta", "bright_cyan",  "bright_white",
    };

    // Print regular and bright colors
    try printColorSection(stdout, "Regular Colors", &regular_colors, &regular_names);
    try printColorSection(stdout, "Bright Colors", &bright_colors, &bright_names);

    // Print 8-bit color palette (selected ranges)
    try stdout.writeAll("\n8-bit Colors (16-31):\n");
    var i: u8 = 16;
    while (i < 32) : (i += 1) {
        try print8BitColor(stdout, i);
        if ((i + 1) % 8 == 0) try stdout.writeAll("\n");
    }

    // Print some formatting examples
    try stdout.writeAll("\nFormatting Examples:\n");
    try stdout.print("{s}Bold{s}    ", .{ Colors.bold, Colors.reset });
    try stdout.print("{s}Dim{s}     ", .{ Colors.dim, Colors.reset });
    try stdout.print("{s}Italic{s}  ", .{ Colors.italic, Colors.reset });
    try stdout.print("{s}Underline{s}\n", .{ Colors.underline, Colors.reset });

    // Reset all formatting
    try stdout.writeAll(Colors.reset);
}
