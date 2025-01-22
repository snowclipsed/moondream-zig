const std = @import("std");

pub fn main() !void {
    const stdout = std.io.getStdOut().writer();

    // Tokyo Night theme colors
    const shadow_color = "\x1b[38;5;60m"; // Muted blue for shadows
    const main_color = "\x1b[38;5;189m"; // Bright foreground
    const reset_color = "\x1b[0m";

    try stdout.writeAll(main_color);
    try stdout.writeAll(
        \\███╗   ███╗  ██████╗   ██████╗  ███╗   ██╗██████╗  ██████╗  ███████╗  █████╗  ███╗   ███╗
    );
    try stdout.writeAll("\n");

    try stdout.writeAll(main_color);
    try stdout.writeAll(
        \\████╗ ████║██╔═══██╗██╔═══██╗████╗  ██║██╔══██╗██╔══██╗██╔════╝██╔══██╗████╗ ████║
    );
    try stdout.writeAll("\n");

    try stdout.writeAll(main_color);
    try stdout.writeAll(
        \\██╔████╔██║██║   ██║██║   ██║██╔██╗ ██║██║  ██║██████╔╝█████╗  ███████║██╔████╔██║
    );
    try stdout.writeAll("\n");

    try stdout.writeAll(main_color);
    try stdout.writeAll(
        \\██║╚██╔╝██║██║   ██║██║   ██║██║╚██╗██║██║  ██║██╔══██╗██╔══╝  ██╔══██║██║╚██╔╝██║
    );
    try stdout.writeAll("\n");

    try stdout.writeAll(main_color);
    try stdout.writeAll(
        \\██║ ╚═╝ ██║╚██████╔╝╚██████╔╝██║ ╚████║██████╔╝██║  ██║███████╗██║  ██║██║ ╚═╝ ██║
    );
    try stdout.writeAll("\n");

    try stdout.writeAll(shadow_color);
    try stdout.writeAll(
        \\╚═╝     ╚═╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝
    );
    try stdout.writeAll(reset_color);
    try stdout.writeAll("\n");
}
