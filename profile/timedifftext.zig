const std = @import("std");
const Timer = std.time.Timer;

pub fn printTimeDiff(timer: *Timer, start_time: i128, step_name: []const u8) !void {
    const stdout = std.io.getStdOut().writer();
    const end_time = timer.read();
    const diff_ns = end_time - start_time;
    const diff_ms = @as(f64, @floatFromInt(diff_ns)) / 1_000_000.0;
    try stdout.print("\x1b[93m[TEXT PROFILE] {s}: {d:.2}ms\x1b[0m\n", .{
        step_name, diff_ms,
    });
}

// If you need the end time separately, you can create a helper that returns both:
pub fn getAndPrintTimeDiff(timer: *Timer, start_time: i128, step_name: []const u8) !i128 {
    try printTimeDiff(timer, start_time, step_name);
    return timer.read();
}
