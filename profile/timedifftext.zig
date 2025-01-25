const std = @import("std");
const Timer = std.time.Timer;

// Configuration struct to control logging
const config = struct {
    pub var enableLogging: bool = true; // Set this to false to disable output
};

pub fn printTimeDiff(timer: *Timer, start_time: i128, step_name: []const u8) !void {
    if (!config.enableLogging) return; // Skip printing if disabled

    const stdout = std.io.getStdOut().writer();
    const end_time = timer.read();
    const diff_ns = end_time - start_time;
    const diff_ms = @as(f64, @floatFromInt(diff_ns)) / 1_000_000.0;
    try stdout.print("\x1b[92m[TEXT PROFILE] {s}: {d:.2}ms\x1b[0m\n", .{
        step_name, diff_ms,
    });
}

// Helper function remains unchanged (it leverages the check in printTimeDiff)
pub fn getAndPrintTimeDiff(timer: *Timer, start_time: i128, step_name: []const u8) !i128 {
    try printTimeDiff(timer, start_time, step_name);
    return timer.read();
}
