const std = @import("std");
const assert = std.debug.assert;

fn rmsnorm(o: []f32, x: []f32, w: []f32) void {
    assert(o.len == x.len);
    assert(o.len == w.len);

    // sum of squares
    var sum: f32 = 0.0;
    for (x) |val| {
        sum += val * val;
    }
    sum /= @floatFromInt(x.len);
    sum += 1e-5;
    sum = 1.0 / std.math.sqrt(sum);

    // normalize and scale
    for (0..o.len) |i| {
        o[i] = x[i] * sum * w[i];
    }
}

pub fn checkVectorStability(vector: []const f32, name: []const u8) !void {
    for (vector, 0..) |value, index| {
        if (std.math.isNan(value)) {
            std.debug.print("Warning: NaN detected in {s} at index {d}\n", .{ name, index });
            return error.NaNDetected;
        }
        if (std.math.isInf(value)) {
            if (value > 0) {
                std.debug.print("Warning: +Inf detected in {s} at index {d}\n", .{ name, index });
            } else {
                std.debug.print("Warning: -Inf detected in {s} at index {d}\n", .{ name, index });
            }
            return error.InfDetected;
        }
    }
}
