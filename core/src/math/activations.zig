const std = @import("std");
const assert = std.debug.assert;

fn softmax(x: []f32) !void {
    assert(x.len > 0);
    // max of x for numerical stability
    var max: f32 = x[0];
    for (x[1..]) |val| {
        if (val > max) {
            max = val;
        }
    }
    // exp and sum
    var sum: f32 = 0.0;
    for (x) |*val| {
        val.* = std.math.exp(val.* - max); // https://stackoverflow.com/questions/42599498/numerically-stable-softmax
        sum += val.*;
    }
    // normalize
    for (x) |*val| {
        val.* /= sum;
    }
}

pub fn gelu(input: []f32) void {
    const sqrt_2_over_pi = std.math.sqrt(2.0 / std.math.pi);
    const factor = 0.044715;

    for (input) |*value| {
        const x = value.*;
        const term = sqrt_2_over_pi * (x + factor * x * x * x);
        const tanh_term = std.math.tanh(term);
        value.* = 0.5 * x * (1.0 + tanh_term);
    }
}
