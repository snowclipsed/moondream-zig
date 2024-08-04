const std = @import("std");
const assert = std.debug.assert;
// vector math functions

// function taken from cgbur/llama2.zig
fn softmax(x: []f32) void {
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

// weights

// runstate

// tokenizer

// attention

// transformer

// vision encoder

// inference

// tests
test "softmax" {
    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };

    softmax(&x);

    var sum: f32 = 0.0;
    for (0..x.len) |value| {
        sum += x[value];
    }

    try std.testing.expect(sum == 1.0);
}
// main
