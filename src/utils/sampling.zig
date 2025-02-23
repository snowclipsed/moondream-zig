const std = @import("std");
const Tensor = @import("../core/tensor.zig").Tensor;
const Allocator = std.mem.Allocator;

// ---------- Sampling ---------- //

pub fn sample_from_probs(comptime T: type, tensor: *Tensor(T), rng: std.rand.Random) !usize {
    if (tensor.shape.len != 2 or tensor.shape[0] != 1) {
        return error.InvalidInputShape;
    }

    var cumsum: T = 0;
    const r = rng.float(T);

    for (tensor.data, 0..) |p, i| {
        cumsum += p;
        if (r < cumsum) {
            return i;
        }
    }

    // If we somehow get here (floating point rounding), return last index
    return tensor.data.len - 1;
}
/// Defines different sampling methods available for token selection
pub const SamplingMethod = enum {
    greedy, // Always select highest probability token (argmax)
    multinomial, // Sample from full distribution
    top_k, // Sample from top k tokens only
};

/// Configuration for sampling parameters
pub const SamplingConfig = struct {
    method: SamplingMethod,
    temperature: f32 = 0.5, // Temperature for softmax
    top_k: ?usize = 3, // Number of top tokens to consider (only for top_k)
};

/// Returns the index of the maximum value in the tensor
pub fn argmax(comptime T: type, tensor: *const Tensor(T)) !usize {
    if (tensor.data.len == 0) {
        return error.EmptyTensor;
    }

    var max_idx: usize = 0;
    var max_val = tensor.data[0];

    for (tensor.data, 0..) |val, i| {
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }

    return max_idx;
}

/// Performs multinomial sampling on a tensor of probabilities
fn multinomial_sampling(comptime T: type, tensor: *const Tensor(T), rng: std.rand.Random) !usize {
    if (tensor.shape.len != 2 or tensor.shape[0] != 1) {
        return error.InvalidInputShape;
    }

    var sum: T = 0;
    for (tensor.data) |val| {
        sum += val;
    }

    const r: T = @floatCast(rng.float(f32) * sum);
    var cumsum: T = 0;

    for (tensor.data, 0..) |val, i| {
        cumsum += val;
        if (r < cumsum) {
            return i;
        }
    }

    return tensor.data.len - 1;
}

/// Performs top-k sampling on a tensor of probabilities
fn top_k_sampling(comptime T: type, tensor: *const Tensor(T), k: usize, rng: std.rand.Random, allocator: Allocator) !usize {
    if (tensor.shape.len != 2 or tensor.shape[0] != 1) {
        return error.InvalidInputShape;
    }

    const vocab_size = tensor.shape[1];
    const k_actual = @min(k, vocab_size);

    var indices = try std.ArrayList(usize).initCapacity(allocator, vocab_size);
    defer indices.deinit();

    for (0..vocab_size) |i| {
        try indices.append(i);
    }

    std.mem.sort(usize, indices.items, tensor, struct {
        fn compare(context: *const Tensor(T), a: usize, b: usize) bool {
            return context.data[a] > context.data[b];
        }
    }.compare);

    const top_k_indices = indices.items[0..k_actual];

    var sum: T = 0;
    for (top_k_indices) |idx| {
        sum += tensor.data[idx];
    }

    const r: T = @floatCast(rng.float(f32) * sum);
    var cumsum: T = 0;

    for (top_k_indices) |idx| {
        cumsum += tensor.data[idx];
        if (r < cumsum) {
            return idx;
        }
    }

    return top_k_indices[k_actual - 1];
}

/// Apply temperature to logits
fn apply_temperature(comptime T: type, tensor: *Tensor(T), temperature: f32) !void {
    if (temperature <= 0) {
        return error.InvalidTemperature;
    }

    const temp = @as(T, @floatCast(temperature));
    for (tensor.data) |*val| {
        val.* = val.* / temp;
    }
}

/// Main sampling function that handles all sampling methods
pub fn sample(
    comptime T: type,
    tensor: *Tensor(T),
    config: SamplingConfig,
    rng: std.rand.Random,
    allocator: Allocator,
) !usize {
    var working_tensor = try tensor.copy();
    defer working_tensor.deinit();

    // Apply temperature scaling if not using greedy sampling
    if (config.method != .greedy and config.temperature != 1.0) {
        try apply_temperature(T, &working_tensor, config.temperature);
    }

    // Apply softmax if not using greedy sampling
    if (config.method != .greedy) {
        // TODO: Implement softmax function
        // try softmax(T, &working_tensor);
    }

    return switch (config.method) {
        .greedy => argmax(T, &working_tensor),
        .multinomial => multinomial_sampling(T, &working_tensor, rng),
        .top_k => if (config.top_k) |k|
            top_k_sampling(T, &working_tensor, k, rng, allocator)
        else
            error.MissingTopKValue,
    };
}

// Helper function to renormalize probabilities after top-k selection
pub fn renormalize_probs(comptime T: type, tensor: *Tensor(T), indices: []const usize, allocator: Allocator) !void {
    var sum: T = 0;

    // Calculate sum of selected probabilities
    for (indices) |idx| {
        sum += tensor.data[idx];
    }

    // Normalize selected probabilities
    if (sum > 0) {
        const scale = 1.0 / sum;
        for (indices) |idx| {
            tensor.data[idx] *= scale;
        }
    }

    // Zero out non-selected probabilities
    var mask = try allocator.alloc(bool, tensor.data.len);
    defer allocator.free(mask);
    @memset(mask, false);

    for (indices) |idx| {
        mask[idx] = true;
    }

    for (tensor.data, 0..) |*val, i| {
        if (!mask[i]) {
            val.* = 0;
        }
    }
}

pub fn scale_logits(comptime T: type, tensor: *Tensor(T), scale_factor: T) !void {
    for (tensor.data) |*value| {
        value.* *= scale_factor;
    }
}
