const std = @import("std");
const Tensor = @import("../core/tensor.zig").Tensor;
const ops = @import("../core/ops.zig");
const Allocator = std.mem.Allocator;

// ---------- Sampling ---------- //

/// Performs min-p sampling on a tensor of logits or probabilities
// fn min_p_sampling(comptime T: type, tensor: *const Tensor(T), min_p: T, rng: std.rand.Random, allocator: Allocator) !usize {
//     if (tensor.shape.len != 2 or tensor.shape[0] != 1) {
//         return error.InvalidInputShape;
//     }

//     // Ensure we're working with f32 since the softmax function is type-specific
//     if (T != f32) {
//         @compileError("min_p_sampling only supports f32 tensors due to softmax constraints");
//     }

//     // Make a copy of the tensor to work with (logits)
//     var logits = try tensor.copy();
//     defer logits.deinit();

//     // Calculate probabilities by applying softmax
//     var probs = try tensor.copy();
//     defer probs.deinit();

//     // Apply softmax on the last dimension (dim=1)
//     try ops.softmax(&probs, 1, allocator);

//     // Find the maximum probability (top_prob)
//     var top_prob: f32 = std.math.floatMin(f32);
//     for (probs.data) |val| {
//         top_prob = @max(top_prob, val);
//     }

//     // Calculate scaled threshold: min_p * top_prob
//     const scaled_min_p = min_p * top_prob;

//     // Create tokens_to_remove mask for probabilities < scaled_min_p
//     var tokens_to_remove = try allocator.alloc(bool, probs.data.len);
//     defer allocator.free(tokens_to_remove);

//     for (probs.data, 0..) |val, i| {
//         tokens_to_remove[i] = val < scaled_min_p;
//     }

//     // Sort indices by logits in descending order
//     var sorted_indices = try allocator.alloc(usize, logits.data.len);
//     defer allocator.free(sorted_indices);

//     for (0..logits.data.len) |i| {
//         sorted_indices[i] = i;
//     }

//     // Create a sorting context for descending order
//     const LogitsContext = struct {
//         logits: []const f32,

//         pub fn lessThan(self: @This(), a: usize, b: usize) bool {
//             return self.logits[a] > self.logits[b]; // Descending order
//         }
//     };

//     std.sort.insertion(usize, sorted_indices, LogitsContext{ .logits = logits.data }, LogitsContext.lessThan);

//     // Get sorted tokens_to_remove
//     var sorted_indices_to_remove = try allocator.alloc(bool, tokens_to_remove.len);
//     defer allocator.free(sorted_indices_to_remove);

//     for (sorted_indices, 0..) |idx, i| {
//         sorted_indices_to_remove[i] = tokens_to_remove[idx];
//     }

//     // Ensure at least min_tokens_to_keep tokens remain
//     const min_tokens_to_keep: usize = 1;

//     for (0..@min(min_tokens_to_keep, sorted_indices_to_remove.len)) |i| {
//         sorted_indices_to_remove[i] = false;
//     }

//     // Apply filtered values back to original ordering (scatter operation)
//     @memset(tokens_to_remove, false); // Reset all to false

//     for (sorted_indices, 0..) |idx, i| {
//         tokens_to_remove[idx] = sorted_indices_to_remove[i];
//     }

//     // Filter the logits (set removed tokens to negative infinity)
//     const neg_inf: f32 = -std.math.inf(f32);
//     for (logits.data, 0..) |*val, i| {
//         if (tokens_to_remove[i]) {
//             val.* = neg_inf;
//         }
//     }

//     // Apply softmax again for sampling
//     try ops.softmax(&logits, 1, allocator);

//     // Multinomial sampling on the filtered distribution
//     const r = rng.float(f32);
//     var cumsum: f32 = 0;

//     for (logits.data, 0..) |val, i| {
//         cumsum += val;
//         if (r < cumsum) {
//             return i;
//         }
//     }

//     // Fallback: return the highest probability token
//     return sorted_indices[0];
// }

fn min_p_sampling(tensor: *const Tensor(f32), min_p: f32, rng: std.rand.Random, allocator: Allocator) !usize {
    if (tensor.shape.len != 2 or tensor.shape[0] != 1) {
        return error.InvalidInputShape;
    }

    const vocab_size = tensor.shape[1];
    const data = tensor.data;

    // Find max logit value for numerical stability (used in softmax)
    var max_logit: f32 = -std.math.inf(f32);
    for (data) |val| {
        max_logit = @max(max_logit, val);
    }

    // OPTIMIZATION 1: Pre-allocate all needed arrays at once
    // Allocate a single buffer for all our temporary arrays to reduce allocation overhead
    const buf_size = vocab_size * 3; // probs + indices + mask
    var buffer = try allocator.alloc(f32, buf_size);
    defer allocator.free(buffer);

    // Split the buffer into our needed arrays
    var probs = buffer[0..vocab_size];

    // OPTIMIZATION 2: Calculate softmax and find max probability in a single pass
    // Calculate softmax and track the max probability
    var sum: f32 = 0;
    var max_prob: f32 = 0;

    for (data, 0..) |logit, i| {
        // Apply softmax: exp(logit - max_logit) for numerical stability
        const val = if (logit - max_logit > -88.0) std.math.exp(logit - max_logit) else 0;
        probs[i] = val;
        sum += val;

        if (val > max_prob) {
            max_prob = val;
        }
    }

    // Normalize probabilities
    const inv_sum = 1.0 / sum;
    for (probs) |*prob| {
        prob.* *= inv_sum;
    }

    // After normalization, we need to find the max probability again
    max_prob = 0;
    for (probs) |prob| {
        max_prob = @max(max_prob, prob);
    }

    // Calculate the min-p threshold
    const threshold = min_p * max_prob;

    // OPTIMIZATION 3: Use direct token filtering without extra arrays
    // Instead of Boolean masks, track valid indices directly
    var valid_indices = try allocator.alloc(usize, vocab_size);
    defer allocator.free(valid_indices);

    var valid_count: usize = 0;

    for (probs, 0..) |prob, i| {
        if (prob >= threshold) {
            valid_indices[valid_count] = i;
            valid_count += 1;
        }
    }

    // Ensure at least one token is kept (highest probability)
    if (valid_count == 0) {
        // Find the index of the max probability
        var max_idx: usize = 0;
        var max_val: f32 = probs[0];

        for (probs, 0..) |val, i| {
            if (val > max_val) {
                max_val = val;
                max_idx = i;
            }
        }

        valid_indices[0] = max_idx;
        valid_count = 1;
    }

    // OPTIMIZATION 4: Direct sampling without re-softmax
    // Sample from valid indices based on their normalized probabilities
    var valid_sum: f32 = 0;
    for (valid_indices[0..valid_count]) |idx| {
        valid_sum += probs[idx];
    }

    // Generate random number and sample
    const r = rng.float(f32) * valid_sum;
    var cumsum: f32 = 0;

    for (valid_indices[0..valid_count]) |idx| {
        cumsum += probs[idx];
        if (r < cumsum) {
            return idx;
        }
    }

    // Fallback: return the first valid index
    return valid_indices[0];
}

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
    min_p, // Min-p sampling with dynamic threshold
};

/// Configuration for sampling parameters
pub const SamplingConfig = struct {
    method: SamplingMethod,
    temperature: f32 = 0.5, // Temperature for softmax
    top_k: ?usize = 3, // Number of top tokens to consider (only for top_k)
    min_p: ?f32 = 0.05, // Min-p threshold scaling factor (only for min_p)
    min_tokens_to_keep: usize = 1, // Minimum tokens to keep for sampling methods
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
fn apply_temperature(tensor: *Tensor(f32), temperature: f32) !void {
    if (temperature <= 0) {
        return error.InvalidTemperature;
    }

    for (tensor.data) |*val| {
        val.* = val.* / temperature;
    }
}

/// Main sampling function that handles all sampling methods
pub fn sample(
    tensor: *Tensor(f32),
    config: SamplingConfig,
    rng: std.rand.Random,
    allocator: Allocator,
) !usize {
    // Make a copy of the tensor to work with
    var working_tensor = try tensor.copy();
    defer working_tensor.deinit();

    // Apply temperature scaling if not using greedy sampling and temperature != 1.0
    if (config.method != .greedy and config.temperature != 1.0) {
        try apply_temperature(&working_tensor, config.temperature);
    }

    // Choose sampling method
    switch (config.method) {
        .greedy => {
            return argmax(f32, &working_tensor);
        },
        .multinomial => {
            // Apply softmax before sampling
            try ops.softmax(&working_tensor, 1, allocator);
            return multinomial_sampling(f32, &working_tensor, rng);
        },
        .top_k => {
            const k = config.top_k orelse return error.MissingTopKValue;
            // Apply softmax before sampling
            try ops.softmax(&working_tensor, 1, allocator);
            return top_k_sampling(f32, &working_tensor, k, rng, allocator);
        },
        .min_p => {
            const p = config.min_p orelse return error.MissingMinPValue;
            return min_p_sampling(&working_tensor, p, rng, allocator);
        },
    }
}
// ---------- Sampling Utils ---------- //

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
