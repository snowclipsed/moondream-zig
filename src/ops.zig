const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const max_items_per_row = 6; // Number of elements to show per row
const max_rows = 8; // Maximum number of rows to show before truncating
const Tensor = @import("tensor.zig").Tensor;
const StabilityError = @import("tensor.zig").StabilityError;
const simdmatmul = @import("matmul.zig");
const testing = std.testing;
const expectEqual = testing.expectEqual;
const expectError = testing.expectError;

// Tensor Operations
pub fn transpose(comptime T: type, tensor: *Tensor(T)) !void {
    if (tensor.shape.len != 2) return error.UnsupportedDimension;

    const rows = tensor.shape[0];
    const cols = tensor.shape[1];
    var new_data = try tensor.allocator.alignedAlloc(@TypeOf(tensor.data[0]), 32, rows * cols);

    for (0..rows) |i| {
        for (0..cols) |j| {
            new_data[j * rows + i] = tensor.data[i * cols + j];
        }
    }

    tensor.allocator.free(tensor.data);
    tensor.data = new_data;

    // Swap dimensions
    const temp = tensor.shape[0];
    tensor.shape[0] = tensor.shape[1];
    tensor.shape[1] = temp;
}

/// Transposes a tensor by swapping specified dimensions
/// Parameters:
/// - tensor: Input tensor to transpose
/// - dim0: First dimension to swap
/// - dim1: Second dimension to swap
pub fn transposeAxes(comptime T: type, tensor: *Tensor(T), dim0: usize, dim1: usize) !void {
    if (dim0 >= tensor.shape.len or dim1 >= tensor.shape.len) {
        return error.InvalidDimension;
    }

    // Calculate strides for the current shape
    var strides = try tensor.allocator.alloc(usize, tensor.shape.len);
    defer tensor.allocator.free(strides);

    strides[tensor.shape.len - 1] = 1;
    var i: usize = tensor.shape.len - 1;
    while (i > 0) : (i -= 1) {
        strides[i - 1] = strides[i] * tensor.shape[i];
    }

    // Create new shape with swapped dimensions
    var new_shape = try tensor.allocator.alloc(usize, tensor.shape.len);
    errdefer tensor.allocator.free(new_shape);

    for (tensor.shape, 0..) |dim, idx| {
        if (idx == dim0) {
            new_shape[idx] = tensor.shape[dim1];
        } else if (idx == dim1) {
            new_shape[idx] = tensor.shape[dim0];
        } else {
            new_shape[idx] = dim;
        }
    }

    // Allocate memory for transposed data
    var new_data = try tensor.allocator.alignedAlloc(T, 32, tensor.data.len);
    errdefer tensor.allocator.free(new_data);

    // Calculate new strides
    var new_strides = try tensor.allocator.alloc(usize, tensor.shape.len);
    defer tensor.allocator.free(new_strides);

    new_strides[tensor.shape.len - 1] = 1;
    i = tensor.shape.len - 1;
    while (i > 0) : (i -= 1) {
        new_strides[i - 1] = new_strides[i] * new_shape[i];
    }

    // Create coordinate arrays
    var coords = try tensor.allocator.alloc(usize, tensor.shape.len);
    defer tensor.allocator.free(coords);
    @memset(coords, 0);

    // Perform the transpose operation
    const total_elements = tensor.data.len;
    var idx: usize = 0;
    while (idx < total_elements) : (idx += 1) {
        // Calculate source coordinates
        var remaining = idx;
        for (0..tensor.shape.len) |dim| {
            coords[dim] = remaining / new_strides[dim];
            remaining = remaining % new_strides[dim];
        }

        // Swap coordinates for the transposed dimensions
        const temp = coords[dim0];
        coords[dim0] = coords[dim1];
        coords[dim1] = temp;

        // Calculate source index using original strides
        var src_idx: usize = 0;
        for (0..tensor.shape.len) |dim| {
            src_idx += coords[dim] * strides[dim];
        }

        new_data[idx] = tensor.data[src_idx];
    }

    // Update tensor with new data and shape
    tensor.allocator.free(tensor.data);
    tensor.data = new_data;
    tensor.allocator.free(tensor.shape);
    tensor.shape = new_shape;
}

pub fn add(comptime T: type, tensor: *Tensor(T), other: Tensor(T)) !void {
    if (!std.mem.eql(usize, tensor.shape, other.shape)) {
        std.debug.print("tensor shape: {d}\n", .{tensor.shape});
        std.debug.print("other shape: {d}\n", .{other.shape});
        return error.ShapeMismatch;
    }

    for (tensor.data, 0..) |_, i| {
        tensor.data[i] += other.data[i];
    }
}

pub fn subtract(comptime T: type, tensor: *Tensor(T), other: Tensor(T)) !void {
    if (!std.mem.eql(usize, tensor.shape, other.shape)) {
        return error.ShapeMismatch;
    }

    for (tensor.data, 0..) |_, i| {
        tensor.data[i] -= other.data[i];
    }
}

pub fn multiply(comptime T: type, tensor: *Tensor(T), other: Tensor(T)) !void {
    if (!std.mem.eql(usize, tensor.shape, other.shape)) {
        return error.ShapeMismatch;
    }

    for (tensor.data, 0..) |_, i| {
        tensor.data[i] *= other.data[i];
    }
}

pub fn scalarAdd(comptime T: type, tensor: *Tensor(T), scalar: T) void {
    for (tensor.data, 0..) |_, i| {
        tensor.data[i] += scalar;
    }
}

pub fn scalarMultiply(comptime T: type, tensor: *Tensor(T), scalar: T) void {
    for (tensor.data, 0..) |_, i| {
        tensor.data[i] *= scalar;
    }
}

/// Performs broadcasted addition between two tensors.
/// The smaller tensor is broadcast to match the shape of the larger tensor along
/// matching dimensions from right to left.
/// For example: [seq_len, dim] + [dim] -> broadcasts [dim] across seq_len
/// Performs broadcasted addition between two tensors.
/// The smaller tensor is broadcast to match the shape of the larger tensor along
/// matching dimensions from right to left.
/// For example: [seq_len, dim] + [dim] -> broadcasts [dim] across seq_len
pub fn broadcast_add(comptime T: type, a: *Tensor(T), b: Tensor(T)) !void {
    // Check that shapes can be broadcast
    if (b.shape.len > a.shape.len) {
        return error.InvalidBroadcast;
    }

    // Check that dimensions match from right to left
    for (0..b.shape.len) |i| {
        const a_dim = a.shape[a.shape.len - 1 - i];
        const b_dim = b.shape[b.shape.len - 1 - i];
        if (b_dim != a_dim and b_dim != 1) {
            return error.IncompatibleBroadcast;
        }
    }

    // For common case of [seq_len, dim] + [dim]
    if (a.shape.len == 2 and b.shape.len == 1 and b.shape[0] == a.shape[1]) {
        const seq_len = a.shape[0];
        const dim = a.shape[1];

        // Add bias to each row
        var i: usize = 0;
        while (i < seq_len) : (i += 1) {
            const row_start = i * dim;
            for (0..dim) |j| {
                a.data[row_start + j] += b.data[j];
            }
        }
        return;
    }

    // Handle general case
    const total_elements = blk: {
        var prod: usize = 1;
        for (a.shape) |dim| {
            prod *= dim;
        }
        break :blk prod;
    };

    // For each element in the output
    var i: usize = 0;
    while (i < total_elements) : (i += 1) {
        // Calculate indices for both tensors
        var a_coords = try a.allocator.alloc(usize, a.shape.len);
        defer a.allocator.free(a_coords);
        var temp = i;

        // Convert flat index to coordinates
        for (0..a.shape.len) |j| {
            const rev_j = a.shape.len - 1 - j;
            a_coords[rev_j] = temp % a.shape[rev_j];
            temp /= a.shape[rev_j];
        }

        // Calculate corresponding b index
        var b_idx: usize = 0;
        var b_stride: usize = 1;

        for (0..b.shape.len) |j| {
            const b_j = b.shape.len - 1 - j;
            const a_j = a.shape.len - 1 - j;
            const coord = a_coords[a_j] % b.shape[b_j];
            b_idx += coord * b_stride;
            b_stride *= b.shape[b_j];
        }

        // Add values
        a.data[i] += b.data[b_idx];
    }
}

pub fn matmul(comptime T: type, tensor: *Tensor(T), other: Tensor(T)) !Tensor(T) {
    if (tensor.shape.len != 2 or other.shape.len != 2) {
        return error.UnsupportedDimension;
    }
    if (tensor.shape[1] != other.shape[0]) {
        return error.IncompatibleDimensions;
    }

    const m = tensor.shape[0];
    const k = tensor.shape[1];
    const n = other.shape[1];

    var result = try Tensor(@TypeOf(tensor.data[0])).init(tensor.allocator, &[_]usize{ m, n });

    for (0..m) |i| {
        for (0..n) |j| {
            var sum: @TypeOf(tensor.data[0]) = 0;
            for (0..k) |l| {
                sum += tensor.data[i * k + l] * other.data[l * n + j];
            }
            result.data[i * n + j] = sum;
        }
    }

    return result;
}

pub fn outer(comptime T: type, tensor: Tensor(T), other: Tensor(T)) !Tensor(T) {
    if (tensor.shape.len != 1 or other.shape.len != 1) {
        return error.InvalidDimensions;
    }

    const m = tensor.shape[0];
    const n = other.shape[0];

    var result = try Tensor(@TypeOf(tensor.data[0])).init(tensor.allocator, &[_]usize{ m, n });
    errdefer result.deinit();

    for (0..m) |i| {
        for (0..n) |j| {
            result.data[i * n + j] = tensor.data[i] * other.data[j];
        }
    }

    return result;
}

pub fn accumulate(comptime T: type, tensor: *Tensor(T), other: Tensor(T)) !void {
    if (!std.mem.eql(usize, tensor.shape, other.shape)) {
        return error.ShapeMismatch;
    }

    var temp = try tensor.copy();
    defer temp.deinit();

    for (tensor.data, 0..) |_, i| {
        tensor.data[i] = temp.data[i] + other.data[i];
        if (i > 0) {
            tensor.data[i] += tensor.data[i - 1];
        }
    }
}

/// Gets a chunk of a tensor along a specified dimension.
/// For example, if tensor has shape [2,6] and we chunk along dim 1 with chunk_size 2,
/// we get 3 tensors of shape [2,2]
pub fn getChunk(comptime T: type, tensor: Tensor(T), dim: usize, chunk_idx: usize, num_chunks: usize) !Tensor(T) {
    // Validate inputs
    if (dim >= tensor.shape.len) {
        return error.InvalidDimension;
    }

    const dim_size = tensor.shape[dim];
    if (num_chunks == 0 or dim_size < num_chunks) {
        return error.InvalidNumChunks;
    }

    if (chunk_idx >= num_chunks) {
        return error.InvalidChunkIndex;
    }

    // Calculate chunk size and start/end indices
    const chunk_size = dim_size / num_chunks;
    if (chunk_size * num_chunks != dim_size) {
        return error.UnevenChunkSize;
    }

    const start_idx = chunk_idx * chunk_size;

    // Create new shape array
    var new_shape = try tensor.allocator.alloc(usize, tensor.shape.len);
    errdefer tensor.allocator.free(new_shape);

    for (tensor.shape, 0..) |s, i| {
        new_shape[i] = if (i == dim) chunk_size else s;
    }

    // Create result tensor
    var result = try Tensor(T).init(tensor.allocator, new_shape);
    tensor.allocator.free(new_shape);
    errdefer result.deinit();

    // Calculate strides for the input tensor
    var strides = try tensor.allocator.alloc(usize, tensor.shape.len);
    defer tensor.allocator.free(strides);

    strides[tensor.shape.len - 1] = 1;
    var i = tensor.shape.len - 1;
    while (i > 0) : (i -= 1) {
        strides[i - 1] = strides[i] * tensor.shape[i];
    }

    // Copy data
    const total_elements = result.data.len;
    var result_idx: usize = 0;
    var coords = try tensor.allocator.alloc(usize, tensor.shape.len);
    defer tensor.allocator.free(coords);
    @memset(coords, 0);

    while (result_idx < total_elements) : (result_idx += 1) {
        // Calculate source coordinates
        var temp = result_idx;
        var src_idx: usize = 0;

        for (0..tensor.shape.len) |j| {
            const rev_j = tensor.shape.len - 1 - j;
            if (rev_j == dim) {
                coords[rev_j] = temp % chunk_size + start_idx;
            } else {
                coords[rev_j] = temp % tensor.shape[rev_j];
            }
            src_idx += coords[rev_j] * strides[rev_j];
            temp /= if (rev_j == dim) chunk_size else tensor.shape[rev_j];
        }

        result.data[result_idx] = tensor.data[src_idx];
    }

    return result;
}

// Calculate index in flattened array from n-dimensional coordinates
pub fn calculateIndex(shape: []const usize, coords: []const usize) usize {
    var index: usize = 0;
    var stride: usize = 1;
    var i: usize = shape.len;
    while (i > 0) {
        i -= 1;
        index += coords[i] * stride;
        stride *= shape[i];
    }
    return index;
}

pub fn checkStability(comptime T: type, tensor: Tensor(T)) !void {
    const info = try getStabilityInfo(T, tensor);
    if (info.has_nan) {
        return StabilityError.HasNaN;
    }
    if (info.has_pos_inf) {
        return StabilityError.HasPositiveInfinity;
    }
    if (info.has_neg_inf) {
        return StabilityError.HasNegativeInfinity;
    }
}

pub fn getStabilityInfo(comptime T: type, tensor: Tensor(T)) !Tensor(T).StabilityInfo {
    var info = Tensor(@TypeOf(tensor.data[0])).StabilityInfo{};

    switch (@typeInfo(@TypeOf(tensor.data[0]))) {
        .Float => {
            for (tensor.data, 0..) |value, i| {
                if (std.math.isNan(value)) {
                    info.has_nan = true;
                    info.nan_count += 1;
                    if (info.first_nan_index == null) {
                        info.first_nan_index = i;
                    }
                } else if (std.math.isPositiveInf(value)) {
                    info.has_pos_inf = true;
                    info.pos_inf_count += 1;
                    if (info.first_pos_inf_index == null) {
                        info.first_pos_inf_index = i;
                    }
                } else if (std.math.isNegativeInf(value)) {
                    info.has_neg_inf = true;
                    info.neg_inf_count += 1;
                    if (info.first_neg_inf_index == null) {
                        info.first_neg_inf_index = i;
                    }
                }
            }
        },
        else => {},
    }

    return info;
}

pub fn isStable(comptime T: type, tensor: Tensor(T)) !bool {
    const info = try getStabilityInfo(T, tensor);
    return !info.has_nan and !info.has_pos_inf and !info.has_neg_inf;
}

pub fn hasNaN(comptime T: type, tensor: Tensor(T)) !bool {
    const info = try getStabilityInfo(T, tensor);
    return info.has_nan;
}

pub fn hasInf(comptime T: type, tensor: Tensor(T)) !bool {
    const info = try getStabilityInfo(T, tensor);
    return info.has_pos_inf or info.has_neg_inf;
}

pub fn replaceUnstable(comptime T: type, tensor: *Tensor(T), replacement: T) !void {
    switch (@typeInfo(@TypeOf(tensor.data[0]))) {
        .Float => {
            for (tensor.data) |*value| {
                if (std.math.isNan(value.*) or std.math.isInf(value.*)) {
                    value.* = replacement;
                }
            }
        },
        else => {},
    }
}

pub fn concat(comptime T: type, tensor: Tensor(T), other: Tensor(T), dim: usize) !Tensor(T) {
    // Verify tensors can be concatenated
    try verifyCompatibleForConcat(T, tensor, other, dim);

    // Calculate new shape
    var new_shape = try tensor.allocator.alloc(usize, tensor.shape.len);
    errdefer tensor.allocator.free(new_shape);

    for (tensor.shape, 0..) |s, i| {
        new_shape[i] = if (i == dim) s + other.shape[i] else s;
    }

    // Create new tensor with combined shape
    var result = try Tensor(T).init(tensor.allocator, new_shape);
    errdefer result.deinit();
    tensor.allocator.free(new_shape);

    // Early return for zero-sized tensors
    if (calculateSize(result.shape) == 0) {
        return result;
    }

    // Helper function to get strides
    var strides = try tensor.allocator.alloc(usize, tensor.shape.len);
    defer tensor.allocator.free(strides);

    // Calculate strides for the result tensor
    strides[strides.len - 1] = 1;
    var i: usize = strides.len - 1;
    while (i > 0) {
        i -= 1;
        strides[i] = strides[i + 1] * result.shape[i + 1];
    }

    // Copy data from first tensor
    const first_size = calculateSize(tensor.shape);
    if (first_size > 0) {
        var coords = try tensor.allocator.alloc(usize, tensor.shape.len);
        defer tensor.allocator.free(coords);
        @memset(coords, 0);

        var idx: usize = 0;
        while (idx < first_size) : (idx += 1) {
            // Calculate source and destination indices
            var src_idx: usize = 0;
            var dst_idx: usize = 0;

            for (coords, 0..) |c, j| {
                if (j == dim) {
                    src_idx += c * (if (j + 1 < tensor.shape.len) blk: {
                        var prod: usize = 1;
                        for (j + 1..tensor.shape.len) |k| {
                            prod *= tensor.shape[k];
                        }
                        break :blk prod;
                    } else 1);
                    dst_idx += c * strides[j];
                } else {
                    src_idx += c * (if (j + 1 < tensor.shape.len) blk: {
                        var prod: usize = 1;
                        for (j + 1..tensor.shape.len) |k| {
                            prod *= tensor.shape[k];
                        }
                        break :blk prod;
                    } else 1);
                    dst_idx += c * strides[j];
                }
            }

            result.data[dst_idx] = tensor.data[src_idx];

            // Update coordinates
            var j = coords.len;
            while (j > 0) {
                j -= 1;
                coords[j] += 1;
                if (coords[j] < tensor.shape[j]) break;
                coords[j] = 0;
            }
        }
    }

    // Copy data from second tensor
    const second_size = calculateSize(other.shape);
    if (second_size > 0) {
        var coords = try tensor.allocator.alloc(usize, other.shape.len);
        defer tensor.allocator.free(coords);
        @memset(coords, 0);

        var idx: usize = 0;
        while (idx < second_size) : (idx += 1) {
            // Calculate source and destination indices
            var src_idx: usize = 0;
            var dst_idx: usize = 0;

            for (coords, 0..) |c, j| {
                if (j == dim) {
                    src_idx += c * (if (j + 1 < other.shape.len) blk: {
                        var prod: usize = 1;
                        for (j + 1..other.shape.len) |k| {
                            prod *= other.shape[k];
                        }
                        break :blk prod;
                    } else 1);
                    dst_idx += (c + tensor.shape[dim]) * strides[j];
                } else {
                    src_idx += c * (if (j + 1 < other.shape.len) blk: {
                        var prod: usize = 1;
                        for (j + 1..other.shape.len) |k| {
                            prod *= other.shape[k];
                        }
                        break :blk prod;
                    } else 1);
                    dst_idx += c * strides[j];
                }
            }

            result.data[dst_idx] = other.data[src_idx];

            // Update coordinates
            var j = coords.len;
            while (j > 0) {
                j -= 1;
                coords[j] += 1;
                if (coords[j] < other.shape[j]) break;
                coords[j] = 0;
            }
        }
    }

    return result;
}

fn verifyCompatibleForConcat(comptime T: type, tensor: Tensor(T), other: Tensor(T), dim: usize) !void {
    // Check if dimension is valid
    if (dim >= tensor.shape.len) {
        return error.InvalidDimension;
    }

    // Check if tensors have same number of dimensions
    if (tensor.shape.len != other.shape.len) {
        return error.DimensionMismatch;
    }

    // Check if all dimensions except concat dim are equal
    for (tensor.shape, 0..) |s, i| {
        if (i != dim and s != other.shape[i]) {
            std.debug.print("tensor shape: {d}\n", .{tensor.shape});
            std.debug.print("other shape: {d}\n", .{other.shape});
            return error.IncompatibleShapes;
        }
    }
}

fn calculateSize(shape: []const usize) usize {
    var size: usize = 1;
    for (shape) |dim| {
        size *= dim;
    }
    return size;
}

pub fn createRandomTensor(comptime T: type, allocator: std.mem.Allocator, shape: []const usize, seed: u64) !Tensor(T) {
    var tensor = try Tensor(T).init(allocator, shape);
    errdefer tensor.deinit();

    var rng = std.rand.DefaultPrng.init(seed);
    for (tensor.data) |*val| {
        val.* = rng.random().float(T) * 2.0 - 1.0; // Values between -1 and 1
    }
    return tensor;
}

pub fn zeros(comptime T: type, allocator: Allocator, shape: []const usize) !Tensor(T) {
    // Calculate total size
    var total_size: usize = 1;
    for (shape) |dim| {
        total_size *= dim;
    }

    // Allocate aligned data array
    const alignment = 32;
    const data = try allocator.alignedAlloc(T, alignment, total_size);
    // Initialize all elements to zero
    @memset(data, 0);

    // Create tensor shape
    const tensor_shape = try allocator.alloc(usize, shape.len);
    @memcpy(tensor_shape, shape);

    // Return initialized tensor
    return Tensor(T){
        .data = data,
        .shape = tensor_shape,
        .allocator = allocator,
    };
}

// ----------------------------------------------------------------------------

pub fn layerNorm(comptime T: type, input: Tensor(T), weight: Tensor(T), bias: Tensor(T), eps: T) !Tensor(T) {
    // Check input stability
    try checkStability(T, input);
    try checkStability(T, weight);
    try checkStability(T, bias);

    // Validate epsilon
    if (eps <= 0) {
        return error.InvalidEpsilon;
    }

    // Input validation
    if (input.shape.len < 1) {
        return error.InvalidShape;
    }
    const last_dim = input.shape[input.shape.len - 1];

    if (weight.shape.len != 1 or weight.shape[0] != last_dim) {
        return error.InvalidWeightShape;
    }
    if (bias.shape.len != 1 or bias.shape[0] != last_dim) {
        return error.InvalidBiasShape;
    }

    // Calculate size of dimensions before the last dimension
    var leading_dims: usize = 1;
    for (input.shape[0 .. input.shape.len - 1]) |dim| {
        leading_dims *= dim;
    }

    // Create output tensor with same shape as input
    var output = try input.copy();
    errdefer output.deinit();

    // Compute mean and variance for each feature vector
    var i: usize = 0;
    while (i < leading_dims) : (i += 1) {
        const start_idx = i * last_dim;
        const end_idx = start_idx + last_dim;

        // Calculate mean
        var mean: T = 0;
        for (start_idx..end_idx) |j| {
            mean += input.data[j];
        }
        mean /= @as(T, @floatFromInt(last_dim));

        // Calculate variance
        var variance: T = 0;
        for (start_idx..end_idx) |j| {
            const diff = input.data[j] - mean;
            variance += diff * diff;
        }
        variance /= @as(T, @floatFromInt(last_dim));

        // Check for numerical stability in variance
        if (variance < -eps) {
            return error.NegativeVariance;
        }

        // Add stability checks for the normalization process
        const std_dev = @sqrt(variance + eps);
        if (std_dev == 0) {
            return error.ZeroStandardDeviation;
        }

        // Normalize and apply scale and bias
        for (start_idx..end_idx) |j| {
            const feature_idx = j - start_idx;
            const normalized = (input.data[j] - mean) / std_dev;
            const scaled = normalized * weight.data[feature_idx];
            const final_value = scaled + bias.data[feature_idx];

            // Check for stability of computed value
            if (std.math.isNan(final_value)) {
                return error.ComputedNaN;
            }
            if (std.math.isInf(final_value)) {
                return error.ComputedInfinity;
            }

            output.data[j] = final_value;
        }
    }

    // Final stability check on output
    try checkStability(T, output);
    return output;
}

const LayerNormError = error{
    InvalidShape,
    InvalidWeightShape,
    InvalidBiasShape,
    InvalidEpsilon,
    NegativeVariance,
    ZeroStandardDeviation,
    ComputedNaN,
    ComputedInfinity,
} || StabilityError;

/// All possible errors from tensor operations and freqs computation
const FreqsError = error{
    // Tensor initialization errors
    TensorTooLarge,
    IncompatibleShape,

    // Input validation errors
    DimensionTooSmall,
    DimensionNotEven,
    EndTooSmall,
    ThetaTooSmall,
    InvalidShape,

    // Computation errors
    ComputationOverflow,
    NumericalInstability,

    // Memory errors
    OutOfMemory,
};

/// Creates a tensor containing precomputed complex frequencies for rotary embeddings
/// Returns a tensor of shape [end, dim//2, 2] where the last dimension contains [real, imag] parts
pub fn precomputeFreqsCis(
    comptime T: type,
    allocator: std.mem.Allocator,
    dim: usize,
    end: usize,
    theta: T,
) FreqsError!Tensor(T) {
    // Input validation
    if (dim <= 0) return error.DimensionTooSmall;
    if (dim % 2 != 0) return error.DimensionNotEven;
    if (end <= 0) return error.EndTooSmall;
    if (theta <= 0) return error.ThetaTooSmall;

    // 1. Create initial frequencies
    var freqs = try Tensor(T).init(allocator, &[_]usize{dim / 2});
    errdefer freqs.deinit();

    const dim_float: T = @floatFromInt(dim);
    for (0..dim / 2) |i| {
        const idx_float: T = @floatFromInt(i * 2);
        const power = idx_float / dim_float;

        // Check for potential overflow in power calculation
        if (power < -1000 or power > 1000) {
            return error.ComputationOverflow;
        }

        const theta_power = std.math.pow(T, theta, power);
        // Check for division by zero or overflow
        if (theta_power == 0 or !std.math.isFinite(theta_power)) {
            return error.NumericalInstability;
        }

        freqs.data[i] = 1.0 / theta_power;

        // Check for numerical stability
        if (!std.math.isFinite(freqs.data[i])) {
            return error.NumericalInstability;
        }
    }

    // 2. Create time tensor
    var time_range = try Tensor(T).init(allocator, &[_]usize{ end, 1 });
    errdefer time_range.deinit();

    for (0..end) |i| {
        time_range.data[i] = @floatFromInt(i);
    }

    // 3. Reshape freqs and prepare for multiplication
    try freqs.reshape(&[_]usize{ 1, dim / 2 });

    // Initialize result tensor for the multiplication
    var freq_matrix = try Tensor(T).init(allocator, &[_]usize{ end, dim / 2 });
    errdefer freq_matrix.deinit();

    // Perform the outer product manually
    for (0..end) |i| {
        for (0..dim / 2) |j| {
            const product = time_range.data[i] * freqs.data[j];
            if (!std.math.isFinite(product)) {
                return error.NumericalInstability;
            }
            freq_matrix.data[i * (dim / 2) + j] = product;
        }
    }

    // 4. Calculate exp(i * freqs)
    var result = try Tensor(T).init(allocator, &[_]usize{ end, dim / 2, 2 });
    errdefer result.deinit();

    // Calculate cos and sin values
    for (0..end) |i| {
        for (0..dim / 2) |j| {
            const x = freq_matrix.data[i * (dim / 2) + j];
            const cos_val = @cos(x);
            const sin_val = @sin(x);

            // Check for numerical stability
            if (!std.math.isFinite(cos_val) or !std.math.isFinite(sin_val)) {
                return error.NumericalInstability;
            }

            // Real part (cos)
            result.data[i * (dim / 2) * 2 + j * 2] = cos_val;
            // Imaginary part (sin)
            result.data[i * (dim / 2) * 2 + j * 2 + 1] = sin_val;
        }
    }

    // Cleanup intermediate tensors
    freqs.deinit();
    time_range.deinit();
    freq_matrix.deinit();

    return result;
}

const RotaryError = error{
    InvalidDimension,
    InvalidShape,
    ShapeMismatch,
    InvalidPositionIds,
} || FreqsError;

/// Applies rotary position embeddings to the input tensor
///
/// Parameters:
///   x: Input tensor of shape [num_heads, seq_len, head_dim]
///   freqs_cis: Precomputed frequencies of shape [seq_len, rot_dim/2, 2]
///   position_ids: Position indices of shape [seq_len]
///   rot_dim: Dimension to rotate (must be <= head_dim)
///   interleave: Whether complex numbers are stored in interleaved format
///
/// Returns:
///   Tensor with rotary embeddings applied
pub fn applyRotaryEmb(
    comptime T: type,
    x: Tensor(T),
    freqs_cis: Tensor(T),
    position_ids: Tensor(T),
    rot_dim: usize,
    interleave: bool,
    allocator: std.mem.Allocator,
) RotaryError!Tensor(T) {
    // Validate dimensions
    if (x.shape.len != 3) return error.InvalidShape; // [num_heads, seq_len, head_dim]
    if (freqs_cis.shape.len != 3) return error.InvalidShape;
    if (position_ids.shape.len != 1) return error.InvalidShape;

    const num_heads = x.shape[0];
    const seq_len = x.shape[1];
    const head_dim = x.shape[2];

    if (rot_dim > head_dim) return error.InvalidDimension;
    if (rot_dim != freqs_cis.shape[1] * 2) return error.InvalidDimension;

    // Split x into rotated and pass-through parts
    var x_rot = try Tensor(T).init(allocator, &[_]usize{ num_heads, seq_len, rot_dim });
    defer x_rot.deinit();
    var x_pass = try Tensor(T).init(allocator, &[_]usize{ num_heads, seq_len, head_dim - rot_dim });
    defer x_pass.deinit();

    // Copy data to split tensors
    for (0..num_heads) |h| {
        for (0..seq_len) |s| {
            // Copy rotated part
            for (0..rot_dim) |d| {
                const src_idx = h * seq_len * head_dim +
                    s * head_dim + d;
                const dst_idx = h * seq_len * rot_dim +
                    s * rot_dim + d;
                x_rot.data[dst_idx] = x.data[src_idx];
            }
            // Copy pass-through part
            for (0..head_dim - rot_dim) |d| {
                const src_idx = h * seq_len * head_dim +
                    s * head_dim + (d + rot_dim);
                const dst_idx = h * seq_len * (head_dim - rot_dim) +
                    s * (head_dim - rot_dim) + d;
                x_pass.data[dst_idx] = x.data[src_idx];
            }
        }
    }

    // Extract real and imaginary parts
    var xq_r: Tensor(T) = undefined;
    var xq_i: Tensor(T) = undefined;

    if (interleave) {
        // Reshape for interleaved format
        try x_rot.reshape(&[_]usize{ num_heads, seq_len, rot_dim / 2, 2 });
        xq_r = try Tensor(T).init(allocator, &[_]usize{ num_heads, seq_len, rot_dim / 2 });
        xq_i = try Tensor(T).init(allocator, &[_]usize{ num_heads, seq_len, rot_dim / 2 });

        // Extract components
        for (0..num_heads * seq_len * (rot_dim / 2)) |i| {
            xq_r.data[i] = x_rot.data[i * 2];
            xq_i.data[i] = x_rot.data[i * 2 + 1];
        }
    } else {
        const half_dim = rot_dim / 2;
        xq_r = try Tensor(T).init(allocator, &[_]usize{ num_heads, seq_len, half_dim });
        xq_i = try Tensor(T).init(allocator, &[_]usize{ num_heads, seq_len, half_dim });

        // Copy first and second halves
        for (0..num_heads) |h| {
            for (0..seq_len) |s| {
                for (0..half_dim) |d| {
                    const src_idx = h * seq_len * rot_dim +
                        s * rot_dim;
                    xq_r.data[
                        h * seq_len * half_dim +
                            s * half_dim + d
                    ] = x_rot.data[src_idx + d];
                    xq_i.data[
                        h * seq_len * half_dim +
                            s * half_dim + d
                    ] = x_rot.data[src_idx + half_dim + d];
                }
            }
        }
    }
    defer xq_r.deinit();
    defer xq_i.deinit();

    // Get cos and sin components for the specified positions
    var freqs_cos = try Tensor(T).init(allocator, &[_]usize{ 1, seq_len, rot_dim / 2 });
    defer freqs_cos.deinit();
    var freqs_sin = try Tensor(T).init(allocator, &[_]usize{ 1, seq_len, rot_dim / 2 });
    defer freqs_sin.deinit();

    // Copy values based on position_ids
    for (0..seq_len) |s| {
        const pos = @as(usize, @intFromFloat(position_ids.data[s]));
        if (pos >= freqs_cis.shape[0]) return error.InvalidPositionIds;

        for (0..rot_dim / 2) |d| {
            const src_idx = pos * (rot_dim / 2) * 2 + d * 2;
            freqs_cos.data[s * (rot_dim / 2) + d] = freqs_cis.data[src_idx];
            freqs_sin.data[s * (rot_dim / 2) + d] = freqs_cis.data[src_idx + 1];
        }
    }

    // Perform complex multiplication
    var xq_out_r = try Tensor(T).init(allocator, xq_r.shape);
    defer xq_out_r.deinit();
    var xq_out_i = try Tensor(T).init(allocator, xq_r.shape);
    defer xq_out_i.deinit();

    for (0..num_heads) |h| {
        for (0..seq_len) |s| {
            for (0..rot_dim / 2) |d| {
                const idx = h * seq_len * (rot_dim / 2) +
                    s * (rot_dim / 2) + d;
                const cos = freqs_cos.data[s * (rot_dim / 2) + d];
                const sin = freqs_sin.data[s * (rot_dim / 2) + d];

                xq_out_r.data[idx] = xq_r.data[idx] * cos - xq_i.data[idx] * sin;
                xq_out_i.data[idx] = xq_r.data[idx] * sin + xq_i.data[idx] * cos;
            }
        }
    }

    // Combine real and imaginary parts
    var xq_out = try Tensor(T).init(allocator, &[_]usize{ num_heads, seq_len, rot_dim });
    defer xq_out.deinit();

    if (interleave) {
        for (0..num_heads * seq_len * (rot_dim / 2)) |i| {
            xq_out.data[i * 2] = xq_out_r.data[i];
            xq_out.data[i * 2 + 1] = xq_out_i.data[i];
        }
    } else {
        const half_dim = rot_dim / 2;
        for (0..num_heads) |h| {
            for (0..seq_len) |s| {
                for (0..half_dim) |d| {
                    const src_idx = h * seq_len * half_dim +
                        s * half_dim + d;
                    const dst_idx = h * seq_len * rot_dim +
                        s * rot_dim;
                    xq_out.data[dst_idx + d] = xq_out_r.data[src_idx];
                    xq_out.data[dst_idx + half_dim + d] = xq_out_i.data[src_idx];
                }
            }
        }
    }

    // Concatenate rotated and pass-through parts
    var result = try Tensor(T).init(allocator, &[_]usize{ num_heads, seq_len, head_dim });

    for (0..num_heads) |h| {
        for (0..seq_len) |s| {
            // Copy rotated part
            for (0..rot_dim) |d| {
                const src_idx = h * seq_len * rot_dim +
                    s * rot_dim + d;
                const dst_idx = h * seq_len * head_dim +
                    s * head_dim + d;
                result.data[dst_idx] = xq_out.data[src_idx];
            }
            // Copy pass-through part
            for (0..head_dim - rot_dim) |d| {
                const src_idx = h * seq_len * (head_dim - rot_dim) +
                    s * (head_dim - rot_dim) + d;
                const dst_idx = h * seq_len * head_dim +
                    s * head_dim + (d + rot_dim);
                result.data[dst_idx] = x_pass.data[src_idx];
            }
        }
    }

    return result;
}

// Create attention mask for proper causal attention alignment
pub fn createAttentionMask(allocator: Allocator, pos: usize, seq_len: usize) !Tensor(bool) {
    // First create the base mask of shape [seq_len, pos + seq_len]
    var mask = try Tensor(bool).init(allocator, &[_]usize{ seq_len, pos + seq_len });
    errdefer mask.deinit();

    // Fill the first part (before pos) with true
    for (0..seq_len) |i| {
        for (0..pos) |j| {
            const idx = i * (pos + seq_len) + j;
            mask.data[idx] = true;
        }
    }

    // Fill the second part (pos onwards) with lower triangular matrix
    for (0..seq_len) |i| {
        for (0..seq_len) |j| {
            const idx = i * (pos + seq_len) + (j + pos);
            mask.data[idx] = j <= i; // Lower triangular
        }
    }

    // Reshape to add head dimension [1, seq_len, pos + seq_len]
    try mask.reshape(&[_]usize{ 1, seq_len, pos + seq_len });

    return mask;
}

// Scaled Dot Product Attention with mask
pub fn scaledDotProductAttention(
    comptime T: type,
    query: Tensor(T),
    key: Tensor(T),
    value: Tensor(T),
    mask: Tensor(bool),
    allocator: Allocator,
) !Tensor(T) {
    const n_heads = query.shape[0];
    const q_len = query.shape[1];
    const kv_len = key.shape[1];
    const head_dim = query.shape[2];

    const scale: T = 1.0 / @sqrt(@as(T, @floatFromInt(head_dim)));

    var out = try Tensor(T).init(allocator, &[_]usize{ n_heads, q_len, head_dim });
    errdefer out.deinit();

    var key_transpose = try key.copy();
    defer key_transpose.deinit();
    try transposeAxes(T, &key_transpose, 1, 2);

    for (0..n_heads) |h| {
        var query_head = try query.getDimensionSlice(0, h);
        defer query_head.deinit();

        var key_head = try key_transpose.getDimensionSlice(0, h);
        defer key_head.deinit();

        var value_head = try value.getDimensionSlice(0, h);
        defer value_head.deinit();

        var attn_weights_flat = try simdmatmul.matmul(T, query_head, key_head, allocator);
        defer attn_weights_flat.deinit();

        scalarMultiply(T, &attn_weights_flat, scale);

        // Apply mask
        for (0..q_len) |q| {
            for (0..kv_len) |k| {
                const head_offset = 0;
                const mask_index = head_offset * (q_len * kv_len) + q * kv_len + k;
                const flat_index = q * kv_len + k;
                if (!mask.data[mask_index]) {
                    attn_weights_flat.data[flat_index] = -std.math.inf(T);
                }
            }
        }

        try softmax(T, &attn_weights_flat, 1);

        var out_flat = try simdmatmul.matmul(T, attn_weights_flat, value_head, allocator);
        defer out_flat.deinit();

        for (0..q_len) |q| {
            for (0..head_dim) |d| {
                out.data[h * q_len * head_dim + q * head_dim + d] = out_flat.data[q * head_dim + d];
            }
        }
    }

    return out;
}

// Softmax operation along specified dimension
fn softmax(comptime T: type, tensor: *Tensor(T), dim: usize) !void {
    const dim_size = tensor.shape[dim];

    // Calculate stride for the specified dimension
    var stride: usize = 1;
    for (dim + 1..tensor.shape.len) |i| {
        stride *= tensor.shape[i];
    }

    // Calculate number of vectors to process
    var num_vectors: usize = 1;
    for (0..dim) |i| {
        num_vectors *= tensor.shape[i];
    }

    // Process each vector
    for (0..num_vectors) |i| {
        const base_idx = i * dim_size * stride;

        // Find max for numerical stability
        var max: T = -std.math.inf(T);
        for (0..dim_size) |j| {
            const val = tensor.data[base_idx + j * stride];
            if (val > max) max = val;
        }

        // Calculate exp and sum
        var sum: T = 0;
        for (0..dim_size) |j| {
            const idx = base_idx + j * stride;
            tensor.data[idx] = @exp(tensor.data[idx] - max);
            sum += tensor.data[idx];
        }

        // Normalize
        if (sum > 0) {
            for (0..dim_size) |j| {
                const idx = base_idx + j * stride;
                tensor.data[idx] /= sum;
            }
        }
    }
}

pub fn gelu(comptime T: type, tensor: *Tensor(T)) !void {
    if (@typeInfo(T) != .Float) {
        @compileError("GELU operation requires floating-point tensor");
    }

    // Constants for GELU approximation
    const sqrt_2_div_pi: T = @sqrt(2.0 / std.math.pi);
    const alpha: T = 0.044715;

    for (tensor.data) |*x| {
        const val = x.*;
        const x_cubed = val * val * val;
        const inner = sqrt_2_div_pi * (val + alpha * x_cubed);
        x.* = 0.5 * val * (1 + std.math.tanh(inner));
    }
}
