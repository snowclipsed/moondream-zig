const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const max_items_per_row = 6; // Number of elements to show per row
const max_rows = 8; // Maximum number of rows to show before truncating
const Tensor = @import("tensor.zig").Tensor;
const StabilityError = @import("tensor.zig").StabilityError;
const sgemm = @import("sgemm.zig");
const hgemm = @import("hgemm.zig");
const Slice = @import("tensor.zig").Slice;
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
        std.debug.print("Error during addition", .{});
        return error.ShapeMismatch;
    }

    for (tensor.data, 0..) |_, i| {
        tensor.data[i] += other.data[i];
    }
}

pub fn subtract(comptime T: type, tensor: *Tensor(T), other: Tensor(T)) !void {
    if (!std.mem.eql(usize, tensor.shape, other.shape)) {
        std.debug.print("tensor shape: {d}\n", .{tensor.shape});
        std.debug.print("other shape: {d}\n", .{other.shape});
        std.debug.print("Error during subtraction", .{});
        return error.ShapeMismatch;
    }

    for (tensor.data, 0..) |_, i| {
        tensor.data[i] -= other.data[i];
    }
}

pub fn multiply(comptime T: type, tensor: *Tensor(T), other: Tensor(T)) !void {
    if (!std.mem.eql(usize, tensor.shape, other.shape)) {
        std.debug.print("tensor shape: {d}\n", .{tensor.shape});
        std.debug.print("other shape: {d}\n", .{other.shape});
        std.debug.print("Error during multiplication", .{});
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

// Helper function for broadcasting multiplication
pub fn broadcast_multiply(comptime T: type, a: *Tensor(T), b: Tensor(T)) !void {
    // Create a temporary tensor for the result
    var result = try a.copy();
    defer result.deinit();

    // Perform broadcasted multiplication
    const total_elements = a.data.len;
    const b_elements = b.data.len;

    for (0..total_elements) |i| {
        // Calculate the broadcast index for b
        const b_idx = i % b_elements;
        result.data[i] = a.data[i] * b.data[b_idx];
    }

    // Copy result back to a
    @memcpy(a.data, result.data);
}

// Helper function for broadcasting subtraction
pub fn broadcast_subtract(comptime T: type, a: *Tensor(T), b: Tensor(T)) !void {
    var result = try a.copy();
    defer result.deinit();

    const total_elements = a.data.len;
    const b_elements = b.data.len;

    for (0..total_elements) |i| {
        const b_idx = i % b_elements;
        result.data[i] = a.data[i] - b.data[b_idx];
    }

    @memcpy(a.data, result.data);
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
        std.debug.print("tensor shape: {d}\n", .{tensor.shape});
        std.debug.print("other shape: {d}\n", .{other.shape});
        std.debug.print("Error during accumulation", .{});
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

pub fn stack(comptime T: type, tensors: []const Tensor(T), dim: usize) !Tensor(T) {
    if (tensors.len == 0) {
        return error.EmptyTensorList;
    }

    const ref_tensor = tensors[0];
    const ref_shape = ref_tensor.shape;

    // Validate all tensors have the same shape
    for (tensors[1..]) |tensor| {
        if (!std.mem.eql(usize, tensor.shape, ref_shape)) {
            std.debug.print("Error during stacking", .{});
            return error.ShapeMismatch;
        }
    }

    // Validate dimension
    if (dim > ref_shape.len) {
        return error.InvalidDimension;
    }

    // Create new shape with extra dimension
    var new_shape = try ref_tensor.allocator.alloc(usize, ref_shape.len + 1);
    errdefer ref_tensor.allocator.free(new_shape);

    // Copy shape and insert new dimension
    var src_shape_idx: usize = 0;
    var dst_shape_idx: usize = 0;
    while (dst_shape_idx < new_shape.len) : (dst_shape_idx += 1) {
        if (dst_shape_idx == dim) {
            new_shape[dst_shape_idx] = tensors.len; // Size of new dimension
        } else {
            new_shape[dst_shape_idx] = ref_shape[src_shape_idx];
            src_shape_idx += 1;
        }
    }

    // Create result tensor
    var result = try Tensor(T).init(ref_tensor.allocator, new_shape);
    errdefer result.deinit();
    ref_tensor.allocator.free(new_shape);

    // Calculate strides for the result tensor
    var strides = try ref_tensor.allocator.alloc(usize, result.shape.len);
    defer ref_tensor.allocator.free(strides);

    strides[strides.len - 1] = 1;
    var i = strides.len - 1;
    while (i > 0) : (i -= 1) {
        strides[i - 1] = strides[i] * result.shape[i];
    }

    // Copy data from each input tensor
    var coords = try ref_tensor.allocator.alloc(usize, result.shape.len);
    defer ref_tensor.allocator.free(coords);
    @memset(coords, 0);

    const elements_per_tensor = calculateSize(ref_shape);

    // For each input tensor
    for (tensors, 0..) |tensor, tensor_idx| {
        var element_idx: usize = 0;
        while (element_idx < elements_per_tensor) : (element_idx += 1) {
            // Calculate source coordinates (excluding stacked dimension)
            var temp = element_idx;
            var src_coords = try ref_tensor.allocator.alloc(usize, ref_shape.len);
            defer ref_tensor.allocator.free(src_coords);

            var j = ref_shape.len;
            while (j > 0) : (j -= 1) {
                src_coords[j - 1] = temp % ref_shape[j - 1];
                temp /= ref_shape[j - 1];
            }

            // Calculate destination coordinates (including stacked dimension)
            var final_dst_idx: usize = 0;
            var coord_idx: usize = 0;
            for (coords, 0..) |*c, idx| {
                if (idx == dim) {
                    c.* = tensor_idx;
                } else {
                    c.* = src_coords[coord_idx];
                    coord_idx += 1;
                }
                final_dst_idx += c.* * strides[idx];
            }

            // Copy the value
            result.data[final_dst_idx] = tensor.data[element_idx];

            // Update coordinates for next iteration
            var k = coords.len;
            while (k > 0) {
                k -= 1;
                if (k == dim) continue; // Skip the stacked dimension
                coords[k] += 1;
                if (coords[k] < result.shape[k]) break;
                coords[k] = 0;
            }
        }
    }

    return result;
}

/// Convert negative dimension index to positive
pub fn normalizeDim(dim: isize, n_dims: usize) !usize {
    const n_dims_i: isize = @intCast(n_dims);
    if (dim >= 0) {
        if (dim >= n_dims_i) return error.InvalidDimension;
        return @intCast(dim);
    } else {
        const positive_dim = n_dims_i + dim; // -1 becomes n_dims-1
        if (positive_dim < 0 or positive_dim >= n_dims_i) return error.InvalidDimension;
        return @intCast(positive_dim);
    }
}

/// Flattens dimensions from start_dim to end_dim (inclusive)
/// TODO: Convert to tensor intrinsic
pub fn flatten(comptime T: type, tensor: *Tensor(T), start_dim: isize, end_dim: isize) !void {
    const positive_start = try normalizeDim(start_dim, tensor.shape.len);
    const positive_end = try normalizeDim(end_dim, tensor.shape.len);

    if (positive_start > positive_end) {
        return error.InvalidDimRange;
    }

    // Calculate the size of the flattened dimension
    var flat_size: usize = 1;
    for (positive_start..positive_end + 1) |i| {
        flat_size *= tensor.shape[i];
    }

    // Create new shape
    const new_shape_len = tensor.shape.len - (positive_end - positive_start);
    var new_shape = try tensor.allocator.alloc(usize, new_shape_len);
    errdefer tensor.allocator.free(new_shape);

    // Copy dimensions before flattened dimensions
    @memcpy(new_shape[0..positive_start], tensor.shape[0..positive_start]);

    // Add flattened dimension
    new_shape[positive_start] = flat_size;

    // Copy dimensions after flattened dimensions
    if (positive_end + 1 < tensor.shape.len) {
        @memcpy(
            new_shape[positive_start + 1 ..],
            tensor.shape[positive_end + 1 ..],
        );
    }

    // Free old shape and update with new shape
    tensor.allocator.free(tensor.shape);
    tensor.shape = new_shape;
}

// Usage example:
pub fn stackAndFlatten(comptime T: type, r: Tensor(T), i: Tensor(T), dim: isize) !Tensor(T) {
    // Convert negative dimension to positive
    const positive_dim = if (dim >= 0)
        @as(usize, @intCast(dim))
    else blk: {
        const n_dims: isize = @intCast(r.shape.len);
        // -1 means last dimension + 1 (where we'll insert)
        const adjusted_dim = n_dims + 1 + dim;
        if (adjusted_dim < 0) return error.InvalidDimension;
        break :blk @as(usize, @intCast(adjusted_dim));
    };

    // Stack the tensors along specified dimension
    var tensors = [_]Tensor(T){ r, i };
    var result = try stack(T, &tensors, positive_dim);
    errdefer result.deinit();

    // Flatten the last two dimensions
    try flatten(T, &result, @intCast(result.shape.len - 2), @intCast(result.shape.len - 1));

    return result;
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

    // Compute mean and variance for each feature vector using Welford's online algorithm
    // Use f32 for intermediate computations regardless of input type
    var i: usize = 0;
    while (i < leading_dims) : (i += 1) {
        const start_idx = i * last_dim;
        const end_idx = start_idx + last_dim;

        // Initialize Welford's algorithm variables
        var mean: f32 = 0;
        var m2: f32 = 0; // Second moment
        var count: f32 = 0;

        // First pass: Compute mean and M2 (sum of squared differences)
        for (start_idx..end_idx) |j| {
            count += 1;
            // Cast input to f32 for higher precision intermediate calculations
            const x: f32 = @floatCast(input.data[j]);
            const delta = x - mean;
            mean += delta / count;
            const delta2 = x - mean;
            m2 += delta * delta2;
        }

        // Calculate variance from M2
        const variance = m2 / count;

        // Check for numerical stability
        if (variance < -eps) {
            return error.NegativeVariance;
        }

        // Calculate standard deviation with epsilon for numerical stability
        // Keep in f32 for better precision
        const std_dev = @sqrt(variance + @as(f32, @floatCast(eps)));
        if (std_dev == 0) {
            return error.ZeroStandardDeviation;
        }

        // Normalize and apply scale and bias
        // Do computations in f32 and cast back to T at the end
        for (start_idx..end_idx) |j| {
            const feature_idx = j - start_idx;

            // Cast all values to f32 for intermediate calculations
            const input_val: f32 = @floatCast(input.data[j]);
            const weight_val: f32 = @floatCast(weight.data[feature_idx]);
            const bias_val: f32 = @floatCast(bias.data[feature_idx]);

            // Perform normalization in f32
            const normalized = (input_val - mean) / std_dev;
            const scaled = normalized * weight_val;
            const final_value = scaled + bias_val;

            // Check for stability of computed value
            if (std.math.isNan(final_value)) {
                return error.ComputedNaN;
            }
            if (std.math.isInf(final_value)) {
                return error.ComputedInfinity;
            }

            // Cast back to original type T only at the end
            output.data[j] = @floatCast(final_value);
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
        const power = idx_float / dim_float; // Removed negative sign to match Python

        // Check for potential overflow
        if (power < -1000 or power > 1000) {
            return error.ComputationOverflow;
        }

        const theta_power = std.math.pow(T, theta, power);
        // Check for division by zero or overflow
        if (theta_power == 0 or !std.math.isFinite(theta_power)) {
            return error.NumericalInstability;
        }

        freqs.data[i] = 1.0 / theta_power; // Now matches Python's 1.0 / (theta ** x)

        // Check for numerical stability
        if (!std.math.isFinite(freqs.data[i])) {
            return error.NumericalInstability;
        }
    }

    // 2. Create time tensor [end, 1]
    var time_range = try Tensor(T).init(allocator, &[_]usize{ end, 1 });
    errdefer time_range.deinit();

    for (0..end) |i| {
        time_range.data[i] = @floatFromInt(i);
    }

    // 3. Reshape freqs and prepare for multiplication
    try freqs.reshape(&[_]usize{ 1, dim / 2 });

    // Initialize freq_matrix for the outer product
    var freq_matrix = try Tensor(T).init(allocator, &[_]usize{ end, dim / 2 });
    errdefer freq_matrix.deinit();

    // Perform the outer product (t * freqs)
    for (0..end) |i| {
        for (0..dim / 2) |j| {
            const product = time_range.data[i] * freqs.data[j];
            if (!std.math.isFinite(product)) {
                return error.NumericalInstability;
            }
            freq_matrix.data[i * (dim / 2) + j] = product;
        }
    }

    // 4. Calculate exp(i * freqs) -> [cos(x), sin(x)]
    var result = try Tensor(T).init(allocator, &[_]usize{ end, dim / 2, 2 });
    errdefer result.deinit();

    // Calculate cos and sin values (equivalent to exp(i*x) = cos(x) + i*sin(x))
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
    DimensionMismatch, // Added for concat
    IncompatibleShapes, // Added for concat
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
    allocator: Allocator,
    x: Tensor(f16),
    freqs_cis: Tensor(f32),
    position_ids: Tensor(usize),
    rot_dim: usize,
    interleave: bool,
) !Tensor(f16) {
    // Validate input constraints
    if (x.shape.len != 3) {
        return error.InvalidInputDimensions;
    }
    if (rot_dim != freqs_cis.shape[freqs_cis.shape.len - 2] * 2) {
        return error.InvalidRotationDimension;
    }

    const n_heads = x.shape[0]; // 32
    const seq_len = x.shape[1]; // 13
    const head_dim = x.shape[2]; // 16

    // Split x into rotation and pass-through parts
    var x_rot = try x.getSliceRange(&[_]Slice{
        Slice.full(), // Head (32)
        Slice.full(), // Sequence (13)
        Slice.from(0, rot_dim), // First rot_dim features
    });
    defer x_rot.deinit();

    var x_pass = if (rot_dim < head_dim) blk: {
        const pass = try x.getSliceRange(&[_]Slice{
            Slice.full(), // Head (32)
            Slice.full(), // Sequence (13)
            Slice.from(rot_dim, null), // Remaining features
        });
        break :blk pass;
    } else Tensor(f16).init(allocator, &[_]usize{ n_heads, seq_len, 0 }) catch unreachable;
    defer x_pass.deinit();

    // x_rot and x_pass are correct!

    // Handle interleaved vs non-interleaved cases
    var xq_r: Tensor(f16) = undefined;
    var xq_i: Tensor(f16) = undefined;

    if (interleave) {
        // Reshape x_rot to [n_heads, seq_len, rot_dim/2, 2]
        var reshaped = try x_rot.copy();
        defer reshaped.deinit();
        try reshaped.reshape(&[_]usize{ n_heads, seq_len, rot_dim / 2, 2 });

        // Extract real and imaginary parts (n_heads, seq_len, rot_dim/2)
        xq_r = try reshaped.getSliceRange(&[_]Slice{
            Slice.full(),
            Slice.full(),
            Slice.full(),
            Slice.from(0, 1),
        });
        try xq_r.reshape(&[_]usize{ n_heads, seq_len, rot_dim / 2 });

        xq_i = try reshaped.getSliceRange(&[_]Slice{
            Slice.full(),
            Slice.full(),
            Slice.full(),
            Slice.from(1, 2),
        });
        try xq_i.reshape(&[_]usize{ n_heads, seq_len, rot_dim / 2 });
    } else {
        // Split last dimension in half
        xq_r = try x_rot.getSliceRange(&[_]Slice{
            Slice.full(),
            Slice.full(),
            Slice.from(0, rot_dim / 2),
        });
        xq_i = try x_rot.getSliceRange(&[_]Slice{
            Slice.full(),
            Slice.full(),
            Slice.from(rot_dim / 2, null),
        });
    }

    // xq_r and xq_i are correct!
    defer xq_r.deinit();
    defer xq_i.deinit();

    // Get cos and sin from freqs_cis
    var cos_part = try freqs_cis.getSliceRange(&[_]Slice{
        Slice.full(),
        Slice.full(),
        Slice.from(0, 1),
    });
    defer cos_part.deinit();

    var sin_part = try freqs_cis.getSliceRange(&[_]Slice{
        Slice.full(),
        Slice.full(),
        Slice.from(1, 2),
    });
    defer sin_part.deinit();

    // Create freqs_cos and freqs_sin with shape (1, seq_len, rot_dim/2)
    var freqs_cos = try zeros(f32, allocator, &[_]usize{
        1,
        seq_len,
        rot_dim / 2,
    });
    defer freqs_cos.deinit();

    var freqs_sin = try zeros(f32, allocator, &[_]usize{
        1,
        seq_len,
        rot_dim / 2,
    });
    defer freqs_sin.deinit();

    // Fill freqs_cos and freqs_sin using position_ids
    for (0..seq_len) |i| {
        const pos_id = position_ids.data[i];
        const offset = i * (rot_dim / 2);
        @memcpy(freqs_cos.data[offset .. offset + rot_dim / 2], cos_part.data[pos_id * cos_part.shape[1] .. (pos_id + 1) * cos_part.shape[1]]);
        @memcpy(freqs_sin.data[offset .. offset + rot_dim / 2], sin_part.data[pos_id * sin_part.shape[1] .. (pos_id + 1) * sin_part.shape[1]]);
    }

    // freqs sin and cos are correct!

    // Complex multiply with broadcasting across heads
    // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    var xq_out_r = try xq_r.castTo(f32); // Will be (n_heads, seq_len, rot_dim/2)
    defer xq_out_r.deinit();
    try broadcast_multiply(f32, &xq_out_r, freqs_cos);

    var temp = try xq_i.castTo(f32);
    defer temp.deinit();
    try broadcast_multiply(f32, &temp, freqs_sin);
    try broadcast_subtract(f32, &xq_out_r, temp);

    var xq_out_i = try xq_r.castTo(f32); // Will be (n_heads, seq_len, rot_dim/2)
    defer xq_out_i.deinit();
    try broadcast_multiply(f32, &xq_out_i, freqs_sin);

    var temp2 = try xq_i.castTo(f32);
    defer temp2.deinit();
    try broadcast_multiply(f32, &temp2, freqs_cos);
    try broadcast_add(f32, &xq_out_i, temp2);

    // xq_out_r amd xq_out_i are correct!

    // Stack real and imaginary parts -> (n_heads, seq_len, rot_dim)
    var tensors = [_]Tensor(f32){ xq_out_r, xq_out_i };
    var stacked = try stack(f32, &tensors, 3);
    defer stacked.deinit();

    try flatten(f32, &stacked, 2, 3);

    // stacked.print3D();

    // std.debug.print("stacked shape (xq_out) {any} \n", .{stacked.shape});
    // std.debug.print("x_pass shape {any} \n", .{x_pass.shape});

    if (x_pass.data.len > 0) {
        var x_pass_f32 = try x_pass.castTo(f32);
        defer x_pass_f32.deinit();
        var result = try concat(f32, stacked, x_pass_f32, 2);
        defer result.deinit();
        const final_result = try result.castTo(f16);
        return final_result;
    } else {
        var result = try stacked.copy();
        defer result.deinit();
        const final_result = try result.castTo(f16);
        return final_result;
    }
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
    query: Tensor(f16),
    key: Tensor(f16),
    value: Tensor(f16),
    mask: Tensor(bool),
    allocator: Allocator,
) !Tensor(f16) {
    const n_heads = query.shape[0];
    const q_len = query.shape[1];
    const kv_len = key.shape[1];
    const head_dim = query.shape[2];

    // Scale factor
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    // Initialize output tensor
    var out = try Tensor(f16).init(allocator, &[_]usize{ n_heads, q_len, head_dim });
    errdefer out.deinit();

    // Prepare transposed key for all heads
    var key_transpose = try key.copy();
    defer key_transpose.deinit();
    try transposeAxes(f16, &key_transpose, 1, 2);

    // Process each attention head separately
    for (0..n_heads) |h| {
        // Get the query, key, value slices for this head
        var query_head = try query.getDimensionSlice(0, h);
        defer query_head.deinit();

        var key_head = try key_transpose.getDimensionSlice(0, h);
        defer key_head.deinit();

        var value_head = try value.getDimensionSlice(0, h);
        defer value_head.deinit();

        // Initialize attention bias tensor in f32
        var attn_bias = try Tensor(f32).init(allocator, &[_]usize{ q_len, kv_len });
        defer attn_bias.deinit();
        @memset(attn_bias.data, 0);

        // Apply mask to attention bias
        for (0..q_len) |i| {
            for (0..kv_len) |j| {
                const mask_idx = i * mask.shape[2] + j;
                const bias_idx = i * kv_len + j;
                if (!mask.data[mask_idx]) {
                    attn_bias.data[bias_idx] = -std.math.inf(f32);
                }
            }
        }

        // Calculate QK^T and scale in one step using f32 accumulation
        var attn_weights = try Tensor(f32).init(allocator, &[_]usize{ q_len, kv_len });
        defer attn_weights.deinit();
        try hgemm.matmul(allocator, query_head, key_head, attn_weights);

        // Apply scaling
        for (attn_weights.data) |*w| {
            w.* *= scale;
        }

        // Add attention bias
        for (0..attn_weights.data.len) |i| {
            attn_weights.data[i] += attn_bias.data[i];
        }

        // Apply softmax in f32 precision
        try softmax(&attn_weights, 1, allocator);

        // Calculate attention output with f32 accumulation
        var out_head = try Tensor(f32).init(allocator, &[_]usize{ q_len, head_dim });
        defer out_head.deinit();

        // Convert attention weights to f16 for HGEMM
        var attn_weights_f16 = try attn_weights.castTo(f16);
        defer attn_weights_f16.deinit();

        // Compute attention output
        try hgemm.matmul(allocator, attn_weights_f16, value_head, out_head);

        // Copy to output tensor with conversion to f16
        for (0..q_len) |q| {
            for (0..head_dim) |d| {
                const out_idx = h * q_len * head_dim + q * head_dim + d;
                const head_idx = q * head_dim + d;
                out.data[out_idx] = @floatCast(out_head.data[head_idx]);
            }
        }
    }

    return out;
}

pub fn masklessDotProductAttention(
    query: Tensor(f16),
    key: Tensor(f16),
    value: Tensor(f16),
    allocator: Allocator,
) !Tensor(f16) {
    const n_heads = query.shape[0];
    const q_len = query.shape[1];
    const kv_len = key.shape[1];
    const head_dim = query.shape[2];

    // Scale factor
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    // Initialize output tensor
    var out = try Tensor(f16).init(allocator, &[_]usize{ n_heads, q_len, head_dim });
    errdefer out.deinit();

    // Prepare transposed key for all heads
    var key_transpose = try key.copy();
    defer key_transpose.deinit();
    try transposeAxes(f16, &key_transpose, 1, 2);

    // Process each attention head separately
    for (0..n_heads) |h| {
        // Get the query, key, value slices for this head
        var query_head = try query.getDimensionSlice(0, h);
        defer query_head.deinit();

        var key_head = try key_transpose.getDimensionSlice(0, h);
        defer key_head.deinit();

        var value_head = try value.getDimensionSlice(0, h);
        defer value_head.deinit();

        // Calculate QK^T and scale in one step using f32 accumulation
        var attn_weights = try Tensor(f32).init(allocator, &[_]usize{ q_len, kv_len });
        defer attn_weights.deinit();
        try hgemm.matmul(allocator, query_head, key_head, attn_weights);

        // Apply scaling
        for (attn_weights.data) |*w| {
            w.* *= scale;
        }

        // Apply softmax in f32 precision
        try softmax(&attn_weights, 1, allocator);

        // Convert value_head to f32 for SGEMM
        var value_head_f32 = try value_head.castTo(f32);
        defer value_head_f32.deinit();

        // Compute attention output

        const out_head = try sgemm.matmul(f32, attn_weights, value_head_f32, allocator);

        // Copy to output tensor with conversion to f16
        for (0..q_len) |q| {
            for (0..head_dim) |d| {
                const out_idx = h * q_len * head_dim + q * head_dim + d;
                const head_idx = q * head_dim + d;
                out.data[out_idx] = @floatCast(out_head.data[head_idx]);
            }
        }
    }

    return out;
}

// Softmax operation along specified dimension
pub fn softmax(tensor: *Tensor(f32), dim: usize, allocator: Allocator) !void {
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

    // Allocate temporary storage for exp values
    var temp_exp = try allocator.alloc(f32, dim_size);
    defer allocator.free(temp_exp);

    // Process each vector
    for (0..num_vectors) |i| {
        const base_idx = i * dim_size * stride;

        // Find max for numerical stability (in f32)
        var max: f32 = -std.math.inf(f32);
        for (0..dim_size) |j| {
            const val = tensor.data[base_idx + j * stride];
            if (val > max) max = val;
        }

        // Calculate exp and sum (in f32)
        var sum: f32 = 0;
        for (0..dim_size) |j| {
            const idx = base_idx + j * stride;
            const val = tensor.data[idx] - max;
            temp_exp[j] = if (val > -88.0) @exp(val) else 0;
            sum += temp_exp[j];
        }

        // Normalize
        if (sum > 0) {
            const inv_sum = 1.0 / sum;
            for (0..dim_size) |j| {
                const idx = base_idx + j * stride;
                tensor.data[idx] = temp_exp[j] * inv_sum;
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
        // Convert result back to type T after tanh operation
        x.* = @floatCast(0.5 * @as(f32, @floatCast(val)) * (1.0 + std.math.tanh(@as(f32, @floatCast(inner)))));
    }
}

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
    temperature: f32 = 1.0, // Temperature for softmax
    top_k: ?usize = null, // Number of top tokens to consider (only for top_k)
};

/// Returns the index of the maximum value in the tensor
fn argmax(comptime T: type, tensor: *const Tensor(T)) !usize {
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

// ----------- Vision Operations ----------- //

pub fn rearrangeBCHWtoBTC(allocator: std.mem.Allocator, input: Tensor(f16), patch_size: usize) !Tensor(f16) {
    // Input shape: [batch, channels, height, width]
    if (input.shape.len != 4) return error.InvalidInputShape;

    const batch = input.shape[0];
    const channels = input.shape[1];
    const height = input.shape[2];
    const width = input.shape[3];

    // Verify dimensions are divisible by patch size
    if (height % patch_size != 0 or width % patch_size != 0) {
        return error.InvalidPatchSize;
    }

    const h_patches = height / patch_size;
    const w_patches = width / patch_size;
    const num_patches = h_patches * w_patches;
    const patch_dim = channels * patch_size * patch_size;

    // Output shape: [batch, h_patches * w_patches, channels * patch_size * patch_size]
    var output = try Tensor(f16).init(allocator, &[_]usize{ batch, num_patches, patch_dim });
    errdefer output.deinit();

    // For each batch
    var b: usize = 0;
    while (b < batch) : (b += 1) {
        // For each patch position
        var h: usize = 0;
        while (h < h_patches) : (h += 1) {
            var w: usize = 0;
            while (w < w_patches) : (w += 1) {
                const patch_idx = h * w_patches + w;

                // For each pixel in the patch
                var ph: usize = 0;
                while (ph < patch_size) : (ph += 1) {
                    var pw: usize = 0;
                    while (pw < patch_size) : (pw += 1) {
                        // For each channel
                        var c: usize = 0;
                        while (c < channels) : (c += 1) {
                            const input_h = h * patch_size + ph;
                            const input_w = w * patch_size + pw;

                            // Input index: [b, c, h, w]
                            const input_idx = ((b * channels + c) * height + input_h) * width + input_w;

                            // Output index: [b, patch_idx, (c * patch_size + ph) * patch_size + pw]
                            const output_idx = ((b * num_patches + patch_idx) * patch_dim) +
                                ((c * patch_size + ph) * patch_size + pw);

                            output.data[output_idx] = input.data[input_idx];
                        }
                    }
                }
            }
        }
    }

    return output;
}

pub fn normalize_patch(allocator: Allocator, input: Tensor(f16), mean: Tensor(f16), stdev: Tensor(f16)) !Tensor(f16) {
    var result = try Tensor(f16).init(allocator, input.shape);
    errdefer result.deinit();

    // Now using BCHW layout
    const batch = input.shape[0];
    const channels = input.shape[1];
    const height = input.shape[2];
    const width = input.shape[3];

    // // Debug prints
    // std.debug.print("Input shape (BCHW): [{}, {}, {}, {}]\n", .{ batch, channels, height, width });
    // std.debug.print("First 10 raw input values: ", .{});
    // for (input.data[0..10]) |val| {
    //     std.debug.print("{d:.6} ", .{val});
    // }
    // std.debug.print("\n", .{});

    // Process each channel first for debug info
    // for (0..channels) |c| {
    //     const c_mean = mean.data[c];
    //     const c_std = stdev.data[c];
    // std.debug.print("Channel {}: mean={d:.6}, std={d:.6}\n", .{ c, c_mean, c_std });

    // Print debug info for first channel
    // if (c == 0) {
    //     std.debug.print("First 10 values in channel 0: ", .{});
    //     for (input.data[0..10]) |val| {
    //         std.debug.print("{d:.6} ", .{val});
    //     }
    //     std.debug.print("\n", .{});

    //     std.debug.print("First 10 normalized values in channel 0: ", .{});
    //     for (0..10) |i| {
    //         const normalized = (input.data[i] - c_mean) / c_std;
    //         std.debug.print("{d:.6} ", .{normalized});
    //     }
    //     std.debug.print("\n", .{});
    // }
    // }

    // Perform normalization in BCHW format
    for (0..batch) |b| {
        for (0..channels) |c| {
            const c_mean = mean.data[c];
            const c_std = stdev.data[c];

            const channel_size = height * width;
            const batch_offset = b * channels * channel_size;
            const channel_offset = c * channel_size;
            const start_idx = batch_offset + channel_offset;
            const end_idx = start_idx + channel_size;

            var i = start_idx;
            while (i < end_idx) : (i += 1) {
                result.data[i] = (input.data[i] - c_mean) / c_std;
            }
        }
    }

    return result;
}

pub fn convert_bhwc_to_bchw(allocator: Allocator, input: Tensor(f16)) !Tensor(f16) {
    const batch = input.shape[0];
    const height = input.shape[1];
    const width = input.shape[2];
    const channels = input.shape[3];

    var output = try Tensor(f16).init(allocator, &[_]usize{ batch, channels, height, width });
    errdefer output.deinit();

    // Transform from BHWC to BCHW
    for (0..batch) |b| {
        for (0..height) |h| {
            for (0..width) |w| {
                for (0..channels) |c| {
                    const src_idx = b * (height * width * channels) +
                        h * (width * channels) +
                        w * channels +
                        c;

                    const dst_idx = b * (channels * height * width) +
                        c * (height * width) +
                        h * width +
                        w;

                    output.data[dst_idx] = input.data[src_idx];
                }
            }
        }
    }

    return output;
}
