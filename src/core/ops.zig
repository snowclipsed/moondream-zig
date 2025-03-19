const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

const Tensor = @import("../core/tensor.zig").Tensor;
const TensorView = @import("../core/tensor.zig").TensorView;
const Slice = @import("../core/tensor.zig").Slice;
const StabilityError = @import("../core/tensor.zig").StabilityError;

const mode = std.builtin.FloatMode.optimized;
comptime {
    // btw, this line applies to only this scope?
    @setFloatMode(mode);
}

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
        new_shape[idx] = if (idx == dim0) tensor.shape[dim1] else if (idx == dim1) tensor.shape[dim0] else dim;
    }

    // Allocate memory for transposed data
    var new_data = try tensor.allocator.alignedAlloc(T, 32, tensor.data.len);
    errdefer tensor.allocator.free(new_data);

    // Calculate new strides - Moved before SIMD block
    var new_strides = try tensor.allocator.alloc(usize, tensor.shape.len);
    defer tensor.allocator.free(new_strides);

    new_strides[tensor.shape.len - 1] = 1;
    i = tensor.shape.len - 1;
    while (i > 0) : (i -= 1) {
        new_strides[i - 1] = new_strides[i] * new_shape[i];
    }

    // SIMD optimization wrapped in a block
    {
        if (tensor.shape.len == 3 and dim0 == 0 and dim1 == 1 and
            (T == f16 or T == f32))
        {
            const batch_size = tensor.shape[0];
            const rows = tensor.shape[1];
            const cols = tensor.shape[2];
            const vector_size = if (T == f16) 16 else 8; // AVX2: 16 fp16 or 8 fp32

            if (cols >= vector_size) {
                const Vector = @Vector(vector_size, T);

                var col: usize = 0;
                while (col < cols) : (col += vector_size) {
                    const vec_size = @min(vector_size, cols - col);

                    for (0..batch_size) |b| {
                        for (0..rows) |r| {
                            const src_idx = b * rows * cols + r * cols + col;
                            var vec: Vector = undefined;

                            if (vec_size == vector_size) {
                                vec = tensor.data[src_idx..][0..vector_size].*;
                            } else {
                                var temp: [vector_size]T = undefined;
                                @memcpy(temp[0..vec_size], tensor.data[src_idx..][0..vec_size]);
                                vec = temp;
                            }

                            const dst_idx = r * batch_size * cols + b * cols + col;
                            if (vec_size == vector_size) {
                                new_data[dst_idx..][0..vector_size].* = vec;
                            } else {
                                @memcpy(new_data[dst_idx..][0..vec_size], @as([vector_size]T, vec)[0..vec_size]);
                            }
                        }
                    }
                }

                tensor.allocator.free(tensor.data);
                tensor.data = new_data;
                tensor.allocator.free(tensor.shape);
                tensor.shape = new_shape;
                return;
            }
        }
    }

    // General case implementation
    var coords = try tensor.allocator.alloc(usize, tensor.shape.len);
    defer tensor.allocator.free(coords);
    @memset(coords, 0);

    const total_elements = tensor.data.len;
    var idx: usize = 0;
    while (idx < total_elements) : (idx += 1) {
        var remaining = idx;
        for (0..tensor.shape.len) |dim| {
            coords[dim] = remaining / new_strides[dim];
            remaining %= new_strides[dim];
        }

        // Swap coordinates
        const temp = coords[dim0];
        coords[dim0] = coords[dim1];
        coords[dim1] = temp;

        // Calculate source index
        var src_idx: usize = 0;
        for (0..tensor.shape.len) |dim| {
            src_idx += coords[dim] * strides[dim];
        }

        new_data[idx] = tensor.data[src_idx];
    }

    tensor.allocator.free(tensor.data);
    tensor.data = new_data;
    tensor.allocator.free(tensor.shape);
    tensor.shape = new_shape;
}
pub fn transposeF16SIMD(tensor: *Tensor(f16), batch_size: usize, rows: usize, cols: usize, new_data: []align(32) f16) void {
    const vec_size = 8;

    var col: usize = 0;
    while (col < cols) : (col += vec_size) {
        const remaining = cols - col;
        const current_vec_size = @min(vec_size, remaining);

        for (0..batch_size) |b| {
            for (0..rows) |r| {
                const src_idx = b * rows * cols + r * cols + col;
                const dst_idx = r * batch_size * cols + b * cols + col;

                // Check for special values in this vector
                var has_special = false;
                var i: usize = 0;
                while (i < current_vec_size) : (i += 1) {
                    const val = tensor.data[src_idx + i];
                    if (std.math.isNan(val) or std.math.isInf(val)) {
                        has_special = true;
                        break;
                    }
                }

                if (has_special) {
                    // Handle scalar-wise for vectors containing special values
                    for (0..current_vec_size) |idx| {
                        new_data[dst_idx + idx] = tensor.data[src_idx + idx];
                    }
                } else {
                    // Use SIMD for normal values
                    var src_batch: [8]f16 align(32) = undefined;
                    var dst_batch: [8]f16 align(32) = undefined;
                    var temp_f32: [8]f32 align(32) = undefined;

                    if (current_vec_size == vec_size) {
                        @memcpy(&src_batch, tensor.data[src_idx..][0..8]);
                        asm volatile (
                            \\vmovups (%[src]), %%xmm0
                            \\vcvtph2ps %%xmm0, %%ymm1
                            \\vmovups %%ymm1, (%[dst])
                            :
                            : [src] "r" (&src_batch),
                              [dst] "r" (&temp_f32),
                            : "xmm0", "ymm1", "memory"
                        );
                        asm volatile (
                            \\vmovups (%[src]), %%ymm0
                            \\vcvtps2ph $0, %%ymm0, %%xmm1
                            \\vmovups %%xmm1, (%[dst])
                            :
                            : [src] "r" (&temp_f32),
                              [dst] "r" (&dst_batch),
                            : "ymm0", "xmm1", "memory"
                        );
                        @memcpy(new_data[dst_idx..][0..8], &dst_batch);
                    } else {
                        @memset(&src_batch, 0);
                        @memcpy(src_batch[0..current_vec_size], tensor.data[src_idx..][0..current_vec_size]);

                        asm volatile (
                            \\vmovups (%[src]), %%xmm0
                            \\vcvtph2ps %%xmm0, %%ymm1
                            \\vmovups %%ymm1, (%[dst])
                            :
                            : [src] "r" (&src_batch),
                              [dst] "r" (&temp_f32),
                            : "xmm0", "ymm1", "memory"
                        );
                        asm volatile (
                            \\vmovups (%[src]), %%ymm0
                            \\vcvtps2ph $0, %%ymm0, %%xmm1
                            \\vmovups %%xmm1, (%[dst])
                            :
                            : [src] "r" (&temp_f32),
                              [dst] "r" (&dst_batch),
                            : "ymm0", "xmm1", "memory"
                        );
                        @memcpy(new_data[dst_idx..][0..current_vec_size], dst_batch[0..current_vec_size]);
                    }
                }
            }
        }
    }
}

fn transposeAxesGeneric(comptime T: type, tensor: *Tensor(T), dim0: usize, dim1: usize) !void {
    // Original implementation for the general case
    var strides = try tensor.allocator.alloc(usize, tensor.shape.len);
    defer tensor.allocator.free(strides);

    strides[tensor.shape.len - 1] = 1;
    var i: usize = tensor.shape.len - 1;
    while (i > 0) : (i -= 1) {
        strides[i - 1] = strides[i] * tensor.shape[i];
    }

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

    var new_data = try tensor.allocator.alignedAlloc(T, 32, tensor.data.len);
    errdefer tensor.allocator.free(new_data);

    var new_strides = try tensor.allocator.alloc(usize, tensor.shape.len);
    defer tensor.allocator.free(new_strides);

    new_strides[tensor.shape.len - 1] = 1;
    i = tensor.shape.len - 1;
    while (i > 0) : (i -= 1) {
        new_strides[i - 1] = new_strides[i] * new_shape[i];
    }

    var coords = try tensor.allocator.alloc(usize, tensor.shape.len);
    defer tensor.allocator.free(coords);
    @memset(coords, 0);

    const total_elements = tensor.data.len;
    var idx: usize = 0;
    while (idx < total_elements) : (idx += 1) {
        var remaining = idx;
        for (0..tensor.shape.len) |dim| {
            coords[dim] = remaining / new_strides[dim];
            remaining = remaining % new_strides[dim];
        }

        const temp = coords[dim0];
        coords[dim0] = coords[dim1];
        coords[dim1] = temp;

        var src_idx: usize = 0;
        for (0..tensor.shape.len) |dim| {
            src_idx += coords[dim] * strides[dim];
        }

        new_data[idx] = tensor.data[src_idx];
    }

    tensor.allocator.free(tensor.data);
    tensor.data = new_data;
    tensor.allocator.free(tensor.shape);
    tensor.shape = new_shape;
}

pub fn add(comptime T: type, tensor: *Tensor(T), other: Tensor(T)) !void {
    if (!std.mem.eql(usize, tensor.shape, other.shape)) {
        std.debug.print("tensor shape: {d}\n", .{tensor.shape});
        std.debug.print("other shape: {d}\n", .{other.shape});
        std.debug.print("Error during addition\n", .{});
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
        std.debug.print("Error during subtraction\n", .{});
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
        std.debug.print("Error during multiplication\n", .{});
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
        std.debug.print("Error during accumulation\n", .{});
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
            std.debug.print("Error during stacking\n", .{});
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

// ----------------------------------------------------------------------------

pub fn layerNorm(comptime T: type, input: Tensor(T), weight: Tensor(T), bias: Tensor(T), eps: T) !Tensor(T) {
    if (builtin.cpu.arch == .aarch64) {
        // empirically determined settings for NEON (Apple M1): N_A=0, N_B=10, N_S2 = 3
        return layerNormInner(T, 0, 10, 3, false, input, weight, bias, eps);
    }
    // my guess for avx2??
    return layerNormInner(T, 6, 0, 3, false, input, weight, bias, eps);
}

pub fn layerNormCheckEverything(comptime T: type, input: Tensor(T), weight: Tensor(T), bias: Tensor(T), eps: T) !Tensor(T) {
    if (builtin.cpu.arch == .aarch64) {
        return layerNormInner(T, 0, 10, 3, true, input, weight, bias, eps);
    }
    return layerNormInner(T, 6, 0, 3, true, input, weight, bias, eps);
}

pub fn layerNormInner(
    comptime T: type,
    // number of unrolls of one type for welford's algorithm update
    comptime N_A: usize,
    // number of unrolls of another type for welford's algorithm update
    comptime N_B: usize,
    // number of unrolls for second pass
    comptime N_S2: usize,
    comptime CHECK_EVERYTHING: bool,
    input: Tensor(T),
    weight: Tensor(T),
    bias: Tensor(T),
    eps: T) !Tensor(T)
{
    // Check input stability
    if (CHECK_EVERYTHING) {
        try checkStability(T, input);
        try checkStability(T, weight);
        try checkStability(T, bias);
    }

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

    const N_U = N_A + N_B;
    const N_FAKE_ZEROS: f32 = 100_000;
    // NOTE: not using AVX-512 until the gemms do.
    // To use AVX-512, get rid of the @min.
    const VLEN = @min(std.simd.suggestVectorLength(f32) orelse 4, 8);
    const f32v = @Vector(VLEN, f32);
    const Tv = @Vector(VLEN, T);

    // Compute mean and variance for each feature vector using Welford's online algorithm
    // Use f32 for intermediate computations regardless of input type
    var i: usize = 0;
    while (i < leading_dims) : (i += 1) {
        const start_idx = i * last_dim;
        const end_idx = start_idx + last_dim;

        // Initialize Welford's algorithm variables
        const leftover_n = last_dim % (VLEN * N_U);
        var count: f32v = @splat(@as(f32, N_FAKE_ZEROS));
        var rcp: f32v = @splat(@as(f32, 1.0/N_FAKE_ZEROS));
        var mean: [N_U]f32v = @bitCast([1]f32{0} ** (VLEN * N_U));
        var m2: [N_U]f32v = @bitCast([1]f32{0} ** (VLEN * N_U));

        // First pass: Compute mean and M2 (sum of squared differences)
        var start_ptr: [*]T = input.data.ptr + start_idx;
        var end_ptr: [*]T = input.data.ptr + end_idx;
        end_ptr = end_ptr - VLEN * N_U + 1;
        while (@intFromPtr(start_ptr) < @intFromPtr(end_ptr)) {
            @setFloatMode(.optimized);
            const prev_count = count;
            count += @splat(@as(f32, 1));
            rcp = rcp * (@as(f32v, @splat(@as(f32, 2))) - count * rcp);
            inline for (0..N_A) |j| {
                const x: f32v = @floatCast(@as(Tv, start_ptr[0..VLEN].*));
                const delta = x - mean[j];
                mean[j] += delta * rcp;
                const delta2 = x - mean[j];
                m2[j] += delta * delta2;
                start_ptr += VLEN;
            }
            inline for (N_A..N_U) |j| {
                const x: f32v = @floatCast(@as(Tv, start_ptr[0..VLEN].*));
                const delta = x - mean[j];
                const delta_div_n = delta * rcp;
                mean[j] += delta_div_n;
                const delta_sq = delta * delta;
                m2[j] += prev_count * delta_sq * rcp;
                start_ptr += VLEN;
            }
        }
        // done with the fully unrolled part - accumulate remaining elements and some fake zeros
        if (leftover_n > 0) {
            @setFloatMode(.optimized);
            var fake_xs: [VLEN*N_U]T = [1]T{0} ** (VLEN * N_U);
            @memcpy(@as([*]T, @ptrCast(&fake_xs)), start_ptr[0..leftover_n]);
            start_ptr = @ptrCast(&fake_xs);
            const prev_count = count;
            count += @splat(@as(f32, 1));
            rcp = rcp * (@as(f32v, @splat(@as(f32, 2))) - count * rcp);
            inline for (0..N_A) |j| {
                const x: f32v = @floatCast(@as(Tv, start_ptr[0..VLEN].*));
                const delta = x - mean[j];
                mean[j] += delta * rcp;
                const delta2 = x - mean[j];
                m2[j] += delta * delta2;
                start_ptr += VLEN;
            }
            inline for (N_A..N_U) |j| {
                const x: f32v = @floatCast(@as(Tv, start_ptr[0..VLEN].*));
                const delta = x - mean[j];
                const delta_div_n = delta * rcp;
                mean[j] += delta_div_n;
                const delta_sq = delta * delta;
                m2[j] += prev_count * delta_sq * rcp;
                start_ptr += VLEN;
            }
        }
        // tree reduction
        const n_tree_iters = @bitSizeOf(usize) - @clz(N_U - 1);
        var counts: [N_U]f32v = .{count} ** N_U;
        inline for (0..n_tree_iters) |tree_iter| {
            const step: usize = 1 << tree_iter;
            inline for (0..N_U) |j| {
                if (@ctz(j) > tree_iter and j + step < N_U) {
                    @setFloatMode(.optimized);
                    const dmean = mean[j+step] - mean[j];
                    const total_n = counts[j] + counts[j+step];
                    const selfnothern = counts[j] * counts[j+step];
                    const dmeanovern = dmean / total_n;
                    const selfnotherndmean2overn = selfnothern * dmean * dmeanovern;
                    counts[j] = total_n;
                    mean[j] += counts[j+step] * dmeanovern;
                    m2[j] += m2[j+step] + selfnotherndmean2overn;
                }
            }
        }
        // add all the states in the first register together
        var scounts: [VLEN]f32 = counts[0];
        var smean: [VLEN]f32 = mean[0];
        var sm2: [VLEN]f32 = m2[0];
        // tree reduction again!
        const n_stree_iters = @bitSizeOf(usize) - @clz(@as(usize, VLEN - 1));
        inline for (0..n_stree_iters) |tree_iter| {
            const step: usize = 1 << tree_iter;
            inline for (0..VLEN) |j| {
                if (@ctz(j) > tree_iter and j + step < VLEN) {
                    @setFloatMode(.optimized);
                    const dmean = smean[j+step] - smean[j];
                    const total_n = scounts[j] + scounts[j+step];
                    const selfnothern = scounts[j] * scounts[j+step];
                    const dmeanovern = dmean / total_n;
                    const selfnotherndmean2overn = selfnothern * dmean * dmeanovern;

                    scounts[j] = total_n;
                    smean[j] += scounts[j+step] * dmeanovern;
                    sm2[j] += sm2[j+step] + selfnotherndmean2overn;
                }
            }
        }
        // subtract out the fake zeros
        {
            var other_count: f32 = -N_FAKE_ZEROS * VLEN * N_U;
            if (leftover_n > 0) {
                const non_leftover_n = VLEN * N_U - leftover_n;
                other_count -= @as(f32, @floatFromInt(non_leftover_n));
            }
            const other_m: f32 = 0;
            const dmean = other_m - smean[0];
            const total_n = scounts[0] + other_count;
            const selfnothern = scounts[0] * other_count;
            const dmean_over_n = dmean / total_n;
            const selfnotherndmean2overn = selfnothern * dmean * dmean_over_n;
            scounts[0] = total_n;
            smean[0] += other_count * dmean_over_n;
            sm2[0] += selfnotherndmean2overn;
        }

        // Calculate variance from M2
        const variance = sm2[0] / scounts[0];

        // Check for numerical stability
        if (variance < -eps) {
            return error.NegativeVariance;
        }

        // Calculate standard deviation with epsilon for numerical stability
        // Keep in f32 for better precision
        const std_dev = @sqrt(variance + @as(f32, @floatCast(eps)));
        // if (i == 0) {
        //     std.log.err("Yolo found mean={d} and std_dev={d}", .{smean[0], std_dev});
        // }

        if (std_dev == 0) {
            return error.ZeroStandardDeviation;
        }
        const rcp_std_dev = 1.0 / std_dev;

        // Normalize and apply scale and bias
        // Do computations in f32 and cast back to T at the end
        start_ptr = input.data.ptr + start_idx;
        end_ptr = input.data.ptr + end_idx;
        end_ptr = end_ptr - VLEN * N_S2 + 1;
        var weight_ptr: [*]T = weight.data.ptr;
        var bias_ptr: [*]T = bias.data.ptr;
        var out_ptr: [*]T = output.data.ptr + start_idx;
        const mean_v: f32v = @splat(@as(f32, smean[0]));
        const rcp_std_dev_v: f32v = @splat(@as(f32, rcp_std_dev));
        while (@intFromPtr(start_ptr) < @intFromPtr(end_ptr)) {
            @setFloatMode(.optimized);
            var input_val: [N_S2]f32v = undefined;
            var weight_val: [N_S2]f32v = undefined;
            var bias_val: [N_S2]f32v = undefined;
            var final_value: [N_S2]f32v = undefined;
            inline for (0..N_S2) |j| {
                // Cast all values to f32 for intermediate calculations
                input_val[j] = @floatCast(@as(Tv, start_ptr[0..VLEN].*));
                weight_val[j] = @floatCast(@as(Tv, weight_ptr[0..VLEN].*));
                bias_val[j] = @floatCast(@as(Tv, bias_ptr[0..VLEN].*));
                start_ptr = start_ptr + VLEN;
                weight_ptr = weight_ptr + VLEN;
                bias_ptr = bias_ptr + VLEN;
            }
            inline for (0..N_S2) |j| {
                // Perform normalization in f32
                const normalized = (input_val[j] - mean_v) * rcp_std_dev_v;
                const scaled = normalized * weight_val[j];
                final_value[j] = scaled + bias_val[j];
            }
            if (CHECK_EVERYTHING) {
                const check_me: [VLEN*N_S2]f32 = @bitCast(final_value);
                for (check_me) |val| {
                    if (std.math.isNan(val)) {
                        return error.ComputedNaN;
                    }
                    if (std.math.isInf(val)) {
                        return error.ComputedInfinity;
                    }
                }
            }
            inline for (0..N_S2) |j| {
                // Cast back to original type T only at the end
                out_ptr[0..VLEN].* = @as(Tv, @floatCast(final_value[j]));
                out_ptr += VLEN;
            }
        }
        end_ptr = input.data.ptr + end_idx;
        end_ptr = end_ptr - VLEN + 1;
        while (@intFromPtr(start_ptr) < @intFromPtr(end_ptr)) {
            @setFloatMode(.optimized);
            // Cast all values to f32 for intermediate calculations
            const input_val: f32v = @floatCast(@as(Tv, start_ptr[0..VLEN].*));
            const weight_val: f32v = @floatCast(@as(Tv, weight_ptr[0..VLEN].*));
            const bias_val: f32v = @floatCast(@as(Tv, bias_ptr[0..VLEN].*));

            // Perform normalization in f32
            const normalized = (input_val - mean_v) * rcp_std_dev_v;
            const scaled = normalized * weight_val;
            const final_value = scaled + bias_val;

            if (CHECK_EVERYTHING) {
                const check_me: [VLEN]f32 = @bitCast(final_value);
                for (check_me) |val| {
                    if (std.math.isNan(val)) {
                        return error.ComputedNaN;
                    }
                    if (std.math.isInf(val)) {
                        return error.ComputedInfinity;
                    }
                }
            }

            // Cast back to original type T only at the end
            out_ptr[0..VLEN].* = @as(Tv, @floatCast(final_value));
            out_ptr += VLEN;
            start_ptr += VLEN;
            weight_ptr += VLEN;
            bias_ptr += VLEN;
        }
        end_ptr = input.data.ptr + end_idx;
        while (@intFromPtr(start_ptr) < @intFromPtr(end_ptr)) {
            @setFloatMode(.optimized);
            // Cast all values to f32 for intermediate calculations
            const input_val: f32 = @floatCast(start_ptr[0]);
            const weight_val: f32 = @floatCast(weight_ptr[0]);
            const bias_val: f32 = @floatCast(bias_ptr[0]);

            // Perform normalization in f32
            const normalized = (input_val - smean[0]) * rcp_std_dev;
            const scaled = normalized * weight_val;
            const final_value = scaled + bias_val;

            // Check for stability of computed value
            if (CHECK_EVERYTHING) {
                if (std.math.isNan(final_value)) {
                    return error.ComputedNaN;
                }
                if (std.math.isInf(final_value)) {
                    return error.ComputedInfinity;
                }
            }

            out_ptr[0] = @floatCast(final_value);
            out_ptr += 1;
            start_ptr += 1;
            weight_ptr += 1;
            bias_ptr += 1;
        }
    }

    // Final stability check on output
    if (CHECK_EVERYTHING) {
        try checkStability(T, output);
    }
    return output;
}

pub fn layerNormOld(comptime T: type, input: Tensor(T), weight: Tensor(T), bias: Tensor(T), eps: T) !Tensor(T) {
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

    // Allocate temporary buffer for exponentials
    var temp_exp = try allocator.alloc(f32, dim_size);
    defer allocator.free(temp_exp);

    const data = tensor.data.ptr; // Pointer to the tensor data

    var vec: usize = 0;
    while (vec < num_vectors) : (vec += 1) {
        const base_offset = vec * dim_size * stride;

        // Optimize for contiguous case (stride == 1)
        if (stride == 1) {
            const vec_start = base_offset;

            // Find max value in the vector
            var max: f32 = data[vec_start];
            var i: usize = 1;
            while (i < dim_size) : (i += 1) {
                const val = data[vec_start + i];
                if (val > max) max = val;
            }

            // Calculate exp and sum
            var sum: f32 = 0;
            i = 0;
            while (i < dim_size) : (i += 1) {
                const val = data[vec_start + i] - max;
                temp_exp[i] = if (val > -88.0) @exp(val) else 0;
                sum += temp_exp[i];
            }

            // Normalize in one pass if sum > 0
            if (sum > 0) {
                const inv_sum = 1.0 / sum;
                i = 0;
                while (i < dim_size) : (i += 1) {
                    data[vec_start + i] = temp_exp[i] * inv_sum;
                }
            }
        } else {
            // Non-contiguous case (stride > 1)
            var i: usize = 0;
            var max: f32 = data[base_offset + i * stride];
            i = 1;

            // Find max value in the vector
            while (i < dim_size) : (i += 1) {
                const val = data[base_offset + i * stride];
                if (val > max) max = val;
            }

            // Calculate exp and sum
            var sum: f32 = 0;
            i = 0;
            while (i < dim_size) : (i += 1) {
                const val = data[base_offset + i * stride] - max;
                temp_exp[i] = if (val > -88.0) @exp(val) else 0;
                sum += temp_exp[i];
            }

            // Normalize in one pass if sum > 0
            if (sum > 0) {
                const inv_sum = 1.0 / sum;
                i = 0;
                while (i < dim_size) : (i += 1) {
                    data[base_offset + i * stride] = temp_exp[i] * inv_sum;
                }
            }
        }
    }
}

pub fn gelu(comptime T: type, tensor: *Tensor(T)) !void {
    if (@typeInfo(T) != .Float) {
        @compileError("GELU operation requires floating-point tensor");
    }

    // Constants for GELU approximation
    const sqrt_2_div_pi: T = @sqrt(@as(T, 2.0) / std.math.pi);
    const alpha: T = 0.044715;

    // For small tensors, process directly without threading overhead
    if (tensor.data.len < 1024) {
        for (tensor.data) |*x| {
            const val = x.*;
            if (T == f16) {
                const val_f32 = @as(f32, val);
                const x_cubed_f32 = val_f32 * val_f32 * val_f32;
                const inner_f32 = sqrt_2_div_pi * (val_f32 + alpha * x_cubed_f32);
                const tanh_inner_f32 = std.math.tanh(inner_f32);
                x.* = @floatCast(0.5 * val_f32 * (1.0 + tanh_inner_f32));
            } else {
                const x_cubed = val * val * val;
                const inner = sqrt_2_div_pi * (val + alpha * x_cubed);
                const tanh_inner = std.math.tanh(inner);
                x.* = 0.5 * val * (1.0 + tanh_inner);
            }
        }
        return;
    }

    // For large tensors, use parallel processing
    const ThreadContext = struct {
        data: []T,
        start: usize,
        end: usize,
        const_sqrt_2_div_pi: T,
        const_alpha: T,
    };

    // Calculate thread count and ensure we don't create too many threads
    const thread_count = @min(@min(try std.Thread.getCpuCount(), tensor.data.len / 1024), 32);

    // Ensure at least one thread
    if (thread_count == 0) {
        // Handle the tensor with a single thread if it's too small
        for (tensor.data) |*x| {
            const val = x.*;
            if (T == f16) {
                const val_f32 = @as(f32, val);
                const x_cubed_f32 = val_f32 * val_f32 * val_f32;
                const inner_f32 = sqrt_2_div_pi * (val_f32 + alpha * x_cubed_f32);
                const tanh_inner_f32 = std.math.tanh(inner_f32);
                x.* = @floatCast(0.5 * val_f32 * (1.0 + tanh_inner_f32));
            } else {
                const x_cubed = val * val * val;
                const inner = sqrt_2_div_pi * (val + alpha * x_cubed);
                const tanh_inner = std.math.tanh(inner);
                x.* = 0.5 * val * (1.0 + tanh_inner);
            }
        }
        return;
    }

    var threads = try std.ArrayList(std.Thread).initCapacity(std.heap.page_allocator, thread_count);
    defer threads.deinit();

    // Calculate chunk size ensuring it's a multiple of 16 for better alignment
    const base_chunk_size = (tensor.data.len + thread_count - 1) / thread_count;
    const chunk_size = base_chunk_size + (16 - (base_chunk_size % 16)) % 16;

    const WorkerFn = struct {
        fn worker(ctx: *const ThreadContext) void {
            var i = ctx.start;
            const end = ctx.end;
            const data = ctx.data;

            // Process chunks of 16 elements at a time for better cache utilization
            const chunk_end = @min(end, (end & ~@as(usize, 15)));
            while (i < chunk_end) : (i += 16) {
                comptime var j: usize = 0;
                inline while (j < 16) : (j += 1) {
                    const val = data[i + j];
                    if (T == f16) {
                        const val_f32 = @as(f32, val);
                        const x_cubed_f32 = val_f32 * val_f32 * val_f32;
                        const inner_f32 = ctx.const_sqrt_2_div_pi * (val_f32 + ctx.const_alpha * x_cubed_f32);
                        const tanh_inner_f32 = std.math.tanh(inner_f32);
                        data[i + j] = @floatCast(0.5 * val_f32 * (1.0 + tanh_inner_f32));
                    } else {
                        const x_cubed = val * val * val;
                        const inner = ctx.const_sqrt_2_div_pi * (val + ctx.const_alpha * x_cubed);
                        const tanh_inner = std.math.tanh(inner);
                        data[i + j] = 0.5 * val * (1.0 + tanh_inner);
                    }
                }
            }

            // Handle remaining elements
            while (i < end) : (i += 1) {
                const val = data[i];
                if (T == f16) {
                    const val_f32 = @as(f32, val);
                    const x_cubed_f32 = val_f32 * val_f32 * val_f32;
                    const inner_f32 = ctx.const_sqrt_2_div_pi * (val_f32 + ctx.const_alpha * x_cubed_f32);
                    const tanh_inner_f32 = std.math.tanh(inner_f32);
                    data[i] = @floatCast(0.5 * val_f32 * (1.0 + tanh_inner_f32));
                } else {
                    const x_cubed = val * val * val;
                    const inner = ctx.const_sqrt_2_div_pi * (val + ctx.const_alpha * x_cubed);
                    const tanh_inner = std.math.tanh(inner);
                    data[i] = 0.5 * val * (1.0 + tanh_inner);
                }
            }
        }
    };

    var contexts = try std.ArrayList(ThreadContext).initCapacity(std.heap.page_allocator, thread_count);
    defer contexts.deinit();

    // Create thread contexts and spawn threads
    var i: usize = 0;
    while (i < thread_count) : (i += 1) {
        const start = i * chunk_size;
        const end = if (i == thread_count - 1) tensor.data.len else @min((i + 1) * chunk_size, tensor.data.len);

        try contexts.append(ThreadContext{
            .data = tensor.data,
            .start = start,
            .end = end,
            .const_sqrt_2_div_pi = sqrt_2_div_pi,
            .const_alpha = alpha,
        });

        try threads.append(try std.Thread.spawn(.{}, WorkerFn.worker, .{&contexts.items[i]}));
    }

    // Wait for all threads to complete
    for (threads.items) |thread| {
        thread.join();
    }
}

//// SIMD Operations ////

// AVX2 optimized broadcast add for f16

pub fn broadcast_add_simd(a: *Tensor(f16), b: Tensor(f16)) !void {
    // Validate broadcast compatibility
    if (b.shape.len > a.shape.len) {
        return error.InvalidBroadcast;
    }

    // Check dimensions match from right to left
    for (0..b.shape.len) |i| {
        const a_dim = a.shape[a.shape.len - 1 - i];
        const b_dim = b.shape[b.shape.len - 1 - i];
        if (b_dim != a_dim and b_dim != 1) {
            return error.IncompatibleBroadcast;
        }
    }

    // Special case for positional embeddings [B,M,N] + [1,M,N]
    if (a.shape.len == 3 and b.shape.len == 3 and
        b.shape[0] == 1 and
        b.shape[1] == a.shape[1] and
        b.shape[2] == a.shape[2])
    {
        const batch = a.shape[0];
        const seq_len = a.shape[1];
        const dim = a.shape[2];
        const elements_per_batch = seq_len * dim;

        // AVX2 uses 8 x f32 vectors
        const Vec = @Vector(8, f32);
        const vec_width = 8;
        const vec_count = elements_per_batch / vec_width;

        // Temporary buffers for f32 conversion
        var temp_a: [8]f32 = undefined;
        var temp_b: [8]f32 = undefined;
        var temp_result: [8]f32 = undefined;

        // Process each batch
        var batch_idx: usize = 0;
        while (batch_idx < batch) : (batch_idx += 1) {
            const batch_offset = batch_idx * elements_per_batch;

            // Vectorized portion
            var vec_idx: usize = 0;
            while (vec_idx < vec_count) : (vec_idx += 1) {
                const i = vec_idx * vec_width;

                // Convert f16 to f32 for a
                for (0..8) |j| {
                    temp_a[j] = @floatCast(a.data[batch_offset + i + j]);
                }

                // Convert f16 to f32 for b
                for (0..8) |j| {
                    temp_b[j] = @floatCast(b.data[i + j]);
                }

                // Perform SIMD addition in f32
                const vec_a = @as(Vec, temp_a);
                const vec_b = @as(Vec, temp_b);
                const result = vec_a + vec_b;

                // Store result and convert back to f16
                for (0..8) |j| {
                    temp_result[j] = result[j];
                    a.data[batch_offset + i + j] = @floatCast(temp_result[j]);
                }
            }

            // Handle remainder elements
            var i: usize = vec_count * vec_width;
            while (i < elements_per_batch) : (i += 1) {
                const a_val: f32 = @floatCast(a.data[batch_offset + i]);
                const b_val: f32 = @floatCast(b.data[i]);
                a.data[batch_offset + i] = @floatCast(a_val + b_val);
            }
        }
        return;
    }

    // Special case for [seq_len, dim] + [dim]
    if (a.shape.len == 2 and b.shape.len == 1 and b.shape[0] == a.shape[1]) {
        const seq_len = a.shape[0];
        const dim = a.shape[1];

        // AVX2 uses 8 x f32 vectors
        const Vec = @Vector(8, f32);
        const vec_width = 8;
        const vec_count = dim / vec_width;
        // Temporary buffers for f32 conversion
        var temp_a: [8]f32 = undefined;
        var temp_b: [8]f32 = undefined;
        var temp_result: [8]f32 = undefined;

        var row: usize = 0;
        while (row < seq_len) : (row += 1) {
            const row_offset = row * dim;

            // Vectorized portion
            var vec_idx: usize = 0;
            while (vec_idx < vec_count) : (vec_idx += 1) {
                const i = vec_idx * vec_width;

                // Convert f16 to f32 for a
                for (0..8) |j| {
                    temp_a[j] = @floatCast(a.data[row_offset + i + j]);
                }

                // Convert f16 to f32 for b
                for (0..8) |j| {
                    temp_b[j] = @floatCast(b.data[i + j]);
                }

                // Perform SIMD addition in f32
                const vec_a = @as(Vec, temp_a);
                const vec_b = @as(Vec, temp_b);
                const result = vec_a + vec_b;

                // Store result and convert back to f16
                for (0..8) |j| {
                    temp_result[j] = result[j];
                    a.data[row_offset + i + j] = @floatCast(temp_result[j]);
                }
            }

            // Handle remainder
            var i: usize = vec_count * vec_width;
            while (i < dim) : (i += 1) {
                const a_val: f32 = @floatCast(a.data[row_offset + i]);
                const b_val: f32 = @floatCast(b.data[i]);
                a.data[row_offset + i] = @floatCast(a_val + b_val);
            }
        }
        return;
    }

    // Fallback for general case
    const total_elements = blk: {
        var prod: usize = 1;
        for (a.shape) |dim| {
            prod *= dim;
        }
        break :blk prod;
    };

    var a_coords = try a.allocator.alloc(usize, a.shape.len);
    defer a.allocator.free(a_coords);

    var i: usize = 0;
    while (i < total_elements) : (i += 1) {
        var temp = i;

        for (0..a.shape.len) |j| {
            const rev_j = a.shape.len - 1 - j;
            a_coords[rev_j] = temp % a.shape[rev_j];
            temp /= a.shape[rev_j];
        }

        var b_idx: usize = 0;
        var b_stride: usize = 1;

        for (0..b.shape.len) |j| {
            const b_j = b.shape.len - 1 - j;
            const a_j = a.shape.len - 1 - j;
            const coord = a_coords[a_j] % b.shape[b_j];
            b_idx += coord * b_stride;
            b_stride *= b.shape[b_j];
        }

        // Convert to f32, add, then back to f16
        const a_val: f32 = @floatCast(a.data[i]);
        const b_val: f32 = @floatCast(b.data[b_idx]);
        a.data[i] = @floatCast(a_val + b_val);
    }
}
