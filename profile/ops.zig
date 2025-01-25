const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const max_items_per_row = 6; // Number of elements to show per row
const max_rows = 8; // Maximum number of rows to show before truncating
const Tensor = @import("tensor.zig").Tensor;
const builtin = @import("builtin");
const StabilityError = @import("tensor.zig").StabilityError;
const sgemm = @import("sgemm.zig");
const matmulInPlace = @import("sgemminplace.zig").matmulInPlace;
const hgemminplace = @import("hgemminplace.zig");
const hgemm = @import("hgemm.zig");
const Slice = @import("tensor.zig").Slice;
const testing = std.testing;
const expectEqual = testing.expectEqual;
const expectError = testing.expectError;
const Timer = std.time.Timer;
const printTimeDiff = @import("timeattention.zig").printTimeDiff;
const TensorView = @import("tensor.zig").TensorView;

const mode = std.builtin.FloatMode.optimized;
comptime {
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

/// Transposes a tensor by swapping specified dimensions
/// Parameters:
/// - tensor: Input tensor to transpose
/// - dim0: First dimension to swap
/// - dim1: Second dimension to swap
///
// pub fn transposeAxes(comptime T: type, tensor: *Tensor(T), dim0: usize, dim1: usize) !void {
//     if (dim0 >= tensor.shape.len or dim1 >= tensor.shape.len) {
//         return error.InvalidDimension;
//     }

//     // Calculate strides for the current shape
//     var strides = try tensor.allocator.alloc(usize, tensor.shape.len);
//     defer tensor.allocator.free(strides);

//     strides[tensor.shape.len - 1] = 1;
//     var i: usize = tensor.shape.len - 1;
//     while (i > 0) : (i -= 1) {
//         strides[i - 1] = strides[i] * tensor.shape[i];
//     }

//     // Create new shape with swapped dimensions
//     var new_shape = try tensor.allocator.alloc(usize, tensor.shape.len);
//     errdefer tensor.allocator.free(new_shape);

//     for (tensor.shape, 0..) |dim, idx| {
//         new_shape[idx] = if (idx == dim0) tensor.shape[dim1] else if (idx == dim1) tensor.shape[dim0] else dim;
//     }

//     // Allocate memory for transposed data
//     var new_data = try tensor.allocator.alignedAlloc(T, 32, tensor.data.len);
//     errdefer tensor.allocator.free(new_data);

//     // Calculate new strides
//     var new_strides = try tensor.allocator.alloc(usize, tensor.shape.len);
//     defer tensor.allocator.free(new_strides);

//     new_strides[tensor.shape.len - 1] = 1;
//     i = tensor.shape.len - 1;
//     while (i > 0) : (i -= 1) {
//         new_strides[i - 1] = new_strides[i] * new_shape[i];
//     }

//     // SIMD optimization for 3D tensors swapping first two dimensions
//     if (tensor.shape.len == 3 and dim0 == 0 and dim1 == 1 and
//         (T == f16 or T == f32))
//     {
//         const batch_size = tensor.shape[0];
//         const rows = tensor.shape[1];
//         const cols = tensor.shape[2];
//         const vector_size = if (T == f16) 16 else 8; // AVX2: 16 fp16 or 8 fp32

//         if (cols >= vector_size) {
//             const Vector = @Vector(vector_size, T);

//             var col: usize = 0;
//             while (col < cols) : (col += vector_size) {
//                 const vec_size = @min(vector_size, cols - col);

//                 // Process each column vector block
//                 for (0..batch_size) |b| {
//                     for (0..rows) |r| {
//                         const src_idx = b * rows * cols + r * cols + col;
//                         var vec: Vector = undefined;

//                         // Load with potential partial vector
//                         if (vec_size == vector_size) {
//                             vec = tensor.data[src_idx..][0..vector_size].*;
//                         } else {
//                             var temp: [vector_size]T = undefined;
//                             @memcpy(temp[0..vec_size], tensor.data[src_idx..][0..vec_size]);
//                             vec = temp;
//                         }

//                         // Store transposed
//                         const dst_idx = r * batch_size * cols + b * cols + col;
//                         if (vec_size == vector_size) {
//                             new_data[dst_idx..][0..vector_size].* = vec;
//                         } else {
//                             @memcpy(new_data[dst_idx..][0..vec_size], @as([vector_size]T, vec)[0..vec_size]);
//                         }
//                     }
//                 }
//             }
//             // Update tensor and return early since we handled this case
//             tensor.allocator.free(tensor.data);
//             tensor.data = new_data;
//             tensor.allocator.free(tensor.shape);
//             tensor.shape = new_shape;
//             return;
//         }
//     }

//     // General case implementation
//     var coords = try tensor.allocator.alloc(usize, tensor.shape.len);
//     defer tensor.allocator.free(coords);
//     @memset(coords, 0);

//     const total_elements = tensor.data.len;
//     var idx: usize = 0;
//     while (idx < total_elements) : (idx += 1) {
//         var remaining = idx;
//         for (0..tensor.shape.len) |dim| {
//             coords[dim] = remaining / new_strides[dim];
//             remaining %= new_strides[dim];
//         }

//         // Swap coordinates
//         const temp = coords[dim0];
//         coords[dim0] = coords[dim1];
//         coords[dim1] = temp;

//         // Calculate source index
//         var src_idx: usize = 0;
//         for (0..tensor.shape.len) |dim| {
//             src_idx += coords[dim] * strides[dim];
//         }

//         new_data[idx] = tensor.data[src_idx];
//     }

//     // Update tensor metadata
//     tensor.allocator.free(tensor.data);
//     tensor.data = new_data;
//     tensor.allocator.free(tensor.shape);
//     tensor.shape = new_shape;
// }
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

    // Update tensor metadata
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

const Thread = std.Thread;

// AVX2 vector type (8 x float32)
const Vec8f32 = @Vector(8, f32);
// Average time per token: 23377.30ms

// pub fn applyRotaryEmb(
//     allocator: Allocator,
//     x: Tensor(f16),
//     freqs_cis: Tensor(f32),
//     position_ids: Tensor(usize),
//     rot_dim: usize,
// ) !Tensor(f16) {
//     // Validate input constraints - exact same as original
//     if (x.shape.len != 3) return error.InvalidInputDimensions;
//     if (rot_dim != freqs_cis.shape[freqs_cis.shape.len - 2] * 2) {
//         return error.InvalidRotationDimension;
//     }

//     const n_heads = x.shape[0];
//     const seq_len = x.shape[1];
//     const head_dim = x.shape[2];

//     // Split x into rotation and pass-through parts exactly as original
//     var x_rot = try x.getSliceRange(&[_]Slice{
//         Slice.full(), // Head
//         Slice.full(), // Sequence
//         Slice.from(0, rot_dim), // First rot_dim features
//     });
//     defer x_rot.deinit();

//     var x_pass = if (rot_dim < head_dim) blk: {
//         const pass = try x.getSliceRange(&[_]Slice{
//             Slice.full(),
//             Slice.full(),
//             Slice.from(rot_dim, null),
//         });
//         break :blk pass;
//     } else Tensor(f16).init(allocator, &[_]usize{ n_heads, seq_len, 0 }) catch unreachable;
//     defer x_pass.deinit();

//     // Extract real and imaginary parts directly
//     var xq_r = try x_rot.getSliceRange(&[_]Slice{
//         Slice.full(),
//         Slice.full(),
//         Slice.from(0, rot_dim / 2),
//     });
//     var xq_i = try x_rot.getSliceRange(&[_]Slice{
//         Slice.full(),
//         Slice.full(),
//         Slice.from(rot_dim / 2, null),
//     });
//     defer xq_r.deinit();
//     defer xq_i.deinit();

//     // Extract cos and sin exactly as original
//     var cos_part = try freqs_cis.getSliceRange(&[_]Slice{
//         Slice.full(),
//         Slice.full(),
//         Slice.from(0, 1),
//     });
//     defer cos_part.deinit();

//     var sin_part = try freqs_cis.getSliceRange(&[_]Slice{
//         Slice.full(),
//         Slice.full(),
//         Slice.from(1, 2),
//     });
//     defer sin_part.deinit();

//     // Create freqs with exact same shape as original
//     var freqs_cos = try zeros(f32, allocator, &[_]usize{ 1, seq_len, rot_dim / 2 });
//     var freqs_sin = try zeros(f32, allocator, &[_]usize{ 1, seq_len, rot_dim / 2 });
//     defer freqs_cos.deinit();
//     defer freqs_sin.deinit();

//     // Fill frequencies exactly as original
//     for (0..seq_len) |i| {
//         const pos_id = position_ids.data[i];
//         const offset = i * (rot_dim / 2);
//         @memcpy(freqs_cos.data[offset .. offset + rot_dim / 2], cos_part.data[pos_id * cos_part.shape[1] .. (pos_id + 1) * cos_part.shape[1]]);
//         @memcpy(freqs_sin.data[offset .. offset + rot_dim / 2], sin_part.data[pos_id * sin_part.shape[1] .. (pos_id + 1) * sin_part.shape[1]]);
//     }

//     // Complex multiplication using SIMD but maintaining exact precision
//     const vec_size: usize = 8;
//     const rot_dim_half = rot_dim / 2;
//     var xq_out_r = try xq_r.castWithSimd(f32);
//     defer xq_out_r.deinit();
//     var xq_out_i = try xq_i.castWithSimd(f32);
//     defer xq_out_i.deinit();

//     // Process in SIMD-sized chunks
//     for (0..n_heads) |h| {
//         for (0..seq_len) |s| {
//             var f: usize = 0;
//             while (f + vec_size <= rot_dim_half) : (f += vec_size) {
//                 const base_idx = h * seq_len * rot_dim_half + s * rot_dim_half + f;
//                 const freq_idx = s * rot_dim_half + f;

//                 // Load vectors
//                 const xr = Vec8f32{
//                     xq_out_r.data[base_idx],
//                     xq_out_r.data[base_idx + 1],
//                     xq_out_r.data[base_idx + 2],
//                     xq_out_r.data[base_idx + 3],
//                     xq_out_r.data[base_idx + 4],
//                     xq_out_r.data[base_idx + 5],
//                     xq_out_r.data[base_idx + 6],
//                     xq_out_r.data[base_idx + 7],
//                 };

//                 const xi = Vec8f32{
//                     xq_out_i.data[base_idx],
//                     xq_out_i.data[base_idx + 1],
//                     xq_out_i.data[base_idx + 2],
//                     xq_out_i.data[base_idx + 3],
//                     xq_out_i.data[base_idx + 4],
//                     xq_out_i.data[base_idx + 5],
//                     xq_out_i.data[base_idx + 6],
//                     xq_out_i.data[base_idx + 7],
//                 };

//                 const cos = Vec8f32{
//                     freqs_cos.data[freq_idx],
//                     freqs_cos.data[freq_idx + 1],
//                     freqs_cos.data[freq_idx + 2],
//                     freqs_cos.data[freq_idx + 3],
//                     freqs_cos.data[freq_idx + 4],
//                     freqs_cos.data[freq_idx + 5],
//                     freqs_cos.data[freq_idx + 6],
//                     freqs_cos.data[freq_idx + 7],
//                 };

//                 const sin = Vec8f32{
//                     freqs_sin.data[freq_idx],
//                     freqs_sin.data[freq_idx + 1],
//                     freqs_sin.data[freq_idx + 2],
//                     freqs_sin.data[freq_idx + 3],
//                     freqs_sin.data[freq_idx + 4],
//                     freqs_sin.data[freq_idx + 5],
//                     freqs_sin.data[freq_idx + 6],
//                     freqs_sin.data[freq_idx + 7],
//                 };

//                 // Complex multiply with exact same operations as original
//                 const out_r = xr * cos - xi * sin;
//                 const out_i = xr * sin + xi * cos;

//                 // Store results
//                 @memcpy(xq_out_r.data[base_idx..][0..vec_size], @as([8]f32, out_r)[0..]);
//                 @memcpy(xq_out_i.data[base_idx..][0..vec_size], @as([8]f32, out_i)[0..]);
//             }

//             // Handle remaining elements exactly
//             while (f < rot_dim_half) : (f += 1) {
//                 const base_idx = h * seq_len * rot_dim_half + s * rot_dim_half + f;
//                 const freq_idx = s * rot_dim_half + f;

//                 const xr = xq_out_r.data[base_idx];
//                 const xi = xq_out_i.data[base_idx];
//                 const cos = freqs_cos.data[freq_idx];
//                 const sin = freqs_sin.data[freq_idx];

//                 xq_out_r.data[base_idx] = xr * cos - xi * sin;
//                 xq_out_i.data[base_idx] = xr * sin + xi * cos;
//             }
//         }
//     }

//     // Stack results exactly as original
//     var tensors = [_]Tensor(f32){ xq_out_r, xq_out_i };
//     var stacked = try stack(f32, &tensors, 3);
//     defer stacked.deinit();
//     try flatten(f32, &stacked, 2, 3);

//     // Combine with pass-through part exactly as original
//     if (x_pass.data.len > 0) {
//         var x_pass_f32 = try x_pass.castWithSimd(f32);
//         defer x_pass_f32.deinit();
//         var result = try concat(f32, stacked, x_pass_f32, 2);
//         defer result.deinit();
//         return try result.castWithSimd(f16);
//     } else {
//         var result = try stacked.copy();
//         defer result.deinit();
//         return try result.castWithSimd(f16);
//     }
// }

pub fn applyRotaryEmb(
    allocator: Allocator,
    x: Tensor(f16),
    freqs_cis: Tensor(f32),
    position_ids: Tensor(usize),
    rot_dim: usize,
) !Tensor(f16) {
    // Validation remains the same
    if (x.shape.len != 3) return error.InvalidInputDimensions;
    if (rot_dim != freqs_cis.shape[freqs_cis.shape.len - 2] * 2) {
        return error.InvalidRotationDimension;
    }

    const n_heads = x.shape[0];
    const seq_len = x.shape[1];
    const head_dim = x.shape[2];
    const half_rot = rot_dim / 2;

    // Allocate single output buffer
    var output = try Tensor(f16).init(allocator, x.shape);
    errdefer output.deinit();

    // Process in head-major order for better memory locality
    const block_size = 32; // Adjust based on cache line size
    const vec_size = 8;

    for (0..n_heads) |h| {
        // Process sequence in blocks for better cache utilization
        var seq_block: usize = 0;
        while (seq_block < seq_len) : (seq_block += block_size) {
            const seq_end = @min(seq_block + block_size, seq_len);

            // Process rotation dimension in SIMD-friendly chunks
            var f: usize = 0;
            while (f + vec_size <= half_rot) : (f += vec_size) {
                // Process each sequence position in the current block
                for (seq_block..seq_end) |s| {
                    const pos_id = position_ids.data[s];
                    const head_offset = h * seq_len * head_dim;
                    const seq_offset = s * head_dim;
                    const x_base = head_offset + seq_offset;
                    const freq_base = pos_id * half_rot * 2; // *2 for cos/sin pairs

                    // Load input vectors for real and imaginary parts
                    const xr = @Vector(8, f32){
                        @floatCast(x.data[x_base + f + 0]),
                        @floatCast(x.data[x_base + f + 1]),
                        @floatCast(x.data[x_base + f + 2]),
                        @floatCast(x.data[x_base + f + 3]),
                        @floatCast(x.data[x_base + f + 4]),
                        @floatCast(x.data[x_base + f + 5]),
                        @floatCast(x.data[x_base + f + 6]),
                        @floatCast(x.data[x_base + f + 7]),
                    };

                    const xi = @Vector(8, f32){
                        @floatCast(x.data[x_base + half_rot + f + 0]),
                        @floatCast(x.data[x_base + half_rot + f + 1]),
                        @floatCast(x.data[x_base + half_rot + f + 2]),
                        @floatCast(x.data[x_base + half_rot + f + 3]),
                        @floatCast(x.data[x_base + half_rot + f + 4]),
                        @floatCast(x.data[x_base + half_rot + f + 5]),
                        @floatCast(x.data[x_base + half_rot + f + 6]),
                        @floatCast(x.data[x_base + half_rot + f + 7]),
                    };

                    // Load precomputed frequency vectors
                    // Note: freqs_cis is already in [end, dim/2, 2] format
                    const cos = @Vector(8, f32){
                        freqs_cis.data[freq_base + f * 2 + 0],
                        freqs_cis.data[freq_base + (f + 1) * 2 + 0],
                        freqs_cis.data[freq_base + (f + 2) * 2 + 0],
                        freqs_cis.data[freq_base + (f + 3) * 2 + 0],
                        freqs_cis.data[freq_base + (f + 4) * 2 + 0],
                        freqs_cis.data[freq_base + (f + 5) * 2 + 0],
                        freqs_cis.data[freq_base + (f + 6) * 2 + 0],
                        freqs_cis.data[freq_base + (f + 7) * 2 + 0],
                    };

                    const sin = @Vector(8, f32){
                        freqs_cis.data[freq_base + f * 2 + 1],
                        freqs_cis.data[freq_base + (f + 1) * 2 + 1],
                        freqs_cis.data[freq_base + (f + 2) * 2 + 1],
                        freqs_cis.data[freq_base + (f + 3) * 2 + 1],
                        freqs_cis.data[freq_base + (f + 4) * 2 + 1],
                        freqs_cis.data[freq_base + (f + 5) * 2 + 1],
                        freqs_cis.data[freq_base + (f + 6) * 2 + 1],
                        freqs_cis.data[freq_base + (f + 7) * 2 + 1],
                    };

                    // Compute rotations
                    const out_r = xr * cos - xi * sin;
                    const out_i = xr * sin + xi * cos;

                    // Store results
                    for (0..8) |i| {
                        output.data[x_base + f + i] = @floatCast(out_r[i]);
                        output.data[x_base + half_rot + f + i] = @floatCast(out_i[i]);
                    }
                }
            }

            // Handle remaining elements in rotation dimension
            while (f < half_rot) : (f += 1) {
                for (seq_block..seq_end) |s| {
                    const pos_id = position_ids.data[s];
                    const head_offset = h * seq_len * head_dim;
                    const seq_offset = s * head_dim;
                    const x_base = head_offset + seq_offset;
                    const freq_base = pos_id * half_rot * 2;

                    const xr: f32 = @floatCast(x.data[x_base + f]);
                    const xi: f32 = @floatCast(x.data[x_base + half_rot + f]);
                    const cos = freqs_cis.data[freq_base + f * 2];
                    const sin = freqs_cis.data[freq_base + f * 2 + 1];

                    output.data[x_base + f] = @floatCast(xr * cos - xi * sin);
                    output.data[x_base + half_rot + f] = @floatCast(xr * sin + xi * cos);
                }
            }
        }
    }

    // Handle pass-through part if needed
    if (rot_dim < head_dim) {
        const pass_dim = head_dim - rot_dim;
        for (0..n_heads) |h| {
            const head_offset = h * seq_len * head_dim;
            for (0..seq_len) |s| {
                const src_base = head_offset + s * head_dim + rot_dim;
                const dst_base = src_base;
                @memcpy(
                    output.data[dst_base .. dst_base + pass_dim],
                    x.data[src_base .. src_base + pass_dim],
                );
            }
        }
    }

    return output;
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
        try hgemminplace.matmul(allocator, query_head, key_head, attn_weights);

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
        var attn_weights_f16 = try attn_weights.castWithSimd(f16);
        defer attn_weights_f16.deinit();

        // Compute attention output
        try hgemminplace.matmul(allocator, attn_weights_f16, value_head, out_head);

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
    var timer = try Timer.start();
    const total_start = timer.read();

    const n_heads = query.shape[0];
    const q_len = query.shape[1];
    const kv_len = key.shape[1];
    const head_dim = query.shape[2];

    // Scale factor
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    // Initialize output tensor
    const init_start = timer.read();
    var out = try Tensor(f16).init(allocator, &[_]usize{ n_heads, q_len, head_dim });
    errdefer out.deinit();
    try printTimeDiff(&timer, init_start, "Output Tensor Initialization");

    // Prepare transposed key for all heads
    const transpose_start = timer.read();
    var key_transpose = try key.copy();
    defer key_transpose.deinit();
    try transposeAxes(f16, &key_transpose, 1, 2);
    try printTimeDiff(&timer, transpose_start, "Key Transpose");

    // Track total times for operations across all heads
    var total_slice_time: i128 = 0;
    var total_qk_time: i128 = 0;
    var total_scale_time: i128 = 0;
    var total_softmax_time: i128 = 0;
    var total_attn_time: i128 = 0;
    var total_copy_time: i128 = 0;

    // Process each attention head separately
    const heads_start = timer.read();
    for (0..n_heads) |h| {
        // Get the query, key, value slices for this head
        const slice_start = timer.read();
        var query_head = try query.getDimensionSlice(0, h);
        defer query_head.deinit();
        var key_head = try key_transpose.getDimensionSlice(0, h);
        defer key_head.deinit();
        var value_head = try value.getDimensionSlice(0, h);
        defer value_head.deinit();
        total_slice_time += timer.read() - slice_start;

        // Calculate QK^T
        const qk_start = timer.read();
        var attn_weights = try Tensor(f32).init(allocator, &[_]usize{ q_len, kv_len });
        defer attn_weights.deinit();
        try hgemminplace.matmul(allocator, query_head, key_head, attn_weights);
        total_qk_time += timer.read() - qk_start;

        // Apply scaling
        const scale_start = timer.read();
        for (attn_weights.data) |*w| {
            w.* *= scale;
        }
        total_scale_time += timer.read() - scale_start;

        // Apply softmax
        const softmax_start = timer.read();
        try softmax(&attn_weights, 1, allocator);
        total_softmax_time += timer.read() - softmax_start;

        // Compute attention output
        const attn_start = timer.read();
        var value_head_f32 = try value_head.castWithSimd(f32);
        defer value_head_f32.deinit();
        const out_head = try sgemm.matmul(f32, attn_weights, value_head_f32, allocator);
        total_attn_time += timer.read() - attn_start;

        // Copy to output tensor
        const copy_start = timer.read();
        for (0..q_len) |q| {
            for (0..head_dim) |d| {
                const out_idx = h * q_len * head_dim + q * head_dim + d;
                const head_idx = q * head_dim + d;
                out.data[out_idx] = @floatCast(out_head.data[head_idx]);
            }
        }
        total_copy_time += timer.read() - copy_start;
    }

    // Print average times per head
    const stdout = std.io.getStdOut().writer();
    const heads_f64 = @as(f64, @floatFromInt(n_heads));
    try stdout.print("\x1b[91m [VISION PROFILE] Average times per attention head:\x1b[0m\n", .{});
    try stdout.print("\x1b[91m   Slice Operations: {d:.2}ms\x1b[0m\n", .{@as(f64, @floatFromInt(total_slice_time)) / heads_f64 / 1_000_000.0});
    try stdout.print("\x1b[91m   QK Multiplication: {d:.2}ms\x1b[0m\n", .{@as(f64, @floatFromInt(total_qk_time)) / heads_f64 / 1_000_000.0});
    try stdout.print("\x1b[91m   Scale Application: {d:.2}ms\x1b[0m\n", .{@as(f64, @floatFromInt(total_scale_time)) / heads_f64 / 1_000_000.0});
    try stdout.print("\x1b[91m   Softmax: {d:.2}ms\x1b[0m\n", .{@as(f64, @floatFromInt(total_softmax_time)) / heads_f64 / 1_000_000.0});
    try stdout.print("\x1b[91m   Attention Computation: {d:.2}ms\x1b[0m\n", .{@as(f64, @floatFromInt(total_attn_time)) / heads_f64 / 1_000_000.0});
    try stdout.print("\x1b[91m   Output Copy: {d:.2}ms\x1b[0m\n", .{@as(f64, @floatFromInt(total_copy_time)) / heads_f64 / 1_000_000.0});

    try printTimeDiff(&timer, heads_start, "Total Attention Heads Processing");
    try printTimeDiff(&timer, total_start, "Total Maskless Attention");

    return out;
}

// pub fn softmax(tensor: *Tensor(f32), dim: usize, allocator: Allocator) !void {
//     const dim_size = tensor.shape[dim];

//     // Calculate stride for the specified dimension
//     var stride: usize = 1;
//     for (dim + 1..tensor.shape.len) |i| {
//         stride *= tensor.shape[i];
//     }

//     // Calculate number of vectors to process
//     var num_vectors: usize = 1;
//     for (0..dim) |i| {
//         num_vectors *= tensor.shape[i];
//     }

//     // Allocate temporary storage for exp values
//     var temp_exp = try allocator.alloc(f32, dim_size);
//     defer allocator.free(temp_exp);

//     // Process each vector
//     for (0..num_vectors) |i| {
//         const base_idx = i * dim_size * stride;

//         // Find max for numerical stability (in f32)
//         var max: f32 = -std.math.inf(f32);
//         for (0..dim_size) |j| {
//             const val = tensor.data[base_idx + j * stride];
//             if (val > max) max = val;
//         }

//         // Calculate exp and sum (in f32)
//         var sum: f32 = 0;
//         for (0..dim_size) |j| {
//             const idx = base_idx + j * stride;
//             const val = tensor.data[idx] - max;
//             temp_exp[j] = if (val > -88.0) @exp(val) else 0;
//             sum += temp_exp[j];
//         }

//         // Normalize
//         if (sum > 0) {
//             const inv_sum = 1.0 / sum;
//             for (0..dim_size) |j| {
//                 const idx = base_idx + j * stride;
//                 tensor.data[idx] = temp_exp[j] * inv_sum;
//             }
//         }
//     }
// }

// pub fn softmax(tensor: *Tensor(f32), dim: usize, allocator: Allocator) !void {
//     const dim_size = tensor.shape[dim];

//     // Calculate stride for the specified dimension
//     var stride: usize = 1;
//     for (dim + 1..tensor.shape.len) |i| {
//         stride *= tensor.shape[i];
//     }

//     // Calculate number of vectors to process
//     var num_vectors: usize = 1;
//     for (0..dim) |i| {
//         num_vectors *= tensor.shape[i];
//     }

//     // Check if we can use SIMD
//     const can_use_simd = dim_size >= VECTOR_WIDTH and stride == 1;

//     if (can_use_simd) {
//         try softmaxSIMD(tensor, dim_size, num_vectors, allocator);
//     } else {
//         try softmaxScalar(tensor, dim_size, stride, num_vectors, allocator);
//     }
// }

// pub fn softmax(tensor: *Tensor(f32), dim: usize, allocator: Allocator) !void {
//     const dim_size = tensor.shape[dim];

//     // Calculate stride for the specified dimension
//     var stride: usize = 1;
//     for (dim + 1..tensor.shape.len) |i| {
//         stride *= tensor.shape[i];
//     }

//     // Calculate number of vectors to process
//     var num_vectors: usize = 1;
//     for (0..dim) |i| {
//         num_vectors *= tensor.shape[i];
//     }

//     var temp_exp = try allocator.alloc(f32, dim_size);
//     defer allocator.free(temp_exp);

//     // Simple optimizations:
//     // 1. Use while loops instead of for
//     // 2. Minimize array bounds checking with pointers
//     // 3. Let compiler auto-vectorize
//     // 4. Keep memory access patterns simple
//     var vec: usize = 0;
//     while (vec < num_vectors) : (vec += 1) {
//         const base_idx = vec * dim_size * stride;

//         // Find max
//         var max: f32 = tensor.data[base_idx];
//         var i: usize = 1;
//         while (i < dim_size) : (i += 1) {
//             const val = tensor.data[base_idx + i * stride];
//             if (val > max) max = val;
//         }

//         // Calculate exp and sum
//         var sum: f32 = 0;
//         i = 0;
//         while (i < dim_size) : (i += 1) {
//             const val = tensor.data[base_idx + i * stride] - max;
//             // Avoid exp calculation if value too small
//             temp_exp[i] = if (val > -88.0) @exp(val) else 0;
//             sum += temp_exp[i];
//         }

//         // Normalize in one pass if sum > 0
//         if (sum > 0) {
//             const inv_sum = 1.0 / sum;
//             i = 0;
//             while (i < dim_size) : (i += 1) {
//                 tensor.data[base_idx + i * stride] = temp_exp[i] * inv_sum;
//             }
//         }
//     }
// }

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
        // var a_coords = try a.allocator.alloc(usize, a.shape.len);
        // defer a.allocator.free(a_coords);
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

// Thread workspace pre-allocated for each thread
const ThreadWorkspace = struct {
    allocator: Allocator,
    query_head: Tensor(f32),
    key_head: Tensor(f32),
    value_head: Tensor(f32),
    attn_weights: Tensor(f32),
    out_head: Tensor(f32),

    pub fn init(allocator: Allocator, max_seq_len: usize, head_dim: usize) !ThreadWorkspace {
        return ThreadWorkspace{
            .allocator = allocator,
            // Pre-allocate tensors with maximum possible sizes
            .query_head = try Tensor(f32).init(allocator, &[_]usize{ max_seq_len, head_dim }),
            .key_head = try Tensor(f32).init(allocator, &[_]usize{ head_dim, max_seq_len }),
            .value_head = try Tensor(f32).init(allocator, &[_]usize{ max_seq_len, head_dim }),
            .attn_weights = try Tensor(f32).init(allocator, &[_]usize{ max_seq_len, max_seq_len }),
            .out_head = try Tensor(f32).init(allocator, &[_]usize{ max_seq_len, head_dim }),
        };
    }

    pub fn deinit(self: *ThreadWorkspace) void {
        self.query_head.deinit();
        self.key_head.deinit();
        self.value_head.deinit();
        self.attn_weights.deinit();
        self.out_head.deinit();
    }
};

// // Persistent thread pool context
const ThreadPoolContext = struct {
    workspaces: []ThreadWorkspace,
    allocator: Allocator,

    pub fn init(allocator: Allocator, thread_count: usize, max_seq_len: usize, head_dim: usize) !ThreadPoolContext {
        const workspaces = try allocator.alloc(ThreadWorkspace, thread_count);
        errdefer allocator.free(workspaces);

        var initialized: usize = 0;
        errdefer {
            // Clean up any successfully initialized workspaces
            for (workspaces[0..initialized]) |*workspace| {
                workspace.deinit();
            }
        }

        // Initialize workspaces for each thread
        for (workspaces) |*workspace| {
            workspace.* = try ThreadWorkspace.init(allocator, max_seq_len, head_dim);
            initialized += 1;
        }

        return ThreadPoolContext{
            .workspaces = workspaces,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ThreadPoolContext) void {
        for (self.workspaces) |*workspace| {
            workspace.deinit();
        }
        self.allocator.free(self.workspaces);
    }
};

// // Updated thread context with workspace
const AttnThreadContext = struct {
    start_head: usize,
    end_head: usize,
    query: Tensor(f32),
    key: Tensor(f32),
    value: Tensor(f32),
    out: *Tensor(f16),
    scale: f32,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    workspace: *ThreadWorkspace,
};

// pub fn multimasklessDotProductAttention(
//     query: Tensor(f16),
//     key: Tensor(f16),
//     value: Tensor(f16),
//     allocator: Allocator,
// ) !Tensor(f16) {
//     var timer = try Timer.start();
//     const total_start = timer.read();

//     const n_heads = query.shape[0];
//     const q_len = query.shape[1];
//     const kv_len = key.shape[1];
//     const head_dim = query.shape[2];

//     // Scale factor
//     const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

//     // Initialize output tensor
//     const init_and_cast_time = timer.read();
//     var out = try Tensor(f16).init(allocator, &[_]usize{ n_heads, q_len, head_dim });
//     errdefer out.deinit();

//     // Pre-cast query and value to f32 (reused across threads)
//     var query_f32 = try query.castWithSimd(f32);
//     defer query_f32.deinit();
//     var value_f32 = try value.castWithSimd(f32);
//     defer value_f32.deinit();
//     try printTimeDiff(&timer, init_and_cast_time, "Init and Query and Value tensors");

//     // Prepare transposed key for all heads
//     const transpose_time = timer.read();
//     var key_transpose = try key.copy();
//     defer key_transpose.deinit();
//     try transposeAxes(f16, &key_transpose, 1, 2);
//     var key_transpose_f32 = try key_transpose.castWithSimd(f32);
//     defer key_transpose_f32.deinit();
//     try printTimeDiff(&timer, transpose_time, "Transpose + Cast key");

//     // Initialize thread pool and workspaces
//     const pool_init_time = timer.read();
//     const thread_count = @min(n_heads, try Thread.getCpuCount());
//     var thread_pool = try ThreadPoolContext.init(allocator, thread_count, q_len, head_dim);
//     defer thread_pool.deinit();

//     var threads = try allocator.alloc(Thread, thread_count);
//     defer allocator.free(threads);
//     try printTimeDiff(&timer, pool_init_time, "Init thread pool");

//     // Divide work among threads
//     const divide_work_time = timer.read();
//     const heads_per_thread = n_heads / thread_count;
//     const remaining_heads = n_heads % thread_count;

//     var current_head: usize = 0;
//     var thread_contexts = try allocator.alloc(AttnThreadContext, thread_count);
//     defer allocator.free(thread_contexts);
//     try printTimeDiff(&timer, divide_work_time, "Divide work among threads");

//     // Thread worker function
//     const worker = struct {
//         fn process(ctx: AttnThreadContext) !void {
//             var processtimer = try Timer.start();
//             const workspace = ctx.workspace;

//             for (ctx.start_head..ctx.end_head) |h| {
//                 // Get slices for this head using pre-allocated workspace
//                 const copy_head_time = processtimer.read();
//                 try copyHeadData(f32, &workspace.query_head, ctx.query, h);
//                 try copyHeadData(f32, &workspace.key_head, ctx.key, h);
//                 try copyHeadData(f32, &workspace.value_head, ctx.value, h);
//                 try printTimeDiff(&processtimer, copy_head_time, "Copy head data");

//                 // Calculate QK^T directly in f32 using workspace tensors
//                 const matmul_time = processtimer.read();
//                 workspace.attn_weights = try sgemm.matmul(f32, workspace.query_head, workspace.key_head, workspace.allocator);
//                 try printTimeDiff(&processtimer, matmul_time, "Matmul QK^T");

//                 // Apply scaling and softmax in-place
//                 const scale_time = processtimer.read();
//                 for (workspace.attn_weights.data) |*w| {
//                     w.* *= ctx.scale;
//                 }
//                 try printTimeDiff(&processtimer, scale_time, "Scale Time");
//                 const softmax_time = processtimer.read();
//                 try softmax(&workspace.attn_weights, 1, workspace.allocator);
//                 try printTimeDiff(&processtimer, softmax_time, "Softmax");

//                 // Compute attention output using workspace
//                 const attn_time = processtimer.read();
//                 workspace.out_head = try sgemm.matmul(f32, workspace.attn_weights, workspace.value_head, workspace.allocator);
//                 try printTimeDiff(&processtimer, attn_time, "Matmul attention output");

//                 // Copy to output tensor with casting
//                 const copy_out_time = processtimer.read();
//                 const out_slice = h * ctx.q_len * ctx.head_dim;
//                 for (0..ctx.q_len * ctx.head_dim) |i| {
//                     ctx.out.data[out_slice + i] = @floatCast(workspace.out_head.data[i]);
//                 }
//                 try printTimeDiff(&processtimer, copy_out_time, "Copy output data");
//             }
//         }
//     }.process;

//     // Launch threads
//     for (0..thread_count) |t| {
//         const start_head = current_head;
//         const extra_head: usize = if (t < remaining_heads) 1 else 0;
//         current_head += heads_per_thread + extra_head;

//         thread_contexts[t] = AttnThreadContext{
//             .start_head = start_head,
//             .end_head = current_head,
//             .query = query_f32,
//             .key = key_transpose_f32,
//             .value = value_f32,
//             .out = &out,
//             .scale = scale,
//             .q_len = q_len,
//             .kv_len = kv_len,
//             .head_dim = head_dim,
//             .workspace = &thread_pool.workspaces[t],
//         };

//         threads[t] = try Thread.spawn(.{}, worker, .{thread_contexts[t]});
//     }

//     // Wait for all threads to complete
//     for (threads) |thread| {
//         thread.join();
//     }

//     try printTimeDiff(&timer, total_start, "Total Maskless Attention");

//     return out;
// }

pub fn multimasklessDotProductAttention(
    query: Tensor(f16),
    key: Tensor(f16),
    value: Tensor(f16),
    allocator: Allocator,
) !Tensor(f16) {
    var timer = try Timer.start();
    const total_start = timer.read();

    const n_heads = query.shape[0];
    const q_len = query.shape[1];
    const kv_len = key.shape[1];
    const head_dim = query.shape[2];

    // Scale factor
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    // Initialize output tensor
    const init_and_cast_time = timer.read();
    var out = try Tensor(f16).init(allocator, &[_]usize{ n_heads, q_len, head_dim });
    errdefer out.deinit();

    // Pre-cast query and value to f32 (reused across threads)
    var query_f32 = try query.castWithSimd(f32);
    defer query_f32.deinit();
    var value_f32 = try value.castWithSimd(f32);
    defer value_f32.deinit();
    try printTimeDiff(&timer, init_and_cast_time, "Init and Query and Value tensors");

    // Prepare transposed key for all heads
    const transpose_time = timer.read();
    var key_transpose = try key.copy();
    defer key_transpose.deinit();
    try transposeAxes(f16, &key_transpose, 1, 2);
    var key_transpose_f32 = try key_transpose.castWithSimd(f32);
    defer key_transpose_f32.deinit();
    try printTimeDiff(&timer, transpose_time, "Transpose + Cast key");

    // Initialize thread pool and workspaces
    const pool_init_time = timer.read();
    const thread_count = @min(n_heads, try Thread.getCpuCount());
    var thread_pool = try ThreadPoolContext.init(allocator, thread_count, q_len, head_dim);
    defer thread_pool.deinit();

    var threads = try allocator.alloc(Thread, thread_count);
    defer allocator.free(threads);
    try printTimeDiff(&timer, pool_init_time, "Init thread pool");

    // Divide work among threads
    const divide_work_time = timer.read();
    const heads_per_thread = n_heads / thread_count;
    const remaining_heads = n_heads % thread_count;

    var current_head: usize = 0;
    var thread_contexts = try allocator.alloc(AttnThreadContext, thread_count);
    defer allocator.free(thread_contexts);
    try printTimeDiff(&timer, divide_work_time, "Divide work among threads");

    // Thread worker function
    // Modified worker function with proper shape updates
    const worker = struct {
        fn process(ctx: AttnThreadContext) !void {
            var processtimer = try Timer.start();
            const workspace = ctx.workspace;

            for (ctx.start_head..ctx.end_head) |h| {
                // Copy head data and adjust shapes
                const copy_head_time = processtimer.read();
                try copyHeadData(f32, &workspace.query_head, ctx.query, h);
                try copyHeadData(f32, &workspace.key_head, ctx.key, h);
                try copyHeadData(f32, &workspace.value_head, ctx.value, h);
                try printTimeDiff(&processtimer, copy_head_time, "Copy head data");

                // OPTIMIZATION: Update existing shape arrays instead of reassigning
                // Query: [q_len, head_dim]
                workspace.query_head.shape[0] = ctx.q_len;
                workspace.query_head.shape[1] = ctx.head_dim;

                // Key: [head_dim, kv_len] (already transposed)
                workspace.key_head.shape[0] = ctx.head_dim;
                workspace.key_head.shape[1] = ctx.kv_len;

                // Value: [kv_len, head_dim]
                workspace.value_head.shape[0] = ctx.kv_len;
                workspace.value_head.shape[1] = ctx.head_dim;

                // Calculate QK^T with corrected shapes
                const matmul_time = processtimer.read();
                workspace.attn_weights.shape[0] = ctx.q_len;
                workspace.attn_weights.shape[1] = ctx.kv_len;
                try matmulInPlace(f32, workspace.query_head, workspace.key_head, &workspace.attn_weights, workspace.allocator);
                try printTimeDiff(&processtimer, matmul_time, "Matmul QK^T");

                // Apply scaling and softmax in-place
                const scale_time = processtimer.read();
                for (workspace.attn_weights.data[0 .. ctx.q_len * ctx.kv_len]) |*w| {
                    w.* *= ctx.scale;
                }
                try printTimeDiff(&processtimer, scale_time, "Scale Time");

                const softmax_time = processtimer.read();
                try softmax(&workspace.attn_weights, 1, workspace.allocator);
                try printTimeDiff(&processtimer, softmax_time, "Softmax");

                // Compute attention output
                const attn_time = processtimer.read();
                workspace.out_head.shape[0] = ctx.q_len;
                workspace.out_head.shape[1] = ctx.head_dim;
                try matmulInPlace(f32, workspace.attn_weights, workspace.value_head, &workspace.out_head, workspace.allocator);
                try printTimeDiff(&processtimer, attn_time, "Matmul attention output");

                // Copy to output tensor with casting
                const copy_out_time = processtimer.read();
                const out_slice = h * ctx.q_len * ctx.head_dim;
                for (0..ctx.q_len * ctx.head_dim) |i| {
                    ctx.out.data[out_slice + i] = @floatCast(workspace.out_head.data[i]);
                }
                try printTimeDiff(&processtimer, copy_out_time, "Copy output data");
            }
        }
    }.process;
    // Launch threads
    for (0..thread_count) |t| {
        const start_head = current_head;
        const extra_head: usize = if (t < remaining_heads) 1 else 0;
        current_head += heads_per_thread + extra_head;

        thread_contexts[t] = AttnThreadContext{
            .start_head = start_head,
            .end_head = current_head,
            .query = query_f32,
            .key = key_transpose_f32,
            .value = value_f32,
            .out = &out,
            .scale = scale,
            .q_len = q_len,
            .kv_len = kv_len,
            .head_dim = head_dim,
            .workspace = &thread_pool.workspaces[t],
        };

        threads[t] = try Thread.spawn(.{}, worker, .{thread_contexts[t]});
    }

    // Wait for all threads to complete
    for (threads) |thread| {
        thread.join();
    }

    try printTimeDiff(&timer, total_start, "Total Maskless Attention");

    return out;
}

// // Helper function to copy head data into workspace tensors
fn copyHeadData(comptime T: type, dst: *Tensor(T), src: Tensor(T), head_idx: usize) !void {
    const slice_size = src.shape[1] * src.shape[2];
    const start_idx = head_idx * slice_size;
    @memcpy(dst.data[0..slice_size], src.data[start_idx..][0..slice_size]);
}

/////////////////////////////////////////////// Masked Attention //////////
// Thread context for masked attention
const MaskedAttnThreadContext = struct {
    start_head: usize,
    end_head: usize,
    query: Tensor(f32),
    key: Tensor(f32),
    value: Tensor(f32),
    mask: Tensor(bool),
    out: *Tensor(f16),
    scale: f32,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    workspace: *MaskedThreadWorkspace,
};

// Thread pool context for masked attention
const MaskedThreadPoolContext = struct {
    workspaces: []MaskedThreadWorkspace,
    allocator: Allocator,

    pub fn init(allocator: Allocator, thread_count: usize, q_len: usize, kv_len: usize, head_dim: usize) !MaskedThreadPoolContext {
        var workspaces = try allocator.alloc(MaskedThreadWorkspace, thread_count);
        errdefer allocator.free(workspaces);

        for (0..thread_count) |i| {
            workspaces[i] = try MaskedThreadWorkspace.init(allocator, q_len, kv_len, head_dim);
        }

        return MaskedThreadPoolContext{
            .workspaces = workspaces,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *MaskedThreadPoolContext) void {
        for (self.workspaces) |*workspace| {
            workspace.deinit();
        }
        self.allocator.free(self.workspaces);
    }
};

const MaskedThreadWorkspace = struct {
    query_head: Tensor(f32),
    key_head: Tensor(f32),
    value_head: Tensor(f32),
    attn_weights: Tensor(f32),
    out_head: Tensor(f32),
    allocator: Allocator,

    pub fn init(allocator: Allocator, q_len: usize, kv_len: usize, head_dim: usize) !MaskedThreadWorkspace {
        return MaskedThreadWorkspace{
            .query_head = try Tensor(f32).init(allocator, &[_]usize{ q_len, head_dim }),
            .key_head = try Tensor(f32).init(allocator, &[_]usize{ head_dim, kv_len }),
            .value_head = try Tensor(f32).init(allocator, &[_]usize{ kv_len, head_dim }),
            .attn_weights = try Tensor(f32).init(allocator, &[_]usize{ q_len, kv_len }),
            .out_head = try Tensor(f32).init(allocator, &[_]usize{ q_len, head_dim }),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *MaskedThreadWorkspace) void {
        self.query_head.deinit();
        self.key_head.deinit();
        self.value_head.deinit();
        self.attn_weights.deinit();
        self.out_head.deinit();
    }
};

pub fn multiscaledDotProductAttention(
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

    // Pre-cast query and value to f32
    var query_f32 = try query.castWithSimd(f32);
    defer query_f32.deinit();
    var value_f32 = try value.castWithSimd(f32);
    defer value_f32.deinit();

    // Handle key tensor: cast to f32 first, then copy and transpose
    var key_f32 = try key.castWithSimd(f32);
    defer key_f32.deinit();
    var key_transpose = try Tensor(f32).init(allocator, &[_]usize{ n_heads, head_dim, kv_len });
    defer key_transpose.deinit();

    // Manual transpose since we know the exact layout
    for (0..n_heads) |h| {
        for (0..kv_len) |k| {
            for (0..head_dim) |d| {
                const src_idx = h * kv_len * head_dim + k * head_dim + d;
                const dst_idx = h * head_dim * kv_len + d * kv_len + k;
                key_transpose.data[dst_idx] = key_f32.data[src_idx];
            }
        }
    }

    // Initialize thread pool and workspaces with correct dimensions
    const thread_count = @min(n_heads, try Thread.getCpuCount());
    var thread_pool = try MaskedThreadPoolContext.init(allocator, thread_count, q_len, kv_len, head_dim);
    defer thread_pool.deinit();

    var threads = try allocator.alloc(Thread, thread_count);
    defer allocator.free(threads);

    // Divide work among threads
    const heads_per_thread = n_heads / thread_count;
    const remaining_heads = n_heads % thread_count;

    var current_head: usize = 0;
    var thread_contexts = try allocator.alloc(MaskedAttnThreadContext, thread_count);
    defer allocator.free(thread_contexts);

    // Thread worker function
    const worker = struct {
        fn process(ctx: MaskedAttnThreadContext) !void {
            const workspace = ctx.workspace;

            for (ctx.start_head..ctx.end_head) |h| {
                // Get query slice
                const query_slice = h * ctx.q_len * ctx.head_dim;
                @memcpy(workspace.query_head.data, ctx.query.data[query_slice..][0 .. ctx.q_len * ctx.head_dim]);

                // Get key slice (already transposed)
                const key_slice = h * ctx.head_dim * ctx.kv_len;
                @memcpy(workspace.key_head.data, ctx.key.data[key_slice..][0 .. ctx.head_dim * ctx.kv_len]);

                // Get value slice
                const value_slice = h * ctx.kv_len * ctx.head_dim;
                @memcpy(workspace.value_head.data, ctx.value.data[value_slice..][0 .. ctx.kv_len * ctx.head_dim]);

                if (workspace.attn_weights.data.len > 0) {
                    workspace.attn_weights.deinit();
                }
                // Calculate QK^T using sgemm
                workspace.attn_weights = try sgemm.matmul(f32, workspace.query_head, workspace.key_head, ctx.workspace.allocator);

                // Apply scaling
                for (workspace.attn_weights.data) |*w| {
                    w.* *= ctx.scale;
                }

                // Apply attention mask
                for (0..ctx.q_len) |i| {
                    for (0..ctx.kv_len) |j| {
                        const mask_idx = i * ctx.mask.shape[2] + j;
                        const weights_idx = i * ctx.kv_len + j;
                        if (!ctx.mask.data[mask_idx]) {
                            workspace.attn_weights.data[weights_idx] = -std.math.inf(f32);
                        }
                    }
                }

                try softmax(&workspace.attn_weights, 1, ctx.workspace.allocator);
                if (workspace.out_head.data.len > 0) {
                    workspace.out_head.deinit();
                }

                // Compute attention output using sgemm
                workspace.out_head = try sgemm.matmul(f32, workspace.attn_weights, workspace.value_head, ctx.workspace.allocator);

                // Copy to output tensor with casting
                const out_slice = h * ctx.q_len * ctx.head_dim;
                for (0..ctx.q_len * ctx.head_dim) |i| {
                    ctx.out.data[out_slice + i] = @floatCast(workspace.out_head.data[i]);
                }
            }
        }
    }.process;

    // Launch threads
    for (0..thread_count) |t| {
        const start_head = current_head;
        const extra_head: usize = if (t < remaining_heads) 1 else 0;
        current_head += heads_per_thread + extra_head;

        thread_contexts[t] = MaskedAttnThreadContext{
            .start_head = start_head,
            .end_head = current_head,
            .query = query_f32,
            .key = key_transpose,
            .value = value_f32,
            .mask = mask,
            .out = &out,
            .scale = scale,
            .q_len = q_len,
            .kv_len = kv_len,
            .head_dim = head_dim,
            .workspace = &thread_pool.workspaces[t],
        };

        threads[t] = try Thread.spawn(.{}, worker, .{thread_contexts[t]});
    }

    // Wait for all threads to complete
    for (threads) |thread| {
        thread.join();
    }

    return out;
}
