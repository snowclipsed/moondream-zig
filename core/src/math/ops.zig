const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;
const T: usize = 64; // Tile size (adjust as necessary)
const V: usize = 32; // Vector size (adjust as necessary)
const simd_align = @alignOf(@Vector(V, f32));

fn transposeSimd(allocator: std.mem.Allocator, matrix: []const f32, rows: usize, cols: usize) ![]f32 {
    const transposed = try allocator.alloc(f32, rows * cols);
    errdefer allocator.free(transposed);

    const VectorType = @Vector(4, f32);
    const simd_width = 4;

    var i: usize = 0;
    while (i < rows) : (i += simd_width) {
        var j: usize = 0;
        while (j < cols) : (j += 1) {
            if (i + simd_width <= rows) {
                const v0 = matrix[(i + 0) * cols + j];
                const v1 = matrix[(i + 1) * cols + j];
                const v2 = matrix[(i + 2) * cols + j];
                const v3 = matrix[(i + 3) * cols + j];
                const vec = VectorType{ v0, v1, v2, v3 };

                @as(*VectorType, @alignCast(@ptrCast(transposed.ptr + j * rows + i))).* = vec;
            } else {
                var k: usize = 0;
                while (k < rows - i) : (k += 1) {
                    transposed[j * rows + (i + k)] = matrix[(i + k) * cols + j];
                }
            }
        }
    }

    return transposed;
}

pub fn accumulate(accum: []f32, x: []const f32) void {
    std.debug.assert(accum.len == x.len);
    const len = accum.len;

    const VectorSize = std.simd.suggestVectorLength(f32) orelse 8;

    const VectorType = @Vector(VectorSize, f32);
    var i: usize = 0;

    // Process in SIMD-sized chunks
    while (i + VectorSize <= len) : (i += VectorSize) {
        var v_accum = @as(*align(4) VectorType, @ptrCast(accum[i..].ptr)).*;
        const v_x = @as(*align(4) const VectorType, @ptrCast(x[i..].ptr)).*;
        v_accum += v_x;
        @as(*align(4) VectorType, @ptrCast(accum[i..].ptr)).* = v_accum;
    }

    // Handle remaining elements
    while (i < len) : (i += 1) {
        accum[i] += x[i];
    }
}

fn argmax(x: []f32) usize {
    assert(x.len > 0);
    var max: f32 = x[0];
    var maxi: usize = 0;
    for (1..x.len) |i| {
        if (x[i] > max) {
            max = x[i];
            maxi = i;
        }
    }
    return maxi;
}

pub fn cos_2d(in: []const f32, out: []f32) !void {
    std.debug.assert(in.len == out.len);
    const len = in.len;

    const VectorSize = std.simd.suggestVectorLength(f32) orelse 8;
    const VectorType = @Vector(VectorSize, f32);
    var i: usize = 0;

    while (i + VectorSize <= len) : (i += VectorSize) {
        // Keep the const qualifier here
        const v_in = @as(*align(4) const VectorType, @ptrCast(in[i..].ptr)).*;
        @as(*align(4) VectorType, @ptrCast(out[i..].ptr)).* = @cos(v_in);
    }

    while (i < len) : (i += 1) {
        out[i] = std.math.sin(in[i]);
    }
}

pub fn sin_2d(in: []const f32, out: []f32) !void {
    std.debug.assert(in.len == out.len);
    const len = in.len;

    const VectorSize = std.simd.suggestVectorLength(f32) orelse 8;
    const VectorType = @Vector(VectorSize, f32);
    var i: usize = 0;

    while (i + VectorSize <= len) : (i += VectorSize) {
        // Keep the const qualifier here
        const v_in = @as(*align(4) const VectorType, @ptrCast(in[i..].ptr)).*;
        @as(*align(4) VectorType, @ptrCast(out[i..].ptr)).* = @sin(v_in);
    }

    while (i < len) : (i += 1) {
        out[i] = std.math.sin(in[i]);
    }
}

fn vector_dot_product(x: []const f32, y: []const f32) f32 {
    assert(x.len == y.len);
    const vector_width = V;
    const vec_len = x.len / vector_width; // num of SIMD vectors
    const vec_rem = x.len % vector_width; // num of f32 in the last SIMD vector

    // do the bulk of the work with SIMD
    var sum: @Vector(vector_width, f32) = @splat(0.0);
    var offset: usize = 0;
    for (0..vec_len) |_| {
        const xvec: @Vector(vector_width, f32) = x[offset..][0..vector_width].*;
        const yvec: @Vector(vector_width, f32) = y[offset..][0..vector_width].*;
        sum += xvec * yvec;
        offset += vector_width;
    }

    // handle the last few elements with normal scalar ops
    var sum_rem: f32 = 0.0;
    for (0..vec_rem) |i| {
        sum_rem += x[offset + i] * y[offset + i];
    }

    // reduce the SIMD vector to a scalar
    return @reduce(.Add, sum) + sum_rem;
}

fn vector_weighted_sum(xout: []f32, x: []const f32, y: f32) void {
    assert(xout.len == x.len);
    const vector_width = V;
    const vec_len = x.len / vector_width; // num of SIMD vectors
    const vec_rem = x.len % vector_width; // num of f32 in the last SIMD vector

    // do the bulk of the work with SIMD
    var offset: usize = 0;
    const yvector: @Vector(vector_width, f32) = @splat(y);
    for (0..vec_len) |_| {
        var xoutvec: @Vector(vector_width, f32) = xout[offset..][0..vector_width].*;
        const xvec: @Vector(vector_width, f32) = x[offset..][0..vector_width].*;
        xoutvec += xvec * yvector;
        xout[offset..][0..vector_width].* = xoutvec;
        offset += vector_width;
    }

    // handle the last few elements with normal scalar operations
    for (0..vec_rem) |i| {
        xout[offset + i] += x[offset + i] * y;
    }
}

fn outer(A: []const f32, B: []const f32, C: []f32) void {
    std.debug.assert(A.len * B.len == C.len);

    const VectorSize = 32;
    const VectorType = @Vector(VectorSize, f32);

    // operate over A's rows
    for (0..A.len) |i| {
        var j: usize = 0;
        // broadcast the selected element of A to a vector
        const a_splat = @as(VectorType, @splat(A[i]));

        // we then process B in chunks of VectorSize
        while (j + VectorSize <= B.len) : (j += VectorSize) {
            // load B directly as a vector, assuming alignment
            const b_vec = @as(*align(4) const VectorType, @ptrCast(B[j..].ptr)).*;
            // multiply and store directly
            @as(*align(4) VectorType, @ptrCast(C[i * B.len + j ..].ptr)).* = a_splat * b_vec;
        }

        // handle remaining elements
        while (j < B.len) : (j += 1) {
            C[i * B.len + j] = A[i] * B[j];
        }
    }
}

pub fn outerConventional(A: []const f32, B: []const f32, C: []f32, M: usize, N: usize) !void {
    // Ensure the input vectors and matrix have the correct sizes
    if (A.len != M or B.len != N) {
        return error.InvalidInputSize;
    }

    // Compute the outer product using conventional method
    var i: usize = 0;
    while (i < M) : (i += 1) {
        var j: usize = 0;
        while (j < N) : (j += 1) {
            C[i * N + j] = A[i] * B[j];
        }
    }
}

pub fn cat(allocator: Allocator, A: []const f32, B: []const f32, A_M: usize, A_N: usize, B_M: usize, B_N: usize, dim: usize) ![]f32 {

    // we have two cases, we can either concatenate along the rows or the columns
    // first case: concatenate along the rows

    if (dim == 0) {
        // check if the number of columns of the two matrices are the same
        if (A_N != B_N) {
            return error.InvalidInputSize;
        }

        // create a new matrix with the correct dimensions
        // we check whose number of rows is greater
        const M: usize = A_M + B_M;

        const N = A_N;

        const C = try allocator.alloc(f32, M * N);

        @memcpy(C[0 .. A_M * N], A[0..]);
        @memcpy(C[A_M * N ..], B[0..]);

        return C;
    }

    if (dim == 1) {
        // check if the number of rows of the two matrices are the same
        if (A_M != B_M) {
            return error.InvalidInputSize;
        }

        // create a new matrix with the correct dimensions
        // we check whose number of columns is greater
        const M: usize = A_M;

        const N = A_N + B_N;

        const C = try allocator.alloc(f32, M * N);

        for (0..M) |i| {
            // Copy the i-th row of matrix A
            @memcpy(C[i * N .. i * N + A_N], A[i * A_N .. (i + 1) * A_N]);
            // Copy the i-th row of matrix B
            @memcpy(C[i * N + A_N .. (i + 1) * N], B[i * B_N .. (i + 1) * B_N]);
        }

        return C;
    }

    return error.InvalidDimension;
}
