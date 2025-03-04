const std = @import("std");
const Tensor = @import("../core/tensor.zig").Tensor;
const Allocator = std.mem.Allocator;
const Thread = std.Thread;

// AVX2 vector type (8 x float32)
const Vec8f32 = @Vector(8, f32);

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

pub fn applyRotEmb(
    allocator: Allocator,
    x: Tensor(f16),
    n_heads: usize,
    head_dim: usize,
    freqs_cis: Tensor(f32),
    position_ids: Tensor(usize),
    rot_dim: usize,
) !Tensor(f16) {
    // Validation remains the same
    if (x.shape.len != 3) return error.InvalidInputDimensions;
    if (rot_dim != freqs_cis.shape[freqs_cis.shape.len - 2] * 2) {
        return error.InvalidRotationDimension;
    }

    // const n_heads = x.shape[0];
    const seq_len = x.shape[1];
    // const head_dim = x.shape[2];
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
