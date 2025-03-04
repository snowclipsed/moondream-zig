const std = @import("std");
const Tensor = @import("../core/tensor.zig").Tensor;
const Allocator = std.mem.Allocator;

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

pub fn normalizePatch(allocator: Allocator, input: Tensor(f16), mean: Tensor(f16), stdev: Tensor(f16)) !Tensor(f16) {
    var result = try Tensor(f16).init(allocator, input.shape);
    errdefer result.deinit();

    const batch = input.shape[0];
    const channels = input.shape[1];
    const height = input.shape[2];
    const width = input.shape[3];

    // Perform normalization in BCHW format (batch, channels, height, width)
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

pub fn convertBHWCtoBCHW(allocator: Allocator, input: Tensor(f16)) !Tensor(f16) {
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
