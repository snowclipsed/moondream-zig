const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const max_items_per_row = 6; // Number of elements to show per row
const max_rows = 8; // Maximum number of rows to show before truncating

pub fn Tensor(comptime DataType: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,
        shape: []usize,
        data: []align(32) DataType,
        // id: usize = undefined,

        // var id_mutex = std.Thread.Mutex{};
        // var next_id: usize = 0;

        // Original functions...
        /// Initializes a tensor with the given shape and allocates memory for its data.
        ///
        /// This function calculates the total size of the tensor by multiplying the dimensions
        /// provided in the `shape` array. It then allocates memory for the shape and data of the tensor
        /// using the provided allocator. The data memory is zero-initialized.
        ///
        /// Parameters:
        /// - `allocator`: The allocator to use for memory allocation.
        /// - `shape`: An array of dimensions specifying the shape of the tensor.
        ///
        /// Returns:
        /// - `Self`: A new instance of the tensor with allocated and zero-initialized data.
        ///
        /// Errors:
        /// - Returns an error if memory allocation for the shape or data fails.
        pub fn init(allocator: Allocator, shape: []const usize) !Self {
            var size: u128 = 1;
            for (shape) |dim| {
                size = size * dim;
                // Check if we would overflow usize
                if (size > std.math.maxInt(usize)) {
                    std.debug.print("size entered = {any}", .{size});
                    return error.TensorTooLarge;
                }
            }
            const shape_copy = try allocator.alloc(usize, shape.len);
            errdefer allocator.free(shape_copy);
            @memcpy(shape_copy, shape);

            // Now we know size fits in usize
            const final_size: usize = @intCast(size);
            const data = try allocator.alignedAlloc(DataType, 32, final_size);
            if (DataType == bool) {
                @memset(data, false);
            } else {
                @memset(data, 0);
            }

            const self = Tensor(DataType){
                .allocator = allocator,
                .shape = shape_copy,
                .data = data,
            };

            // // Assign unique ID
            // id_mutex.lock();
            // defer id_mutex.unlock();
            // self.id = next_id;
            // next_id += 1;

            // std.debug.print("Created tensor {d} at {x} with shape {any}\n", .{ self.id, @returnAddress(), shape });

            return self;
        }

        /// Deinitializes the tensor by freeing its allocated memory for shape and data.
        ///
        /// This function should be called when the tensor is no longer needed to
        /// release the memory allocated for its shape and data arrays.
        ///
        /// Parameters:
        /// - `self`: A pointer to the tensor instance to be deinitialized.
        ///
        /// Note:
        /// Ensure that the allocator used for allocating the shape and data is the same
        /// allocator used for freeing them to avoid undefined behavior.
        pub fn deinit(self: *Self) void {
            // std.debug.print("Destroying tensor {d} at {x} with shape {any}\n", .{ self.id, @returnAddress(), self.shape });
            self.allocator.free(self.shape);
            self.allocator.free(self.data);
        }

        /// Cast the tensor to a different data type.
        /// This function handles conversion between different numeric types,
        /// with checks for potential data loss.
        pub fn castTo(self: Self, comptime TargetType: type) !Tensor(TargetType) {
            // Create new tensor with same shape but target type
            var result = try Tensor(TargetType).init(self.allocator, self.shape);
            errdefer result.deinit();

            // This handles the casting based on type combinations
            for (self.data, 0..) |value, i| {
                result.data[i] = switch (@TypeOf(value)) {
                    // From floating point types
                    f16, f32, f64, f128 => switch (TargetType) {
                        // Float to float
                        f16, f32, f64, f128 => @floatCast(value),
                        // Float to signed int
                        i8, i16, i32, i64, i128 => blk: {
                            if (value != @trunc(value)) return error.LossyConversion;
                            if (value > @as(@TypeOf(value), @floatFromInt(std.math.maxInt(TargetType))) or
                                value < @as(@TypeOf(value), @floatFromInt(std.math.minInt(TargetType))))
                            {
                                return error.OutOfRange;
                            }
                            break :blk @intFromFloat(@trunc(value));
                        },
                        // Float to unsigned int
                        u8, u16, u32, u64, u128 => blk: {
                            if (value != @trunc(value)) return error.LossyConversion;
                            if (value < 0 or
                                value > @as(@TypeOf(value), @floatFromInt(std.math.maxInt(TargetType))))
                            {
                                return error.OutOfRange;
                            }
                            break :blk @intFromFloat(@trunc(value));
                        },
                        // Float to bool
                        bool => value != 0,
                        else => @compileError("Unsupported target type for float casting"),
                    },

                    // From signed integer types
                    i8, i16, i32, i64, i128 => switch (TargetType) {
                        // Signed int to float
                        f16, f32, f64, f128 => @floatFromInt(value),
                        // Signed int to signed int
                        i8, i16, i32, i64, i128 => blk: {
                            if (value > std.math.maxInt(TargetType) or
                                value < std.math.minInt(TargetType))
                            {
                                return error.OutOfRange;
                            }
                            break :blk @intCast(value);
                        },
                        // Signed int to unsigned int
                        u8, u16, u32, u64, u128 => blk: {
                            if (value < 0 or
                                value > std.math.maxInt(TargetType))
                            {
                                return error.OutOfRange;
                            }
                            break :blk @intCast(value);
                        },
                        // Signed int to bool
                        bool => value != 0,
                        else => @compileError("Unsupported target type for signed integer casting"),
                    },

                    // From unsigned integer types
                    u8, u16, u32, u64, u128 => switch (TargetType) {
                        // Unsigned int to float
                        f16, f32, f64, f128 => @floatFromInt(value),
                        // Unsigned int to signed int
                        i8, i16, i32, i64, i128 => blk: {
                            if (value > @as(u128, @intCast(std.math.maxInt(TargetType)))) {
                                return error.OutOfRange;
                            }
                            break :blk @intCast(value);
                        },
                        // Unsigned int to unsigned int
                        u8, u16, u32, u64, u128 => blk: {
                            if (value > std.math.maxInt(TargetType)) {
                                return error.OutOfRange;
                            }
                            break :blk @intCast(value);
                        },
                        // Unsigned int to bool
                        bool => value != 0,
                        else => @compileError("Unsupported target type for unsigned integer casting"),
                    },

                    // From boolean type
                    bool => switch (TargetType) {
                        // Bool to float
                        f16, f32, f64, f128 => if (value) @as(TargetType, 1) else @as(TargetType, 0),
                        // Bool to signed int
                        i8, i16, i32, i64, i128 => if (value) @as(TargetType, 1) else @as(TargetType, 0),
                        // Bool to unsigned int
                        u8, u16, u32, u64, u128 => if (value) @as(TargetType, 1) else @as(TargetType, 0),
                        // Bool to bool
                        bool => value,
                        else => @compileError("Unsupported target type for boolean casting"),
                    },

                    else => @compileError("Unsupported source type for casting"),
                };
            }

            return result;
        }
        pub fn castWithSimd(self: Self, comptime TargetType: type) !Tensor(TargetType) {
            // First create the shape copy
            const shape_copy = try self.allocator.alloc(usize, self.shape.len);
            errdefer self.allocator.free(shape_copy);
            @memcpy(shape_copy, self.shape);

            // Calculate total size
            var size: usize = 1;
            for (self.shape) |dim| {
                size *= dim;
            }

            // Allocate data array
            const data = try self.allocator.alignedAlloc(TargetType, 32, size);
            errdefer self.allocator.free(data);

            // Create result tensor
            const result = Tensor(TargetType){
                .allocator = self.allocator,
                .shape = shape_copy,
                .data = data,
            };

            if (hasAVX2() and self.data.len >= 8) {
                if (DataType == f16 and TargetType == f32) {
                    convertF16ToF32Simd(self.data, result.data);
                    return result;
                } else if (DataType == f32 and TargetType == f16) {
                    convertF32ToF16Simd(self.data, result.data);
                    return result;
                }
            }

            // Fallback to scalar conversion
            for (self.data, 0..) |val, i| {
                result.data[i] = @floatCast(val);
            }

            return result;
        }

        /// Convert 8 f32 values to f16 using AVX2
        inline fn convertF32ToF16Batch(src: *const [8]f32, dst: *[8]f16) void {
            asm volatile (
                \\vmovups (%[src]), %%ymm0
                \\vcvtps2ph $0, %%ymm0, %%xmm1
                \\vmovups %%xmm1, (%[dst])
                :
                : [dst] "r" (dst),
                  [src] "r" (src),
                : "ymm0", "xmm1", "memory"
            );
        }

        /// Convert 8 f16 values to f32 using AVX2
        inline fn convertF16ToF32Batch(src: *const [8]f16, dst: *[8]f32) void {
            asm volatile (
                \\vmovups (%[src]), %%xmm0
                \\vcvtph2ps %%xmm0, %%ymm1
                \\vmovups %%ymm1, (%[dst])
                :
                : [dst] "r" (dst),
                  [src] "r" (src),
                : "xmm0", "ymm1", "memory"
            );
        }

        fn convertF32ToF16Simd(src: []const f32, dst: []f16) void {
            const size = src.len;
            std.debug.assert(size == dst.len);

            if (size < 8) {
                // we handle small arrays with scalar operations
                for (src, 0..) |val, i| {
                    dst[i] = @floatCast(val);
                }
                return;
            }

            var src_batch: [8]f32 align(32) = undefined;
            var dst_batch: [8]f16 align(32) = undefined;

            const fullAlignedSize = size & ~@as(usize, 31); // Size aligned to 32
            const partialAlignedSize = size & ~@as(usize, 7); // Size aligned to 8

            var i: usize = 0;

            // Process 32 elements at a time
            while (i < fullAlignedSize) : (i += 32) {
                inline for ([_]usize{ 0, 8, 16, 24 }) |offset| {
                    @memcpy(&src_batch, src[i + offset ..][0..8]);
                    convertF32ToF16Batch(&src_batch, &dst_batch);
                    @memcpy(dst[i + offset ..][0..8], &dst_batch);
                }
            }

            // Process remaining aligned chunks of 8
            while (i < partialAlignedSize) : (i += 8) {
                @memcpy(&src_batch, src[i..][0..8]);
                convertF32ToF16Batch(&src_batch, &dst_batch);
                @memcpy(dst[i..][0..8], &dst_batch);
            }

            // Handle remaining elements with scalar operations
            while (i < size) : (i += 1) {
                dst[i] = @floatCast(src[i]);
            }
        }

        fn convertF16ToF32Simd(src: []const f16, dst: []f32) void {
            const size = src.len;
            std.debug.assert(size == dst.len);

            if (size < 8) {
                // Handle small arrays with scalar operations
                for (src, 0..) |val, i| {
                    dst[i] = @floatCast(val);
                }
                return;
            }

            var src_batch: [8]f16 align(32) = undefined;
            var dst_batch: [8]f32 align(32) = undefined;

            const fullAlignedSize = size & ~@as(usize, 31);
            const partialAlignedSize = size & ~@as(usize, 7);

            var i: usize = 0;

            // Process 32 elements at a time
            while (i < fullAlignedSize) : (i += 32) {
                inline for ([_]usize{ 0, 8, 16, 24 }) |offset| {
                    @memcpy(&src_batch, src[i + offset ..][0..8]);
                    convertF16ToF32Batch(&src_batch, &dst_batch);
                    @memcpy(dst[i + offset ..][0..8], &dst_batch);
                }
            }

            // Process remaining aligned chunks of 8
            while (i < partialAlignedSize) : (i += 8) {
                @memcpy(&src_batch, src[i..][0..8]);
                convertF16ToF32Batch(&src_batch, &dst_batch);
                @memcpy(dst[i..][0..8], &dst_batch);
            }

            // Handle remaining elements with scalar operations
            while (i < size) : (i += 1) {
                dst[i] = @floatCast(src[i]);
            }
        }
        // CPU feature detection helper
        inline fn hasAVX2() bool {
            if (!builtin.cpu.arch.isX86()) return false;
            return comptime std.Target.x86.featureSetHas(builtin.cpu.features, .avx2) and
                std.Target.x86.featureSetHas(builtin.cpu.features, .f16c);
        }

        /// Returns a slice of the tensor's data.
        ///
        /// This function provides access to the underlying data of the tensor as a slice.
        ///
        /// Returns:
        /// - `[]DataType`: A slice containing the tensor's data.
        pub fn getSlice(self: Self) []DataType {
            return self.data;
        }

        /// Returns the dimensions of the tensor as a struct with fields `m` and `n`.
        ///
        /// The dimensions are derived from the `shape` array of the tensor.
        ///
        /// Returns:
        /// - A struct with two fields:
        ///   - `m`: The size of the first dimension.
        ///   - `n`: The size of the second dimension.
        pub fn getDimensions(self: Self) struct { m: usize, n: usize } {
            return .{
                .m = self.shape[0],
                .n = self.shape[1],
            };
        }

        /// Returns the total number of elements in the tensor.
        /// This function calculates the total number of elements in the tensor by multiplying
        /// all the dimensions specified in the `shape` array.
        fn calculateSize(shape: []const usize) usize {
            var size: usize = 1;
            for (shape) |dim| {
                size *= dim;
            }
            return size;
        }

        /// Reshapes the tensor to the specified new shape.
        ///
        /// This function changes the shape of the tensor to the given `new_shape`.
        /// It first calculates the total size of the new shape and verifies that it
        /// matches the current size of the tensor. If the sizes do not match, it
        /// returns an `IncompatibleShape` error. If the sizes match, it updates the
        /// tensor's shape to the new shape.
        ///
        /// # Parameters
        /// - `self`: A pointer to the tensor object.
        /// - `new_shape`: A slice of `usize` representing the new shape.
        ///
        /// # Returns
        /// - `!void`: Returns an error if the new shape is incompatible with the current size.
        ///
        /// # Errors
        /// - `IncompatibleShape`: Returned if the total size of the new shape does not match the current size of the tensor.
        ///
        /// # Example
        /// ```zig
        /// var tensor = Tensor.init(allocator, shape, data);
        /// try tensor.reshape(new_shape);
        /// ```
        // Shape operations
        pub fn reshape(self: *Self, new_shape: []const usize) !void {
            var new_size: usize = 1;
            for (new_shape) |dim| {
                new_size *= dim;
            }

            // Verify that the new shape is compatible
            const current_size = @as(usize, @intCast(self.data.len));
            if (new_size != current_size) {
                return error.IncompatibleShape;
            }

            // Update shape
            const new_shape_copy = try self.allocator.alloc(usize, new_shape.len);
            @memcpy(new_shape_copy, new_shape);

            self.allocator.free(self.shape);
            self.shape = new_shape_copy;
        }

        // Add to your Tensor struct:
        pub fn asView(self: *const Self) !TensorView(DataType) {
            return TensorView(DataType).fromTensor(self);
        }

        // Optional: Add a convenience wrapper that uses views internally
        pub fn getChunkFast(self: *const Self, dim: usize, chunk_idx: usize, num_chunks: usize) !Self {
            var view = try self.asView();
            defer view.deinit();

            var chunk_view = try view.getChunkView(dim, chunk_idx, num_chunks);
            defer chunk_view.deinit();

            return chunk_view.toTensor();
        }

        // Inside the Tensor struct definition:

        /// Adds a dimension of size 1 at the specified position.
        /// Negative dimensions are supported - they count from the end.
        /// For example: -1 means insert at the end, -2 means insert at second to last position
        pub fn unsqueeze(self: *Self, dim: isize) !void {
            const positive_dim = if (dim >= 0)
                @as(usize, @intCast(dim))
            else blk: {
                const n_dims: isize = @intCast(self.shape.len);
                const adjusted_dim = n_dims + 1 + dim;
                if (adjusted_dim < 0) return error.InvalidDimension;
                break :blk @as(usize, @intCast(adjusted_dim));
            };

            // Verify dimension is valid
            if (positive_dim > self.shape.len) return error.InvalidDimension;

            // Create new shape with extra dimension
            var new_shape = try self.allocator.alloc(usize, self.shape.len + 1);
            errdefer self.allocator.free(new_shape);

            // Copy shape values with new dimension of size 1
            @memcpy(new_shape[0..positive_dim], self.shape[0..positive_dim]);
            new_shape[positive_dim] = 1;
            if (positive_dim < self.shape.len) {
                @memcpy(new_shape[positive_dim + 1 ..], self.shape[positive_dim..]);
            }

            // Update tensor shape
            self.allocator.free(self.shape);
            self.shape = new_shape;
        }

        pub fn getDimensionSlice(self: Self, dim: usize, index: usize) !Self {
            // Verify dimension is valid
            if (dim >= self.shape.len) {
                return error.InvalidDimension;
            }

            // Verify index is within bounds
            if (index >= self.shape[dim]) {
                return error.IndexOutOfBounds;
            }

            // Special case: 1D tensor becomes scalar (0D tensor)
            if (self.shape.len == 1) {
                var result = try Self.init(self.allocator, &[_]usize{});
                result.data[0] = self.data[index];
                return result;
            }

            // Create new shape by removing the specified dimension
            var new_shape = try self.allocator.alloc(usize, self.shape.len - 1);
            errdefer self.allocator.free(new_shape);

            // Copy shape excluding the specified dimension
            var new_idx: usize = 0;
            for (self.shape, 0..) |size, i| {
                if (i != dim) {
                    new_shape[new_idx] = size;
                    new_idx += 1;
                }
            }

            // Create new tensor with reduced dimensions
            var result = try Self.init(self.allocator, new_shape);
            errdefer result.deinit();
            self.allocator.free(new_shape);

            // Calculate strides for source tensor
            var src_strides = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(src_strides);

            src_strides[self.shape.len - 1] = 1;
            var i: usize = self.shape.len - 1;
            while (i > 0) {
                i -= 1;
                src_strides[i] = src_strides[i + 1] * self.shape[i + 1];
            }

            // For dimensions > 1, calculate destination strides
            if (result.shape.len > 0) {
                var dst_strides = try self.allocator.alloc(usize, result.shape.len);
                defer self.allocator.free(dst_strides);

                dst_strides[result.shape.len - 1] = 1;
                i = result.shape.len - 1;
                while (i > 0) {
                    i -= 1;
                    dst_strides[i] = dst_strides[i + 1] * result.shape[i + 1];
                }

                // Create coordinate arrays
                var src_coords = try self.allocator.alloc(usize, self.shape.len);
                defer self.allocator.free(src_coords);
                @memset(src_coords, 0);

                // Set the fixed dimension to the specified index
                src_coords[dim] = index;

                var dst_coords = try self.allocator.alloc(usize, result.shape.len);
                defer self.allocator.free(dst_coords);
                @memset(dst_coords, 0);

                // Copy data
                const total_elements = calculateSize(result.shape);
                var result_idx: usize = 0;

                while (result_idx < total_elements) : (result_idx += 1) {
                    // Calculate source index
                    var src_idx: usize = 0;
                    var dst_coord_idx: usize = 0;

                    for (src_coords, 0..) |*coord, d| {
                        if (d != dim) {
                            coord.* = dst_coords[dst_coord_idx];
                            dst_coord_idx += 1;
                        }
                        src_idx += coord.* * src_strides[d];
                    }

                    // Copy data
                    result.data[result_idx] = self.data[src_idx];

                    // Update destination coordinates
                    var j = dst_coords.len;
                    while (j > 0) {
                        j -= 1;
                        dst_coords[j] += 1;
                        if (dst_coords[j] < result.shape[j]) break;
                        dst_coords[j] = 0;
                    }
                }
            }

            return result;
        }

        // Extension to the Tensor type
        pub fn getSliceRange(self: Self, slices: []const Slice) !Self {
            if (slices.len > self.shape.len) {
                return error.TooManySlices;
            }

            // Calculate new shape
            var new_shape = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(new_shape);

            // Calculate actual start and end indices for each dimension
            var actual_starts = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(actual_starts);
            var actual_ends = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(actual_ends);

            // Initialize with full ranges for dimensions not specified
            for (self.shape, 0..) |dim_size, i| {
                if (i < slices.len) {
                    const slice = slices[i];
                    actual_starts[i] = slice.start orelse 0;
                    actual_ends[i] = slice.end orelse dim_size;

                    // Bounds checking
                    if (actual_starts[i] > dim_size or actual_ends[i] > dim_size) {
                        return error.SliceOutOfBounds;
                    }
                    if (actual_starts[i] > actual_ends[i]) {
                        return error.InvalidSlice;
                    }
                } else {
                    actual_starts[i] = 0;
                    actual_ends[i] = dim_size;
                }
                new_shape[i] = actual_ends[i] - actual_starts[i];
            }

            // Create new tensor with calculated shape
            var result = try Self.init(self.allocator, new_shape);
            errdefer result.deinit();

            // Calculate strides for the original tensor
            var strides = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(strides);

            strides[self.shape.len - 1] = 1;
            var i = self.shape.len - 1;
            while (i > 0) : (i -= 1) {
                strides[i - 1] = strides[i] * self.shape[i];
            }

            // Copy data with proper indexing
            var coords = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(coords);
            @memset(coords, 0);

            var result_idx: usize = 0;
            while (true) {
                // Calculate source index
                var src_idx: usize = 0;
                for (coords, 0..) |coord, dim| {
                    src_idx += (coord + actual_starts[dim]) * strides[dim];
                }

                // Copy data
                result.data[result_idx] = self.data[src_idx];
                result_idx += 1;

                // Update coordinates
                var dim = self.shape.len;
                while (dim > 0) {
                    dim -= 1;
                    coords[dim] += 1;
                    if (coords[dim] < new_shape[dim]) break;
                    coords[dim] = 0;
                    if (dim == 0) return result;
                }
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

        // Utility functions
        pub fn fill(self: *Self, value: DataType) void {
            for (self.data, 0..) |_, i| {
                self.data[i] = value;
            }
        }

        pub fn copy(self: Self) !Self {
            const new_tensor = try Self.init(self.allocator, self.shape);
            @memcpy(new_tensor.data, self.data);
            return new_tensor;
        }

        fn calculateStrides(shape: []const usize, allocator: Allocator) ![]usize {
            var strides = try allocator.alloc(usize, shape.len);
            errdefer allocator.free(strides);

            if (shape.len == 0) return strides;

            strides[shape.len - 1] = 1;
            var i = shape.len - 1;
            while (i > 0) : (i -= 1) {
                strides[i - 1] = strides[i] * shape[i];
            }
            return strides;
        }
        /// Format a single value with proper precision
        pub fn printF16WithFullPrecision(self: Self) void {
            if (DataType != f16) {
                std.debug.print("Error: This function is only for f16 tensors\n", .{});
                return;
            }

            std.debug.print("[\n", .{});
            for (self.data) |value| {
                const bits = @as(u16, @bitCast(value));
                const sign_bit = bits >> 15;
                const exp = @as(i32, @intCast((bits >> 10) & 0x1F)) - 15;
                const frac = @as(f64, @floatFromInt(bits & 0x3FF)) / 1024.0;

                var sign_mult: f64 = 1.0;
                if (sign_bit == 1) {
                    sign_mult = -1.0;
                }

                const value_f64 = sign_mult * (1.0 + frac) * std.math.pow(f64, 2.0, @floatFromInt(exp));

                // Print with maximum precision
                std.debug.print("{d:.16}", .{value_f64});
                std.debug.print("\n", .{});
            }
            std.debug.print("]\n", .{});
        }
        pub fn debugF16(value: f16) void {
            const bits = @as(u16, @bitCast(value));
            const sign = (bits >> 15) & 1;
            const exp = (bits >> 10) & 0x1F;
            const frac = bits & 0x3FF;

            std.debug.print(
                \\f16 value analysis:
                \\  Raw bits: 0x{X:0>4}
                \\  Sign bit: {d}
                \\  Exponent: {d} (raw: {d})
                \\  Fraction: 0x{X:0>3}
                \\  Value: {d}
                \\
            , .{
                bits,
                sign,
                @as(i32, @intCast(exp)) - 15, // Biased exponent
                exp,
                frac,
                @as(f64, @floatCast(value)),
            });
        }
        /// Format a single value with proper precision
        fn formatValue(value: DataType, writer: anytype, options: PrintOptions) !void {
            switch (@typeInfo(DataType)) {
                .Float => {
                    if (std.math.isNan(value)) {
                        try writer.writeAll("nan");
                    } else if (std.math.isPositiveInf(value)) {
                        try writer.writeAll("inf");
                    } else if (std.math.isNegativeInf(value)) {
                        try writer.writeAll("-inf");
                    } else {
                        // For f16, we need special handling to maintain precision
                        if (DataType == f16) {
                            // Convert to f64 first to maintain precision during printing
                            var value_f64: f64 = @floatCast(@as(f64, value));

                            // For very small values near zero, use more precise conversion
                            if (value_f64 != 0 and @abs(value_f64) < 1e-4) {
                                const bits = @as(u16, @bitCast(value));
                                const sign_bit = bits >> 15;
                                const exp = @as(i32, @intCast((bits >> 10) & 0x1F)) - 15;
                                const frac = @as(f64, @floatFromInt(bits & 0x3FF)) / 1024.0;

                                var sign_mult: f64 = 1.0;
                                if (sign_bit == 1) {
                                    sign_mult = -1.0;
                                }

                                value_f64 = sign_mult * (1.0 + frac) * std.math.pow(f64, 2.0, @floatFromInt(exp));
                            }

                            // Print with scientific notation to show full precision
                            const precision = options.precision;
                            // Print with extended precision for f16
                            switch (precision) {
                                0...8 => try writer.print("{d:.8}", .{value_f64}),
                                9 => try writer.print("{d:.9}", .{value_f64}),
                                10 => try writer.print("{d:.10}", .{value_f64}),
                                11 => try writer.print("{d:.11}", .{value_f64}),
                                12 => try writer.print("{d:.12}", .{value_f64}),
                                13 => try writer.print("{d:.13}", .{value_f64}),
                                14 => try writer.print("{d:.14}", .{value_f64}),
                                15 => try writer.print("{d:.15}", .{value_f64}),
                                else => try writer.print("{d:.16}", .{value_f64}),
                            }
                        } else {
                            // For other float types, use standard formatting
                            const precision = options.precision;
                            switch (precision) {
                                0 => try writer.print("{d:.0}", .{value}),
                                1 => try writer.print("{d:.1}", .{value}),
                                2 => try writer.print("{d:.2}", .{value}),
                                3 => try writer.print("{d:.3}", .{value}),
                                4 => try writer.print("{d:.4}", .{value}),
                                5 => try writer.print("{d:.5}", .{value}),
                                6 => try writer.print("{d:.6}", .{value}),
                                7 => try writer.print("{d:.7}", .{value}),
                                8 => try writer.print("{d:.8}", .{value}),
                                else => try writer.print("{d:.4}", .{value}),
                            }
                        }
                    }
                },
                .Int => {
                    try writer.print("{d}", .{value});
                },
                else => {
                    try writer.print("{any}", .{value});
                },
            }
        }
        // Calculate strides for each dimension - using original implementation

        /// Format tensor recursively
        fn formatRecursive(
            self: Self,
            writer: anytype,
            options: PrintOptions,
            current_dim: usize,
            offset: usize,
            strides: []const usize,
        ) !void {
            if (current_dim == self.shape.len) {
                try formatValue(self.data[offset], writer, options);
                return;
            }

            try writer.writeAll("[");
            const dim_size = self.shape[current_dim];

            for (0..dim_size) |i| {
                const new_offset = offset + i * strides[current_dim];
                try formatRecursive(self, writer, options, current_dim + 1, new_offset, strides);
                if (i < dim_size - 1) {
                    try writer.writeAll(", ");
                }
            }

            try writer.writeAll("]");
        }

        /// Internal formatting function
        fn formatTensor(self: Self, writer: anytype) !void {
            const options = PrintOptions{};

            try writer.writeAll("tensor(");

            // Handle empty tensor
            if (self.data.len == 0) {
                try writer.writeAll("[]");
            } else {
                const strides = try calculateStrides(self.shape, self.allocator);
                defer self.allocator.free(strides);

                try formatRecursive(self, writer, options, 0, 0, strides);
            }

            try writer.print(", dtype={s})", .{@typeName(DataType)});
        }

        /// Implement std.fmt.format interface
        pub fn format(
            self: Self,
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            _ = fmt;
            _ = options;
            try self.formatTensor(writer);
        }

        /// Print the tensor to stdout
        pub fn print(self: Self) void {
            std.debug.print("{}", .{self});
        }
        /// Print a 2D tensor to stdout with truncated rows if necessary
        pub fn print2D(self: Self) void {
            if (self.shape.len != 2) {
                std.debug.print("Error: Not a 2D tensor\n", .{});
                return;
            }

            const rows = self.shape[0];
            const cols = self.shape[1];
            const options = PrintOptions{};

            // Print shape information
            std.debug.print("Tensor {d}x{d}:\n", .{ rows, cols });

            for (0..rows) |i| {
                std.debug.print("[ ", .{});

                // Calculate how many items to show at start and end
                const items_per_side = max_items_per_row / 2;
                const show_ellipsis = cols > max_items_per_row;

                // Print first few items
                const start_items = @min(items_per_side, cols);
                for (0..start_items) |j| {
                    const idx = i * cols + j;
                    formatValue(self.data[idx], std.io.getStdOut().writer(), options) catch {};
                    std.debug.print(" ", .{});
                }

                // Print ellipsis if needed
                if (show_ellipsis) {
                    std.debug.print("... ", .{});

                    // Print last few items
                    const remaining_items = @min(items_per_side, cols - start_items);
                    const start_idx = cols - remaining_items;
                    for (start_idx..cols) |j| {
                        const idx = i * cols + j;
                        formatValue(self.data[idx], std.io.getStdOut().writer(), options) catch {};
                        std.debug.print(" ", .{});
                    }
                }

                std.debug.print("]\n", .{});
            }
        }

        /// Print a 3D tensor to stdout with truncated rows and slices if necessary
        pub fn print3D(self: Self) void {
            if (self.shape.len != 3) {
                std.debug.print("Error: Not a 3D tensor\n", .{});
                return;
            }

            const depth = self.shape[0];
            const rows = self.shape[1];
            const cols = self.shape[2];
            const options = PrintOptions{};

            // Print shape information
            std.debug.print("Tensor {d}x{d}x{d}:\n", .{ depth, rows, cols });

            // Calculate how many slices to show
            const max_slices = max_rows;
            const show_slice_ellipsis = depth > max_slices;
            const slices_to_show = @min(max_slices, depth);

            for (0..slices_to_show) |d| {
                std.debug.print("\nSlice [{d}]:\n", .{d});

                for (0..rows) |i| {
                    std.debug.print("[ ", .{});

                    // Calculate how many items to show at start and end
                    const items_per_side = max_items_per_row / 2;
                    const show_ellipsis = cols > max_items_per_row;

                    // Print first few items
                    const start_items = @min(items_per_side, cols);
                    for (0..start_items) |j| {
                        const idx = (d * rows * cols) + (i * cols) + j;
                        formatValue(self.data[idx], std.io.getStdOut().writer(), options) catch {};
                        std.debug.print(" ", .{});
                    }

                    // Print ellipsis if needed
                    if (show_ellipsis) {
                        std.debug.print("... ", .{});

                        // Print last few items
                        const remaining_items = @min(items_per_side, cols - start_items);
                        const start_idx = cols - remaining_items;
                        for (start_idx..cols) |j| {
                            const idx = (d * rows * cols) + (i * cols) + j;
                            formatValue(self.data[idx], std.io.getStdOut().writer(), options) catch {};
                            std.debug.print(" ", .{});
                        }
                    }

                    std.debug.print("]\n", .{});
                }
            }

            // Print ellipsis for slices if needed
            if (show_slice_ellipsis) {
                std.debug.print("\n...\n\n", .{});

                // Print last slice
                const last_slice = depth - 1;
                std.debug.print("Slice [{d}]:\n", .{last_slice});

                for (0..rows) |i| {
                    std.debug.print("[ ", .{});

                    // Calculate how many items to show at start and end
                    const items_per_side = max_items_per_row / 2;
                    const show_ellipsis = cols > max_items_per_row;

                    // Print first few items
                    const start_items = @min(items_per_side, cols);
                    for (0..start_items) |j| {
                        const idx = (last_slice * rows * cols) + (i * cols) + j;
                        formatValue(self.data[idx], std.io.getStdOut().writer(), options) catch {};
                        std.debug.print(" ", .{});
                    }

                    // Print ellipsis if needed
                    if (show_ellipsis) {
                        std.debug.print("... ", .{});

                        // Print last few items
                        const remaining_items = @min(items_per_side, cols - start_items);
                        const start_idx = cols - remaining_items;
                        for (start_idx..cols) |j| {
                            const idx = (last_slice * rows * cols) + (i * cols) + j;
                            formatValue(self.data[idx], std.io.getStdOut().writer(), options) catch {};
                            std.debug.print(" ", .{});
                        }
                    }

                    std.debug.print("]\n", .{});
                }
            }
        }

        /// Print a 4D tensor to stdout with truncated dimensions if necessary
        pub fn print4D(self: Self) void {
            if (self.shape.len != 4) {
                std.debug.print("Error: Not a 4D tensor\n", .{});
                return;
            }

            const time = self.shape[0];
            const depth = self.shape[1];
            const rows = self.shape[2];
            const cols = self.shape[3];
            const options = PrintOptions{};

            // Print shape information
            std.debug.print("Tensor {d}x{d}x{d}x{d}:\n", .{ time, depth, rows, cols });

            // Calculate how many time slices to show
            const max_time_slices = max_rows;
            const show_time_ellipsis = time > max_time_slices;
            const time_slices_to_show = @min(max_time_slices, time);

            // Calculate how many depth slices to show
            const max_depth_slices = max_rows;
            const show_depth_ellipsis = depth > max_depth_slices;
            const depth_slices_to_show = @min(max_depth_slices, depth);

            // For each time slice
            for (0..time_slices_to_show) |t| {
                std.debug.print("\nTime [{d}]:\n", .{t});

                // For each depth slice
                for (0..depth_slices_to_show) |d| {
                    std.debug.print("\nDepth [{d}]:\n", .{d});

                    // For each row
                    for (0..rows) |i| {
                        std.debug.print("[ ", .{});

                        // Calculate how many items to show at start and end
                        const items_per_side = max_items_per_row / 2;
                        const show_ellipsis = cols > max_items_per_row;

                        // Print first few items
                        const start_items = @min(items_per_side, cols);
                        for (0..start_items) |j| {
                            const idx = (t * depth * rows * cols) + (d * rows * cols) + (i * cols) + j;
                            formatValue(self.data[idx], std.io.getStdOut().writer(), options) catch {};
                            std.debug.print(" ", .{});
                        }

                        // Print ellipsis if needed
                        if (show_ellipsis) {
                            std.debug.print("... ", .{});

                            // Print last few items
                            const remaining_items = @min(items_per_side, cols - start_items);
                            const start_idx = cols - remaining_items;
                            for (start_idx..cols) |j| {
                                const idx = (t * depth * rows * cols) + (d * rows * cols) + (i * cols) + j;
                                formatValue(self.data[idx], std.io.getStdOut().writer(), options) catch {};
                                std.debug.print(" ", .{});
                            }
                        }

                        std.debug.print("]\n", .{});
                    }
                }

                // Print ellipsis for depth slices if needed
                if (show_depth_ellipsis) {
                    std.debug.print("\n...\n\n", .{});

                    // Print last depth slice
                    const last_depth = depth - 1;
                    std.debug.print("Depth [{d}]:\n", .{last_depth});

                    for (0..rows) |i| {
                        std.debug.print("[ ", .{});

                        // Calculate how many items to show at start and end
                        const items_per_side = max_items_per_row / 2;
                        const show_ellipsis = cols > max_items_per_row;

                        // Print first few items
                        const start_items = @min(items_per_side, cols);
                        for (0..start_items) |j| {
                            const idx = (t * depth * rows * cols) + (last_depth * rows * cols) + (i * cols) + j;
                            formatValue(self.data[idx], std.io.getStdOut().writer(), options) catch {};
                            std.debug.print(" ", .{});
                        }

                        // Print ellipsis if needed
                        if (show_ellipsis) {
                            std.debug.print("... ", .{});

                            // Print last few items
                            const remaining_items = @min(items_per_side, cols - start_items);
                            const start_idx = cols - remaining_items;
                            for (start_idx..cols) |j| {
                                const idx = (t * depth * rows * cols) + (last_depth * rows * cols) + (i * cols) + j;
                                formatValue(self.data[idx], std.io.getStdOut().writer(), options) catch {};
                                std.debug.print(" ", .{});
                            }
                        }

                        std.debug.print("]\n", .{});
                    }
                }
            }

            // Print ellipsis for time slices if needed
            if (show_time_ellipsis) {
                std.debug.print("\n...\n\n", .{});

                // Print last time slice
                const last_time = time - 1;
                std.debug.print("Time [{d}]:\n", .{last_time});

                // Print depth slices for the last time slice
                for (0..depth_slices_to_show) |d| {
                    std.debug.print("\nDepth [{d}]:\n", .{d});

                    for (0..rows) |i| {
                        std.debug.print("[ ", .{});

                        // Calculate how many items to show at start and end
                        const items_per_side = max_items_per_row / 2;
                        const show_ellipsis = cols > max_items_per_row;

                        // Print first few items
                        const start_items = @min(items_per_side, cols);
                        for (0..start_items) |j| {
                            const idx = (last_time * depth * rows * cols) + (d * rows * cols) + (i * cols) + j;
                            formatValue(self.data[idx], std.io.getStdOut().writer(), options) catch {};
                            std.debug.print(" ", .{});
                        }

                        // Print ellipsis if needed
                        if (show_ellipsis) {
                            std.debug.print("... ", .{});

                            // Print last few items
                            const remaining_items = @min(items_per_side, cols - start_items);
                            const start_idx = cols - remaining_items;
                            for (start_idx..cols) |j| {
                                const idx = (last_time * depth * rows * cols) + (d * rows * cols) + (i * cols) + j;
                                formatValue(self.data[idx], std.io.getStdOut().writer(), options) catch {};
                                std.debug.print(" ", .{});
                            }
                        }

                        std.debug.print("]\n", .{});
                    }
                }
            }
        }

        /// Convert tensor to string
        pub fn toString(self: Self, allocator: std.mem.Allocator) ![]const u8 {
            var list = std.ArrayList(u8).init(allocator);
            errdefer list.deinit();

            try list.writer().print("{}", .{self});
            return list.toOwnedSlice();
        }

        /// StabilityInfo struct to hold detailed information about tensor stability
        pub const StabilityInfo = struct {
            has_nan: bool = false,
            has_pos_inf: bool = false,
            has_neg_inf: bool = false,
            first_nan_index: ?usize = null,
            first_pos_inf_index: ?usize = null,
            first_neg_inf_index: ?usize = null,
            nan_count: usize = 0,
            pos_inf_count: usize = 0,
            neg_inf_count: usize = 0,
        };
    };
}

/// Helper struct to hold formatting options
const PrintOptions = struct {
    precision: u8 = 8,
};
pub const StabilityError = error{
    HasNaN,
    HasPositiveInfinity,
    HasNegativeInfinity,
    HasInfinity,
};

// Slice type to represent Python-style slice operations
pub const Slice = struct {
    start: ?usize,
    end: ?usize,

    pub fn full() Slice {
        return .{ .start = null, .end = null };
    }

    pub fn from(start: ?usize, end: ?usize) Slice {
        return .{ .start = start, .end = end };
    }
};

// Error set for casting operations
pub const CastError = error{
    LossyConversion, // When precision would be lost (e.g., float to int with decimals)
    OutOfRange, // When value is outside the target type's range
};

/// A view into a tensor that doesn't own its data
pub fn TensorView(comptime DataType: type) type {
    return struct {
        const Self = @This();

        data: []align(32) DataType, // Reference to original aligned data
        shape: []usize, // View dimensions
        strides: []usize, // Stride for each dimension
        offset: usize, // Offset into original data
        allocator: Allocator, // For managing shape/strides arrays

        /// Create a view from an existing tensor
        pub fn fromTensor(tensor: *const Tensor(DataType)) !Self {
            const strides = try calculateStrides(tensor.shape, tensor.allocator);
            errdefer tensor.allocator.free(strides);

            const shape = try tensor.allocator.alloc(usize, tensor.shape.len);
            errdefer tensor.allocator.free(shape);
            @memcpy(shape, tensor.shape);

            return Self{
                .data = tensor.data,
                .shape = shape,
                .strides = strides,
                .offset = 0,
                .allocator = tensor.allocator,
            };
        }

        /// Get a chunk view without copying data
        pub fn getChunkView(self: *const Self, dim: usize, chunk_idx: usize, num_chunks: usize) !Self {
            if (dim >= self.shape.len) return error.InvalidDimension;

            const dim_size = self.shape[dim];
            if (num_chunks == 0 or dim_size < num_chunks) return error.InvalidNumChunks;
            if (chunk_idx >= num_chunks) return error.InvalidChunkIndex;

            const chunk_size = dim_size / num_chunks;
            if (chunk_size * num_chunks != dim_size) return error.UnevenChunkSize;

            const start_idx = chunk_idx * chunk_size;

            // Create new shape array
            var new_shape = try self.allocator.alloc(usize, self.shape.len);
            errdefer self.allocator.free(new_shape);
            @memcpy(new_shape, self.shape);
            new_shape[dim] = chunk_size;

            // Calculate new strides (reuse existing ones)
            const new_strides = try self.allocator.alloc(usize, self.strides.len);
            errdefer self.allocator.free(new_strides);
            @memcpy(new_strides, self.strides);

            // Calculate new offset
            const new_offset = self.offset + start_idx * self.strides[dim];

            return Self{
                .data = self.data,
                .shape = new_shape,
                .strides = new_strides,
                .offset = new_offset,
                .allocator = self.allocator,
            };
        }

        /// Convert view back to owned tensor (copies data)
        pub fn toTensor(self: Self) !Tensor(DataType) {
            var result = try Tensor(DataType).init(self.allocator, self.shape);
            errdefer result.deinit();

            // Copy data using view's layout
            var coords = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(coords);
            @memset(coords, 0);

            const total_elements = calculateSize(self.shape);
            var i: usize = 0;
            while (i < total_elements) : (i += 1) {
                const src_idx = self.getIndex(coords);
                result.data[i] = self.data[src_idx];

                // Update coordinates
                var dim = self.shape.len;
                while (dim > 0) {
                    dim -= 1;
                    coords[dim] += 1;
                    if (coords[dim] < self.shape[dim]) break;
                    coords[dim] = 0;
                }
            }

            return result;
        }

        /// Calculate actual data index from coordinates
        pub inline fn getIndex(self: Self, coords: []const usize) usize {
            var index = self.offset;
            for (coords, 0..) |coord, dim| {
                index += coord * self.strides[dim];
            }
            return index;
        }

        /// Free allocated memory
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.shape);
            self.allocator.free(self.strides);
        }

        /// Calculate strides for given shape
        fn calculateStrides(shape: []const usize, allocator: Allocator) ![]usize {
            var strides = try allocator.alloc(usize, shape.len);
            errdefer allocator.free(strides);

            if (shape.len == 0) return strides;

            strides[shape.len - 1] = 1;
            var i = shape.len - 1;
            while (i > 0) : (i -= 1) {
                strides[i - 1] = strides[i] * shape[i];
            }
            return strides;
        }

        /// Calculate total size from shape
        fn calculateSize(shape: []const usize) usize {
            var size: usize = 1;
            for (shape) |dim| {
                size *= dim;
            }
            return size;
        }

        pub fn transposeAxes(self: *Self, axis1: usize, axis2: usize) !void {
            if (axis1 >= self.shape.len or axis2 >= self.shape.len) {
                return error.InvalidDimension;
            }

            // Swap shapes
            const temp_shape = self.shape[axis1];
            self.shape[axis1] = self.shape[axis2];
            self.shape[axis2] = temp_shape;

            // Swap strides
            const temp_stride = self.strides[axis1];
            self.strides[axis1] = self.strides[axis2];
            self.strides[axis2] = temp_stride;
        }

        /// Fast indexing for transposed/reshaped views
        pub fn getDataIndex(self: Self, coords: []const usize) usize {
            var index = self.offset;
            for (coords, 0..) |coord, dim| {
                index += coord * self.strides[dim];
            }
            return index;
        }

        pub fn reshape(self: *Self, new_shape: []const usize) !void {
            // Calculate new size
            var new_size: usize = 1;
            for (new_shape) |dim| {
                new_size *= dim;
            }

            // Verify compatible size
            const current_size = calculateSize(self.shape);
            if (new_size != current_size) {
                return error.IncompatibleShape;
            }

            // Update shape array
            if (new_shape.len != self.shape.len) {
                const new_shape_copy = try self.allocator.alloc(usize, new_shape.len);
                errdefer self.allocator.free(new_shape_copy);
                @memcpy(new_shape_copy, new_shape);
                self.allocator.free(self.shape);
                self.shape = new_shape_copy;
            } else {
                @memcpy(self.shape, new_shape);
            }

            // Update strides for new shape
            const new_strides = try calculateStrides(self.shape, self.allocator);
            self.allocator.free(self.strides);
            self.strides = new_strides;
        }
        /// Add to TensorView struct
        pub fn toContiguousTensor(self: Self) !Tensor(DataType) {
            // Create new tensor with current shape
            var result = try Tensor(DataType).init(self.allocator, self.shape);
            errdefer result.deinit();

            // Copy data using view's layout
            var coords = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(coords);
            @memset(coords, 0);

            const total_elements = calculateSize(self.shape);
            var i: usize = 0;

            switch (self.shape.len) {
                // Specialized for common tensor dimensions
                1 => while (i < total_elements) : (i += 1) {
                    const src_idx = self.offset + coords[0] * self.strides[0];
                    result.data[i] = self.data[src_idx];
                    coords[0] += 1;
                },
                2 => while (i < total_elements) : (i += 1) {
                    const src_idx = self.offset +
                        coords[0] * self.strides[0] +
                        coords[1] * self.strides[1];
                    result.data[i] = self.data[src_idx];

                    coords[1] += 1;
                    if (coords[1] >= self.shape[1]) {
                        coords[1] = 0;
                        coords[0] += 1;
                    }
                },
                3 => while (i < total_elements) : (i += 1) {
                    const src_idx = self.offset +
                        coords[0] * self.strides[0] +
                        coords[1] * self.strides[1] +
                        coords[2] * self.strides[2];
                    result.data[i] = self.data[src_idx];

                    coords[2] += 1;
                    if (coords[2] >= self.shape[2]) {
                        coords[2] = 0;
                        coords[1] += 1;
                        if (coords[1] >= self.shape[1]) {
                            coords[1] = 0;
                            coords[0] += 1;
                        }
                    }
                },
                4 => while (i < total_elements) : (i += 1) {
                    const src_idx = self.offset +
                        coords[0] * self.strides[0] +
                        coords[1] * self.strides[1] +
                        coords[2] * self.strides[2] +
                        coords[3] * self.strides[3];
                    result.data[i] = self.data[src_idx];

                    coords[3] += 1;
                    if (coords[3] >= self.shape[3]) {
                        coords[3] = 0;
                        coords[2] += 1;
                        if (coords[2] >= self.shape[2]) {
                            coords[2] = 0;
                            coords[1] += 1;
                            if (coords[1] >= self.shape[1]) {
                                coords[1] = 0;
                                coords[0] += 1;
                            }
                        }
                    }
                },
                else => {
                    // Fallback for higher dimensions
                    while (i < total_elements) : (i += 1) {
                        const src_idx = self.getDataIndex(coords);
                        result.data[i] = self.data[src_idx];

                        var dim = self.shape.len;
                        while (dim > 0) {
                            dim -= 1;
                            coords[dim] += 1;
                            if (coords[dim] < self.shape[dim]) break;
                            coords[dim] = 0;
                        }
                    }
                },
            }

            return result;
        }
    };
}
