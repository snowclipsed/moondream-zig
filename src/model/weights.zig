const std = @import("std");
const Allocator = std.mem.Allocator;
const Tensor = @import("../core/tensor.zig").Tensor;
const Config = @import("config.zig").Config;

/// Quantization parameters for INT8 tensors
pub const QuantParams = struct {
    scale: Tensor(f16), // Scale factors for dequantization
    zero_point: Tensor(i8), // Zero point offsets
};

pub const Weights = struct {
    const Self = @This();

    // Magic constants for format validation
    pub const WEIGHTS_MAGIC = 0x4D4F4F4E; // "MOON" in ASCII
    pub const WEIGHTS_VERSION = 3; // Version 3 for quantized tensor format

    allocator: Allocator,

    // Text model weights
    word_token_embedding: Tensor(f16),
    t_ln_w: Tensor(f16),
    t_ln_b: Tensor(f16),
    t_Wqkv_w: Tensor(i8),
    t_Wqkv_b: Tensor(f16),
    t_out_proj_w: Tensor(i8),
    t_out_proj_bias: Tensor(f16),
    t_fc1_w: Tensor(i8),
    t_fc1_b: Tensor(f16),
    t_fc2_w: Tensor(i8),
    t_fc2_b: Tensor(f16),
    t_linear_w: Tensor(i8),
    t_linear_b: Tensor(f16),
    t_ln_out_w: Tensor(f16),
    t_ln_out_b: Tensor(f16),

    // Vision model weights
    v_patch_embedding_linear_w: Tensor(f16),
    v_patch_embedding_linear_b: Tensor(f16),
    v_pos_embedding: Tensor(f16),
    v_Wqkv_w: Tensor(i8),
    v_Wqkv_b: Tensor(f16),
    v_out_proj_w: Tensor(i8),
    v_out_proj_b: Tensor(f16),
    v_fc1_w: Tensor(i8),
    v_fc1_b: Tensor(f16),
    v_fc2_w: Tensor(i8),
    v_fc2_b: Tensor(f16),
    v_norm1_w: Tensor(f16),
    v_norm1_b: Tensor(f16),
    v_norm2_w: Tensor(f16),
    v_norm2_b: Tensor(f16),
    v_norm_out_w: Tensor(f16),
    v_norm_out_b: Tensor(f16),
    v_proj_fc1_w: Tensor(i8),
    v_proj_fc1_b: Tensor(f16),
    v_proj_fc2_w: Tensor(i8),
    v_proj_fc2_b: Tensor(f16),

    // Quantization parameters for INT8 weights
    quant_params: std.StringHashMap(QuantParams),

    const WeightHeader = struct {
        name_length: u32,
        shape_length: u32,
        data_length: u32,
    };

    /// Check if a tensor name is for quantized weights, scale, or zero point
    fn isQuantizedParam(name: []const u8) struct { is_quant: bool, param_type: enum { Weight, Scale, Zero }, base_name: []const u8 } {
        if (std.mem.endsWith(u8, name, "_q")) {
            return .{ .is_quant = true, .param_type = .Weight, .base_name = name[0 .. name.len - 2] };
        } else if (std.mem.endsWith(u8, name, "_s")) {
            return .{ .is_quant = true, .param_type = .Scale, .base_name = name[0 .. name.len - 2] };
        } else if (std.mem.endsWith(u8, name, "_z")) {
            return .{ .is_quant = true, .param_type = .Zero, .base_name = name[0 .. name.len - 2] };
        } else {
            return .{ .is_quant = false, .param_type = .Weight, .base_name = name };
        }
    }

    /// Dequantize an INT8 tensor back to FP16 using scale and zero point
    pub fn dequantize(self: Self, base_name: []const u8, tensor_i8: Tensor(i8)) !Tensor(f16) {
        // Get quantization parameters for this tensor
        const params = self.quant_params.get(base_name) orelse return error.QuantParamsNotFound;

        // Create a new tensor for F16 data
        var tensor_f16 = try Tensor(f16).init(self.allocator, tensor_i8.shape);

        // Loop through each channel/row
        const row_size = tensor_i8.data.len / params.scale.data.len;
        for (0..params.scale.data.len) |row_idx| {
            const scale = params.scale.data[row_idx];
            const zero_point = params.zero_point.data[row_idx];

            // Calculate the start and end indices for this row
            const start = row_idx * row_size;
            const end = start + row_size;

            // Dequantize all values in this row
            for (start..end) |i| {
                // Apply dequantization: (value - zero_point) * scale
                tensor_f16.data[i] = @as(f16, @floatFromInt(tensor_i8.data[i] - zero_point)) * scale;
            }
        }

        return tensor_f16;
    }

    pub fn init(config: Config, filename: []const u8, allocator: Allocator) !Self {
        const file = try std.fs.cwd().openFile(filename, .{});
        defer file.close();

        var current_pos: u64 = 0;

        // Magic number and version check
        var magic_buf: [@sizeOf(u32)]u8 = undefined;
        const magic_bytes = try file.read(&magic_buf);
        if (magic_bytes != @sizeOf(u32)) return error.InvalidRead;
        const magic = std.mem.readInt(u32, &magic_buf, .little);
        if (magic != WEIGHTS_MAGIC) {
            return error.InvalidMagicNumber;
        }
        current_pos += @sizeOf(u32);

        var version_buf: [@sizeOf(u32)]u8 = undefined;
        const version_bytes = try file.read(&version_buf);
        if (version_bytes != @sizeOf(u32)) return error.InvalidRead;
        const version = std.mem.readInt(u32, &version_buf, .little);
        if (version != WEIGHTS_VERSION) {
            return error.InvalidVersion;
        }
        current_pos += @sizeOf(u32);

        const expected_shapes = try calculateShapes(allocator, config);
        defer {
            inline for (@typeInfo(@TypeOf(expected_shapes)).Struct.fields) |field| {
                allocator.free(@field(expected_shapes, field.name));
            }
        }

        // Initialize weights struct with hashmap for quantization parameters
        var self = Self{
            .allocator = allocator,
            .word_token_embedding = undefined,
            .t_ln_w = undefined,
            .t_ln_b = undefined,
            .t_Wqkv_w = undefined,
            .t_Wqkv_b = undefined,
            .t_out_proj_w = undefined,
            .t_out_proj_bias = undefined,
            .t_fc1_w = undefined,
            .t_fc1_b = undefined,
            .t_fc2_w = undefined,
            .t_fc2_b = undefined,
            .t_linear_w = undefined,
            .t_linear_b = undefined,
            .t_ln_out_w = undefined,
            .t_ln_out_b = undefined,
            .v_patch_embedding_linear_w = undefined,
            .v_patch_embedding_linear_b = undefined,
            .v_pos_embedding = undefined,
            .v_Wqkv_w = undefined,
            .v_Wqkv_b = undefined,
            .v_out_proj_w = undefined,
            .v_out_proj_b = undefined,
            .v_fc1_w = undefined,
            .v_fc1_b = undefined,
            .v_fc2_w = undefined,
            .v_fc2_b = undefined,
            .v_norm1_w = undefined,
            .v_norm1_b = undefined,
            .v_norm2_w = undefined,
            .v_norm2_b = undefined,
            .v_norm_out_w = undefined,
            .v_norm_out_b = undefined,
            .v_proj_fc1_w = undefined,
            .v_proj_fc1_b = undefined,
            .v_proj_fc2_w = undefined,
            .v_proj_fc2_b = undefined,
            .quant_params = std.StringHashMap(QuantParams).init(allocator),
        };

        // Temporary storage for quantization parameters
        var temp_quant_params = std.StringHashMap(struct {
            scale: ?Tensor(f16) = null,
            zero: ?Tensor(i8) = null,
        }).init(allocator);
        defer {
            var it = temp_quant_params.iterator();
            while (it.next()) |entry| {
                if (entry.value_ptr.scale) |scale| {
                    scale.deinit();
                }
                if (entry.value_ptr.zero) |zero| {
                    zero.deinit();
                }
                allocator.free(entry.key_ptr.*);
            }
            temp_quant_params.deinit();
        }

        while (true) {
            var header_buf: [@sizeOf(WeightHeader)]u8 = undefined;
            const header_bytes = try file.read(&header_buf);
            if (header_bytes == 0) break;
            if (header_bytes != @sizeOf(WeightHeader)) return error.InvalidHeader;

            const name_length = std.mem.readInt(u32, header_buf[0..4], .little);
            const shape_length = std.mem.readInt(u32, header_buf[4..8], .little);

            // Allow names up to 256 bytes - this is plenty for our tensor names
            if (name_length > 256) {
                std.debug.print("Invalid name length encountered: {d}\n", .{name_length});
                return error.InvalidNameLength;
            }
            if (shape_length > 4) return error.InvalidShapeLength;

            current_pos += @sizeOf(WeightHeader);

            var name_buf = try allocator.alloc(u8, name_length);
            defer allocator.free(name_buf);

            const name_bytes = try file.read(name_buf);
            if (name_bytes != name_length) return error.InvalidRead;
            current_pos += name_length;

            const name = name_buf[0..name_length];

            var shape = try allocator.alloc(usize, shape_length);
            defer allocator.free(shape);

            for (0..shape_length) |i| {
                var dim_bytes: [8]u8 = undefined;
                const bytes_read = try file.read(&dim_bytes);
                if (bytes_read != 8) return error.InvalidRead;

                const dim = std.mem.readInt(u64, &dim_bytes, .little);
                shape[i] = @intCast(dim);
                current_pos += 8;
            }

            // Check if this is a quantized parameter
            const quant_info = isQuantizedParam(name);

            // Skip shape check for quantization parameters
            if (!quant_info.is_quant) {
                // Only check expected shape for non-quantized tensors
                const expected_shape = try getExpectedShape(name, expected_shapes);
                if (!std.mem.eql(usize, shape, expected_shape)) {
                    printShapeMismatchError(name, shape, expected_shape);
                    return error.ShapeMismatch;
                }
            }

            // Determine the tensor type based on the name
            if (quant_info.is_quant) {
                // Handle quantized tensor components
                switch (quant_info.param_type) {
                    .Weight => {
                        // Create and read the quantized int8 tensor
                        const tensor = try Tensor(i8).init(allocator, shape);
                        const data_size_bytes = tensor.data.len * @sizeOf(i8);
                        const data_bytes = try file.read(std.mem.sliceAsBytes(tensor.data));
                        if (data_bytes != data_size_bytes) {
                            return error.InvalidRead;
                        }
                        current_pos += data_size_bytes;

                        // Store in the appropriate field
                        try self.storeTensor(quant_info.base_name, tensor);
                    },
                    .Scale => {
                        // Create and read the scale tensor (f16)
                        const tensor = try Tensor(f16).init(allocator, shape);
                        const data_size_bytes = tensor.data.len * @sizeOf(f16);
                        const data_bytes = try file.read(std.mem.sliceAsBytes(tensor.data));
                        if (data_bytes != data_size_bytes) {
                            return error.InvalidRead;
                        }
                        current_pos += data_size_bytes;

                        // Store in temp storage
                        const key = try allocator.dupe(u8, quant_info.base_name);
                        var entry = try temp_quant_params.getOrPut(key);
                        if (!entry.found_existing) {
                            entry.value_ptr.* = .{};
                        } else {
                            // Free the key if entry already exists
                            allocator.free(key);
                        }
                        entry.value_ptr.scale = tensor;
                    },
                    .Zero => {
                        // Create and read the zero point tensor (i8)
                        const tensor = try Tensor(i8).init(allocator, shape);
                        const data_size_bytes = tensor.data.len * @sizeOf(i8);
                        const data_bytes = try file.read(std.mem.sliceAsBytes(tensor.data));
                        if (data_bytes != data_size_bytes) {
                            return error.InvalidRead;
                        }
                        current_pos += data_size_bytes;

                        // Store in temp storage
                        const key = try allocator.dupe(u8, quant_info.base_name);
                        var entry = try temp_quant_params.getOrPut(key);
                        if (!entry.found_existing) {
                            entry.value_ptr.* = .{};
                        } else {
                            // Free the key if entry already exists
                            allocator.free(key);
                        }
                        entry.value_ptr.zero = tensor;
                    },
                }
            } else {
                // Regular non-quantized tensor (f16)
                const tensor = try Tensor(f16).init(allocator, shape);
                const data_size_bytes = tensor.data.len * @sizeOf(f16);
                const data_bytes = try file.read(std.mem.sliceAsBytes(tensor.data));
                if (data_bytes != data_size_bytes) {
                    return error.InvalidRead;
                }
                current_pos += data_size_bytes;

                // Store the tensor in the appropriate field
                try self.storeTensor(name, tensor);
            }

            // Verify our position tracking matches the file
            const actual_pos = try file.getPos();
            if (actual_pos != current_pos) {
                std.debug.print("Position mismatch after tensor {s}!\n", .{name});
                std.debug.print("Expected: {d}, Actual: {d}\n", .{ current_pos, actual_pos });
                return error.InvalidPosition;
            }
        }

        // Transfer completed quantization parameters to the main storage
        var it = temp_quant_params.iterator();
        while (it.next()) |entry| {
            if (entry.value_ptr.scale != null and entry.value_ptr.zero != null) {
                // Create a new QuantParams entry
                const key = try allocator.dupe(u8, entry.key_ptr.*);
                try self.quant_params.put(key, .{
                    .scale = entry.value_ptr.scale.?,
                    .zero_point = entry.value_ptr.zero.?,
                });

                // Mark as transferred to avoid double-freeing
                entry.value_ptr.scale = null;
                entry.value_ptr.zero = null;
            } else {
                std.debug.print("Warning: Incomplete quantization parameters for {s}\n", .{entry.key_ptr.*});
            }
        }

        return self;
    }

    fn checkPrecision(tensor: Tensor(f16)) void {
        const first_few = 5;
        std.debug.print("First {d} values in binary format:\n", .{first_few});
        for (tensor.data[0..first_few]) |value| {
            // Get raw bits of f16
            const bits = @as(u16, @bitCast(value));
            const sign = (bits >> 15) & 1;
            const exp = (bits >> 10) & 0x1F;
            const frac = bits & 0x3FF;

            std.debug.print("Value: {e:8}, Sign: {d}, Exp: {d}, Frac: 0x{X:3}\n", .{ value, sign, exp, frac });
        }
    }

    fn printShapeMismatchError(name: []const u8, got: []const usize, expected: []const usize) void {
        std.debug.print("Shape mismatch for tensor '{s}':\n", .{name});
        std.debug.print("  Got shape: [", .{});
        for (got, 0..) |dim, i| {
            if (i > 0) std.debug.print(", ", .{});
            std.debug.print("{d}", .{dim});
        }
        std.debug.print("]\n", .{});

        std.debug.print("  Expected shape: [", .{});
        for (expected, 0..) |dim, i| {
            if (i > 0) std.debug.print(", ", .{});
            std.debug.print("{d}", .{dim});
        }
        std.debug.print("]\n", .{});
    }

    /// Store a tensor in the appropriate field based on its name
    fn storeTensor(self: *Self, name: []const u8, tensor: anytype) !void {
        // Text model tensors
        if (std.mem.eql(u8, name, "word_token_embedding")) {
            self.word_token_embedding = @as(Tensor(f16), tensor);
        } else if (std.mem.eql(u8, name, "t_ln_w")) {
            self.t_ln_w = @as(Tensor(f16), tensor);
        } else if (std.mem.eql(u8, name, "t_ln_b")) {
            self.t_ln_b = @as(Tensor(f16), tensor);
        } else if (std.mem.eql(u8, name, "t_Wqkv_w")) {
            self.t_Wqkv_w = @as(Tensor(i8), tensor);
        } else if (std.mem.eql(u8, name, "t_Wqkv_b")) {
            self.t_Wqkv_b = @as(Tensor(f16), tensor);
        } else if (std.mem.eql(u8, name, "t_out_proj_w")) {
            self.t_out_proj_w = @as(Tensor(i8), tensor);
        } else if (std.mem.eql(u8, name, "t_out_proj_bias")) {
            self.t_out_proj_bias = @as(Tensor(f16), tensor);
        } else if (std.mem.eql(u8, name, "t_fc1_w")) {
            self.t_fc1_w = @as(Tensor(i8), tensor);
        } else if (std.mem.eql(u8, name, "t_fc1_b")) {
            self.t_fc1_b = @as(Tensor(f16), tensor);
        } else if (std.mem.eql(u8, name, "t_fc2_w")) {
            self.t_fc2_w = @as(Tensor(i8), tensor);
        } else if (std.mem.eql(u8, name, "t_fc2_b")) {
            self.t_fc2_b = @as(Tensor(f16), tensor);
        } else if (std.mem.eql(u8, name, "t_linear_w")) {
            self.t_linear_w = @as(Tensor(i8), tensor);
        } else if (std.mem.eql(u8, name, "t_linear_b")) {
            self.t_linear_b = @as(Tensor(f16), tensor);
        } else if (std.mem.eql(u8, name, "t_ln_out_w")) {
            self.t_ln_out_w = @as(Tensor(f16), tensor);
        } else if (std.mem.eql(u8, name, "t_ln_out_b")) {
            self.t_ln_out_b = @as(Tensor(f16), tensor);
        }
        // Vision model tensors
        else if (std.mem.eql(u8, name, "v_patch_embedding_linear_w")) {
            self.v_patch_embedding_linear_w = @as(Tensor(f16), tensor);
        } else if (std.mem.eql(u8, name, "v_patch_embedding_linear_b")) {
            self.v_patch_embedding_linear_b = @as(Tensor(f16), tensor);
        } else if (std.mem.eql(u8, name, "v_pos_embedding")) {
            self.v_pos_embedding = @as(Tensor(f16), tensor);
        } else if (std.mem.eql(u8, name, "v_Wqkv_w")) {
            self.v_Wqkv_w = @as(Tensor(i8), tensor);
        } else if (std.mem.eql(u8, name, "v_Wqkv_b")) {
            self.v_Wqkv_b = @as(Tensor(f16), tensor);
        } else if (std.mem.eql(u8, name, "v_out_proj_w")) {
            self.v_out_proj_w = @as(Tensor(i8), tensor);
        } else if (std.mem.eql(u8, name, "v_out_proj_b")) {
            self.v_out_proj_b = @as(Tensor(f16), tensor);
        } else if (std.mem.eql(u8, name, "v_fc1_w")) {
            self.v_fc1_w = @as(Tensor(i8), tensor);
        } else if (std.mem.eql(u8, name, "v_fc1_b")) {
            self.v_fc1_b = @as(Tensor(f16), tensor);
        } else if (std.mem.eql(u8, name, "v_fc2_w")) {
            self.v_fc2_w = @as(Tensor(i8), tensor);
        } else if (std.mem.eql(u8, name, "v_fc2_b")) {
            self.v_fc2_b = @as(Tensor(f16), tensor);
        } else if (std.mem.eql(u8, name, "v_norm1_w")) {
            self.v_norm1_w = @as(Tensor(f16), tensor);
        } else if (std.mem.eql(u8, name, "v_norm1_b")) {
            self.v_norm1_b = @as(Tensor(f16), tensor);
        } else if (std.mem.eql(u8, name, "v_norm2_w")) {
            self.v_norm2_w = @as(Tensor(f16), tensor);
        } else if (std.mem.eql(u8, name, "v_norm2_b")) {
            self.v_norm2_b = @as(Tensor(f16), tensor);
        } else if (std.mem.eql(u8, name, "v_norm_out_w")) {
            self.v_norm_out_w = @as(Tensor(f16), tensor);
        } else if (std.mem.eql(u8, name, "v_norm_out_b")) {
            self.v_norm_out_b = @as(Tensor(f16), tensor);
        }
        // Projection layer tensors
        else if (std.mem.eql(u8, name, "v_proj_fc1_w")) {
            self.v_proj_fc1_w = @as(Tensor(i8), tensor);
        } else if (std.mem.eql(u8, name, "v_proj_fc1_b")) {
            self.v_proj_fc1_b = @as(Tensor(f16), tensor);
        } else if (std.mem.eql(u8, name, "v_proj_fc2_w")) {
            self.v_proj_fc2_w = @as(Tensor(i8), tensor);
        } else if (std.mem.eql(u8, name, "v_proj_fc2_b")) {
            self.v_proj_fc2_b = @as(Tensor(f16), tensor);
        } else {
            return error.UnknownTensorName;
        }
    }

    pub fn deinit(self: *Self) void {
        // Text model tensors
        self.word_token_embedding.deinit();
        self.t_ln_w.deinit();
        self.t_ln_b.deinit();
        self.t_Wqkv_w.deinit();
        self.t_Wqkv_b.deinit();
        self.t_out_proj_w.deinit();
        self.t_out_proj_bias.deinit();
        self.t_fc1_w.deinit();
        self.t_fc1_b.deinit();
        self.t_fc2_w.deinit();
        self.t_fc2_b.deinit();
        self.t_linear_w.deinit();
        self.t_linear_b.deinit();
        self.t_ln_out_w.deinit();
        self.t_ln_out_b.deinit();

        // Vision model tensors
        self.v_patch_embedding_linear_w.deinit();
        self.v_patch_embedding_linear_b.deinit();
        self.v_pos_embedding.deinit();
        self.v_Wqkv_w.deinit();
        self.v_Wqkv_b.deinit();
        self.v_out_proj_w.deinit();
        self.v_out_proj_b.deinit();
        self.v_fc1_w.deinit();
        self.v_fc1_b.deinit();
        self.v_fc2_w.deinit();
        self.v_fc2_b.deinit();
        self.v_norm1_w.deinit();
        self.v_norm1_b.deinit();
        self.v_norm2_w.deinit();
        self.v_norm2_b.deinit();
        self.v_norm_out_w.deinit();
        self.v_norm_out_b.deinit();

        // Projection layer tensors
        self.v_proj_fc1_w.deinit();
        self.v_proj_fc1_b.deinit();
        self.v_proj_fc2_w.deinit();
        self.v_proj_fc2_b.deinit();

        // Clean up quantization parameters
        var it = self.quant_params.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.scale.deinit();
            entry.value_ptr.zero_point.deinit();
            self.allocator.free(entry.key_ptr.*);
        }
        self.quant_params.deinit();

        // Set self to undefined after cleaning up
        self.* = undefined;
    }

    fn getExpectedShape(name: []const u8, shapes: anytype) ![]const usize {
        inline for (@typeInfo(@TypeOf(shapes)).Struct.fields) |field| {
            if (std.mem.eql(u8, field.name, name)) {
                const shape_ptr = @field(shapes, field.name);

                // Additional safety checks
                if (@TypeOf(shape_ptr) != []const usize) {
                    std.debug.print("Unexpected shape type for {s}\n", .{name});
                    return error.InvalidShapeType;
                }

                return shape_ptr;
            }
        }

        std.debug.print("Could not find shape for tensor name: {s}\n", .{name});
        return error.UnknownTensorName;
    }

    /// Calculate the shapes of all tensors based on the model configuration
    fn calculateShapes(allocator: Allocator, config: Config) !struct {
        // Text model shapes
        word_token_embedding: []const usize,
        t_ln_w: []const usize,
        t_ln_b: []const usize,
        t_Wqkv_w: []const usize,
        t_Wqkv_b: []const usize,
        t_out_proj_w: []const usize,
        t_out_proj_bias: []const usize,
        t_fc1_w: []const usize,
        t_fc1_b: []const usize,
        t_fc2_w: []const usize,
        t_fc2_b: []const usize,
        t_linear_w: []const usize,
        t_linear_b: []const usize,
        t_ln_out_w: []const usize,
        t_ln_out_b: []const usize,

        // Vision model shapes
        v_patch_embedding_linear_w: []const usize,
        v_patch_embedding_linear_b: []const usize,
        v_pos_embedding: []const usize,
        v_Wqkv_w: []const usize,
        v_Wqkv_b: []const usize,
        v_out_proj_w: []const usize,
        v_out_proj_b: []const usize,
        v_fc1_w: []const usize,
        v_fc1_b: []const usize,
        v_fc2_w: []const usize,
        v_fc2_b: []const usize,
        v_norm1_w: []const usize,
        v_norm1_b: []const usize,
        v_norm2_w: []const usize,
        v_norm2_b: []const usize,
        v_norm_out_w: []const usize,
        v_norm_out_b: []const usize,
        v_proj_fc1_w: []const usize,
        v_proj_fc1_b: []const usize,
        v_proj_fc2_w: []const usize,
        v_proj_fc2_b: []const usize,
    } {
        // Text model shapes allocation
        const word_token_embedding = try allocator.dupe(usize, &[_]usize{ config.vocab, config.dim });
        const t_ln_w = try allocator.dupe(usize, &[_]usize{ config.n_layers, config.dim });
        const t_ln_b = try allocator.dupe(usize, &[_]usize{ config.n_layers, config.dim });
        const t_Wqkv_w = try allocator.dupe(usize, &[_]usize{ config.n_layers, config.n_heads * config.head_dim * 3, config.dim });
        const t_Wqkv_b = try allocator.dupe(usize, &[_]usize{ config.n_layers, config.n_heads * config.head_dim * 3 });
        const t_out_proj_w = try allocator.dupe(usize, &[_]usize{ config.n_layers, config.dim, config.dim });
        const t_out_proj_bias = try allocator.dupe(usize, &[_]usize{ config.n_layers, config.dim });
        const t_fc1_w = try allocator.dupe(usize, &[_]usize{ config.n_layers, config.hidden_dim, config.dim });
        const t_fc1_b = try allocator.dupe(usize, &[_]usize{ config.n_layers, config.hidden_dim });
        const t_fc2_w = try allocator.dupe(usize, &[_]usize{ config.n_layers, config.dim, config.hidden_dim });
        const t_fc2_b = try allocator.dupe(usize, &[_]usize{ config.n_layers, config.dim });
        const t_linear_w = try allocator.dupe(usize, &[_]usize{ config.vocab, config.dim });
        const t_linear_b = try allocator.dupe(usize, &[_]usize{config.vocab});
        const t_ln_out_w = try allocator.dupe(usize, &[_]usize{config.dim});
        const t_ln_out_b = try allocator.dupe(usize, &[_]usize{config.dim});

        // Vision model shapes allocation
        const v_patch_embedding_linear_w = try allocator.dupe(usize, &[_]usize{ config.patch_size * config.patch_size * config.img_channels, config.vit_dim });
        const v_patch_embedding_linear_b = try allocator.dupe(usize, &[_]usize{config.vit_dim});
        const v_pos_embedding = try allocator.dupe(usize, &[_]usize{ 1, config.num_patches, config.vit_dim });
        const v_Wqkv_w = try allocator.dupe(usize, &[_]usize{ config.n_vit_layers, config.vit_dim, config.n_vit_heads * config.vit_head_dim * 3 });
        const v_Wqkv_b = try allocator.dupe(usize, &[_]usize{ config.n_vit_layers, config.n_vit_heads * config.vit_head_dim * 3 });
        const v_out_proj_w = try allocator.dupe(usize, &[_]usize{ config.n_vit_layers, config.vit_dim, config.vit_dim });
        const v_out_proj_b = try allocator.dupe(usize, &[_]usize{ config.n_vit_layers, config.vit_dim });
        const v_fc1_w = try allocator.dupe(usize, &[_]usize{ config.n_vit_layers, config.vit_dim, config.hidden_features });
        const v_fc1_b = try allocator.dupe(usize, &[_]usize{ config.n_vit_layers, config.hidden_features });
        const v_fc2_w = try allocator.dupe(usize, &[_]usize{ config.n_vit_layers, config.hidden_features, config.vit_dim });
        const v_fc2_b = try allocator.dupe(usize, &[_]usize{ config.n_vit_layers, config.vit_dim });
        const v_norm1_w = try allocator.dupe(usize, &[_]usize{ config.n_vit_layers, config.vit_dim });
        const v_norm1_b = try allocator.dupe(usize, &[_]usize{ config.n_vit_layers, config.vit_dim });
        const v_norm2_w = try allocator.dupe(usize, &[_]usize{ config.n_vit_layers, config.vit_dim });
        const v_norm2_b = try allocator.dupe(usize, &[_]usize{ config.n_vit_layers, config.vit_dim });
        const v_norm_out_w = try allocator.dupe(usize, &[_]usize{config.vit_dim});
        const v_norm_out_b = try allocator.dupe(usize, &[_]usize{config.vit_dim});

        // Projection layer shapes allocation
        const v_proj_fc1_w = try allocator.dupe(usize, &[_]usize{ config.vit_head_dim * config.n_heads, config.hidden_dim });
        const v_proj_fc1_b = try allocator.dupe(usize, &[_]usize{config.hidden_dim});
        const v_proj_fc2_w = try allocator.dupe(usize, &[_]usize{ config.hidden_dim, config.dim });
        const v_proj_fc2_b = try allocator.dupe(usize, &[_]usize{config.dim});

        return .{
            // Text model shapes
            .word_token_embedding = word_token_embedding,
            .t_ln_w = t_ln_w,
            .t_ln_b = t_ln_b,
            .t_Wqkv_w = t_Wqkv_w,
            .t_Wqkv_b = t_Wqkv_b,
            .t_out_proj_w = t_out_proj_w,
            .t_out_proj_bias = t_out_proj_bias,
            .t_fc1_w = t_fc1_w,
            .t_fc1_b = t_fc1_b,
            .t_fc2_w = t_fc2_w,
            .t_fc2_b = t_fc2_b,
            .t_linear_w = t_linear_w,
            .t_linear_b = t_linear_b,
            .t_ln_out_w = t_ln_out_w,
            .t_ln_out_b = t_ln_out_b,

            // Vision model shapes
            .v_patch_embedding_linear_w = v_patch_embedding_linear_w,
            .v_patch_embedding_linear_b = v_patch_embedding_linear_b,
            .v_pos_embedding = v_pos_embedding,
            .v_Wqkv_w = v_Wqkv_w,
            .v_Wqkv_b = v_Wqkv_b,
            .v_out_proj_w = v_out_proj_w,
            .v_out_proj_b = v_out_proj_b,
            .v_fc1_w = v_fc1_w,
            .v_fc1_b = v_fc1_b,
            .v_fc2_w = v_fc2_w,
            .v_fc2_b = v_fc2_b,
            .v_norm1_w = v_norm1_w,
            .v_norm1_b = v_norm1_b,
            .v_norm2_w = v_norm2_w,
            .v_norm2_b = v_norm2_b,
            .v_norm_out_w = v_norm_out_w,
            .v_norm_out_b = v_norm_out_b,

            // Projection layer shapes
            .v_proj_fc1_w = v_proj_fc1_w,
            .v_proj_fc1_b = v_proj_fc1_b,
            .v_proj_fc2_w = v_proj_fc2_w,
            .v_proj_fc2_b = v_proj_fc2_b,
        };
    }
};
