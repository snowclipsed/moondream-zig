const std = @import("std");
const Allocator = std.mem.Allocator;
const Tensor = @import("tensor.zig").Tensor;
const Config = @import("config.zig").Config;

pub const Weights = struct {
    const Self = @This();

    // Magic constants for format validation
    pub const WEIGHTS_MAGIC = 0x4D4F4F4E; // "MOON" in ASCII
    pub const WEIGHTS_VERSION = 2; // Incrementing version for tensor format

    allocator: Allocator,

    // Text model weights
    word_token_embedding: Tensor(f16),
    t_ln_w: Tensor(f16),
    t_ln_b: Tensor(f16),
    t_Wqkv_w: Tensor(f16),
    t_Wqkv_b: Tensor(f16),
    t_out_proj_w: Tensor(f16),
    t_out_proj_bias: Tensor(f16),
    t_fc1_w: Tensor(f16),
    t_fc1_b: Tensor(f16),
    t_fc2_w: Tensor(f16),
    t_fc2_b: Tensor(f16),
    t_linear_w: Tensor(f16),
    t_linear_b: Tensor(f16),
    t_ln_out_w: Tensor(f16),
    t_ln_out_b: Tensor(f16),

    // Vision model weights
    v_patch_embedding_linear_w: Tensor(f16),
    v_patch_embedding_linear_b: Tensor(f16),
    v_pos_embedding: Tensor(f16),
    v_Wqkv_w: Tensor(f16),
    v_Wqkv_b: Tensor(f16),
    v_out_proj_w: Tensor(f16),
    v_out_proj_b: Tensor(f16),
    v_fc1_w: Tensor(f16),
    v_fc1_b: Tensor(f16),
    v_fc2_w: Tensor(f16),
    v_fc2_b: Tensor(f16),
    v_norm1_w: Tensor(f16),
    v_norm1_b: Tensor(f16),
    v_norm2_w: Tensor(f16),
    v_norm2_b: Tensor(f16),
    v_norm_out_w: Tensor(f16),
    v_norm_out_b: Tensor(f16),
    v_proj_fc1_w: Tensor(f16),
    v_proj_fc1_b: Tensor(f16),
    v_proj_fc2_w: Tensor(f16),
    v_proj_fc2_b: Tensor(f16),

    const WeightHeader = struct {
        name_length: u32,
        shape_length: u32,
        data_length: u32,
    };

    pub fn init(config: Config, filename: []const u8, allocator: Allocator) !Self {
        const file = try std.fs.cwd().openFile(filename, .{});
        defer file.close();

        var current_pos: u64 = 0;

        // Magic number and version check remains the same
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

        // Initialize weights struct
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
        };

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
            // std.debug.print("Reading tensor: {s} (length: {d})\n", .{ name, name_length });

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

            const expected_shape = try getExpectedShape(name, expected_shapes);
            if (!std.mem.eql(usize, shape, expected_shape)) {
                printShapeMismatchError(name, shape, expected_shape);
                return error.ShapeMismatch;
            }

            // Create f16 tensor and read data
            const tensor = try Tensor(f16).init(allocator, shape);
            // Calculate total data size in bytes, accounting for f16 size
            const data_size_bytes = tensor.data.len * @sizeOf(f16);
            // std.debug.print("Reading {d} bytes of data for tensor {s}\n\n", .{ data_size_bytes, name });

            // Read raw bytes into the f16 tensor data buffer
            const data_bytes = try file.read(std.mem.sliceAsBytes(tensor.data));
            if (data_bytes != data_size_bytes) {
                std.debug.print("Data read mismatch! Expected {d} bytes, got {d}\n", .{ data_size_bytes, data_bytes });
                return error.InvalidRead;
            }
            current_pos += data_size_bytes;

            // Verify our position tracking matches the file
            const actual_pos = try file.getPos();
            if (actual_pos != current_pos) {
                std.debug.print("Position mismatch after tensor {s}!\n", .{name});
                std.debug.print("Expected: {d}, Actual: {d}\n", .{ current_pos, actual_pos });
                return error.InvalidPosition;
            }

            try self.storeTensor(name, tensor);
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
    fn storeTensor(self: *Self, name: []const u8, tensor: Tensor(f16)) !void {
        // Text model tensors
        if (std.mem.eql(u8, name, "word_token_embedding")) {
            self.word_token_embedding = tensor;
        } else if (std.mem.eql(u8, name, "t_ln_w")) {
            self.t_ln_w = tensor;
        } else if (std.mem.eql(u8, name, "t_ln_b")) {
            self.t_ln_b = tensor;
        } else if (std.mem.eql(u8, name, "t_Wqkv_w")) {
            self.t_Wqkv_w = tensor;
        } else if (std.mem.eql(u8, name, "t_Wqkv_b")) {
            self.t_Wqkv_b = tensor;
        } else if (std.mem.eql(u8, name, "t_out_proj_w")) {
            self.t_out_proj_w = tensor;
        } else if (std.mem.eql(u8, name, "t_out_proj_bias")) {
            self.t_out_proj_bias = tensor;
        } else if (std.mem.eql(u8, name, "t_fc1_w")) {
            self.t_fc1_w = tensor;
        } else if (std.mem.eql(u8, name, "t_fc1_b")) {
            self.t_fc1_b = tensor;
        } else if (std.mem.eql(u8, name, "t_fc2_w")) {
            self.t_fc2_w = tensor;
        } else if (std.mem.eql(u8, name, "t_fc2_b")) {
            self.t_fc2_b = tensor;
        } else if (std.mem.eql(u8, name, "t_linear_w")) {
            self.t_linear_w = tensor;
        } else if (std.mem.eql(u8, name, "t_linear_b")) {
            self.t_linear_b = tensor;
        } else if (std.mem.eql(u8, name, "t_ln_out_w")) {
            self.t_ln_out_w = tensor;
        } else if (std.mem.eql(u8, name, "t_ln_out_b")) {
            self.t_ln_out_b = tensor;
        }
        // Vision model tensors
        else if (std.mem.eql(u8, name, "v_patch_embedding_linear_w")) {
            self.v_patch_embedding_linear_w = tensor;
        } else if (std.mem.eql(u8, name, "v_patch_embedding_linear_b")) {
            self.v_patch_embedding_linear_b = tensor;
        } else if (std.mem.eql(u8, name, "v_pos_embedding")) {
            self.v_pos_embedding = tensor;
        } else if (std.mem.eql(u8, name, "v_Wqkv_w")) {
            self.v_Wqkv_w = tensor;
        } else if (std.mem.eql(u8, name, "v_Wqkv_b")) {
            self.v_Wqkv_b = tensor;
        } else if (std.mem.eql(u8, name, "v_out_proj_w")) {
            self.v_out_proj_w = tensor;
        } else if (std.mem.eql(u8, name, "v_out_proj_b")) {
            self.v_out_proj_b = tensor;
        } else if (std.mem.eql(u8, name, "v_fc1_w")) {
            self.v_fc1_w = tensor;
        } else if (std.mem.eql(u8, name, "v_fc1_b")) {
            self.v_fc1_b = tensor;
        } else if (std.mem.eql(u8, name, "v_fc2_w")) {
            self.v_fc2_w = tensor;
        } else if (std.mem.eql(u8, name, "v_fc2_b")) {
            self.v_fc2_b = tensor;
        } else if (std.mem.eql(u8, name, "v_norm1_w")) {
            self.v_norm1_w = tensor;
        } else if (std.mem.eql(u8, name, "v_norm1_b")) {
            self.v_norm1_b = tensor;
        } else if (std.mem.eql(u8, name, "v_norm2_w")) {
            self.v_norm2_w = tensor;
        } else if (std.mem.eql(u8, name, "v_norm2_b")) {
            self.v_norm2_b = tensor;
        } else if (std.mem.eql(u8, name, "v_norm_out_w")) {
            self.v_norm_out_w = tensor;
        } else if (std.mem.eql(u8, name, "v_norm_out_b")) {
            self.v_norm_out_b = tensor;
        }
        // Projection layer tensors
        else if (std.mem.eql(u8, name, "v_proj_fc1_w")) {
            self.v_proj_fc1_w = tensor;
        } else if (std.mem.eql(u8, name, "v_proj_fc1_b")) {
            self.v_proj_fc1_b = tensor;
        } else if (std.mem.eql(u8, name, "v_proj_fc2_w")) {
            self.v_proj_fc2_w = tensor;
        } else if (std.mem.eql(u8, name, "v_proj_fc2_b")) {
            self.v_proj_fc2_b = tensor;
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

        // Set self to undefined after cleaning up
        self.* = undefined;
    }

    fn getExpectedShape(name: []const u8, shapes: anytype) ![]const usize {
        // Debug print the name and available field names
        // inline for (@typeInfo(@TypeOf(shapes)).Struct.fields) |field| {
        //     std.debug.print("Available field: {s}\n", .{field.name});
        // }

        // Use a more robust matching approach
        inline for (@typeInfo(@TypeOf(shapes)).Struct.fields) |field| {
            if (std.mem.eql(u8, field.name, name)) {
                const shape_ptr = @field(shapes, field.name);

                // Additional safety checks
                if (@TypeOf(shape_ptr) != []const usize) {
                    std.debug.print("Unexpected shape type for {s}\n", .{name});
                    return error.InvalidShapeType;
                }

                // std.debug.print("Found shape for {s}: {any}\n", .{ name, shape_ptr });
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
        const t_Wqkv_w = try allocator.dupe(usize, &[_]usize{ config.n_layers, config.dim, config.n_heads * config.head_dim * 3 });
        const t_Wqkv_b = try allocator.dupe(usize, &[_]usize{ config.n_layers, config.n_heads * config.head_dim * 3 });
        const t_out_proj_w = try allocator.dupe(usize, &[_]usize{ config.n_layers, config.dim, config.dim });
        const t_out_proj_bias = try allocator.dupe(usize, &[_]usize{ config.n_layers, config.dim });
        const t_fc1_w = try allocator.dupe(usize, &[_]usize{ config.n_layers, config.dim, config.hidden_dim });
        const t_fc1_b = try allocator.dupe(usize, &[_]usize{ config.n_layers, config.hidden_dim });
        const t_fc2_w = try allocator.dupe(usize, &[_]usize{ config.n_layers, config.hidden_dim, config.dim });
        const t_fc2_b = try allocator.dupe(usize, &[_]usize{ config.n_layers, config.dim });
        const t_linear_w = try allocator.dupe(usize, &[_]usize{ config.dim, config.vocab });
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
