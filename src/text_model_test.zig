const std = @import("std");
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;
const expectError = std.testing.expectError;
const Config = @import("config.zig").Config;
const ConfigReader = @import("config.zig").ConfigReader;
const Weights = @import("weights.zig").Weights;
const TextModel = @import("text_model.zig").TextModel;
const Tensor = @import("tensor.zig").Tensor;

test "TextModel - text_encoder with real config and weights" {
    // Setup allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Load config
    const config_path = "../model_config.json";
    const config_file = try std.fs.cwd().openFile(config_path, .{});
    defer config_file.close();

    const config_size = (try config_file.stat()).size;
    const config_buffer = try allocator.alloc(u8, config_size);
    defer allocator.free(config_buffer);
    _ = try config_file.readAll(config_buffer);

    var json_tree = try std.json.parseFromSlice(ConfigReader, allocator, config_buffer, .{});
    defer json_tree.deinit();

    const config = json_tree.value.config();

    // Load weights
    const bin_path = "../moondream_f32.bin";
    const weights = try Weights.init(config, bin_path, allocator);

    var model = try TextModel.init(config, weights, allocator);
    defer model.deinit();

    // Create test input with direct data access
    const input_data = try allocator.alloc(f32, config.seq_len);
    defer allocator.free(input_data);

    for (input_data, 0..) |*val, i| {
        val.* = @floatFromInt(i % config.vocab);
    }

    var input_ids = try Tensor(f32).init(allocator, &[_]usize{config.seq_len});
    @memcpy(input_ids.data, input_data);
    defer input_ids.deinit();

    // Run encoder
    var output = try model.text_encoder(input_ids);
    defer output.deinit();
    // Verify output shape
    try expectEqual(output.shape.len, 2);
    try expectEqual(output.shape[0], config.seq_len);
    try expectEqual(output.shape[1], config.dim);
}

test "TextModel - text_encoder error cases with real config" {
    // Setup
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Load config
    const config_path = "../model_config.json";
    const config_file = try std.fs.cwd().openFile(config_path, .{});
    defer config_file.close();

    const config_size = (try config_file.stat()).size;
    const config_buffer = try allocator.alloc(u8, config_size);
    defer allocator.free(config_buffer);
    _ = try config_file.readAll(config_buffer);

    var json_tree = try std.json.parseFromSlice(ConfigReader, allocator, config_buffer, .{});
    defer json_tree.deinit();

    const config = json_tree.value.config();

    // Load weights
    const bin_path = "../moondream_f32.bin";
    const weights = try Weights.init(config, bin_path, allocator);

    var model = try TextModel.init(config, weights, allocator);
    defer model.deinit();

    // Test 1: Invalid input shape (2D instead of 1D)
    {
        var invalid_shape_input = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        @memset(invalid_shape_input.data, undefined);
        defer invalid_shape_input.deinit();

        try expectError(error.InvalidInputShape, model.text_encoder(invalid_shape_input));
    }

    // Test 3: Sequence length longer than config.seq_len
    {
        var too_long_input = try Tensor(f32).init(allocator, &[_]usize{config.seq_len + 1});
        @memset(too_long_input.data, undefined);
        defer too_long_input.deinit();

        try expectError(error.SequenceTooLong, model.text_encoder(too_long_input));
    }
}
