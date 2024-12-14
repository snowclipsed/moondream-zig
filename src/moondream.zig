const std = @import("std");
const Config = @import("config.zig").Config;
const ConfigReader = @import("config.zig").ConfigReader;
const Weights = @import("weights.zig").Weights;
const Tensor = @import("tensor.zig").Tensor;
const ops = @import("ops.zig");
const TextModel = @import("text_model.zig").TextModel;
const PrintOptions = @import("tensor.zig").PrintOptions;

pub fn main() !void {
    // Get allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Constants
    const bin_path: []const u8 = "../moondream_f32.bin";
    const config_path: []const u8 = "../model_config.json";

    // Load and parse config file
    const config_file = try std.fs.cwd().openFile(config_path, .{});
    defer config_file.close();

    const config_size = (try config_file.stat()).size;

    const config_buffer = try allocator.alloc(u8, config_size);
    defer allocator.free(config_buffer);
    _ = try config_file.readAll(config_buffer);

    var json_tree = try std.json.parseFromSlice(ConfigReader, allocator, config_buffer, .{});
    defer json_tree.deinit();

    const config = json_tree.value.config();
    // Print some debug information
    std.debug.print("Config loaded successfully\n", .{});
    std.debug.print("Text model dimensions: {d}\n", .{config.dim});
    std.debug.print("Vision model dimensions: {d}\n", .{config.vit_dim});

    // Load model weights
    const weights = try Weights.init(config, bin_path, allocator);

    std.debug.print("Weights loaded successfully\n", .{});

    // Optional: Print shapes of some tensors to verify loading
    std.debug.print("\nSample tensor shapes:\n", .{});
    std.debug.print("word_token_embedding: {any}\n", .{weights.word_token_embedding.shape});
    std.debug.print("t_ln_w: {any}\n", .{weights.t_ln_w.shape});
    std.debug.print("v_patch_embedding_linear_w: {any}\n", .{weights.v_patch_embedding_linear_w.shape});

    // Initialize text model
    var text_model = TextModel.init(config, weights, allocator);
    var input_embeds: Tensor(f32) = try Tensor(f32).init(allocator, &[_]usize{ config.seq_len, config.dim });
    input_embeds.fill(0.0);
    try text_model.text_decoder(input_embeds);
    defer text_model.deinit();
    defer input_embeds.deinit();
}

// test "basic config and weights loading" {
//     const allocator = std.testing.allocator;

//     const bin_path: []const u8 = "/home/snow/projects/moondream-zig/model.bin";
//     const config_path: []const u8 = "/home/snow/projects/moondream-zig/model_config.json";

//     // Load and parse config
//     const config_file = try std.fs.cwd().openFile(config_path, .{});
//     defer config_file.close();

//     const config_size = (try config_file.stat()).size;
//     const config_buffer = try allocator.alloc(u8, config_size);
//     defer allocator.free(config_buffer);
//     _ = try config_file.readAll(config_buffer);

//     var json_tree = try std.json.parseFromSlice(ConfigReader, allocator, config_buffer, .{});
//     defer json_tree.deinit();

//     const config = json_tree.value.config();

//     // Load weights
//     var weights = try Weights.init(config, bin_path, allocator);
//     defer weights.deinit();

//     // Add basic shape verifications
//     try std.testing.expectEqual(config.vocab, weights.word_token_embedding.shape[0]);
//     try std.testing.expectEqual(config.dim, weights.word_token_embedding.shape[1]);
// }
