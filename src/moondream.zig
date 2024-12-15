const std = @import("std");
const Config = @import("config.zig").Config;
const ConfigReader = @import("config.zig").ConfigReader;
const Weights = @import("weights.zig").Weights;
const Tensor = @import("tensor.zig").Tensor;
const ops = @import("ops.zig");
const TextModel = @import("text_model.zig").TextModel;
const KVCache = @import("text_model.zig").KVCache;
const LayerCache = @import("text_model.zig").LayerCache;
const PrintOptions = @import("tensor.zig").PrintOptions;

pub fn main() !void {
    // Get allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{
        .enable_memory_limit = true,
        .verbose_log = true,
    }){};
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
    var text_model = try TextModel.init(config, weights, allocator);
    defer text_model.deinit();

    // Initialize input embeddings
    var input_embeds = try Tensor(f32).init(allocator, &[_]usize{ config.seq_len, config.dim });

    input_embeds.fill(0.0);
    defer input_embeds.deinit();

    std.debug.print("config.seq_len: {d}\n", .{config.seq_len});
    std.debug.print("config.dim: {d}\n", .{config.dim});

    std.debug.print("Input embeddings initialized successfully with shape {any}\n", .{input_embeds.shape});

    // Initialize KV cache (optional - pass null if you don't want to use cache)
    var kv_cache = try KVCache.init(allocator, config.n_layers, config.n_heads, config.head_dim);
    defer kv_cache.deinit();

    // Call text_decoder with KV cache
    var result = try text_model.text_decoder(input_embeds, &kv_cache);
    // Print output shape or other debug info if needed
    std.debug.print("Output tensor shape: {any}\n", .{result.output.shape});

    defer result.output.deinit();
    defer result.cache.deinit();
    // find the memory leak??
    // Note: don't need to defer result.cache.deinit() as it's the same as kv_cache and we can call deinit on it

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
