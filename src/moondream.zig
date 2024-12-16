const std = @import("std");
const Config = @import("config.zig").Config;
const ConfigReader = @import("config.zig").ConfigReader;
const Weights = @import("weights.zig").Weights;
const Tensor = @import("tensor.zig").Tensor;
const ops = @import("ops.zig");
const TextModel = @import("text_model.zig").TextModel;
const KVCache = @import("text_model.zig").KVCache;
const LayerCache = @import("text_model.zig").LayerCache;
const Tokenizer = @import("tokenizer.zig").Tokenizer;
const print = std.debug.print;

pub fn main() !void {
    // Initialize allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Constants
    const bin_path: []const u8 = "../moondream_f32.bin";
    const config_path: []const u8 = "../model_config.json";
    const tokenizer_path: []const u8 = "../tokenizer.bin";
    const max_tokens: usize = 10; // Equivalent to args.max_tokens

    // Load tokenizer
    var tokenizer = try Tokenizer.fromFile(tokenizer_path, allocator);
    defer tokenizer.deinit();

    // Load and parse config
    const config_file = try std.fs.cwd().openFile(config_path, .{});
    defer config_file.close();
    const config_size = (try config_file.stat()).size;
    const config_buffer = try allocator.alloc(u8, config_size);
    defer allocator.free(config_buffer);
    _ = try config_file.readAll(config_buffer);

    var json_tree = try std.json.parseFromSlice(ConfigReader, allocator, config_buffer, .{});
    defer json_tree.deinit();
    const config = json_tree.value.config();

    // Load model weights
    const weights = try Weights.init(config, bin_path, allocator);
    print("lm head weights shape: {any}\n", .{weights.t_linear_w.shape});

    // Initialize text model
    var text_model = try TextModel.init(config, weights, allocator);
    defer text_model.deinit();

    // Initialize prompt
    const prompt = "\n\nQuestion?\n\nAnswer:";
    var token_ids = try tokenizer.encode(prompt);
    defer token_ids.deinit();

    // Add EOS token at the beginning
    try token_ids.insert(0, tokenizer.eos_token);

    // Convert token_ids to tensor
    var input_ids = try Tensor(f32).init(allocator, &[_]usize{token_ids.items.len});
    defer input_ids.deinit();
    for (token_ids.items, 0..) |id, i| {
        input_ids.data[i] = @floatFromInt(id);
    }

    // Get input embeddings
    var input_embeds = try text_model.text_encoder(input_ids);
    defer input_embeds.deinit();

    // Initialize KV cache
    var kv_cache = try KVCache.init(allocator, config.n_layers, config.n_heads, config.head_dim);
    defer kv_cache.deinit();

    var pos: usize = 0;
    var output_buffer = std.ArrayList(u8).init(allocator);
    defer output_buffer.deinit();

    print("Generated text:\n", .{});

    // Generation loop
    for (0..max_tokens) |_| {
        // Run the model
        var result = try text_model.text_decoder(input_embeds, &kv_cache);
        defer {
            result.output.deinit();
            result.cache.deinit(); // Assuming there's a cache field that needs cleanup
        }

        var logits = try text_model.lm_head(result.output);
        defer logits.deinit();

        // Simple greedy sampling (equivalent to args.sampler == "greedy")
        const next_token_id = try ops.argmax(f32, logits);

        // Check for EOS token
        if (next_token_id == tokenizer.eos_token) {
            break;
        }

        // Convert next token to tensor for encoding
        var next_token_tensor = try Tensor(f32).init(allocator, &[_]usize{1});
        defer next_token_tensor.deinit();
        next_token_tensor.data[0] = @floatFromInt(next_token_id);

        // Get embeddings for next token
        const new_embeds = try text_model.text_encoder(next_token_tensor);
        input_embeds.deinit();
        input_embeds = new_embeds;

        // Decode and print the token
        var next_token_list = std.ArrayList(u32).init(allocator);
        defer next_token_list.deinit();
        try next_token_list.append(@intCast(next_token_id));

        const decoded = try tokenizer.decode(next_token_list);
        defer allocator.free(decoded);

        try output_buffer.appendSlice(decoded);
        try std.io.getStdOut().writer().writeAll(decoded);

        // Update position counter
        pos += 1;
    }

    try std.io.getStdOut().writer().writeByte('\n');
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
