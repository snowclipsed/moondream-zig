const std = @import("std");
const Config = @import("config.zig").Config;
const ConfigReader = @import("config.zig").ConfigReader;
const Weights = @import("weights.zig").Weights;
const Tensor = @import("tensor.zig").Tensor;
const Slice = @import("tensor.zig").Slice;
const ops = @import("ops.zig");
const TextModel = @import("text_model.zig").TextModel;
const VisionModel = @import("vision_model.zig").VisionModel;
const KVCache = @import("text_model.zig").KVCache;
const LayerCache = @import("text_model.zig").LayerCache;
const Tokenizer = @import("tokenizer.zig").Tokenizer;
const print = std.debug.print;
const main_color = "\x1b[38;5;214m"; // Using 8-bit color code 214 for orange
const reset_color = "\x1b[0m";

pub fn main() !void {
    const stdout = std.io.getStdOut().writer();

    try stdout.print("{s}", .{main_color});
    try stdout.writeAll(
        \\
        \\███╗   ███╗ ██████╗  ██████╗ ███╗   ██╗██████╗ ██████╗ ███████╗ █████╗ ███╗   ███╗ ▪    ███████╗██╗ ██████╗ 
        \\████╗ ████║██╔═══██╗██╔═══██╗████╗  ██║██╔══██╗██╔══██╗██╔════╝██╔══██╗████╗ ████║ ▪    ╚══███╔╝██║██╔════╝ 
        \\██╔████╔██║██║   ██║██║   ██║██╔██╗ ██║██║  ██║██████╔╝█████╗  ███████║██╔████╔██║ ▪      ███╔╝ ██║██║  ███╗
        \\██║╚██╔╝██║██║   ██║██║   ██║██║╚██╗██║██║  ██║██╔══██╗██╔══╝  ██╔══██║██║╚██╔╝██║ ▪    ███╔╝   ██║██║   ██║
        \\██║ ╚═╝ ██║╚██████╔╝╚██████╔╝██║ ╚████║██████╔╝██║  ██║███████╗██║  ██║██║ ╚═╝ ██║ ▪    ███████╗██║╚██████╔╝
        \\╚═╝     ╚═╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝ ▪    ╚══════╝╚═╝ ╚═════╝ •
        \\
    );
    try stdout.print("{s}", .{reset_color});

    // Initialize allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize RNG
    const seed: u64 = @as(u64, @truncate(@as(u128, @bitCast(std.time.nanoTimestamp()))));
    var rng = std.rand.DefaultPrng.init(seed);
    const random = rng.random();

    // Constants and configuration
    const bin_path: []const u8 = "../moondream.bin";
    const config_path: []const u8 = "../model_config.json";
    const tokenizer_path: []const u8 = "../tokenizer.bin";
    const max_tokens: usize = 10;
    const sampling_config = ops.SamplingConfig{ .method = .greedy };

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

    // Initialize vision model and encode image
    var vision_model = try VisionModel.init(config, weights, allocator);
    defer vision_model.deinit();
    var image_tensor = try vision_model.encode_image("/home/snow/projects/moondream-zig/images/catmonitor.png");
    defer image_tensor.deinit();

    // Initialize text model
    var text_model = try TextModel.init(config, weights, allocator);
    defer text_model.deinit();

    // Initialize tokens
    var token_ids = std.ArrayList(u32).init(allocator);
    defer token_ids.deinit();
    try token_ids.appendSlice(&[_]u32{ 50256, 198, 198, 24361, 25, 39373, 4892, 262, 2939, 198, 198, 33706, 25 });

    // Create input tensor and encode tokens
    var input_ids = try Tensor(u32).init(allocator, &[_]usize{token_ids.items.len});
    defer input_ids.deinit();
    @memcpy(input_ids.data, token_ids.items);

    // Get initial text embeddings
    var text_embeds = try text_model.text_encoder(input_ids);
    defer text_embeds.deinit();

    // Split embeddings for image insertion
    var first_slice = try text_embeds.getSliceRange(&[_]Slice{
        Slice.from(0, 1),
        Slice.full(),
    });
    defer first_slice.deinit();

    var last_slice = try text_embeds.getSliceRange(&[_]Slice{
        Slice.from(1, null),
        Slice.full(),
    });
    defer last_slice.deinit();

    // Combine text and image embeddings
    var leading_embed = try ops.concat(f16, first_slice, image_tensor, 0);
    defer leading_embed.deinit();
    var input_embeds = try ops.concat(f16, leading_embed, last_slice, 0);
    defer input_embeds.deinit();

    // Initialize KV cache
    var kv_cache = try KVCache.init(allocator, config.n_layers, config.n_heads, config.head_dim);
    defer kv_cache.deinit();

    var pos: usize = 0;
    var output_buffer = std.ArrayList(u8).init(allocator);
    defer output_buffer.deinit();

    try stdout.writeAll("\nGenerated text: ");

    // Generation loop
    for (0..max_tokens) |_| {
        if (pos >= 2048) break;

        // Run decoder
        var result = try text_model.text_decoder(input_embeds, &kv_cache);
        defer result.output.deinit();
        defer result.cache.deinit();

        // Get logits and sample next token
        var logits = try text_model.lm_head(result.output);
        defer logits.deinit();
        const next_token_id = try ops.sample(f16, &logits, sampling_config, random, allocator);

        // Check for EOS token
        if (next_token_id == tokenizer.eos_token) break;

        // Convert next token to tensor
        var next_token_tensor = try Tensor(u32).init(allocator, &[_]usize{1});
        defer next_token_tensor.deinit();
        next_token_tensor.data[0] = @intCast(next_token_id);

        // Get embeddings for next token
        var new_embeds = try text_model.text_encoder(next_token_tensor);
        errdefer new_embeds.deinit();
        input_embeds.deinit();
        input_embeds = new_embeds;

        // Decode and print the token
        var next_token_list = std.ArrayList(u32).init(allocator);
        defer next_token_list.deinit();
        try next_token_list.append(@intCast(next_token_id));

        const decoded = try tokenizer.decode(next_token_list);
        defer allocator.free(decoded);

        try output_buffer.appendSlice(decoded);
        try stdout.writeAll(decoded);

        pos += 1;
    }

    try stdout.writeByte('\n');
}
// TODO list
// - Debug and Fix generation loop
// - Try specifically with tokens : [ 50256,   198,   198, 24361,    25, 39373,  4892,   262,  2939,   198, 198, 33706,    25]
// - Check correctness of tokenizer using the output tokens of the python version.
// - Check correctness of lm head
// - Check correctness of all weights
// - add epsilon to config
// - Add vision encoder
// - Add proper tests for the generation loop
// - Conver flatten to intrinsic op for tensor class
// - Make precompute freqs cis a little simpler

// verified list:
// - layer norm
// - qkv projection and transform
// - position ids are correct
// - precompute freqs cis looks good!
// - rotary embedding looks like it's working
// - attn mask looks good
// - attention looks good
// - transpose looks good
// - linear looks good
