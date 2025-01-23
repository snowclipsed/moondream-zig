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
const displayImage = @import("imagedisplay.zig").displayImage;
const print = std.debug.print;
const Timer = std.time.Timer;

const main_color = "\x1b[38;5;214m";
const reset_color = "\x1b[0m";
const time_color = "\x1b[96m";

fn printTimeDiff(start_time: i128, end_time: i128, step_name: []const u8) !void {
    const stdout = std.io.getStdOut().writer();
    const diff_ns = end_time - start_time;
    const diff_ms = @as(f64, @floatFromInt(diff_ns)) / 1_000_000.0;
    try stdout.print("{s}[PROFILE] {s}: {d:.2}ms{s}\n", .{
        time_color, step_name, diff_ms, reset_color,
    });
}

pub fn main() !void {
    const stdout = std.io.getStdOut().writer();
    var timer = try Timer.start();
    const total_start = timer.read();

    // ASCII Art Display
    const display_start = timer.read();
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
    try printTimeDiff(display_start, timer.read(), "ASCII Art Display");

    // Initialize allocator
    const init_start = timer.read();
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize RNG
    const seed: u64 = @as(u64, @truncate(@as(u128, @bitCast(std.time.nanoTimestamp()))));
    var rng = std.rand.DefaultPrng.init(seed);
    const random = rng.random();
    try printTimeDiff(init_start, timer.read(), "Initialization");

    // Constants and configuration
    const bin_path: []const u8 = "../moondream.bin";
    const config_path: []const u8 = "../model_config.json";
    const tokenizer_path: []const u8 = "../tokenizer.bin";
    const image_path: []const u8 = "../images/catmonitor.png";
    const max_tokens: usize = 10;
    const sampling_config = ops.SamplingConfig{ .method = .greedy };

    // Load tokenizer
    const tokenizer_start = timer.read();
    var tokenizer = try Tokenizer.fromFile(tokenizer_path, allocator);
    defer tokenizer.deinit();
    try printTimeDiff(tokenizer_start, timer.read(), "Tokenizer Loading");

    // Load and parse config
    const config_start = timer.read();
    const config_file = try std.fs.cwd().openFile(config_path, .{});
    defer config_file.close();
    const config_size = (try config_file.stat()).size;
    const config_buffer = try allocator.alloc(u8, config_size);
    defer allocator.free(config_buffer);
    _ = try config_file.readAll(config_buffer);

    var json_tree = try std.json.parseFromSlice(ConfigReader, allocator, config_buffer, .{});
    defer json_tree.deinit();
    const config = json_tree.value.config();
    try printTimeDiff(config_start, timer.read(), "Config Loading");

    // Load model weights
    const weights_start = timer.read();
    const weights = try Weights.init(config, bin_path, allocator);
    try printTimeDiff(weights_start, timer.read(), "Model Weights Loading");

    std.debug.print("\n\n", .{});
    std.debug.print("Loading image, displaying at scale {d:3}x.\n", .{0.75});

    // Display and process image
    const image_start = timer.read();
    try displayImage(allocator, image_path, 0.75);
    try printTimeDiff(image_start, timer.read(), "Image Display");

    // Initialize vision model and encode image
    const vision_start = timer.read();
    var vision_model = try VisionModel.init(config, weights, allocator);
    defer vision_model.deinit();
    var image_tensor = try vision_model.encode_image(image_path);
    defer image_tensor.deinit();
    try printTimeDiff(vision_start, timer.read(), "Vision Model Init and Image Encoding");

    // Initialize text model
    const text_model_start = timer.read();
    var text_model = try TextModel.init(config, weights, allocator);
    defer text_model.deinit();
    try printTimeDiff(text_model_start, timer.read(), "Text Model Initialization");

    // Initialize tokens and create input tensor
    const token_init_start = timer.read();
    var token_ids = std.ArrayList(u32).init(allocator);
    defer token_ids.deinit();
    try token_ids.appendSlice(&[_]u32{ 50256, 198, 198, 24361, 25, 39373, 4892, 262, 2939, 198, 198, 33706, 25 });

    var input_ids = try Tensor(u32).init(allocator, &[_]usize{token_ids.items.len});
    defer input_ids.deinit();
    @memcpy(input_ids.data, token_ids.items);
    try printTimeDiff(token_init_start, timer.read(), "Token Initialization");

    // Get initial text embeddings and process
    const embedding_start = timer.read();
    var text_embeds = try text_model.text_encoder(input_ids);
    defer text_embeds.deinit();

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

    var leading_embed = try ops.concat(f16, first_slice, image_tensor, 0);
    defer leading_embed.deinit();
    var input_embeds = try ops.concat(f16, leading_embed, last_slice, 0);
    defer input_embeds.deinit();
    try printTimeDiff(embedding_start, timer.read(), "Embedding Processing");

    // Initialize KV cache
    const cache_start = timer.read();
    var kv_cache = try KVCache.init(allocator, config.n_layers, config.n_heads, config.head_dim);
    defer kv_cache.deinit();
    try printTimeDiff(cache_start, timer.read(), "KV Cache Initialization");

    var pos: usize = 0;
    var output_buffer = std.ArrayList(u8).init(allocator);
    defer output_buffer.deinit();

    try stdout.writeAll("\nGenerated text: ");

    // Generation loop
    const generation_start = timer.read();
    var total_decoder_time: i128 = 0;
    var total_sampling_time: i128 = 0;
    var total_embedding_time: i128 = 0;
    var token_count: usize = 0;

    for (0..max_tokens) |_| {
        if (pos >= 2048) break;

        const decoder_start = timer.read();
        var result = try text_model.text_decoder(input_embeds, &kv_cache);
        defer result.output.deinit();
        defer result.cache.deinit();
        total_decoder_time += timer.read() - decoder_start;

        const sampling_start = timer.read();
        var logits = try text_model.lm_head(result.output);
        defer logits.deinit();
        const next_token_id = try ops.sample(f16, &logits, sampling_config, random, allocator);
        total_sampling_time += timer.read() - sampling_start;

        if (next_token_id == tokenizer.eos_token) break;

        const embed_start = timer.read();
        var next_token_tensor = try Tensor(u32).init(allocator, &[_]usize{1});
        defer next_token_tensor.deinit();
        next_token_tensor.data[0] = @intCast(next_token_id);

        var new_embeds = try text_model.text_encoder(next_token_tensor);
        errdefer new_embeds.deinit();
        input_embeds.deinit();
        input_embeds = new_embeds;
        total_embedding_time += timer.read() - embed_start;

        var next_token_list = std.ArrayList(u32).init(allocator);
        defer next_token_list.deinit();
        try next_token_list.append(@intCast(next_token_id));

        const decoded = try tokenizer.decode(next_token_list);
        defer allocator.free(decoded);

        try output_buffer.appendSlice(decoded);
        try stdout.writeAll(decoded);

        pos += 1;
        token_count += 1;
    }

    const generation_time = timer.read() - generation_start;
    try stdout.writeByte('\n');

    // Print generation statistics
    try stdout.print("\n{s}Generation Statistics:{s}\n", .{ time_color, reset_color });
    try stdout.print("Total tokens generated: {d}\n", .{token_count});
    try stdout.print("Average time per token: {d:.2}ms\n", .{
        @as(f64, @floatFromInt(generation_time)) / @as(f64, @floatFromInt(token_count)) / 1_000_000.0,
    });
    try stdout.print("Average decoder time: {d:.2}ms\n", .{
        @as(f64, @floatFromInt(total_decoder_time)) / @as(f64, @floatFromInt(token_count)) / 1_000_000.0,
    });
    try stdout.print("Average sampling time: {d:.2}ms\n", .{
        @as(f64, @floatFromInt(total_sampling_time)) / @as(f64, @floatFromInt(token_count)) / 1_000_000.0,
    });
    try stdout.print("Average embedding time: {d:.2}ms\n", .{
        @as(f64, @floatFromInt(total_embedding_time)) / @as(f64, @floatFromInt(token_count)) / 1_000_000.0,
    });

    // Print total execution time
    try printTimeDiff(total_start, timer.read(), "Total Execution");
}
