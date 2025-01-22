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

// Timing helper function
fn printTiming(start: i128, end: i128, label: []const u8) void {
    const duration_ns = end - start;
    const duration_ms = @as(f64, @floatFromInt(duration_ns)) / 1_000_000.0;
    print("{s}: {d:.2}ms\n", .{ label, duration_ms });
}

pub fn main() !void {
    const stdout = std.io.getStdOut().writer();

    // ANSI escape codes for orange (using bright yellow as approximation)
    const orange_color = "\x1b[38;5;214m"; // Using 8-bit color code 214 for orange
    const reset_color = "\x1b[0m";

    try stdout.print("{s}", .{orange_color});
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

    // printing header

    const total_start = std.time.nanoTimestamp();

    // Initialize allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize RNG
    const seed: u64 = @as(u64, @truncate(@as(u128, @bitCast(std.time.nanoTimestamp()))));
    var rng = std.rand.DefaultPrng.init(seed);
    const random = rng.random();

    // Constants
    const bin_path: []const u8 = "../moondream.bin";
    const config_path: []const u8 = "../model_config.json";
    const tokenizer_path: []const u8 = "../tokenizer.bin";
    const max_tokens: usize = 20;

    const sampling_config = ops.SamplingConfig{
        .method = .greedy,
    };

    // Load and initialize components with timing
    const load_start = std.time.nanoTimestamp();

    var tokenizer = try Tokenizer.fromFile(tokenizer_path, allocator);
    defer tokenizer.deinit();
    const tokenizer_end = std.time.nanoTimestamp();
    printTiming(load_start, tokenizer_end, "Tokenizer loading");

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
    const config_end = std.time.nanoTimestamp();
    printTiming(tokenizer_end, config_end, "Config loading");

    // Load model weights
    const weights_start = std.time.nanoTimestamp();
    const weights = try Weights.init(config, bin_path, allocator);
    const weights_end = std.time.nanoTimestamp();
    printTiming(weights_start, weights_end, "Weights loading");

    // Initialize vision model and encode image
    const vision_start = std.time.nanoTimestamp();
    var vision_model = try VisionModel.init(config, weights, allocator);
    defer vision_model.deinit();

    var image_tensor = try vision_model.encode_image("/home/snow/projects/moondream-zig/images/demo-1.jpg");
    defer image_tensor.deinit();
    const vision_end = std.time.nanoTimestamp();
    printTiming(vision_start, vision_end, "Vision model init and image encoding");

    // Initialize text model
    const text_model_start = std.time.nanoTimestamp();
    var text_model = try TextModel.init(config, weights, allocator);
    defer text_model.deinit();
    const text_model_end = std.time.nanoTimestamp();
    printTiming(text_model_start, text_model_end, "Text model initialization");

    // Initialize tokens
    var token_ids = std.ArrayList(u32).init(allocator);
    defer token_ids.deinit();
    try token_ids.appendSlice(&[_]u32{ 50256, 198, 198, 24361, 25, 39373, 4892, 262, 2939, 198, 198, 33706, 25 });

    // Text encoding and embedding preparation
    const embed_start = std.time.nanoTimestamp();
    var input_ids = try Tensor(u32).init(allocator, &[_]usize{token_ids.items.len});
    defer input_ids.deinit();
    @memcpy(input_ids.data, token_ids.items);

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
    errdefer input_embeds.deinit();
    const embed_end = std.time.nanoTimestamp();
    printTiming(embed_start, embed_end, "Initial embedding preparation");

    // Initialize KV cache
    var kv_cache = try KVCache.init(allocator, config.n_layers, config.n_heads, config.head_dim);
    defer kv_cache.deinit();

    var pos: usize = 0;
    var output_buffer = std.ArrayList(u8).init(allocator);
    defer output_buffer.deinit();

    print("\nGenerated text: ", .{});

    // Generation loop with timing for each token
    const generation_start = std.time.nanoTimestamp();
    var total_inference_time: i128 = 0;
    var total_sampling_time: i128 = 0;
    var total_encoding_time: i128 = 0;

    for (0..max_tokens) |token_idx| {
        const token_start = std.time.nanoTimestamp();

        // Run the model
        const inference_start = std.time.nanoTimestamp();
        var result = try text_model.text_decoder(input_embeds, &kv_cache);
        const inference_end = std.time.nanoTimestamp();
        total_inference_time += inference_end - inference_start;

        defer {
            result.output.deinit();
            result.cache.deinit();
        }

        // Sampling
        const sampling_start = std.time.nanoTimestamp();
        var lm_result = try text_model.lm_head(result.output);
        defer lm_result.deinit();
        const next_token_id = try ops.sample(f16, &lm_result, sampling_config, random, allocator);
        const sampling_end = std.time.nanoTimestamp();
        total_sampling_time += sampling_end - sampling_start;

        if (next_token_id == tokenizer.eos_token) {
            print("eos token found\n", .{});
            break;
        }

        // Token encoding and embedding
        const encoding_start = std.time.nanoTimestamp();
        var next_token_tensor = try Tensor(u32).init(allocator, &[_]usize{1});
        defer next_token_tensor.deinit();
        next_token_tensor.data[0] = @intCast(next_token_id);

        var new_embeds = try text_model.text_encoder(next_token_tensor);
        errdefer new_embeds.deinit();
        input_embeds.deinit();
        input_embeds = new_embeds;
        const encoding_end = std.time.nanoTimestamp();
        total_encoding_time += encoding_end - encoding_start;

        // Token output
        var next_token_list = std.ArrayList(u32).init(allocator);
        defer next_token_list.deinit();
        try next_token_list.append(@intCast(next_token_id));

        const decoded = try tokenizer.decode(next_token_list);
        defer allocator.free(decoded);

        try output_buffer.appendSlice(decoded);
        try std.io.getStdOut().writer().writeAll(decoded);

        const token_end = std.time.nanoTimestamp();
        var token_label_buf: [32]u8 = undefined;
        const token_label = try std.fmt.bufPrint(&token_label_buf, "Token {d} generation", .{token_idx});
        printTiming(token_start, token_end, token_label);

        pos += 1;
    }

    const generation_end = std.time.nanoTimestamp();

    try std.io.getStdOut().writer().writeByte('\n');
    print("\nGeneration Statistics:\n", .{});
    printTiming(generation_start, generation_end, "Total generation time");
    print("Average per token:\n", .{});
    if (pos > 0) {
        const avg_inference = @as(f64, @floatFromInt(total_inference_time)) / (@as(f64, @floatFromInt(pos)) * 1_000_000.0);
        const avg_sampling = @as(f64, @floatFromInt(total_sampling_time)) / (@as(f64, @floatFromInt(pos)) * 1_000_000.0);
        const avg_encoding = @as(f64, @floatFromInt(total_encoding_time)) / (@as(f64, @floatFromInt(pos)) * 1_000_000.0);
        print("  Inference: {d:.2}ms\n", .{avg_inference});
        print("  Sampling: {d:.2}ms\n", .{avg_sampling});
        print("  Encoding: {d:.2}ms\n", .{avg_encoding});
    }

    const total_end = std.time.nanoTimestamp();
    printTiming(total_start, total_end, "\nTotal execution time");
}
