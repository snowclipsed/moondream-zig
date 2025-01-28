const std = @import("std");

const Config = @import("config.zig").Config;
const model_config = @import("config.zig").MODEL_CONFIG;
const ConfigReader = @import("config.zig").ConfigReader;
const Weights = @import("weights.zig").Weights;
const Tokenizer = @import("tokenizer.zig").Tokenizer;
const VisionModel = @import("vision_model_new.zig").VisionModel;
const TextModel = @import("text_model_new.zig").TextModel;
const KVCache = @import("text_model_new.zig").KVCache;

const Tensor = @import("tensor.zig").Tensor;
const Slice = @import("tensor.zig").Slice;
const ops = @import("ops.zig");

const displayImage = @import("imagedisplay.zig").displayImage;
const print = std.debug.print;
const Timer = std.time.Timer;

// CONSTANTS //

const ENABLE_STREAMING = false; // Comptime flag to control token streaming

const HEADER_ART =
    \\███╗   ███╗ ██████╗  ██████╗ ███╗   ██╗██████╗ ██████╗ ███████╗ █████╗ ███╗   ███╗ ▪    ███████╗██╗ ██████╗ 
    \\████╗ ████║██╔═══██╗██╔═══██╗████╗  ██║██╔══██╗██╔══██╗██╔════╝██╔══██╗████╗ ████║ ▪    ╚══███╔╝██║██╔════╝ 
    \\██╔████╔██║██║   ██║██║   ██║██╔██╗ ██║██║  ██║██████╔╝█████╗  ███████║██╔████╔██║ ▪      ███╔╝ ██║██║  ███╗
    \\██║╚██╔╝██║██║   ██║██║   ██║██║╚██╗██║██║  ██║██╔══██╗██╔══╝  ██╔══██║██║╚██╔╝██║ ▪    ███╔╝   ██║██║   ██║
    \\██║ ╚═╝ ██║╚██████╔╝╚██████╔╝██║ ╚████║██████╔╝██║  ██║███████╗██║  ██║██║ ╚═╝ ██║ ▪    ███████╗██║╚██████╔╝
    \\╚═╝     ╚═╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝ ▪    ╚══════╝╚═╝ ╚═════╝ •
;

// Default paths and configurations
const DEFAULT_MODEL_PATH = "../moondream.bin";
const DEFAULT_TOKENIZER_PATH = "../tokenizer.bin";
const DEFAULT_IMAGE_PATH = "../images/demo-1.jpg";
const DEFAULT_PROMPT = "describe the image";
const DEFAULT_MAX_TOKENS: usize = 200;

const Args = struct {
    model_path: []const u8,
    tokenizer_path: []const u8,
    image_path: []const u8,
    prompt: []const u8,
    max_tokens: usize,
    show_stats: bool,
    sampling_method: []const u8,
    temperature: f32,
    top_k: usize,
    show_header: bool,
};

fn parseArgs(allocator: std.mem.Allocator) !Args {
    var args = Args{
        .model_path = DEFAULT_MODEL_PATH,
        .tokenizer_path = DEFAULT_TOKENIZER_PATH,
        .image_path = DEFAULT_IMAGE_PATH,
        .prompt = DEFAULT_PROMPT,
        .max_tokens = DEFAULT_MAX_TOKENS,
        .show_stats = true,
        .sampling_method = "greedy",
        .temperature = 0.5,
        .top_k = 3,
        .show_header = true,
    };

    var arg_it = try std.process.argsWithAllocator(allocator);
    defer arg_it.deinit();

    // Skip executable name
    _ = arg_it.skip();

    while (arg_it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--model")) {
            args.model_path = arg_it.next() orelse return error.MissingValue;
        } else if (std.mem.eql(u8, arg, "--tokenizer")) {
            args.tokenizer_path = arg_it.next() orelse return error.MissingValue;
        } else if (std.mem.eql(u8, arg, "--image")) {
            args.image_path = arg_it.next() orelse return error.MissingValue;
        } else if (std.mem.eql(u8, arg, "--prompt")) {
            args.prompt = arg_it.next() orelse return error.MissingValue;
        } else if (std.mem.eql(u8, arg, "--max-tokens")) {
            const tokens_str = arg_it.next() orelse return error.MissingValue;
            args.max_tokens = try std.fmt.parseInt(usize, tokens_str, 10);
        } else if (std.mem.eql(u8, arg, "--sampling")) {
            args.sampling_method = arg_it.next() orelse return error.MissingValue;
            if (!std.mem.eql(u8, args.sampling_method, "greedy") and !std.mem.eql(u8, args.sampling_method, "topk")) {
                return error.InvalidSamplingMethod;
            }
        } else if (std.mem.eql(u8, arg, "--temperature")) {
            const temp_str = arg_it.next() orelse return error.MissingValue;
            args.temperature = try std.fmt.parseFloat(f32, temp_str);
            if (args.temperature <= 0.0 or args.temperature > 2.0) {
                return error.InvalidTemperature;
            }
        } else if (std.mem.eql(u8, arg, "--top-k")) {
            const k_str = arg_it.next() orelse return error.MissingValue;
            args.top_k = try std.fmt.parseInt(usize, k_str, 10);
            if (args.top_k < 1) {
                return error.InvalidTopK;
            }
        } else if (std.mem.eql(u8, arg, "--noheader")) {
            args.show_header = false;
        } else if (std.mem.eql(u8, arg, "--stats")) {
            args.show_stats = true;
        } else if (std.mem.eql(u8, arg, "--help")) {
            try printUsage();
            std.process.exit(0);
        }
    }

    return args;
}

fn printUsage() !void {
    const stdout = std.io.getStdOut().writer();
    try stdout.writeAll(
        \\Usage: moondream [options]
        \\
        \\Options:
        \\  --model <path>       Path to model weights (default: ../moondream.bin)
        \\  --tokenizer <path>   Path to tokenizer file (default: ../tokenizer.bin)
        \\  --image <path>       Path to input image (default: ../images/demo-1.jpg)
        \\  --prompt <text>      Prompt for the model (default: "describe the image")
        \\  --max-tokens <num>   Maximum number of tokens to generate (default: 200)
        \\  --sampling <method>  Sampling method: 'greedy' or 'topk' (default: greedy)
        \\  --temperature <val> Temperature for top-k sampling (0-2.0, default: 0.5)
        \\  --top-k <num>       Top-k value for sampling (default: 3)
        \\  --noheader          Disable ASCII header display
        \\  --stats             Enable timing statistics output (default: disabled)
        \\  --help              Show this help message
        \\
    );
}

fn writeToken(stdout: anytype, token: []const u8) !void {
    if (ENABLE_STREAMING) {
        try stdout.writeAll(token);
    }
}

const mode = std.builtin.FloatMode.optimized;
comptime {
    @setFloatMode(mode);
}

const main_color = "\x1b[95m";
const reset_color = "\x1b[0m";
const time_color = "\x1b[94m";

fn printTimeDiff(start_time: i128, end_time: i128, step_name: []const u8, show_stats: bool) !void {
    if (!show_stats) return;

    const stdout = std.io.getStdOut().writer();
    const diff_ns = end_time - start_time;
    const diff_ms = @as(f64, @floatFromInt(diff_ns)) / 1_000_000.0;
    try stdout.print("{s}\n[STATS] {s}: {d:.2}ms{s}\n", .{
        time_color, step_name, diff_ms, reset_color,
    });
}

pub fn main() !void {
    const stdout = std.io.getStdOut().writer();
    var timer = try Timer.start();
    const total_start = timer.read();

    // Initialize allocator
    const init_start = timer.read();
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Parse command line arguments
    const args = try parseArgs(allocator);
    try stdout.writeAll("\n");
    try stdout.print("{s}", .{main_color});
    if (args.show_header) {
        try stdout.writeAll(HEADER_ART);
    }
    try stdout.writeAll("\n");
    try stdout.print("{s}", .{reset_color});

    // Initialize RNG
    const seed: u64 = @as(u64, @truncate(@as(u128, @bitCast(std.time.nanoTimestamp()))));
    var rng = std.rand.DefaultPrng.init(seed);
    const random = rng.random();
    try printTimeDiff(init_start, timer.read(), "Initialization", args.show_stats);

    // Configure sampling based on arguments
    const sampling_config = if (std.mem.eql(u8, args.sampling_method, "greedy"))
        ops.SamplingConfig{ .method = .greedy }
    else
        ops.SamplingConfig{
            .method = .top_k,
            .top_k = args.top_k,
            .temperature = args.temperature,
        };

    // Load tokenizer
    const tokenizer_start = timer.read();
    var tokenizer = try Tokenizer.fromFile(args.tokenizer_path, allocator);
    defer tokenizer.deinit();
    try printTimeDiff(tokenizer_start, timer.read(), "Tokenizer Loading", args.show_stats);

    const config = model_config;

    // Load model weights
    const weights_start = timer.read();
    var weights = try Weights.init(config, args.model_path, allocator);
    defer weights.deinit();
    try printTimeDiff(weights_start, timer.read(), "Model Weights Loading", args.show_stats);

    std.debug.print("\n\n", .{});
    const scale = 0.75;

    // Display and process image
    const image_start = timer.read();
    try displayImage(allocator, args.image_path, scale);
    std.debug.print("\n \n", .{});

    try printTimeDiff(image_start, timer.read(), "Image Display", args.show_stats);

    // Initialize vision model and encode image
    const modelinit_start = timer.read();
    const VisionModelType = VisionModel(model_config);
    var vision_model = try VisionModelType.init(weights, allocator);
    defer vision_model.deinit();
    const TextModelType = TextModel(model_config);
    var text_model = try TextModelType.init(weights, allocator);
    defer text_model.deinit();

    try printTimeDiff(modelinit_start, timer.read(), "Models Init", args.show_stats);

    const image_encode_start = timer.read();
    var image_tensor = try vision_model.encode_image(args.image_path);
    defer image_tensor.deinit();
    try printTimeDiff(image_encode_start, timer.read(), "Image Encoding", args.show_stats);

    // Initialize tokens and create input tensor
    const token_init_start = timer.read();

    // Format the prompt with Question/Answer format
    var prompt_buf = std.ArrayList(u8).init(allocator);
    defer prompt_buf.deinit();
    try prompt_buf.writer().print("\n\nQuestion: {s}\n\nAnswer:", .{args.prompt});

    // Encode the formatted prompt
    var encoded = try tokenizer.encode(prompt_buf.items);
    defer encoded.deinit();

    // Initialize token IDs list with special token and encoded prompt
    var token_ids = std.ArrayList(u32).init(allocator);
    defer token_ids.deinit();
    try token_ids.append(50256);
    try token_ids.appendSlice(encoded.items);

    // Create input tensor from token IDs
    var input_ids = try Tensor(u32).init(allocator, &[_]usize{token_ids.items.len});
    defer input_ids.deinit();
    @memcpy(input_ids.data, token_ids.items);

    try printTimeDiff(token_init_start, timer.read(), "Token Initialization", args.show_stats);

    // Generation loop
    var pos: usize = 0;
    var output_buffer = std.ArrayList(u8).init(allocator);
    defer output_buffer.deinit();

    try stdout.print("{s}{s}{s}", .{ main_color, prompt_buf.items, reset_color });

    // Pre-allocate buffers that will be reused in the loop
    var next_token_tensor = try Tensor(u32).init(allocator, &[_]usize{1});
    defer next_token_tensor.deinit();

    var next_token_list = std.ArrayList(u32).init(allocator);
    defer next_token_list.deinit();
    try next_token_list.ensureTotalCapacity(1);

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
    try printTimeDiff(embedding_start, timer.read(), "Embedding Processing", args.show_stats);

    // Initialize KV cache
    const cache_start = timer.read();
    const MyKVCache = KVCache(config);
    var kv_cache = try MyKVCache.init(allocator);
    defer kv_cache.deinit();
    try printTimeDiff(cache_start, timer.read(), "KV Cache Initialization", args.show_stats);

    // Generation statistics
    var total_decoder_time: i128 = 0;
    var total_sampling_time: i128 = 0;
    var total_embedding_time: i128 = 0;
    var token_count: usize = 0;
    var first_token_time: i128 = undefined;
    var post_first_token_start: i128 = undefined;
    var is_first_token = true;

    for (0..args.max_tokens) |_| {
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
        next_token_tensor.data[0] = @intCast(next_token_id);

        var new_embeds = try text_model.text_encoder(next_token_tensor);
        errdefer new_embeds.deinit();
        input_embeds.deinit();
        input_embeds = new_embeds;
        total_embedding_time += timer.read() - embed_start;

        next_token_list.clearRetainingCapacity();
        try next_token_list.append(@intCast(next_token_id));

        const decoded = try tokenizer.decode(next_token_list);
        defer allocator.free(decoded);

        try output_buffer.appendSlice(decoded);
        try writeToken(stdout, decoded);

        if (is_first_token) {
            first_token_time = timer.read() - image_encode_start;
            post_first_token_start = timer.read();
            is_first_token = false;
        }

        pos += 1;
        token_count += 1;
    }

    if (!ENABLE_STREAMING) {
        try stdout.writeAll(output_buffer.items);
    }
    try stdout.writeByte('\n');

    // Print statistics if enabled
    if (args.show_stats) {
        try stdout.print("\n{s}Generation Statistics:{s}\n", .{ time_color, reset_color });
        try stdout.print("{s}Total tokens generated: {d}{s}\n", .{ time_color, token_count, reset_color });

        try stdout.print("{s}Time to first token: {d}s{s}\n", .{
            time_color,
            @as(f64, @floatFromInt(first_token_time)) / 1_000_000_000.0,
            reset_color,
        });

        if (token_count > 1) {
            const remaining_tokens = token_count - 1;
            const remaining_time = timer.read() - post_first_token_start;
            const tokens_per_second = @as(f64, @floatFromInt(remaining_tokens)) /
                (@as(f64, @floatFromInt(remaining_time)) / 1_000_000_000.0);
            try stdout.print("{s}Tokens per second (after first token): {d:.2} Tok/s{s}\n", .{ time_color, tokens_per_second, reset_color });
        }

        try stdout.print("{s}Average decoder time per token:{s} {d:.4}s\n", .{
            time_color,
            reset_color,
            @as(f64, @floatFromInt(total_decoder_time)) / @as(f64, @floatFromInt(token_count)) / 1_000_000_000.0,
        });
        try stdout.print("{s}Average sampling time per token:{s} {d:.4}s\n", .{
            time_color,
            reset_color,
            @as(f64, @floatFromInt(total_sampling_time)) / @as(f64, @floatFromInt(token_count)) / 1_000_000_000.0,
        });
        try stdout.print("{s}Average embedding time per token:{s} {d:.4}s\n", .{
            time_color,
            reset_color,
            @as(f64, @floatFromInt(total_embedding_time)) / @as(f64, @floatFromInt(token_count)) / 1_000_000_000.0,
        });

        // Print total execution time
        try printTimeDiff(total_start, timer.read(), "Total Execution", true);
    }
}
