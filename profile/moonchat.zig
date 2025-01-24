const std = @import("std");
const Config = @import("config.zig").Config;
const ConfigReader = @import("config.zig").ConfigReader;
const Weights = @import("weights.zig").Weights;
const Tensor = @import("tensor.zig").Tensor;
const Slice = @import("tensor.zig").Slice;
const ops = @import("ops.zig");
const TextModel = @import("text_model_new.zig").TextModel;
const VisionModel = @import("vision_model_new.zig").VisionModel;
const KVCache = @import("text_model_new.zig").KVCache;
const LayerCache = @import("text_model_new.zig").LayerCache;
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

fn readUserInput(allocator: std.mem.Allocator) ![]const u8 {
    const stdin = std.io.getStdIn().reader();
    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();
    try stdin.streamUntilDelimiter(buffer.writer(), '\n', null);
    return buffer.toOwnedSlice();
}
fn chatLoop(
    allocator: std.mem.Allocator,
    text_model: *TextModel,
    vision_model: *VisionModel,
    tokenizer: *Tokenizer,
    config: Config,
    sampling_config: ops.SamplingConfig,
    random: std.rand.Random,
) !void {
    const stdout = std.io.getStdOut().writer();
    const stdin = std.io.getStdIn().reader();
    var timer = try std.time.Timer.start();

    // Buffer for user input
    var input_buffer: [1024]u8 = undefined;

    mainLoop: while (true) {
        try stdout.writeAll("\nEnter command (/chat or /exit): ");

        const user_input = try stdin.readUntilDelimiter(&input_buffer, '\n');
        const trimmed = std.mem.trim(u8, user_input, &std.ascii.whitespace);

        if (std.mem.eql(u8, trimmed, "/exit")) {
            break :mainLoop;
        } else if (std.mem.eql(u8, trimmed, "/chat")) {
            // Each image session gets its own complete scope
            imageSession: while (true) {
                // Create new scope for each image to ensure complete cleanup
                var image_session = blk: {
                    try stdout.writeAll("\nEnter image path (or /exit to return to main menu): ");
                    const image_path = std.mem.trim(u8, try stdin.readUntilDelimiter(&input_buffer, '\n'), &std.ascii.whitespace);

                    if (std.mem.eql(u8, image_path, "/exit")) {
                        break :imageSession;
                    }

                    // Display image and compute image tensor
                    const image_start = timer.read();

                    // Verify image exists
                    const file = std.fs.openFileAbsolute(image_path, .{}) catch |err| {
                        try stdout.print("\n[ERROR] Cannot open image file: {any}\n", .{err});
                        continue :imageSession;
                    };
                    file.close();

                    try displayImage(allocator, image_path, 0.75);

                    // Encode image - this will be cleaned up at the end of the image session
                    var image_tensor = try vision_model.encode_image(image_path);
                    errdefer image_tensor.deinit();

                    try printTimeDiff(image_start, timer.read(), "Image Processing");

                    break :blk ImageSession{
                        .tensor = image_tensor,
                    };
                };
                // Ensure cleanup of image session resources
                defer image_session.tensor.deinit();

                // Initialize new KV cache for this image session
                var kv_cache = try KVCache.init(allocator, config.n_layers, config.n_heads, config.head_dim);
                defer kv_cache.deinit();

                // Chat loop for current image
                chatLoop: while (true) {
                    try stdout.writeAll("\nEnter your prompt (or /newimage for new image, /exit to return to main menu): ");
                    const prompt = std.mem.trim(u8, try stdin.readUntilDelimiter(&input_buffer, '\n'), &std.ascii.whitespace);

                    if (std.mem.eql(u8, prompt, "/exit")) {
                        break :imageSession;
                    } else if (std.mem.eql(u8, prompt, "/newimage")) {
                        break :chatLoop;
                    }

                    // Each conversation turn gets its own scope for cleanup
                    {
                        // Format prompt
                        var prompt_buf = std.ArrayList(u8).init(allocator);
                        defer prompt_buf.deinit();
                        try prompt_buf.writer().print("\n\nQuestion: {s}\n\nAnswer:", .{prompt});

                        // Reset KV cache for new conversation turn
                        kv_cache.reset();

                        // Encode the prompt
                        var encoded = try tokenizer.encode(prompt_buf.items);
                        defer encoded.deinit();

                        var token_ids = std.ArrayList(u32).init(allocator);
                        defer token_ids.deinit();
                        try token_ids.append(50256); // EOS token
                        try token_ids.appendSlice(encoded.items);

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

                        var leading_embed = try ops.concat(f16, first_slice, image_session.tensor, 0);
                        defer leading_embed.deinit();
                        var input_embeds = try ops.concat(f16, leading_embed, last_slice, 0);
                        defer input_embeds.deinit();

                        var output_buffer = std.ArrayList(u8).init(allocator);
                        defer output_buffer.deinit();

                        try stdout.writeAll("\nGenerated response: ");

                        // Generation loop
                        var pos: usize = 0;
                        const max_tokens: usize = 200;
                        for (0..max_tokens) |_| {
                            if (pos >= 2048) break;

                            var result = try text_model.text_decoder(input_embeds, &kv_cache);
                            defer result.output.deinit();
                            defer result.cache.deinit();

                            var logits = try text_model.lm_head(result.output);
                            defer logits.deinit();
                            const next_token_id = try ops.sample(f16, &logits, sampling_config, random, allocator);

                            if (next_token_id == tokenizer.eos_token) break;

                            var next_token_tensor = try Tensor(u32).init(allocator, &[_]usize{1});
                            defer next_token_tensor.deinit();
                            next_token_tensor.data[0] = @intCast(next_token_id);

                            var new_embeds = try text_model.text_encoder(next_token_tensor);
                            errdefer new_embeds.deinit();
                            input_embeds.deinit();
                            input_embeds = new_embeds;

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
                }
            }
        }
    }
}

const ImageSession = struct {
    tensor: Tensor(f16),
};

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

    // Print welcome message
    try stdout.writeAll("\nWelcome to Moondream Chat!\n");
    try stdout.writeAll("Type /chat to start a chat session or /exit to quit.\n");
    try stdout.writeAll("When in a chat session, you'll need to provide an image path first.\n");
    try stdout.writeAll("Then you can chat with the model about the image.\n\n");

    // Initialize vision model
    const vision_start = timer.read();
    var vision_model = try VisionModel.init(config, weights, allocator);
    defer vision_model.deinit();
    try printTimeDiff(vision_start, timer.read(), "Vision Model Init");

    // Initialize text model
    const text_model_start = timer.read();
    var text_model = try TextModel.init(config, weights, allocator);
    defer text_model.deinit();
    try printTimeDiff(text_model_start, timer.read(), "Text Model Init");

    // Start chat loop
    try chatLoop(allocator, &text_model, &vision_model, &tokenizer, config, sampling_config, random);

    // Print total execution time
    try printTimeDiff(total_start, timer.read(), "Total Execution");
}
