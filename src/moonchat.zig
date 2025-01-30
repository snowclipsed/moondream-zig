const std = @import("std");
const Config = @import("config.zig").Config;
const model_config = @import("config.zig").MODEL_CONFIG;
const Weights = @import("weights.zig").Weights;
const Tokenizer = @import("tokenizer.zig").Tokenizer;
const VisionModel = @import("vision_model_new.zig").VisionModel;
const TextModel = @import("text_model_new.zig").TextModel;
const KVCache = @import("text_model_new.zig").KVCache;
const Tensor = @import("tensor.zig").Tensor;
const Slice = @import("tensor.zig").Slice;
const ops = @import("ops.zig");
const displayImage = @import("imagedisplay.zig").displayImage;

// ANSI Color codes
const HEADER_ART =
    \\███╗   ███╗ ██████╗  ██████╗ ███╗   ██╗██████╗ ██████╗ ███████╗ █████╗ ███╗   ███╗ ▪    ███████╗██╗ ██████╗ 
    \\████╗ ████║██╔═══██╗██╔═══██╗████╗  ██║██╔══██╗██╔══██╗██╔════╝██╔══██╗████╗ ████║ ▪    ╚══███╔╝██║██╔════╝ 
    \\██╔████╔██║██║   ██║██║   ██║██╔██╗ ██║██║  ██║██████╔╝█████╗  ███████║██╔████╔██║ ▪      ███╔╝ ██║██║  ███╗
    \\██║╚██╔╝██║██║   ██║██║   ██║██║╚██╗██║██║  ██║██╔══██╗██╔══╝  ██╔══██║██║╚██╔╝██║ ▪    ███╔╝   ██║██║   ██║
    \\██║ ╚═╝ ██║╚██████╔╝╚██████╔╝██║ ╚████║██████╔╝██║  ██║███████╗██║  ██║██║ ╚═╝ ██║ ▪    ███████╗██║╚██████╔╝
    \\╚═╝     ╚═╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝ ▪    ╚══════╝╚═╝ ╚═════╝ •
;

const main_color = "\x1b[95m";
const reset_color = "\x1b[0m";
const time_color = "\x1b[94m";
const question_color = "\x1b[96m";
const answer_color = "\x1b[92m";
const stat_color = "\x1b[93m";
const prompt_color = "\x1b[97m";
const error_color = "\x1b[91m";

// Helper function to display available commands
fn displayCommands(writer: anytype) !void {
    try writer.print("{s}Available Commands:{s}\n", .{ main_color, reset_color });
    try writer.writeAll("  /chat <image-path> - Start new chat with an image\n");
    try writer.writeAll("  /newchat - Start fresh chat (new image)\n");
    try writer.writeAll("  /clear - Clear current chat history\n");
    try writer.writeAll("  /help - Display this help message\n");
    try writer.writeAll("  /sampler - Show sampling method options\n");
    try writer.writeAll("  /sampler <method> - Set sampling method (greedy/multinomial/top_k)\n");
    try writer.writeAll("  /maxlen <n> - Set maximum response length (default: 1024)\n");
    try writer.writeAll("  /exit - Exit the program\n");
}

fn displaySamplerOptions(writer: anytype) !void {
    try writer.print("{s}Available sampling methods:{s}\n", .{ main_color, reset_color });
    try writer.writeAll("  greedy - Always select the most likely token (default)\n");
    try writer.writeAll("  multinomial - Sample from full distribution (temperature=1.0)\n");
    try writer.writeAll("  top_k - Sample from the top K most likely tokens (k=40)\n\n");
    try writer.writeAll("Usage: /sampler <method>\n");
    try writer.writeAll("Example: /sampler top_k\n");
}

fn getCurrentTimestamp() []const u8 {
    const current_time = std.time.timestamp();
    const seconds_since_midnight = @mod(current_time, 86400);
    const hours = @divTrunc(seconds_since_midnight, 3600);
    const minutes = @divTrunc(@mod(seconds_since_midnight, 3600), 60);
    const seconds = @mod(seconds_since_midnight, 60);

    var buf: [32]u8 = undefined;
    const timestamp = std.fmt.bufPrint(&buf, "[{d:0>2}:{d:0>2}:{d:0>2}]", .{ hours, minutes, seconds }) catch return "[]";
    return buf[0..timestamp.len];
}

const ChatState = struct {
    vision_model: *VisionModel(model_config),
    text_model: *TextModel(model_config),
    tokenizer: *Tokenizer,
    weights: *Weights,
    image_tensor: *Tensor(f16),
    kv_cache: *KVCache(model_config),
    allocator: std.mem.Allocator,
    rng: std.rand.DefaultPrng,
    current_image_path: ?[]u8,
    sampling_config: ops.SamplingConfig,
    max_length: usize,

    // Add method to change sampling configuration
    pub fn setSamplingMethod(self: *ChatState, method: []const u8) !void {
        if (std.mem.eql(u8, method, "greedy")) {
            self.sampling_config = .{ .method = .greedy };
        } else if (std.mem.eql(u8, method, "multinomial")) {
            self.sampling_config = .{
                .method = .multinomial,
                .temperature = 1.0,
            };
        } else if (std.mem.eql(u8, method, "top_k")) {
            self.sampling_config = .{
                .method = .top_k,
                .temperature = 1.0,
                .top_k = 40, // You can adjust this default value
            };
        } else {
            return error.InvalidSamplingMethod;
        }
    }

    pub fn getCurrentSamplingMethod(self: *const ChatState) []const u8 {
        return switch (self.sampling_config.method) {
            .greedy => "greedy",
            .multinomial => "multinomial",
            .top_k => "top_k",
        };
    }

    pub fn init(
        allocator: std.mem.Allocator,
        vision_model: *VisionModel(model_config),
        text_model: *TextModel(model_config),
        tokenizer: *Tokenizer,
        weights: *Weights,
    ) !*ChatState {
        const state = try allocator.create(ChatState);
        errdefer allocator.destroy(state);

        // Initialize RNG
        const seed: u64 = @as(u64, @truncate(@as(u128, @bitCast(std.time.nanoTimestamp()))));
        state.rng = std.rand.DefaultPrng.init(seed);

        // Store references to existing models and weights
        state.vision_model = vision_model;
        state.text_model = text_model;
        state.tokenizer = tokenizer;
        state.weights = weights;

        // Initialize KV cache
        const MyKVCache = KVCache(model_config);
        state.kv_cache = try allocator.create(MyKVCache);
        errdefer allocator.destroy(state.kv_cache);
        state.kv_cache.* = try MyKVCache.init(allocator);
        errdefer state.kv_cache.deinit();

        // Initialize image tensor
        state.image_tensor = try allocator.create(Tensor(f16));
        state.image_tensor.* = try Tensor(f16).init(allocator, &[_]usize{ 1, model_config.dim });

        state.current_image_path = null;
        state.sampling_config = .{ .method = .greedy }; // Default to greedy sampling
        state.max_length = 1024; // Default max length
        state.allocator = allocator;
        return state;
    }

    pub fn clearChat(self: *ChatState) !void {
        const stdout = std.io.getStdOut().writer();

        // Clear screen
        try stdout.writeAll("\x1B[2J\x1B[H");

        // Redisplay the current image if we have one
        if (self.current_image_path) |path| {
            try displayImage(self.allocator, path, 0.75);
            std.debug.print("\n", .{});
        }
    }

    pub fn loadImage(self: *ChatState, image_path: []const u8) !void {
        // Store image path
        if (self.current_image_path) |old_path| {
            self.allocator.free(old_path);
        }
        self.current_image_path = try self.allocator.dupe(u8, image_path);

        try displayImage(self.allocator, image_path, 0.75);
        std.debug.print("\n", .{});

        // Clean up old tensor's data
        self.image_tensor.deinit();

        // Encode new image directly into the existing tenso
        self.image_tensor.* = try self.vision_model.encode_image(image_path);
    }

    pub fn processTurn(self: *ChatState, prompt: []const u8) !void {
        const stdout = std.io.getStdOut().writer();
        const timestamp = getCurrentTimestamp();

        // Format prompt with timestamp
        try stdout.print("\n{s}{s}{s} {s}Question:{s} {s}\n", .{ time_color, timestamp, reset_color, question_color, reset_color, prompt });
        try stdout.print("{s}{s}{s} {s}Answer:{s} ", .{ time_color, timestamp, reset_color, answer_color, reset_color });

        var token_count: usize = 0;
        var start_time: i64 = undefined;
        var timing_started = false;

        // ... [existing token generation code]
        var prompt_buf = std.ArrayList(u8).init(self.allocator);
        defer prompt_buf.deinit();
        try prompt_buf.writer().print("\n\nQuestion: {s}\n\nAnswer:", .{prompt});

        var encoded = try self.tokenizer.encode(prompt_buf.items);
        defer encoded.deinit();

        var token_ids = std.ArrayList(u32).init(self.allocator);
        defer token_ids.deinit();
        try token_ids.append(50256);
        try token_ids.appendSlice(encoded.items);

        var input_ids = try Tensor(u32).init(self.allocator, &[_]usize{token_ids.items.len});
        defer input_ids.deinit();
        @memcpy(input_ids.data, token_ids.items);

        var text_embeds = try self.text_model.text_encoder(input_ids);
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

        var leading_embed = try ops.concat(f16, first_slice, self.image_tensor.*, 0);
        defer leading_embed.deinit();
        var input_embeds = try ops.concat(f16, leading_embed, last_slice, 0);
        defer input_embeds.deinit();

        var pos: usize = 0;
        var next_token_tensor = try Tensor(u32).init(self.allocator, &[_]usize{1});
        defer next_token_tensor.deinit();

        var next_token_list = std.ArrayList(u32).init(self.allocator);
        defer next_token_list.deinit();
        try next_token_list.ensureTotalCapacity(1);

        while (pos < self.max_length) : (pos += 1) {
            var result = try self.text_model.text_decoder(input_embeds, self.kv_cache);
            defer result.output.deinit();
            defer result.cache.deinit();

            var logits = try self.text_model.lm_head(result.output);
            defer logits.deinit();
            const next_token_id = try ops.sample(f16, &logits, self.sampling_config, self.rng.random(), self.allocator);

            if (next_token_id == self.tokenizer.eos_token) break;

            next_token_tensor.data[0] = @intCast(next_token_id);
            const new_embeds = try self.text_model.text_encoder(next_token_tensor);
            input_embeds.deinit();
            input_embeds = new_embeds;

            next_token_list.clearRetainingCapacity();
            try next_token_list.append(@intCast(next_token_id));

            const decoded = try self.tokenizer.decode(next_token_list);
            defer self.allocator.free(decoded);
            if (!timing_started) {
                start_time = @intCast(std.time.nanoTimestamp());
                timing_started = true;
            }
            try stdout.writeAll(decoded);
            token_count += 1;
        }

        // Calculate and display stats
        const end_time = std.time.nanoTimestamp();
        const elapsed_ms = @as(f64, @floatFromInt(end_time - start_time)) / 1e6;
        const tokens_per_second = @as(f64, @floatFromInt(token_count)) / (elapsed_ms / 1000.0);

        try stdout.print("\n{s}[{d:.1} tok/s]{s}\n\n", .{ stat_color, tokens_per_second, reset_color });

        // Reset KV cache
        self.kv_cache.deinit();
        const MyKVCache = KVCache(model_config);
        self.kv_cache.* = try MyKVCache.init(self.allocator);
    }

    pub fn deinit(self: *ChatState) void {
        // Only free resources we own
        if (self.current_image_path) |path| {
            self.allocator.free(path);
        }
        self.kv_cache.deinit();
        self.allocator.destroy(self.kv_cache);
        self.image_tensor.deinit();
        self.allocator.destroy(self.image_tensor);
        self.allocator.destroy(self);
    }
};

pub fn main() !void {
    // Initialize allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const stdout = std.io.getStdOut().writer();

    // Display header art
    try stdout.print("{s}{s}{s}\n\n", .{ main_color, HEADER_ART, reset_color });
    try stdout.writeAll("Welcome to the Chat Interface!\n");
    try displayCommands(stdout);
    try stdout.writeAll("\n");

    // Load model and tokenizer first - these stay loaded
    var tokenizer = try allocator.create(Tokenizer);
    defer {
        tokenizer.deinit();
        allocator.destroy(tokenizer);
    }
    tokenizer.* = try Tokenizer.fromFile("../tokenizer.bin", allocator);

    var weights = try allocator.create(Weights);
    defer {
        weights.deinit();
        allocator.destroy(weights);
    }
    weights.* = try Weights.init(model_config, "../moondream.bin", allocator);

    const VisionModelType = VisionModel(model_config);
    var vision_model = try allocator.create(VisionModelType);
    defer {
        vision_model.deinit();
        allocator.destroy(vision_model);
    }
    vision_model.* = try VisionModelType.init(weights.*, allocator);

    const TextModelType = TextModel(model_config);
    var text_model = try allocator.create(TextModelType);
    defer {
        text_model.deinit();
        allocator.destroy(text_model);
    }
    text_model.* = try TextModelType.init(weights.*, allocator);

    var chat_state: ?*ChatState = null;
    defer if (chat_state != null) chat_state.?.deinit();

    var buffer: [1024]u8 = undefined;
    const stdin = std.io.getStdIn().reader();

    while (true) {
        try stdout.print("{s}>{s} ", .{ prompt_color, reset_color });
        const input = try stdin.readUntilDelimiter(&buffer, '\n');

        if (std.mem.startsWith(u8, input, "/exit")) {
            break;
        } else if (std.mem.startsWith(u8, input, "/chat ")) {
            const image_path = std.mem.trim(u8, input[6..], " ");

            // Initialize new chat state if needed
            if (chat_state == null) {
                chat_state = try ChatState.init(allocator, vision_model, text_model, tokenizer, weights);
            }

            try chat_state.?.loadImage(image_path);
            try stdout.writeAll("Image loaded! Enter your prompt:\n");
        } else if (std.mem.eql(u8, input, "/newchat")) {
            if (chat_state != null) {
                chat_state.?.deinit();
                chat_state = null;
            }
            try stdout.writeAll("Start a new chat with /chat <image-path>\n");
        } else if (std.mem.eql(u8, input, "/clear")) {
            if (chat_state != null) {
                try chat_state.?.clearChat();
                try stdout.writeAll("Chat history cleared. Continue chatting about the current image.\n");
            } else {
                try stdout.writeAll("No active chat to clear. Start a chat first with /chat <image-path>\n");
            }
        } else if (std.mem.eql(u8, input, "/help")) {
            try displayCommands(stdout);
        } else if (std.mem.startsWith(u8, input, "/sampler")) {
            if (chat_state == null) {
                try stdout.writeAll("Please start a chat first with /chat <image-path>\n");
                continue;
            }

            const args = std.mem.trim(u8, input[8..], " ");
            if (args.len == 0) {
                // Display current sampling method and options
                try stdout.print("Current sampling method: {s}\n\n", .{chat_state.?.getCurrentSamplingMethod()});
                try displaySamplerOptions(stdout);
            } else {
                // Try to set new sampling method
                chat_state.?.setSamplingMethod(args) catch |err| {
                    if (err == error.InvalidSamplingMethod) {
                        try stdout.writeAll("Invalid sampling method. ");
                        try displaySamplerOptions(stdout);
                    } else {
                        return err;
                    }
                    continue;
                };
                try stdout.print("Sampling method set to: {s}\n", .{args});
            }
        } else if (chat_state != null) {
            try chat_state.?.processTurn(input);
        } else {
            try stdout.writeAll("Please start a chat first with /chat <image-path>\n");
        }
    }
}
