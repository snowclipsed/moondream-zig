const std = @import("std");
const Tensor = @import("core/tensor.zig").Tensor;
const Slice = @import("core/tensor.zig").Slice;
const ops = @import("core/ops.zig");

const Tokenizer = @import("preprocessing/tokenizer.zig").Tokenizer;

const Config = @import("/model/config.zig").Config;
const model_config = @import("model/config.zig").MODEL_CONFIG;
const Weights = @import("model/weights.zig").Weights;
const VisionModel = @import("model/vision_model.zig").VisionModel;
const TextModel = @import("model/text_model.zig").TextModel;
const KVCache = @import("model/text_model.zig").KVCache;

const sampling = @import("utils/sampling.zig");
const displayImage = @import("utils/image_display.zig").displayImage;

const HEADER_ART =
    \\███╗   ███╗ ██████╗  ██████╗ ███╗   ██╗██████╗ ██████╗ ███████╗ █████╗ ███╗   ███╗ ▪    ███████╗██╗ ██████╗ 
    \\████╗ ████║██╔═══██╗██╔═══██╗████╗  ██║██╔══██╗██╔══██╗██╔════╝██╔══██╗████╗ ████║ ▪    ╚══███╔╝██║██╔════╝ 
    \\██╔████╔██║██║   ██║██║   ██║██╔██╗ ██║██║  ██║██████╔╝█████╗  ███████║██╔████╔██║ ▪      ███╔╝ ██║██║  ███╗
    \\██║╚██╔╝██║██║   ██║██║   ██║██║╚██╗██║██║  ██║██╔══██╗██╔══╝  ██╔══██║██║╚██╔╝██║ ▪    ███╔╝   ██║██║   ██║
    \\██║ ╚═╝ ██║╚██████╔╝╚██████╔╝██║ ╚████║██████╔╝██║  ██║███████╗██║  ██║██║ ╚═╝ ██║ ▪    ███████╗██║╚██████╔╝
    \\╚═╝     ╚═╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝ ▪    ╚══════╝╚═╝ ╚═════╝ •
;

// ANSI Color Codes
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
    try writer.writeAll("  top_k - Sample from the top K most likely tokens (k=40)\n");
    try writer.writeAll("  min_p - Sample tokens above min-p * max_probability threshold (min_p=0.05)\n\n");
    try writer.writeAll("Usage: /sampler <method>\n");
    try writer.writeAll("Example: /sampler min_p\n");
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

const LoadingSpinner = struct {
    thread: ?std.Thread,
    stop_flag: std.atomic.Value(bool),
    message: []const u8,

    const frames = [_][]const u8{
        \\ /\___/\
        \\(> .. <)
        \\ (  w )~
        \\  u u
        ,
        \\ /\___/\
        \\(> -- <)
        \\ (  w )~
        \\  u u
        ,
        \\ /\___/\
        \\(> .. <)
        \\ (  w )~
        \\  u u
    };

    // Move up 5 lines from current position and clear to end of each line
    const CLEAR_LINES = "\x1B[5A\x1B[K\x1B[K\x1B[K\x1B[K\x1B[K";

    pub fn init(message: []const u8) LoadingSpinner {
        return .{
            .thread = null,
            .stop_flag = std.atomic.Value(bool).init(false),
            .message = message,
        };
    }

    fn spinnerThread(self: *LoadingSpinner) void {
        const stdout = std.io.getStdOut().writer();
        var frame: usize = 0;

        while (!self.stop_flag.load(.monotonic)) {
            // Clear previous frame
            stdout.writeAll(CLEAR_LINES) catch return;

            // Print current frame with message
            stdout.print("{s}\n{s}\n", .{
                frames[frame],
                self.message,
            }) catch return;

            frame = (frame + 1) % frames.len;
            std.time.sleep(200 * std.time.ns_per_ms);
        }

        // Clear the animation one final time
        stdout.writeAll(CLEAR_LINES) catch return;
    }

    pub fn start(self: *LoadingSpinner) !void {
        if (self.thread == null) {
            const stdout = std.io.getStdOut().writer();
            self.stop_flag.store(false, .monotonic);
            // Print initial newlines for space
            try stdout.writeAll("\n\n\n\n\n");
            self.thread = try std.Thread.spawn(.{}, spinnerThread, .{self});
        }
    }

    pub fn stop(self: *LoadingSpinner) void {
        if (self.thread) |thread| {
            self.stop_flag.store(true, .monotonic);
            thread.join();
            self.thread = null;
        }
    }
};

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
    sampling_config: sampling.SamplingConfig,
    max_length: usize,

    /// Sets the sampling method for the chat state.
    ///
    /// This function allows you to specify the sampling method to be used
    /// for the chat state. The method is provided as a string.
    ///
    /// # Parameters
    /// - `self`: A pointer to the `ChatState` instance.
    /// - `method`: A constant byte slice representing the sampling method.
    ///
    /// # Errors
    /// This function may return an error if the provided sampling method is invalid.
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
                .top_k = 40, // Default value
            };
        } else if (std.mem.eql(u8, method, "min_p")) {
            self.sampling_config = .{
                .method = .min_p,
                .temperature = 1.0,
                .min_p = 0.01, // Default value
            };
        } else {
            return error.InvalidSamplingMethod;
        }
    }

    /// Retrieves the current sampling method used in the chat state.
    ///
    /// This function returns a slice of constant unsigned 8-bit integers
    /// representing the name or identifier of the current sampling method.
    ///
    /// Parameters:
    /// - `self`: A constant pointer to the `ChatState` structure.
    ///
    /// Returns:
    /// - A slice of constant `u8` representing the current sampling method.
    pub fn getCurrentSamplingMethod(self: *const ChatState) []const u8 {
        return switch (self.sampling_config.method) {
            .greedy => "greedy",
            .multinomial => "multinomial",
            .top_k => "top_k",
            .min_p => "min_p",
        };
    }

    /// Initializes the chat state with the provided models, tokenizer, and weights.
    ///
    /// This function sets up the necessary components for the chat system to function,
    /// including the vision model, text model, tokenizer, and weights.
    ///
    /// Parameters:
    /// - `allocator`: The memory allocator to use for allocating resources.
    /// - `vision_model`: A pointer to the vision model configuration.
    /// - `text_model`: A pointer to the text model configuration.
    /// - `tokenizer`: A pointer to the tokenizer to use for processing text.
    /// - `weights`: A pointer to the weights to use for the models.
    ///
    /// Returns:
    /// - A pointer to the initialized `ChatState` on success.
    /// - An error if initialization fails.
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

    /// Clears the chat state.
    ///
    /// This function resets the chat state to its initial state, effectively
    /// clearing any messages or data stored in the chat.
    ///
    /// # Errors
    ///
    /// This function may return an error if the chat state cannot be cleared
    /// due to an underlying issue.
    ///
    /// # Parameters
    ///
    /// - `self`: A pointer to the `ChatState` instance that will be cleared.
    ///
    /// # Returns
    ///
    /// This function returns `!void`, indicating that it may return an error
    /// or nothing if the operation is successful.
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

    /// Loads an image from the specified file path into the chat state.
    ///
    /// This function takes a file path to an image and loads it into the
    /// chat state. It returns an error if the image cannot be loaded.
    ///
    /// # Parameters
    /// - `self`: A pointer to the `ChatState` instance.
    /// - `image_path`: A slice of constant unsigned 8-bit integers representing the file path to the image.
    ///
    /// # Errors
    /// This function returns an error if the image cannot be loaded.
    ///
    /// # Example
    /// ```
    /// const chat_state = ChatState{};
    /// try chat_state.loadImage("/path/to/image.png");
    /// ```
    pub fn loadImage(self: *ChatState, image_path: []const u8) !void {
        const stdout = std.io.getStdOut().writer();

        var spinner = LoadingSpinner.init("Loading image file...");
        try spinner.start();

        // Free previous image path if it exists
        if (self.current_image_path) |old_path| {
            self.allocator.free(old_path);
            self.current_image_path = null; // Clear it immediately to prevent double-free in error cases
        }

        // Validate and normalize the path
        const normalized_path = validateImagePath(self.allocator, image_path) catch |err| {
            spinner.stop();
            try stdout.print("{s}Error: Failed to validate image path: {s}{s}\n", .{ error_color, @errorName(err), reset_color });
            return err;
        };

        // Store the normalized path
        self.current_image_path = try self.allocator.dupe(u8, normalized_path);
        errdefer {
            self.allocator.free(self.current_image_path.?);
            self.current_image_path = null;
        }

        // Stop the spinner before display operations
        spinner.stop();

        // Try to display the image first - use normalized path consistently
        displayImage(self.allocator, normalized_path, 0.75) catch |err| {
            // Clean up the stored path on error
            if (self.current_image_path) |path| {
                self.allocator.free(path);
                self.current_image_path = null;
            }
            self.allocator.free(normalized_path); // Free the normalized path
            return err;
        };

        std.debug.print("\n\n\n", .{});

        // Start new spinner for vision processing
        var vision_spinner = LoadingSpinner.init("Processing image with vision model...");
        try vision_spinner.start();

        // Clean up old tensor's data
        self.image_tensor.deinit();

        // Initialize a placeholder/empty tensor to ensure we don't have a null pointer
        self.image_tensor.* = try Tensor(f16).init(self.allocator, &[_]usize{ 1, model_config.dim });

        // Try to encode the image - use normalized path consistently
        self.image_tensor.deinit(); // Free the placeholder
        self.image_tensor.* = self.vision_model.encode_image(normalized_path) catch |err| {
            // Stop spinner before showing error
            vision_spinner.stop();

            // Clear the lines the spinner was using
            try stdout.writeAll("\x1B[6A\x1B[J"); // Move up 6 lines and clear to end
            try stdout.writeAll("\x1B[0m"); // Reset colors

            // Clean up current image path since we failed to process it
            if (self.current_image_path) |path| {
                self.allocator.free(path);
                self.current_image_path = null;
            }

            self.allocator.free(normalized_path); // Free the normalized path

            // Reinitialize tensor with empty data
            self.image_tensor.* = try Tensor(f16).init(self.allocator, &[_]usize{ 1, model_config.dim });

            return err;
        };

        // Free the normalized path after we're done with it
        self.allocator.free(normalized_path);

        // Explicitly stop spinner and ensure complete cleanup
        vision_spinner.stop();
        // Move cursor up and clear multiple lines to ensure complete cleanup
        try stdout.writeAll("\x1B[6A\x1B[J"); // Move up 6 lines and clear to end
        try stdout.writeAll("\x1B[0m"); // Reset colors
    }

    const ImageError = error{
        FileNotFound,
        ImageNotFound,
        FailedToResizeImage,
        InvalidImageDimensions,
        InvalidResizeDimensions,
        MemoryAllocationFailed,
        EmptyPath,
        InvalidImageFormat,
        // Add any other error types you need
    };

    pub fn normalizePath(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
        // Special handling for Windows UNC paths (\\server\share\...)
        if (path.len >= 2 and path[0] == '\\' and path[1] == '\\') {
            // For UNC paths, preserve the leading \\ but convert all other backslashes
            var normalized = try allocator.alloc(u8, path.len);
            errdefer allocator.free(normalized);

            normalized[0] = '\\';
            normalized[1] = '\\';

            for (path[2..], 0..) |c, i| {
                normalized[i + 2] = if (c == '\\') '/' else c;
            }

            return normalized;
        }

        // For regular paths (including those with drive letters like C:\...)
        var normalized = try allocator.alloc(u8, path.len);
        errdefer allocator.free(normalized);

        // Special handling for drive letters (keep the : but convert \ to /)
        var i: usize = 0;
        while (i < path.len) : (i += 1) {
            if (i == 1 and path[i] == ':' and isAlpha(path[0])) {
                // This is a drive letter (e.g., "C:"), keep it as is
                normalized[i] = path[i];
            } else {
                normalized[i] = if (path[i] == '\\') '/' else path[i];
            }
        }

        return normalized;
    }

    pub fn fileExists(path: []const u8) bool {
        // Check if it's an absolute path
        const is_absolute =
            (path.len >= 1 and path[0] == '/') or // Unix-style absolute path
            (path.len >= 3 and isAlpha(path[0]) and path[1] == ':' and (path[2] == '/' or path[2] == '\\')); // Windows-style

        const file = if (is_absolute)
            std.fs.openFileAbsolute(path, .{}) catch {
                return false;
            }
        else
            std.fs.cwd().openFile(path, .{}) catch {
                return false;
            };

        file.close();
        return true;
    }

    fn isAlpha(c: u8) bool {
        return (c >= 'A' and c <= 'Z') or (c >= 'a' and c <= 'z');
    }

    // Check if a path has a valid image extension
    pub fn hasValidImageExtension(path: []const u8) bool {
        const lower_extensions = [_][]const u8{ ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp" };

        // Convert path to lowercase for case-insensitive comparison
        var lowercase_buf: [1024]u8 = undefined;
        if (path.len >= lowercase_buf.len) return false; // Path too long

        for (path, 0..) |c, i| {
            lowercase_buf[i] = std.ascii.toLower(c);
        }

        const lowercase_path = lowercase_buf[0..path.len];

        // Check against each valid extension
        for (lower_extensions) |ext| {
            if (std.mem.endsWith(u8, lowercase_path, ext)) {
                return true;
            }
        }

        return false;
    }

    // Perform all basic image path validation
    pub fn validateImagePath(allocator: std.mem.Allocator, original_path: []const u8) ![]u8 {
        if (original_path.len == 0) {
            return error.EmptyPath;
        }

        // Always normalize the path first
        const normalized_path = try normalizePath(allocator, original_path);
        errdefer allocator.free(normalized_path);

        // Check if the file exists using the normalized path
        if (!fileExists(normalized_path)) {
            allocator.free(normalized_path);
            return error.FileNotFound;
        }

        // Check if it has a valid image extension
        if (!hasValidImageExtension(normalized_path)) {
            allocator.free(normalized_path);
            return error.InvalidImageFormat;
        }

        return normalized_path;
    }

    /// Processes a turn in the chat state with the given prompt.
    ///
    /// This function takes a prompt as input and processes it within the context
    /// of the current chat state. It performs necessary actions based on the prompt
    /// and updates the chat state accordingly.
    ///
    /// # Parameters
    /// - `self`: A pointer to the `ChatState` instance.
    /// - `prompt`: A slice of constant unsigned 8-bit integers representing the prompt to process.
    ///
    /// @return Returns an error if the processing fails.
    pub fn processTurn(self: *ChatState, prompt: []const u8) !void {
        const stdout = std.io.getStdOut().writer();
        const timestamp = getCurrentTimestamp();

        // Format prompt with timestamp
        try stdout.print("\n{s}{s}{s} {s}Question:{s} {s}\n", .{ time_color, timestamp, reset_color, question_color, reset_color, prompt });
        try stdout.print("{s}{s}{s} {s}Answer:{s} ", .{ time_color, timestamp, reset_color, answer_color, reset_color });

        var token_count: usize = 0;
        var start_time: i64 = undefined;
        var timing_started = false;

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
            var logits_f32 = try logits.castWithSimd(f32);
            defer logits_f32.deinit();
            const next_token_id = try sampling.sample(&logits_f32, self.sampling_config, self.rng.random(), self.allocator);

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

    // Initialize standard I/O
    const stdout = std.io.getStdOut().writer();

    // Display header art
    try stdout.print("{s}{s}{s}\n\n", .{ main_color, HEADER_ART, reset_color });
    try stdout.writeAll("Welcome to the Chat Interface!\n");

    // Add loading spinner for model initialization
    var spinner = LoadingSpinner.init("Loading model...");
    try spinner.start();

    // Load model and tokenizer first - these stay loaded
    var tokenizer = try allocator.create(Tokenizer);
    defer {
        tokenizer.deinit();
        allocator.destroy(tokenizer);
    }
    tokenizer.* = try Tokenizer.fromJson("tokenizer.json", allocator);

    // load weights
    var weights = try allocator.create(Weights);
    defer {
        weights.deinit();
        allocator.destroy(weights);
    }

    weights.* = try Weights.init(model_config, "moondream.bin", allocator);

    // load vision model persistent state
    const VisionModelType = VisionModel(model_config);
    var vision_model = try allocator.create(VisionModelType);
    defer {
        vision_model.deinit();
        allocator.destroy(vision_model);
    }
    vision_model.* = try VisionModelType.init(weights.*, allocator);

    // load text model persistent state
    const TextModelType = TextModel(model_config);
    var text_model = try allocator.create(TextModelType);
    defer {
        text_model.deinit();
        allocator.destroy(text_model);
    }
    text_model.* = try TextModelType.init(weights.*, allocator);

    // Stop the spinner now that loading is complete
    spinner.stop();

    try displayCommands(stdout);
    try stdout.writeAll("\n");
    var chat_state: ?*ChatState = null;
    defer if (chat_state != null) chat_state.?.deinit();

    var buffer: [1024]u8 = undefined;
    const stdin = std.io.getStdIn().reader();

    while (true) {
        // start chat loop

        try stdout.print("{s}>{s} ", .{ prompt_color, reset_color });
        const input = try stdin.readUntilDelimiter(&buffer, '\n');

        if (std.mem.startsWith(u8, input, "/exit")) {
            break;
        } else if (std.mem.startsWith(u8, input, "/chat ")) {
            const image_path = std.mem.trim(u8, input[6..], " ");

            if (image_path.len == 0) {
                try stdout.print("{s}Error: Please provide an image path{s}\n", .{ error_color, reset_color });
                try stdout.print("Usage: /chat <image-path>\n", .{});
                continue;
            }

            // Initialize new chat state if needed
            if (chat_state == null) {
                chat_state = try ChatState.init(allocator, vision_model, text_model, tokenizer, weights);
            }

            // Try to load the image, but handle errors gracefully
            chat_state.?.loadImage(image_path) catch |err| {
                const error_msg = switch (err) {
                    error.FileNotFound => "File not found",
                    error.ImageNotFound => "Image could not be loaded (invalid format or corrupted file)",
                    error.FailedToResizeImage => "Failed to resize image",
                    error.EmptyPath => "Empty image path provided",
                    error.InvalidImageFormat => "Invalid image format (must be jpg, png, etc.)",
                    else => "Unexpected error occurred",
                };

                try stdout.print("\n{s}Error: {s} for path '{s}'{s}\n", .{ error_color, error_msg, image_path, reset_color });

                // Add helpful suggestions based on error type
                if (err == error.FileNotFound or err == error.ImageNotFound) {
                    try stdout.print("Make sure the file exists and the path is correct.\n", .{});
                    try stdout.print("For Windows paths, you can use either forward slashes or backslashes:\n", .{});
                    try stdout.print("Example: /chat C:/images/cat.png  or  /chat C:\\images\\cat.png\n", .{});
                } else if (err == error.InvalidImageFormat) {
                    try stdout.print("Supported formats: jpg, jpeg, png, bmp, gif, tiff, webp\n", .{});
                } else {
                    try stdout.print("Unexpected error occurred. Try a different image.\n", .{});
                }

                continue; // Return to command prompt
            };

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
