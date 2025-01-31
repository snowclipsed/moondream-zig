const std = @import("std");
const mem = std.mem;
const json = std.json;
const Allocator = mem.Allocator;

// Special token handling constants
const GPT2_SPACE_PREFIX = [_]u8{ 0xC4, 0xA0 }; // Ġ in UTF-8
const GPT2_NEWLINE_PREFIX = [_]u8{ 0xC4, 0x82 }; // Ċ in UTF-8

const SpecialToken = struct {
    id: u32,
    content: []const u8,
    is_special: bool,
    single_word: bool,
    lstrip: bool,
    rstrip: bool,
    normalized: bool,
};

pub const Tokenizer = struct {
    const Self = @This();
    tokens: std.StringHashMap(u32),
    special_tokens: std.ArrayList(SpecialToken),
    merges: std.ArrayList([]const u8),
    allocator: Allocator,
    eos_token: u32,
    bos_token: u32,
    pad_token: u32,

    fn init(allocator: Allocator) Tokenizer {
        return .{
            .tokens = std.StringHashMap(u32).init(allocator),
            .special_tokens = std.ArrayList(SpecialToken).init(allocator),
            .merges = std.ArrayList([]const u8).init(allocator),
            .allocator = allocator,
            .eos_token = 50256,
            .bos_token = 50257,
            .pad_token = 50258,
        };
    }

    pub fn fromJson(filename: []const u8, allocator: Allocator) !Tokenizer {
        // Initialize tokenizer
        var tokenizer = Tokenizer.init(allocator);
        errdefer tokenizer.deinit();

        // Read the file
        const file_content = try std.fs.cwd().readFileAlloc(allocator, filename, 10 * 1024 * 1024);
        defer allocator.free(file_content);

        // Parse JSON
        var tree = try std.json.parseFromSlice(std.json.Value, allocator, file_content, .{});
        defer tree.deinit();
        const root = tree.value;

        // Load special tokens
        const added_tokens = root.object.get("added_tokens").?.array;
        for (added_tokens.items) |token_obj| {
            const obj = token_obj.object;
            const id: u32 = @intCast(obj.get("id").?.integer);
            const content = obj.get("content").?.string;
            const is_special = obj.get("special").?.bool;
            const single_word = obj.get("single_word").?.bool;
            const lstrip = obj.get("lstrip").?.bool;
            const rstrip = obj.get("rstrip").?.bool;
            const normalized = obj.get("normalized").?.bool;

            // Allocate and copy content
            const content_copy = try allocator.dupe(u8, content);
            errdefer allocator.free(content_copy);

            try tokenizer.special_tokens.append(.{
                .id = id,
                .content = content_copy,
                .is_special = is_special,
                .single_word = single_word,
                .lstrip = lstrip,
                .rstrip = rstrip,
                .normalized = normalized,
            });
        }

        // Load vocabulary
        const model = root.object.get("model").?.object;
        const vocab = model.get("vocab").?.object;
        var vocab_it = vocab.iterator();
        while (vocab_it.next()) |entry| {
            const token = entry.key_ptr.*;
            const id: u32 = @intCast(entry.value_ptr.*.integer);

            // Allocate and copy token
            const token_copy = try allocator.dupe(u8, token);
            errdefer allocator.free(token_copy);

            try tokenizer.tokens.put(token_copy, id);
        }

        // Load merges
        const merges = model.get("merges").?.array;
        for (merges.items) |merge| {
            const merge_str = try allocator.dupe(u8, merge.string);
            errdefer allocator.free(merge_str);

            try tokenizer.merges.append(merge_str);
        }

        // Set special token IDs based on vocab
        if (vocab.get("<|endoftext|>")) |eos_value| {
            tokenizer.eos_token = @intCast(eos_value.integer);
        }

        return tokenizer;
    }

    pub fn fromFile(filename: []const u8, allocator: Allocator) !Tokenizer {
        var self = Tokenizer.init(allocator);
        errdefer self.deinit();

        const file = try std.fs.cwd().openFile(filename, .{});
        defer file.close();

        var reader = file.reader();

        // Read regular tokens
        const num_tokens = try reader.readInt(u32, .little);
        // std.debug.print("Loading {} regular tokens\n", .{num_tokens});

        for (0..num_tokens) |_| {
            const token_id = try reader.readInt(u32, .little);
            const token_len = try reader.readInt(u32, .little);
            const token_content = try allocator.alloc(u8, token_len);
            errdefer allocator.free(token_content);
            const bytes_read = try reader.readAll(token_content);
            if (bytes_read != token_len) {
                return error.UnexpectedEOF;
            }
            try self.tokens.put(token_content, token_id);

            // if (i % 1000 == 0) {
            //     // std.debug.print("Loaded {} regular tokens...\n", .{i});
            // }
        }

        // Read special tokens
        const num_special = try reader.readInt(u32, .little);
        // std.debug.print("Loading {} special tokens\n", .{num_special});

        for (0..num_special) |_| {
            // Read token metadata
            const token_id = try reader.readInt(u32, .little);
            const token_len = try reader.readInt(u32, .little);

            // Read flags
            const is_special = try reader.readInt(u8, .little) != 0;
            const single_word = try reader.readInt(u8, .little) != 0;
            const lstrip = try reader.readInt(u8, .little) != 0;
            const rstrip = try reader.readInt(u8, .little) != 0;
            const normalized = try reader.readInt(u8, .little) != 0;

            // Read token content
            const content = try allocator.alloc(u8, token_len);
            errdefer allocator.free(content);
            const bytes_read = try reader.readAll(content);
            if (bytes_read != token_len) {
                return error.UnexpectedEOF;
            }

            // Create and store special token
            try self.special_tokens.append(.{
                .id = token_id,
                .content = content,
                .is_special = is_special,
                .single_word = single_word,
                .lstrip = lstrip,
                .rstrip = rstrip,
                .normalized = normalized,
            });

            // std.debug.print("Loaded special token {}: id={}, content='{s}'\n", .{ i, token_id, content });
        }

        // Read merges
        const num_merges = try reader.readInt(u32, .little);
        // std.debug.print("Loading {} merges\n", .{num_merges});

        for (0..num_merges) |_| {
            // Read first part
            const first_len = try reader.readInt(u16, .little);
            const first = try allocator.alloc(u8, first_len);
            errdefer allocator.free(first);
            const first_bytes_read = try reader.readAll(first);
            if (first_bytes_read != first_len) {
                return error.UnexpectedEOF;
            }

            // Read second part
            const second_len = try reader.readInt(u16, .little);
            const second = try allocator.alloc(u8, second_len);
            errdefer allocator.free(second);
            const second_bytes_read = try reader.readAll(second);
            if (second_bytes_read != second_len) {
                return error.UnexpectedEOF;
            }

            // Combine into merge rule
            const merge = try std.fmt.allocPrint(allocator, "{s} {s}", .{ first, second });
            errdefer allocator.free(merge);

            try self.merges.append(merge);

            // Clean up temporary allocations
            allocator.free(first);
            allocator.free(second);
        }

        // Final load summary
        std.debug.print("Tokenizer loaded: {} tokens, {} special tokens, {} merges\n", .{
            self.tokens.count(),
            self.special_tokens.items.len,
            self.merges.items.len,
        });

        return self;
    }

    fn lookup(self: *const Tokenizer, token: []const u8) ?u32 {
        // std.debug.print("Looking up token: '{s}' (len: {})\n", .{ token, token.len });
        var it = self.tokens.iterator();
        while (it.next()) |entry| {
            // std.debug.print("Stored token: '{s}' (len: {})\n", .{ entry.key_ptr.*, entry.key_ptr.len });
            if (std.mem.eql(u8, entry.key_ptr.*, token)) {
                return entry.value_ptr.*;
            }
        }
        return null;
    }

    fn bytesToUtf8(bytes: []const u8) ![]u8 {
        // Convert raw bytes to proper UTF-8 string
        return std.unicode.utf8Encode(bytes);
    }

    // Modified preprocessText to handle GPT-2 specific character encodings
    fn preprocessText(text: []const u8, allocator: Allocator) ![]u8 {
        // Don't add any special prefixes here - we'll handle them during tokenization
        return allocator.dupe(u8, text);
    }

    // Helper function to convert between Ġ and space
    fn convertGpt2Space(text: []const u8, allocator: Allocator) ![]u8 {
        var result = std.ArrayList(u8).init(allocator);
        errdefer result.deinit();

        var i: usize = 0;
        while (i < text.len) {
            // Check for Ġ
            if (i + 1 < text.len and text[i] == 0xC4 and text[i + 1] == 0xA0) {
                try result.append(' '); // Replace with regular space
                i += 2;
            } else {
                try result.append(text[i]);
                i += 1;
            }
        }
        return result.toOwnedSlice();
    }

    fn convertToGpt2Space(text: []const u8, allocator: Allocator) ![]u8 {
        var result = std.ArrayList(u8).init(allocator);
        errdefer result.deinit();

        var i: usize = 0;
        while (i < text.len) {
            if (text[i] == ' ') {
                try result.appendSlice(&GPT2_SPACE_PREFIX); // Replace space with Ġ
            } else {
                try result.append(text[i]);
            }
            i += 1;
        }
        return result.toOwnedSlice();
    }

    pub fn encode(self: *const Tokenizer, text: []const u8) !std.ArrayList(u32) {
        var tokens = std.ArrayList(u32).init(self.allocator);
        errdefer tokens.deinit();

        // Convert spaces to Ġ for matching
        const gpt2_text = try convertToGpt2Space(text, self.allocator);
        defer self.allocator.free(gpt2_text);

        var current_pos: usize = 0;
        while (current_pos < gpt2_text.len) {
            var longest_match: ?struct { id: u32, len: usize } = null;
            var found = false;

            // Try to find the longest matching token
            var it = self.tokens.iterator();
            while (it.next()) |entry| {
                const token = entry.key_ptr.*;
                if (current_pos + token.len <= gpt2_text.len) {
                    const text_slice = gpt2_text[current_pos .. current_pos + token.len];
                    if (std.mem.eql(u8, token, text_slice)) {
                        if (longest_match == null or token.len > longest_match.?.len) {
                            longest_match = .{
                                .id = entry.value_ptr.*,
                                .len = token.len,
                            };
                            found = true;
                        }
                    }
                }
            }

            if (found) {
                try tokens.append(longest_match.?.id);
                current_pos += longest_match.?.len;
            } else {
                // Handle byte fallback
                try tokens.append(gpt2_text[current_pos]);
                current_pos += 1;
            }
        }

        return tokens;
    }

    pub fn decode(self: *const Tokenizer, tokens: std.ArrayList(u32)) ![]const u8 {
        var decoded = std.ArrayList(u8).init(self.allocator);
        errdefer decoded.deinit();

        for (tokens.items) |token_id| {
            if (token_id == self.eos_token or token_id == self.bos_token or token_id == self.pad_token) {
                continue;
            }

            var found = false;
            var it = self.tokens.iterator();
            while (it.next()) |entry| {
                if (entry.value_ptr.* == token_id) {
                    // Convert Ġ to space when showing output
                    const converted = try convertGpt2Space(entry.key_ptr.*, self.allocator);
                    defer self.allocator.free(converted);
                    try decoded.appendSlice(converted);
                    found = true;
                    break;
                }
            }

            if (!found and token_id < 256) {
                try decoded.append(@intCast(token_id));
                found = true;
            }

            if (!found) {
                return error.TokenNotFound;
            }
        }

        return decoded.toOwnedSlice();
    }
    pub fn deinit(self: *Self) void {
        var token_it = self.tokens.keyIterator();
        while (token_it.next()) |key| {
            self.allocator.free(key.*);
        }
        self.tokens.deinit();

        for (self.special_tokens.items) |token| {
            self.allocator.free(token.content);
        }
        self.special_tokens.deinit();

        for (self.merges.items) |merge| {
            self.allocator.free(merge);
        }
        self.merges.deinit();
    }
};
