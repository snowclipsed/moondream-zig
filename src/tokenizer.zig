const std = @import("std");
const mem = std.mem;
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
            .eos_token = 50256, // These values should match your tokenizer.json
            .bos_token = 50257,
            .pad_token = 50258,
        };
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
        var result = std.ArrayList(u8).init(allocator);
        errdefer result.deinit();

        var i: usize = 0;
        while (i < text.len) {
            const c = text[i];

            // Handle different whitespace cases
            if (std.ascii.isWhitespace(c)) {
                if (c == ' ') {
                    // Add GPT-2 space prefix (Ġ)
                    try result.appendSlice(&GPT2_SPACE_PREFIX);
                } else if (c == '\n') {
                    // Add GPT-2 newline prefix (Ċ)
                    try result.appendSlice(&GPT2_NEWLINE_PREFIX);
                }
            } else {
                if (i == 0 or (i > 0 and std.ascii.isWhitespace(text[i - 1]))) {
                    // Add space prefix at the start of words
                    try result.appendSlice(&GPT2_SPACE_PREFIX);
                }
                try result.append(c);
            }
            i += 1;
        }

        return result.toOwnedSlice();
    }

    // Modified encode function to handle special tokens and UTF-8 properly
    pub fn encode(self: *const Tokenizer, text: []const u8) !std.ArrayList(u32) {
        var tokens = std.ArrayList(u32).init(self.allocator);
        errdefer tokens.deinit();

        // First preprocess the text to handle GPT-2 special characters
        const processed_text = try preprocessText(text, self.allocator);
        defer self.allocator.free(processed_text);

        var current_pos: usize = 0;
        while (current_pos < processed_text.len) {
            // First check for special tokens
            var special_token_found = false;
            for (self.special_tokens.items) |special_token| {
                if (current_pos + special_token.content.len <= processed_text.len) {
                    if (mem.eql(u8, special_token.content, processed_text[current_pos .. current_pos + special_token.content.len])) {
                        try tokens.append(special_token.id);
                        current_pos += special_token.content.len;
                        special_token_found = true;
                        break;
                    }
                }
            }
            if (special_token_found) continue;

            // Try to match the longest token
            var longest_match: ?struct { token: []const u8, id: u32, len: usize } = null;
            var token_it = self.tokens.iterator();

            while (token_it.next()) |entry| {
                const token = entry.key_ptr.*;
                if (current_pos + token.len <= processed_text.len) {
                    if (mem.eql(u8, token, processed_text[current_pos .. current_pos + token.len])) {
                        if (longest_match == null or token.len > longest_match.?.len) {
                            longest_match = .{
                                .token = token,
                                .id = entry.value_ptr.*,
                                .len = token.len,
                            };
                        }
                    }
                }
            }

            if (longest_match) |match| {
                try tokens.append(match.id);
                current_pos += match.len;
            } else {
                // Handle unknown bytes
                const byte = processed_text[current_pos];
                if (byte >= 0x80) {
                    const utf8_len = try std.unicode.utf8ByteSequenceLength(byte);
                    if (current_pos + utf8_len <= processed_text.len) {
                        // Create a byte-level token for each byte in the UTF-8 sequence
                        var j: usize = 0;
                        while (j < utf8_len) : (j += 1) {
                            try tokens.append(processed_text[current_pos + j]);
                        }
                        current_pos += utf8_len;
                    } else {
                        try tokens.append(byte);
                        current_pos += 1;
                    }
                } else {
                    try tokens.append(byte);
                    current_pos += 1;
                }
            }
        }

        return tokens;
    }

    fn info(self: *const Tokenizer) void {
        std.debug.print("vocab size: {}\n", .{self.tokens.count()});
        std.debug.print("merge list size: {}\n", .{self.merges.items.len});
    }
    // TODO: Write Decode Function Tests.
    // Modified decode function to properly handle GPT-2 special characters
    pub fn decode(self: *const Tokenizer, tokens: std.ArrayList(u32)) ![]const u8 {
        var decoded_text = std.ArrayList(u8).init(self.allocator);
        errdefer decoded_text.deinit();

        var last_was_space = false;
        for (tokens.items) |token_id| {
            if (token_id == self.bos_token or token_id == self.eos_token or token_id == self.pad_token) {
                continue;
            }

            // First check special tokens
            var special_found = false;
            for (self.special_tokens.items) |special_token| {
                if (special_token.id == token_id) {
                    try decoded_text.appendSlice(special_token.content);
                    special_found = true;
                    break;
                }
            }
            if (special_found) continue;

            // Then check regular tokens
            var found = false;
            var token_it = self.tokens.iterator();
            while (token_it.next()) |entry| {
                if (entry.value_ptr.* == token_id) {
                    const token = entry.key_ptr.*;

                    // Handle GPT-2 special prefixes
                    if (token.len >= 2 and mem.eql(u8, token[0..2], &GPT2_SPACE_PREFIX)) {
                        if (!last_was_space) {
                            try decoded_text.append(' ');
                        }
                        try decoded_text.appendSlice(token[2..]);
                        last_was_space = false;
                    } else if (token.len >= 2 and mem.eql(u8, token[0..2], &GPT2_NEWLINE_PREFIX)) {
                        try decoded_text.append('\n');
                        try decoded_text.appendSlice(token[2..]);
                        last_was_space = true;
                    } else {
                        try decoded_text.appendSlice(token);
                        last_was_space = false;
                    }
                    found = true;
                    break;
                }
            }

            if (!found and token_id < 256) {
                // Handle byte-level tokens
                try decoded_text.append(@intCast(token_id));
                last_was_space = false;
                found = true;
            }

            if (!found) {
                return error.TokenNotFound;
            }
        }

        return decoded_text.toOwnedSlice();
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
