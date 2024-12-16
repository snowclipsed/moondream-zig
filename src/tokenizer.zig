const std = @import("std");
const mem = std.mem;
const Allocator = mem.Allocator;

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
    // TODO : Add handling external tokens to tokenizer
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

    fn preprocessText(text: []const u8, allocator: Allocator) ![]u8 {
        // Add space prefix to words
        var result = std.ArrayList(u8).init(allocator);
        errdefer result.deinit();

        var i: usize = 0;
        var last_was_space = true; // Start with true to handle first word

        while (i < text.len) {
            const c = text[i];
            if (std.ascii.isWhitespace(c)) {
                try result.append(c);
                last_was_space = true;
            } else {
                if (!last_was_space) {
                    try result.append(c);
                } else {
                    // Add the special GPT2 space marker (Ġ) followed by the character
                    try result.appendSlice("Ġ");
                    try result.append(c);
                }
                last_was_space = false;
            }
            i += 1;
        }

        return result.toOwnedSlice();
    }

    pub fn encode(self: *const Tokenizer, text: []const u8) !std.ArrayList(u32) {
        var tokens = std.ArrayList(u32).init(self.allocator);
        errdefer tokens.deinit();

        var current_pos: usize = 0;
        var prev_was_whitespace = true; // Start with true to handle first word

        while (current_pos < text.len) {
            // Special handling for whitespace
            if (std.ascii.isWhitespace(text[current_pos])) {
                if (text[current_pos] == ' ') {
                    try tokens.append(32); // Standard space token
                } else if (text[current_pos] == '\n') {
                    try tokens.append(198); // Newline token
                }
                current_pos += 1;
                prev_was_whitespace = true;
                continue;
            }

            var longest_match: ?struct { token: []const u8, id: u32, len: usize } = null;
            var token_it = self.tokens.iterator();

            // Try to find the longest matching token
            while (token_it.next()) |entry| {
                const token = entry.key_ptr.*;

                // Skip Ġ tokens when not at a word boundary
                if (token.len >= 3 and std.mem.startsWith(u8, token, "Ġ") and !prev_was_whitespace) {
                    continue;
                }

                const token_content = if (token.len >= 3 and std.mem.startsWith(u8, token, "Ġ"))
                    token[3..]
                else
                    token;

                // Check if we have enough text left to match
                if (current_pos + token_content.len <= text.len) {
                    const text_slice = text[current_pos .. current_pos + token_content.len];
                    if (std.mem.eql(u8, token_content, text_slice)) {
                        if (longest_match == null or token_content.len > longest_match.?.len) {
                            longest_match = .{
                                .token = token_content,
                                .id = entry.value_ptr.*,
                                .len = token_content.len,
                            };
                        }
                    }
                }
            }

            if (longest_match) |match| {
                try tokens.append(match.id);
                current_pos += match.len;
                prev_was_whitespace = false;
            } else {
                // Handle UTF-8 sequences properly
                const byte = text[current_pos];
                if (byte >= 0x80) { // UTF-8 leading byte
                    const utf8_len = std.unicode.utf8ByteSequenceLength(byte) catch 1;
                    if (current_pos + utf8_len <= text.len) {
                        const sequence = text[current_pos .. current_pos + utf8_len];
                        // Store the entire UTF-8 sequence as bytes
                        for (sequence) |b| {
                            try tokens.append(b);
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
                prev_was_whitespace = false;
            }
        }

        return tokens;
    }

    fn info(self: *const Tokenizer) void {
        std.debug.print("vocab size: {}\n", .{self.tokens.count()});
        std.debug.print("merge list size: {}\n", .{self.merges.items.len});
    }

    pub fn decode(self: *const Tokenizer, tokens: std.ArrayList(u32)) ![]const u8 {
        var decoded_text = std.ArrayList(u8).init(self.allocator);
        errdefer decoded_text.deinit();

        var i: usize = 0;
        var prev_was_space = false;

        while (i < tokens.items.len) {
            const token_id = tokens.items[i];

            // Handle special tokens first
            if (token_id == self.bos_token or token_id == self.eos_token or token_id == self.pad_token) {
                i += 1;
                continue;
            }

            switch (token_id) {
                32 => { // Space token
                    try decoded_text.append(' ');
                    prev_was_space = true;
                },
                198 => { // Newline token
                    try decoded_text.append('\n');
                    prev_was_space = false;
                },
                else => {
                    var token_found = false;
                    var token_it = self.tokens.iterator();
                    while (token_it.next()) |entry| {
                        if (entry.value_ptr.* == token_id) {
                            const token = entry.key_ptr.*;

                            // Handle Ġ prefix tokens
                            if (token.len >= 3 and std.mem.startsWith(u8, token, "Ġ")) {
                                const content = token[3..];
                                // Only add space if we're not at the start and not after another space
                                if (!prev_was_space and decoded_text.items.len > 0) {
                                    try decoded_text.append(' ');
                                }
                                try decoded_text.appendSlice(content);
                            } else {
                                // For non-Ġ tokens, just append directly
                                try decoded_text.appendSlice(token);
                            }
                            token_found = true;
                            prev_was_space = false;
                            break;
                        }
                    }

                    if (!token_found) {
                        // Handle byte tokens
                        if (token_id < 256) {
                            try decoded_text.append(@intCast(token_id));
                            if (token_id >= 0x80) { // Part of UTF-8 sequence
                                const remaining_bytes = std.unicode.utf8ByteSequenceLength(@intCast(token_id)) catch 1;
                                // Try to collect the full UTF-8 sequence
                                var bytes_collected: usize = 1;
                                while (bytes_collected < remaining_bytes and i + 1 < tokens.items.len) {
                                    const next_byte = tokens.items[i + 1];
                                    if (next_byte < 256) {
                                        try decoded_text.append(@intCast(next_byte));
                                        i += 1;
                                        bytes_collected += 1;
                                    } else {
                                        break;
                                    }
                                }
                            }
                        } else {
                            return error.TokenNotFound;
                        }
                        prev_was_space = false;
                    }
                },
            }
            i += 1;
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
