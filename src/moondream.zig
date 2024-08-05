const std = @import("std");
const assert = std.debug.assert;
const mem = std.mem;
const Allocator = mem.Allocator;
const ThreadPool = mem.Thread.Pool;
// vector math functions

// function taken from cgbur/llama2.zig
// applies softmax on a vector
fn softmax(x: []f32) void {
    assert(x.len > 0);
    // max of x for numerical stability
    var max: f32 = x[0];
    for (x[1..]) |val| {
        if (val > max) {
            max = val;
        }
    }
    // exp and sum
    var sum: f32 = 0.0;
    for (x) |*val| {
        val.* = std.math.exp(val.* - max); // https://stackoverflow.com/questions/42599498/numerically-stable-softmax
        sum += val.*;
    }
    // normalize
    for (x) |*val| {
        val.* /= sum;
    }
}

// This function accumulates an f32 array b into another f32 array a
fn accumulate(a: []f32, b: []f32) void {
    assert(a.len == b.len);
    for (0..a.len) |i| {
        a[i] += b[i];
    }
}

// Returns index of the max value in an f32 array
fn argmax(x: []f32) usize {
    assert(x.len > 0);
    var max: f32 = x[0];
    var maxi: usize = 0;
    for (1..x.len) |i| {
        if (x[i] > max) {
            max = x[i];
            maxi = i;
        }
    }
    return maxi;
}

// weights

// runstate

// tokenizer

// Tokens, their scores, and the max token length. Supports initialization
// from a file and encoding text into tokens via the `encode` method.
const Tokenizer = struct {
    const Self = @This();
    tokens: std.StringHashMap(u32),
    merges: std.ArrayList([]const u8),
    allocator: Allocator,

    fn init(allocator: Allocator) Tokenizer {
        return .{
            .tokens = std.StringHashMap(u32).init(allocator),
            .merges = std.ArrayList([]const u8).init(allocator),
            .allocator = allocator,
        };
    }

    fn fromFile(filename: []const u8, allocator: Allocator) !Tokenizer {
        var self = Tokenizer.init(allocator);
        errdefer self.deinit();

        const file = try std.fs.cwd().openFile(filename, .{});
        defer file.close();

        var reader = file.reader();

        // Read number of tokens
        const num_tokens = try reader.readInt(u32, .little);

        // Read tokens
        var i: u32 = 0;
        while (i < num_tokens) : (i += 1) {
            const token_id = try reader.readInt(u32, .little);
            const token_len = try reader.readInt(u32, .little);
            const token_content = try allocator.alloc(u8, token_len);
            errdefer allocator.free(token_content);
            _ = try reader.readAll(token_content);

            try self.tokens.put(token_content, token_id);
        }

        // Read number of merges
        const num_merges = try reader.readInt(u32, .little);

        // Read merges
        i = 0;
        while (i < num_merges) : (i += 1) {
            const first_len = try reader.readInt(u16, .little);
            const first = try allocator.alloc(u8, first_len);
            _ = try reader.readAll(first);
            defer allocator.free(first);

            const second_len = try reader.readInt(u16, .little);
            const second = try allocator.alloc(u8, second_len);
            _ = try reader.readAll(second);
            defer allocator.free(second);

            const merge = try std.fmt.allocPrint(allocator, "{s} {s}", .{ first, second });
            try self.merges.append(merge);
        }

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

    fn encode(self: *const Tokenizer, text: []const u8) !std.ArrayList(u32) {
        var tokens = std.ArrayList(u32).init(self.allocator);
        errdefer tokens.deinit();

        var words = std.mem.split(u8, text, " ");
        while (words.next()) |word| {
            var current_word = word;
            while (current_word.len > 0) {
                var longest_token: ?[]const u8 = null;
                var longest_token_id: ?u32 = null;

                // Find the longest matching token
                var token_it = self.tokens.iterator();
                while (token_it.next()) |entry| {
                    const token = entry.key_ptr.*;
                    if (std.mem.startsWith(u8, current_word, token)) {
                        if (longest_token == null or token.len > longest_token.?.len) {
                            longest_token = token;
                            longest_token_id = entry.value_ptr.*;
                        }
                    }
                }

                if (longest_token) |token| {
                    try tokens.append(longest_token_id.?);
                    current_word = current_word[token.len..];
                } else {
                    // If no token matches, treat the first byte as an unknown token
                    try tokens.append(current_word[0]);
                    current_word = current_word[1..];
                }
            }
        }

        return tokens;
    }

    fn info(self: *const Tokenizer) void {
        std.debug.print("vocab size: {}\n", .{self.tokens.count()});
        std.debug.print("merge list size: {}\n", .{self.merges.items.len});
    }

    fn deinit(self: *Self) void {
        var token_it = self.tokens.keyIterator();
        while (token_it.next()) |key| {
            self.allocator.free(key.*);
        }
        self.tokens.deinit();

        for (self.merges.items) |merge| {
            self.allocator.free(merge);
        }
        self.merges.deinit();
    }
};

// attention

// transformer

// vision encoder

// inference

// tests
test "softmax" {
    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };

    softmax(&x);

    var sum: f32 = 0.0;
    for (0..x.len) |value| {
        sum += x[value];
    }

    try std.testing.expect(sum == 1.0);
}

test "tokenizer" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    var tokenizer = try Tokenizer.fromFile("tokenizer.bin", allocator);
    defer tokenizer.deinit();

    //checking vocab size and merge list size
    try std.testing.expectEqual(50000, tokenizer.merges.items.len);
    try std.testing.expectEqual(50257, tokenizer.tokens.count());
    //checking some token lookups
    try std.testing.expectEqual(29302, tokenizer.lookup("tube"));
    try std.testing.expectEqual(null, tokenizer.lookup("AAAAAAAAAAAAAAAA")); // null value check
    try std.testing.expectEqual(50256, tokenizer.lookup("<|endoftext|>")); // null value check

    const input: []const u8 = "Moondream is a small VLM that punches above its weight.";
    const tokenization = try tokenizer.encode(input);
    const tokenization_slice: []u32 = tokenization.items;
    defer tokenization.deinit();
    const expected_tokenization: []const u32 = &[_]u32{ 31640, 25966, 271, 64, 17470, 47468, 44, 5562, 35512, 2052, 29370, 896, 6551, 13 };

    for (tokenization_slice, 0..) |token, i| {
        try std.testing.expect(token == expected_tokenization[i]);
    }
}

// main
