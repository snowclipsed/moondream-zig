const std = @import("std");
const mem = std.mem;
const json = std.json;
const Allocator = mem.Allocator;
const Thread = std.Thread;

// Constants for parallel processing
pub const MIN_PARALLEL_TEXT_SIZE = 10_000;
pub const DEFAULT_CHUNK_SIZE = 1024 * 64;
pub const MAX_THREADS = 16;

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

// Trie node structure
const TrieNode = struct {
    children: std.AutoHashMap(u8, *TrieNode),
    token_id: ?u32,
    allocator: Allocator,

    fn init(allocator: Allocator) !*TrieNode {
        const node = try allocator.create(TrieNode);
        node.* = .{
            .children = std.AutoHashMap(u8, *TrieNode).init(allocator),
            .token_id = null,
            .allocator = allocator,
        };
        return node;
    }

    fn deinit(self: *TrieNode) void {
        var it = self.children.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.*.deinit();
            self.allocator.destroy(entry.value_ptr.*);
        }
        self.children.deinit();
    }
};

const WorkerContext = struct {
    tokenizer: *const Tokenizer,
    text: []const u8,
    result: *std.ArrayList(u32),
    start_pos: usize,
    end_pos: usize,
};

pub const Tokenizer = struct {
    const Self = @This();
    trie_root: *TrieNode,
    tokens: std.StringHashMap(u32),
    special_tokens: std.ArrayList(SpecialToken),
    merges: std.ArrayList([]const u8),
    allocator: Allocator,
    eos_token: u32,
    bos_token: u32,
    pad_token: u32,
    thread_pool: ThreadPool,

    const ThreadPool = struct {
        threads: []Thread,
        allocator: Allocator,

        fn init(allocator: Allocator, num_threads: usize) !ThreadPool {
            const threads = try allocator.alloc(Thread, num_threads);
            return ThreadPool{
                .threads = threads,
                .allocator = allocator,
            };
        }

        fn deinit(self: *ThreadPool) void {
            self.allocator.free(self.threads);
        }
    };

    // Constructor and initialization functions
    pub fn init(allocator: Allocator) !Self {
        const num_threads = @min(try Thread.getCpuCount(), MAX_THREADS);
        return Self{
            .trie_root = try TrieNode.init(allocator),
            .tokens = std.StringHashMap(u32).init(allocator),
            .special_tokens = std.ArrayList(SpecialToken).init(allocator),
            .merges = std.ArrayList([]const u8).init(allocator),
            .allocator = allocator,
            .eos_token = 50256,
            .bos_token = 50257,
            .pad_token = 50258,
            .thread_pool = try ThreadPool.init(allocator, num_threads),
        };
    }

    // File loading functions from original
    pub fn fromJson(filename: []const u8, allocator: Allocator) !Tokenizer {
        var tokenizer = try Tokenizer.init(allocator);
        errdefer tokenizer.deinit();

        const file_content = try std.fs.cwd().readFileAlloc(allocator, filename, 10 * 1024 * 1024);
        defer allocator.free(file_content);

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

        // Set special token IDs
        if (vocab.get("<|endoftext|>")) |eos_value| {
            tokenizer.eos_token = @intCast(eos_value.integer);
        }

        // Build the trie after loading vocabulary
        try tokenizer.buildTrie();

        return tokenizer;
    }

    pub fn fromFile(filename: []const u8, allocator: Allocator) !Tokenizer {
        var self = try Tokenizer.init(allocator);
        errdefer self.deinit();

        const file = try std.fs.cwd().openFile(filename, .{});
        defer file.close();

        var reader = file.reader();

        // Read regular tokens
        const num_tokens = try reader.readInt(u32, .little);

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
        }

        // Read special tokens
        const num_special = try reader.readInt(u32, .little);

        for (0..num_special) |_| {
            const token_id = try reader.readInt(u32, .little);
            const token_len = try reader.readInt(u32, .little);

            const is_special = try reader.readInt(u8, .little) != 0;
            const single_word = try reader.readInt(u8, .little) != 0;
            const lstrip = try reader.readInt(u8, .little) != 0;
            const rstrip = try reader.readInt(u8, .little) != 0;
            const normalized = try reader.readInt(u8, .little) != 0;

            const content = try allocator.alloc(u8, token_len);
            errdefer allocator.free(content);
            const bytes_read = try reader.readAll(content);
            if (bytes_read != token_len) {
                return error.UnexpectedEOF;
            }

            try self.special_tokens.append(.{
                .id = token_id,
                .content = content,
                .is_special = is_special,
                .single_word = single_word,
                .lstrip = lstrip,
                .rstrip = rstrip,
                .normalized = normalized,
            });
        }

        // Read merges
        const num_merges = try reader.readInt(u32, .little);

        for (0..num_merges) |_| {
            const first_len = try reader.readInt(u16, .little);
            const first = try allocator.alloc(u8, first_len);
            errdefer allocator.free(first);
            const first_bytes_read = try reader.readAll(first);
            if (first_bytes_read != first_len) {
                return error.UnexpectedEOF;
            }

            const second_len = try reader.readInt(u16, .little);
            const second = try allocator.alloc(u8, second_len);
            errdefer allocator.free(second);
            const second_bytes_read = try reader.readAll(second);
            if (second_bytes_read != second_len) {
                return error.UnexpectedEOF;
            }

            const merge = try std.fmt.allocPrint(allocator, "{s} {s}", .{ first, second });
            errdefer allocator.free(merge);

            try self.merges.append(merge);

            allocator.free(first);
            allocator.free(second);
        }

        // Build the trie after loading vocabulary
        try self.buildTrie();

        return self;
    }

    // Trie-related functions
    pub fn buildTrie(self: *Self) !void {
        var it = self.tokens.iterator();
        while (it.next()) |entry| {
            const token = entry.key_ptr.*;
            const id = entry.value_ptr.*;

            var current = self.trie_root;
            for (token) |byte| {
                const gop = try current.children.getOrPut(byte);
                if (!gop.found_existing) {
                    gop.value_ptr.* = try TrieNode.init(self.allocator);
                }
                current = gop.value_ptr.*;
            }
            current.token_id = id;
        }
    }

    // GPT2-specific functions from original
    fn convertToGpt2Chars(text: []const u8, allocator: Allocator) ![]u8 {
        var result = std.ArrayList(u8).init(allocator);
        errdefer result.deinit();

        var i: usize = 0;
        while (i < text.len) {
            if (text[i] == ' ') {
                try result.appendSlice(&GPT2_SPACE_PREFIX); // Ġ
            } else if (text[i] == '\n') {
                try result.appendSlice(&GPT2_NEWLINE_PREFIX); // Ċ
            } else {
                try result.append(text[i]);
            }
            i += 1;
        }
        return result.toOwnedSlice();
    }

    fn convertFromGpt2Chars(text: []const u8, allocator: Allocator) ![]u8 {
        var result = std.ArrayList(u8).init(allocator);
        errdefer result.deinit();

        var i: usize = 0;
        while (i < text.len) {
            // Check for Ġ
            if (i + 1 < text.len and text[i] == 0xC4) {
                if (text[i + 1] == 0xA0) {
                    try result.append(' '); // Replace Ġ with space
                    i += 2;
                    continue;
                } else if (text[i + 1] == 0x82) {
                    try result.append('\n'); // Replace Ċ with newline
                    i += 2;
                    continue;
                }
            }
            try result.append(text[i]);
            i += 1;
        }
        return result.toOwnedSlice();
    }

    // Parallel processing functions
    fn encoderWorker(context: *WorkerContext) void {
        const text_chunk = context.text[context.start_pos..context.end_pos];

        if (context.tokenizer.encodeChunkWithTrie(text_chunk)) |tokens| {
            context.result.appendSlice(tokens.items) catch return;
            tokens.deinit();
        } else |_| {
            return;
        }
    }

    pub fn lookup(self: *const Tokenizer, token: []const u8) ?u32 {
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

    fn encodeChunkWithTrie(self: *const Self, chunk: []const u8) !std.ArrayList(u32) {
        var tokens = std.ArrayList(u32).init(self.allocator);
        errdefer tokens.deinit();

        const gpt2_text = try convertToGpt2Chars(chunk, self.allocator);
        defer self.allocator.free(gpt2_text);

        var current_pos: usize = 0;
        while (current_pos < gpt2_text.len) {
            var current_node = self.trie_root;
            var longest_match: ?struct { id: u32, len: usize } = null;
            var match_length: usize = 0;

            while (current_pos + match_length < gpt2_text.len) {
                const byte = gpt2_text[current_pos + match_length];
                if (current_node.children.get(byte)) |next_node| {
                    current_node = next_node;
                    match_length += 1;
                    if (current_node.token_id) |id| {
                        longest_match = .{ .id = id, .len = match_length };
                    }
                } else break;
            }

            if (longest_match) |match| {
                try tokens.append(match.id);
                current_pos += match.len;
            } else {
                try tokens.append(gpt2_text[current_pos]);
                current_pos += 1;
            }
        }

        return tokens;
    }

    // Parallel encode function

    pub fn encode(self: *const Self, text: []const u8) !std.ArrayList(u32) {
        if (text.len < MIN_PARALLEL_TEXT_SIZE) {
            return self.encodeChunkWithTrie(text);
        }

        const num_threads = @min(@max(text.len / DEFAULT_CHUNK_SIZE, 1), self.thread_pool.threads.len);

        const base_chunk_size = text.len / num_threads;

        var contexts = try self.allocator.alloc(WorkerContext, num_threads);
        defer self.allocator.free(contexts);

        var results = try self.allocator.alloc(std.ArrayList(u32), num_threads);
        defer self.allocator.free(results);

        // Initialize results arrays
        for (results) |*result| {
            result.* = std.ArrayList(u32).init(self.allocator);
        }
        defer {
            for (results) |*result| {
                result.deinit();
            }
        }

        // Create and start threads
        var start_pos: usize = 0;
        for (0..num_threads) |i| {
            const is_last_chunk = (i == num_threads - 1);
            const approximate_end = if (is_last_chunk)
                text.len
            else
                start_pos + base_chunk_size;

            const end_pos = if (is_last_chunk)
                text.len
            else
                findSplitPoint(text, approximate_end);

            contexts[i] = .{
                .tokenizer = self,
                .text = text,
                .result = &results[i],
                .start_pos = start_pos,
                .end_pos = end_pos,
            };

            self.thread_pool.threads[i] = try Thread.spawn(.{}, encoderWorker, .{&contexts[i]});

            start_pos = end_pos;
        }

        // Wait for all threads
        for (self.thread_pool.threads[0..num_threads]) |thread| {
            thread.join();
        }

        // Combine results
        var final_tokens = std.ArrayList(u32).init(self.allocator);
        errdefer final_tokens.deinit();

        var total_tokens: usize = 0;
        for (results) |result| {
            total_tokens += result.items.len;
        }

        try final_tokens.ensureTotalCapacity(total_tokens);

        for (results) |result| {
            try final_tokens.appendSlice(result.items);
        }

        return final_tokens;
    }

    fn findSplitPoint(text: []const u8, approximate_pos: usize) usize {
        var pos = approximate_pos;

        // Look forward for a space or newline
        while (pos < text.len) : (pos += 1) {
            if (text[pos] == ' ' or text[pos] == '\n') {
                return pos;
            }
        }

        // Look backward if we couldn't find forward
        pos = approximate_pos;
        while (pos > 0) : (pos -= 1) {
            if (text[pos] == ' ' or text[pos] == '\n') {
                return pos;
            }
        }

        return approximate_pos;
    }

    // Decoding function from original
    pub fn decode(self: *const Self, tokens: std.ArrayList(u32)) ![]const u8 {
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
                    const converted = try convertFromGpt2Chars(entry.key_ptr.*, self.allocator);
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

    // Cleanup
    pub fn deinit(self: *Self) void {
        // Clean up trie
        self.trie_root.deinit();
        self.allocator.destroy(self.trie_root);

        // Clean up tokens
        var token_it = self.tokens.keyIterator();
        while (token_it.next()) |key| {
            self.allocator.free(key.*);
        }
        self.tokens.deinit();

        // Clean up special tokens
        for (self.special_tokens.items) |token| {
            self.allocator.free(token.content);
        }
        self.special_tokens.deinit();

        // Clean up merges
        for (self.merges.items) |merge| {
            self.allocator.free(merge);
        }
        self.merges.deinit();

        // Clean up thread pool
        self.thread_pool.deinit();
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize tokenizer

    // Load vocabulary from file
    var tokenizer = try Tokenizer.fromFile("../tokenizer.bin", allocator);
    defer tokenizer.deinit();
    // Or load from JSON
    // try tokenizer.fromJson("../tokenizer.json", allocator);

    // Example text to encode
    const text = "Hello, this is a test of the parallel tokenizer!";
    var tokens = try tokenizer.encode(text);
    defer tokens.deinit();

    // Decode back to text
    const decoded = try tokenizer.decode(tokens);
    defer allocator.free(decoded);

    // Print results
    std.debug.print("Encoded tokens: {any}\n", .{tokens.items});
    std.debug.print("Decoded text: {s}\n", .{decoded});
}
