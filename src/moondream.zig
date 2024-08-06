const std = @import("std");
const assert = std.debug.assert;
const mem = std.mem;
const Allocator = mem.Allocator;
const ThreadPool = mem.Thread.Pool;

const DEFAULT_VECTOR_WIDTH: usize = std.simd.suggestVectorLength(f16) orelse 4;
const simd_align = @alignOf(@Vector(DEFAULT_VECTOR_WIDTH, f16));

comptime {
    @setFloatMode(.optimized);
}

// vector math functions

// function taken from cgbur/llama2.zig
// applies softmax on a vector
fn softmax(x: []f16) void {
    assert(x.len > 0);
    // max of x for numerical stability
    var max: f16 = x[0];
    for (x[1..]) |val| {
        if (val > max) {
            max = val;
        }
    }
    // exp and sum
    var sum: f16 = 0.0;
    for (x) |*val| {
        val.* = std.math.exp(val.* - max); // https://stackoverflow.com/questions/42599498/numerically-stable-softmax
        sum += val.*;
    }
    // normalize
    for (x) |*val| {
        val.* /= sum;
    }
}

// This function accumulates an f16 array b into another f16 array a
fn accumulate(a: []f16, b: []f16) void {
    assert(a.len == b.len);
    for (0..a.len) |i| {
        a[i] += b[i];
    }
}

// Returns index of the max value in an f16 array
fn argmax(x: []f16) usize {
    assert(x.len > 0);
    var max: f16 = x[0];
    var maxi: usize = 0;
    for (1..x.len) |i| {
        if (x[i] > max) {
            max = x[i];
            maxi = i;
        }
    }
    return maxi;
}

//config

// const ConfigReader = extern struct {
//     const Self = @This();
//     dim: i32, // transformer dimension
//     hidden_dim: i32, // for ffn layers
//     n_layers: i32, // number of layers
//     n_heads: i32, // number of query heads
//     n_kv_heads: i32, // number of key/value heads (can be < query heads because of multiquery)
//     vocab_size: i32, // vocabulary size, usually 256 (byte-level)
//     seq_len: i32, // max sequence length

//     fn config(self: Self) Config {
//         return Config{
//             .dim = @intCast(self.dim),
//             .hidden_dim = @intCast(self.hidden_dim),
//             .n_layers = @intCast(self.n_layers),
//             .n_heads = @intCast(self.n_heads),
//             .n_kv_heads = @intCast(self.n_kv_heads),
//             .vocab_size = @intCast(self.vocab_size),
//             .seq_len = @intCast(self.seq_len),
//         };
//     }
// };

const Config = struct {

    // TODO: Add stuff to config as required
    //text
    dim: usize, //text transformer dim, 2048
    hidden_dim: usize, // hidden fc dim
    n_layers: usize, //number of transformer layers, 24 for text model
    n_heads: usize, //number of attn heads per layer
    head_dim: usize, //size of attn heads per layer
    seq_len: usize, // sequence length, 2048

    // vision
    img_channels: usize, // number of channels per patch, RGB has 3
    img_dim: usize,
    patch_size: usize, // size of patch, 14x14 default
    vit_dim: usize, // width of each patch embedding created from linear patch embedding layer, 1152 default
    hidden_features: usize,
};

// weights
/// Struct defines the weights of moondream
/// All weights are transposed
/// Naming convention :
/// "t_" prefix : text_model (phi 1.5)
/// "v_" prefix : vision_model (vision encoder)
/// "_w" suffix : weights
/// "_b" suffix : biases
const Weights = struct {
    //text model weights
    word_token_embedding: [*]f16, //(dim, vocab_size)
    //attn layer norm
    t_ln_w: [*]f16, // (layer, dim)
    t_ln_b: [*]f16, // (layer, dim)
    //attn qkv
    t_Wqkv_w: [*]f16, // (layer, dim, n_heads*head_dim*3)
    t_Wqkv_b: [*]f16, // (layer, n_heads*head_dim*3)
    //fully connected
    t_fc1_w: [*]f16, // (layer, hidden_dim, dim)
    t_fc1_b: [*]f16, // (layer, hidden_dim)
    t_fc2_w: [*]f16, // (layer, dim, hidden_dim)
    t_fc2_b: [*]f16, // (layer, dim)
    // output
    t_out_proj_w: [*]f16, // (layer, seqlen, dim)
    t_out_proj_bias: [*]f16, // (layer, dim)

    // vision model weights;
    v_patch_embedding_linear_w: [*]f16, // (vit_dim, patch_size * path_size * channels)
    v_patch_embedding_linear_b: [*]f16, // (vit_dim)

    v_pos_embedding: [*]f16, // (1, (img_dim/patch_dim)^2, vit_dim)
    // (img_dim/patch_dim)^2 is basically the number of patches in an image of size img_dim x img_dim
    // the first dim 1 allows it to be broadcast across all the images in a batch
    // this is useful if we break down a larger image into multiple images of img_dim x img_dim

    v_Wqkv_w: [*]f16, // (vit_dim, vit_dim*3)
    v_Wqkv_b: [*]f16, // (vit_dim * 3)

    v_fc1_w: [*]f16, // (hidden_features, vit_dim)
    v_fc1_b: [*]f16, // (hidden_features)

    v_fc2_w: [*]f16, // (vit_dim, hidden_features)
    v_fc2_b: [*]f16, // (vit_dim)

    v_out_proj_w: [*]f16, // (vit_dim, vit_dim)
    v_out_proj_b: [*]f16, // (vit_dim)

};

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
    var x = [_]f16{ 1.0, 2.0, 3.0, 4.0 };

    softmax(&x);

    var sum: f16 = 0.0;
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

    // checking encoding
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
