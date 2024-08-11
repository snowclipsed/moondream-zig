const std = @import("std");
const assert = std.debug.assert;
const mem = std.mem;
const Allocator = mem.Allocator;
const Thread = std.Thread;
const c = @cImport({
    @cInclude("ggml.h");
});
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

const ConfigReader = extern struct {
    const Self = @This();
    vocab: i32, // vocabulary size, usually 256 (byte-level)
    dim: i32, // transformer dimension
    hidden_dim: i32, // for ffn layers
    n_layers: i32, // number of transformer layers, 24 for text model
    n_heads: i32, // number of attn heads per layer
    head_dim: i32, // size of attn heads per layer
    seq_len: i32, // max sequence length

    // vision
    img_channels: i32, // number of channels per patch, RGB has 3
    img_dim: i32,
    patch: i32, // size of patch, 14x14 default
    vit_dim: i32, // width of each patch embedding created from linear patch embedding layer, 1152 default
    hidden_features: i32, // the number of hidden features, equivalent to hidden_dim in text model, 4304 default

    fn config(self: Self) Config {
        return Config{
            .vocab = @intCast(self.vocab),
            .dim = @intCast(self.dim),
            .hidden_dim = @intCast(self.hidden_dim),
            .n_layers = @intCast(self.n_layers),
            .n_heads = @intCast(self.n_heads),
            .head_dim = @intCast(self.head_dim),
            .seq_len = @intCast(self.seq_len),
            .img_channels = @intCast(self.img_channels),
            .img_dim = @intCast(self.img_dim),
            .patch = @intCast(self.patch),
            .vit_dim = @intCast(self.vit_dim),
            .hidden_features = @intCast(self.hidden_features),
        };
    }
};

const Config = struct {

    // TODO: Add stuff to config as required
    //text
    vocab: usize,
    dim: usize, //text transformer dim, 2048
    hidden_dim: usize, // hidden fc dim
    n_layers: usize, //number of transformer layers, 24 for text model
    n_heads: usize, //number of attn heads per layer
    head_dim: usize, //size of attn heads per layer
    seq_len: usize, // sequence length, 2048

    // vision
    img_channels: usize, // number of channels per patch, RGB has 3
    img_dim: usize,
    patch: usize, // size of patch, 14x14 default
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

    // Single contiguous memory block for all weights
    memory: []f16,

    // Slices for specific weights

    ///Text model start///
    word_token_embedding: []f16, // (dim, vocab)

    /// Transformer layer start

    // attn layer norm
    t_ln_w: []f16, // (layer, dim)
    t_ln_b: []f16, // (layer, dim)
    // attn qkv
    t_Wqkv_w: []f16, // (layer, dim, n_heads*head_dim*3)
    t_Wqkv_b: []f16, // (layer, n_heads*head_dim*3)
    // output
    t_out_proj_w: []f16, // (layer, seqlen, dim)
    t_out_proj_bias: []f16, // (layer, dim)
    // fully connected
    t_fc1_w: []f16, // (layer, hidden_dim, dim)
    t_fc1_b: []f16, // (layer, hidden_dim)
    t_fc2_w: []f16, // (layer, dim, hidden_dim)
    t_fc2_b: []f16, // (layer, dim)

    ///Transformer layer end ///

    // lm head
    t_linear_w: []f16, //(vocab, dim)
    t_linear_b: []f16, //(vocab)
    t_ln_out_w: []f16, //(dim)
    t_ln_out_b: []f16, //(dim)
    //Text model end///

    /// Vision model start ///

    // combining patch embeddngs and pos
    v_patch_embedding_linear_w: []f16, // (vit_dim, patch * patch * channels)
    v_patch_embedding_linear_b: []f16, // (vit_dim)
    v_pos_embedding: []f16, // (1, (img_dim/patch_dim)^2, vit_dim)

    /// Vision Transformer Start
    // attention qkv
    v_Wqkv_w: []f16, // (vit_dim, vit_dim*3)
    v_Wqkv_b: []f16, // (vit_dim * 3)

    //attn out
    v_out_proj_w: []f16, // (vit_dim, vit_dim)
    v_out_proj_b: []f16, // (vit_dim)

    //ViT fc
    v_fc1_w: []f16, // (hidden_features, vit_dim)
    v_fc1_b: []f16, // (hidden_features)
    v_fc2_w: []f16, // (vit_dim, hidden_features)
    v_fc2_b: []f16, // (vit_dim)

    //ViT norm
    v_norm1_w: []f16, // (hidden_features)
    v_norm1_b: []f16, // (hidden_features)
    v_norm2_w: []f16, // (hidden_features)
    v_norm2_b: []f16, // (hidden_features)

    // Vision Transformer End
    // projection

    fn init(config: *const Config, data: []const u8) !Weights {
        const sizes = calculateSizes(config);
        const total = calculateTotalSize(config);

        if (data.len < total * @sizeOf(f16)) {
            return error.InsufficientData;
        }
        var self: Weights = undefined;
        self.memory = @alignCast(@ptrCast(data));
        var offset: usize = 0;

        // Set slices for each weight
        self.word_token_embedding = self.memory[offset .. offset + sizes.word_token_embedding];

        // text model
        offset += sizes.word_token_embedding;

        self.t_ln_w = self.memory[offset .. offset + sizes.t_ln_w];
        offset += sizes.t_ln_w;

        self.t_ln_b = self.memory[offset .. offset + sizes.t_ln_b];
        offset += sizes.t_ln_b;

        self.t_Wqkv_w = self.memory[offset .. offset + sizes.t_Wqkv_w];
        offset += sizes.t_Wqkv_w;

        self.t_Wqkv_b = self.memory[offset .. offset + sizes.t_Wqkv_b];
        offset += sizes.t_Wqkv_b;

        self.t_out_proj_w = self.memory[offset .. offset + sizes.t_out_proj_w];
        offset += sizes.t_out_proj_w;

        self.t_out_proj_bias = self.memory[offset .. offset + sizes.t_out_proj_bias];
        offset += sizes.t_out_proj_bias;

        self.t_fc1_w = self.memory[offset .. offset + sizes.t_fc1_w];
        offset += sizes.t_fc1_w;

        self.t_fc1_b = self.memory[offset .. offset + sizes.t_fc1_b];
        offset += sizes.t_fc1_b;

        self.t_fc2_w = self.memory[offset .. offset + sizes.t_fc2_w];
        offset += sizes.t_fc2_w;

        self.t_fc2_b = self.memory[offset .. offset + sizes.t_fc2_b];
        offset += sizes.t_fc2_b;

        self.t_linear_w = self.memory[offset .. offset + sizes.t_linear_w];
        offset += sizes.t_linear_w;

        self.t_linear_b = self.memory[offset .. offset + sizes.t_linear_b];
        offset += sizes.t_linear_b;

        self.t_ln_out_w = self.memory[offset .. offset + sizes.t_ln_out_w];
        offset += sizes.t_ln_out_w;

        self.t_ln_out_b = self.memory[offset .. offset + sizes.t_ln_out_b];
        offset += sizes.t_ln_out_b;

        // vision model

        self.v_patch_embedding_linear_w = self.memory[offset .. offset + sizes.v_patch_embedding_linear_w];
        offset += sizes.v_patch_embedding_linear_w;

        self.v_patch_embedding_linear_b = self.memory[offset .. offset + sizes.v_patch_embedding_linear_b];
        offset += sizes.v_patch_embedding_linear_b;

        self.v_pos_embedding = self.memory[offset .. offset + sizes.v_pos_embedding];
        offset += sizes.v_pos_embedding;

        self.v_Wqkv_w = self.memory[offset .. offset + sizes.v_Wqkv_w];
        offset += sizes.v_Wqkv_w;

        self.v_Wqkv_b = self.memory[offset .. offset + sizes.v_Wqkv_b];
        offset += sizes.v_Wqkv_b;

        self.v_out_proj_w = self.memory[offset .. offset + sizes.v_out_proj_w];
        offset += sizes.v_out_proj_w;

        self.v_out_proj_b = self.memory[offset .. offset + sizes.v_out_proj_b];
        offset += sizes.v_out_proj_b;

        self.v_fc1_w = self.memory[offset .. offset + sizes.v_fc1_w];
        offset += sizes.v_fc1_w;

        self.v_fc1_b = self.memory[offset .. offset + sizes.v_fc1_b];
        offset += sizes.v_fc1_b;

        self.v_fc2_w = self.memory[offset .. offset + sizes.v_fc2_w];
        offset += sizes.v_fc2_w;

        self.v_fc2_b = self.memory[offset .. offset + sizes.v_fc2_b];
        offset += sizes.v_fc2_b;

        self.v_norm1_w = self.memory[offset .. offset + sizes.v_norm1_w];
        offset += sizes.v_norm1_w;

        self.v_norm1_b = self.memory[offset .. offset + sizes.v_norm1_b];
        offset += sizes.v_norm1_b;

        self.v_norm2_w = self.memory[offset .. offset + sizes.v_norm2_w];
        offset += sizes.v_norm2_w;

        self.v_norm2_b = self.memory[offset .. offset + sizes.v_norm2_b];
        offset += sizes.v_norm2_b;
        return self;
    }

    fn calculateSizes(config: Config) struct {
        //text
        word_token_embedding: usize,
        t_ln_w: usize,
        t_ln_b: usize,
        t_Wqkv_w: usize,
        t_Wqkv_b: usize,
        t_fc1_w: usize,
        t_fc1_b: usize,
        t_fc2_w: usize,
        t_fc2_b: usize,
        t_out_proj_w: usize,
        t_out_proj_bias: usize,
        t_linear_w: usize,
        t_linear_b: usize,
        t_ln_out_w: usize,
        t_ln_out_b: usize,
        // vision
        v_patch_embedding_linear_w: usize,
        v_patch_embedding_linear_b: usize,
        v_pos_embedding: usize,
        v_Wqkv_w: usize,
        v_Wqkv_b: usize,
        v_fc1_w: usize,
        v_fc1_b: usize,
        v_fc2_w: usize,
        v_fc2_b: usize,
        v_out_proj_w: usize,
        v_out_proj_b: usize, //TODO : move this up before fc1_w
        v_norm1_w: usize,
        v_norm1_b: usize,
        v_norm2_w: usize,
        v_norm2_b: usize,
    } {
        return .{
            // TODO : Recheck this once
            // text model
            .word_token_embedding = config.dim * config.vocab,
            .t_ln_w = config.n_layers * config.dim,
            .t_ln_b = config.n_layers * config.dim,
            .t_Wqkv_w = config.n_layers * config.dim * config.n_heads * config.head_dim * 3,
            .t_Wqkv_b = config.n_layers * config.n_heads * config.head_dim * 3,
            .t_fc1_w = config.n_layers * config.hidden_dim * config.dim,
            .t_fc1_b = config.n_layers * config.hidden_dim,
            .t_fc2_w = config.n_layers * config.dim * config.hidden_dim,
            .t_fc2_b = config.n_layers * config.dim,
            .t_out_proj_w = config.n_layers * config.seq_len * config.dim,
            .t_out_proj_bias = config.n_layers * config.dim,
            .t_linear_w = config.vocab * config.dim,
            .t_linear_b = config.vocab,
            .t_ln_out_w = config.dim,
            .t_ln_out_b = config.dim,

            //vision model
            .v_patch_embedding_linear_w = config.vit_dim * config.patch * config.patch * config.img_channels,
            .v_patch_embedding_linear_b = config.vit_dim,
            .v_pos_embedding = 1 * ((config.img_dim / config.patch) * (config.img_dim / config.patch)) * config.vit_dim,
            .v_Wqkv_w = config.vit_dim * config.vit_dim * 3,
            .v_Wqkv_b = config.vit_dim * 3,
            .v_out_proj_w = config.vit_dim * config.vit_dim,
            .v_out_proj_b = config.vit_dim,
            .v_fc1_w = config.hidden_features * config.vit_dim,
            .v_fc1_b = config.hidden_features,
            .v_fc2_w = config.vit_dim * config.hidden_features,
            .v_fc2_b = config.vit_dim,
            .v_norm1_w = config.hidden_features,
            .v_norm1_b = config.hidden_features,
            .v_norm2_w = config.hidden_features,
            .v_norm2_b = config.hidden_features,
        };
    }

    fn calculateTotalSize(sizes: anytype) usize {
        var total: usize = 0;
        inline for (std.meta.fields(@TypeOf(sizes))) |field| {
            total += @field(sizes, field.name);
        }
        return total;
    }
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

// main

pub fn main() !void {}

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
