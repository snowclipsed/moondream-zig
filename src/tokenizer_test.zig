const std = @import("std");
const Tokenizer = @import("tokenizer.zig").Tokenizer;
const testing = std.testing;
const expectEqual = testing.expectEqual;
const expectError = testing.expectError;

test "tokenizer" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const check = gpa.deinit();
        if (check == .leak) {
            std.debug.print("WARNING: GPA detected memory leaks!\n", .{});
        }
    }
    const allocator = gpa.allocator();

    var tokenizer = try Tokenizer.fromFile("/home/snow/projects/moondream-zig/tokenizer.bin", allocator);
    defer tokenizer.deinit();

    {
        std.debug.print("\n=== Testing Special Character Encoding ===\n", .{});
        const test_str = "a\nb";
        var tokens = try tokenizer.encode(test_str);
        defer tokens.deinit();

        std.debug.print("Input: 'a\\nb'\n", .{});
        std.debug.print("Tokens: ", .{});
        for (tokens.items) |token| {
            std.debug.print("{} ", .{token});
        }
        std.debug.print("\n", .{});

        const decoded = try tokenizer.decode(tokens);
        defer allocator.free(decoded);
        std.debug.print("Decoded: '{s}'\n", .{decoded});
    }

    // First, let's examine special tokens
    {
        std.debug.print("\n=== Special Tokens Analysis ===\n", .{});
        for (tokenizer.special_tokens.items) |special| {
            std.debug.print("Special token ID {}: '{s}' (is_special: {})\n", .{
                special.id,
                special.content,
                special.is_special,
            });
        }
    }

    // Test full prompt tokenization
    {
        std.debug.print("\n=== Testing Full Prompt Tokenization ===\n", .{});
        const prompt = "<image>\nQuestion: what is in this image?\nAnswer:";
        var tokens = try tokenizer.encode(prompt);
        defer tokens.deinit();

        std.debug.print("Input prompt: '{s}'\n", .{prompt});
        std.debug.print("Token sequence ({} tokens):\n", .{tokens.items.len});
        for (tokens.items, 0..) |token, i| {
            // Try to decode each token individually
            var single_token = std.ArrayList(u32).init(allocator);
            defer single_token.deinit();
            try single_token.append(token);
            const decoded = try tokenizer.decode(single_token);
            defer allocator.free(decoded);

            std.debug.print("Token {}: ID {} -> '{s}'\n", .{ i, token, decoded });
        }
    }

    // Test with BOS token prepended
    {
        std.debug.print("\n=== Testing with BOS Token ===\n", .{});
        const prompt = "<image>\nQuestion: what is in this image?\nAnswer:";
        var tokens = std.ArrayList(u32).init(allocator);
        defer tokens.deinit();

        // Add BOS token
        try tokens.append(tokenizer.bos_token);

        // Add encoded prompt
        var prompt_tokens = try tokenizer.encode(prompt);
        defer prompt_tokens.deinit();
        try tokens.appendSlice(prompt_tokens.items);

        std.debug.print("Full token sequence with BOS ({} tokens):\n", .{tokens.items.len});
        for (tokens.items, 0..) |token, i| {
            var single_token = std.ArrayList(u32).init(allocator);
            defer single_token.deinit();
            try single_token.append(token);
            const decoded = try tokenizer.decode(single_token);
            defer allocator.free(decoded);

            std.debug.print("Token {}: ID {} -> '{s}'\n", .{ i, token, decoded });
        }

        // Try decoding the full sequence
        const full_decoded = try tokenizer.decode(tokens);
        defer allocator.free(full_decoded);
        std.debug.print("\nFull sequence decoded: '{s}'\n", .{full_decoded});
    }

    // Test edge cases
    {
        std.debug.print("\n=== Testing Edge Cases ===\n", .{});

        // Test empty string
        {
            var tokens = try tokenizer.encode("");
            defer tokens.deinit();
            std.debug.print("Empty string produces {} tokens\n", .{tokens.items.len});
        }

        // Test whitespace handling
        {
            var tokens = try tokenizer.encode(" \n \t ");
            defer tokens.deinit();
            std.debug.print("Whitespace string produces {} tokens: ", .{tokens.items.len});
            for (tokens.items) |token| {
                std.debug.print("{} ", .{token});
            }
            std.debug.print("\n", .{});
        }
    }
}
