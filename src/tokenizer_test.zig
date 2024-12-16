const std = @import("std");
const testing = std.testing;
const Tokenizer = @import("tokenizer.zig").Tokenizer;
const Allocator = std.mem.Allocator;

test "basic encode-decode roundtrip" {
    const allocator = testing.allocator;
    var tokenizer = try Tokenizer.fromFile("../tokenizer.bin", allocator);
    defer tokenizer.deinit();

    const input = "Hello world!";
    var encoded = try tokenizer.encode(input);
    defer encoded.deinit();

    const decoded = try tokenizer.decode(encoded);
    defer allocator.free(decoded);

    try testing.expectEqualStrings(input, decoded);
}

test "special characters handling" {
    const allocator = testing.allocator;
    var tokenizer = try Tokenizer.fromFile("../tokenizer.bin", allocator);
    defer tokenizer.deinit();

    // Test with newlines and spaces
    const input = "Hello\nworld  test";
    var encoded = try tokenizer.encode(input);
    defer encoded.deinit();

    const decoded = try tokenizer.decode(encoded);
    defer allocator.free(decoded);

    try testing.expectEqualStrings(input, decoded);
}

test "empty string" {
    const allocator = testing.allocator;
    var tokenizer = try Tokenizer.fromFile("../tokenizer.bin", allocator);
    defer tokenizer.deinit();

    const input = "";
    var encoded = try tokenizer.encode(input);
    defer encoded.deinit();

    const decoded = try tokenizer.decode(encoded);
    defer allocator.free(decoded);

    try testing.expectEqualStrings(input, decoded);
}

test "special tokens handling" {
    const allocator = testing.allocator;
    var tokenizer = try Tokenizer.fromFile("../tokenizer.bin", allocator);
    defer tokenizer.deinit();

    // Create array with special tokens
    var tokens = std.ArrayList(u32).init(allocator);
    defer tokens.deinit();

    try tokens.append(tokenizer.bos_token);
    try tokens.append(32); // space
    try tokens.append(tokenizer.eos_token);
    try tokens.append(tokenizer.pad_token);

    const decoded = try tokenizer.decode(tokens);
    defer allocator.free(decoded);

    // Special tokens should be stripped, leaving only the space
    try testing.expectEqualStrings(" ", decoded);
}

test "long text with mixed content" {
    const allocator = testing.allocator;
    var tokenizer = try Tokenizer.fromFile("../tokenizer.bin", allocator);
    defer tokenizer.deinit();

    const input =
        \\This is a longer text with multiple lines,
        \\special characters, and    multiple spaces!
        \\It should handle everything correctly.
    ;
    var encoded = try tokenizer.encode(input);
    defer encoded.deinit();

    const decoded = try tokenizer.decode(encoded);
    defer allocator.free(decoded);

    try testing.expectEqualStrings(input, decoded);
}

test "error handling - invalid token" {
    const allocator = testing.allocator;
    var tokenizer = try Tokenizer.fromFile("../tokenizer.bin", allocator);
    defer tokenizer.deinit();

    var tokens = std.ArrayList(u32).init(allocator);
    defer tokens.deinit();

    // Add an invalid token ID (assuming it's beyond your vocabulary size)
    try tokens.append(999999);

    // Should return error.TokenNotFound
    try testing.expectError(error.TokenNotFound, tokenizer.decode(tokens));
}

test "GPT2 tokenization" {
    const allocator = testing.allocator;
    var tokenizer = try Tokenizer.fromFile("../tokenizer.bin", allocator);
    defer tokenizer.deinit();

    // Test cases
    const test_cases = [_][]const u8{
        "Hello world", // Basic case
        "Hello  world", // Multiple spaces
        "Hello\nworld", // Newline
        "こんにちは", // UTF-8 text
        "Hello world! Testing spaces and UTF-8: 🌍こんにちは", // Mixed content
    };

    for (test_cases) |input| {
        var encoded = try tokenizer.encode(input);
        defer encoded.deinit();

        const decoded = try tokenizer.decode(encoded);
        defer allocator.free(decoded);

        // Print debug info
        std.debug.print("\nTest case: {s}\n", .{input});
        std.debug.print("Encoded tokens: ", .{});
        for (encoded.items) |token| {
            std.debug.print("{d} ", .{token});
        }
        std.debug.print("\nDecoded: {s}\n", .{decoded});

        // Check if the decoded text matches the input
        // Note: we normalize spaces in both input and output for comparison
        const normalized_input = try normalizeSpaces(input, allocator);
        defer allocator.free(normalized_input);
        const normalized_output = try normalizeSpaces(decoded, allocator);
        defer allocator.free(normalized_output);

        try testing.expectEqualStrings(normalized_input, normalized_output);
    }
}

fn normalizeSpaces(text: []const u8, allocator: Allocator) ![]u8 {
    var result = std.ArrayList(u8).init(allocator);
    errdefer result.deinit();

    var last_was_space = false;
    for (text) |c| {
        if (std.ascii.isWhitespace(c)) {
            if (!last_was_space) {
                try result.append(' ');
                last_was_space = true;
            }
        } else {
            try result.append(c);
            last_was_space = false;
        }
    }

    return result.toOwnedSlice();
}

test "consecutive spaces and newlines" {
    const allocator = testing.allocator;
    var tokenizer = try Tokenizer.fromFile("../tokenizer.bin", allocator);
    defer tokenizer.deinit();

    const input = "Hello    world\n\n\ntest";
    var encoded = try tokenizer.encode(input);
    defer encoded.deinit();

    const decoded = try tokenizer.decode(encoded);
    defer allocator.free(decoded);

    try testing.expectEqualStrings(input, decoded);
}

test "token boundaries" {
    const allocator = testing.allocator;
    var tokenizer = try Tokenizer.fromFile("../tokenizer.bin", allocator);
    defer tokenizer.deinit();

    // This test assumes certain token boundaries exist in your vocabulary
    const input = "testing token boundaries";
    var encoded = try tokenizer.encode(input);
    defer encoded.deinit();

    // Verify we got multiple tokens (not just byte-by-byte encoding)
    try testing.expect(encoded.items.len < input.len);

    const decoded = try tokenizer.decode(encoded);
    defer allocator.free(decoded);

    try testing.expectEqualStrings(input, decoded);
}

test "token boundaries stress test" {
    const allocator = testing.allocator;
    var tokenizer = try Tokenizer.fromFile("../tokenizer.bin", allocator);
    defer tokenizer.deinit();

    // Test various word boundaries and combinations
    try testRoundTrip(&tokenizer, "prefix_word word_suffix");
    try testRoundTrip(&tokenizer, "hyphenated-word");
    try testRoundTrip(&tokenizer, "under_scored_word");
    try testRoundTrip(&tokenizer, "camelCaseWord");
    try testRoundTrip(&tokenizer, "PascalCaseWord");
    try testRoundTrip(&tokenizer, "UPPER_CASE_WORD");
    try testRoundTrip(&tokenizer, "snake_case_word");
    try testRoundTrip(&tokenizer, "dot.separated.word");
}

test "error cases" {
    const allocator = testing.allocator;
    var tokenizer = try Tokenizer.fromFile("../tokenizer.bin", allocator);
    defer tokenizer.deinit();

    // Test invalid token ID
    var invalid_tokens = std.ArrayList(u32).init(allocator);
    defer invalid_tokens.deinit();
    try invalid_tokens.append(999999); // Invalid token ID

    try testing.expectError(error.TokenNotFound, tokenizer.decode(invalid_tokens));
}

test "URLs and paths" {
    const allocator = testing.allocator;
    var tokenizer = try Tokenizer.fromFile("../tokenizer.bin", allocator);
    defer tokenizer.deinit();

    // Test URLs and file paths
    try testRoundTrip(&tokenizer, "https://example.com");
    try testRoundTrip(&tokenizer, "file:///usr/local/bin");
    try testRoundTrip(&tokenizer, "/path/to/file.txt");
    try testRoundTrip(&tokenizer, "C:\\Windows\\System32");
    try testRoundTrip(&tokenizer, "user@example.com");
}

test "mixed language content" {
    const allocator = testing.allocator;
    var tokenizer = try Tokenizer.fromFile("../tokenizer.bin", allocator);
    defer tokenizer.deinit();

    // Test mixing different types of content
    try testRoundTrip(&tokenizer,
        \\Code: print("Hello!")
        \\Math: 1 + 1 = 2
        \\URL: https://example.com
        \\Path: /usr/bin/
        \\Quote: "Testing 123"
    );
}

// Helper function for round-trip testing
fn testRoundTrip(tokenizer: *Tokenizer, input: []const u8) !void {
    var encoded = try tokenizer.encode(input);
    defer encoded.deinit();

    const decoded = try tokenizer.decode(encoded);
    defer tokenizer.allocator.free(decoded);

    try testing.expectEqualStrings(input, decoded);
}

test "empty and whitespace strings" {
    const allocator = testing.allocator;
    var tokenizer = try Tokenizer.fromFile("../tokenizer.bin", allocator);
    defer tokenizer.deinit();

    // Test empty string
    try testRoundTrip(&tokenizer, "");

    // Test various whitespace combinations
    try testRoundTrip(&tokenizer, " ");
    try testRoundTrip(&tokenizer, "  ");
    try testRoundTrip(&tokenizer, "   ");
    try testRoundTrip(&tokenizer, "\n");
    try testRoundTrip(&tokenizer, "\n\n");
    try testRoundTrip(&tokenizer, " \n ");
    try testRoundTrip(&tokenizer, "  \n  ");
}

test "punctuation and special characters" {
    const allocator = testing.allocator;
    var tokenizer = try Tokenizer.fromFile("../tokenizer.bin", allocator);
    defer tokenizer.deinit();

    // Test common punctuation
    try testRoundTrip(&tokenizer, "Hello, world!");
    try testRoundTrip(&tokenizer, "Testing: 1, 2, 3...");
    try testRoundTrip(&tokenizer, "What's up? Not much!");
    try testRoundTrip(&tokenizer, "Semi;Colon;Test");
    try testRoundTrip(&tokenizer, "Dash-test");
    try testRoundTrip(&tokenizer, "[Brackets] {and} (parentheses)");
}

test "code-like content" {
    const allocator = testing.allocator;
    var tokenizer = try Tokenizer.fromFile("../tokenizer.bin", allocator);
    defer tokenizer.deinit();

    // Test code snippets
    try testRoundTrip(&tokenizer,
        \\fn main() {
        \\    printf("Hello, world!\n");
        \\    return 0;
        \\}
    );

    try testRoundTrip(&tokenizer, "var x = 42; // Comment");
    try testRoundTrip(&tokenizer, "function test() { return true; }");
    try testRoundTrip(&tokenizer, "SELECT * FROM table WHERE id = 1;");
}

test "numbers and mixed content" {
    const allocator = testing.allocator;
    var tokenizer = try Tokenizer.fromFile("../tokenizer.bin", allocator);
    defer tokenizer.deinit();

    // Test various number formats
    try testRoundTrip(&tokenizer, "12345");
    try testRoundTrip(&tokenizer, "3.14159");
    try testRoundTrip(&tokenizer, "-42");
    try testRoundTrip(&tokenizer, "1,000,000");
    try testRoundTrip(&tokenizer, "1e-10");
    try testRoundTrip(&tokenizer, "0xFF");
    try testRoundTrip(&tokenizer, "Temperature: 98.6°F");
}

test "repetitive patterns" {
    const allocator = testing.allocator;
    var tokenizer = try Tokenizer.fromFile("../tokenizer.bin", allocator);
    defer tokenizer.deinit();

    // Test repeating patterns
    try testRoundTrip(&tokenizer, "ha ha ha ha ha");
    try testRoundTrip(&tokenizer, "test test test");
    try testRoundTrip(&tokenizer, "a b a b a b a b");
    try testRoundTrip(&tokenizer, "......................");
    try testRoundTrip(&tokenizer, "* * * * * * * * * *");
}

test "long texts" {
    const allocator = testing.allocator;
    var tokenizer = try Tokenizer.fromFile("../tokenizer.bin", allocator);
    defer tokenizer.deinit();

    // Test long paragraph
    try testRoundTrip(&tokenizer,
        \\The quick brown fox jumps over the lazy dog. This pangram contains every letter of the English alphabet at least once. 
        \\Jackdaws love my big sphinx of quartz. Pack my box with five dozen liquor jugs. How vexingly quick daft zebras jump! 
        \\The five boxing wizards jump quickly. Sphinx of black quartz, judge my vow.
    );

    // Test repeated words with varying spacing
    try testRoundTrip(&tokenizer,
        \\word word  word   word
        \\word     word      word
        \\word word word
    );
}

test "special token sequences" {
    const allocator = testing.allocator;
    var tokenizer = try Tokenizer.fromFile("../tokenizer.bin", allocator);
    defer tokenizer.deinit();

    // Create array with special tokens
    var tokens = std.ArrayList(u32).init(allocator);
    defer tokens.deinit();

    // Test different combinations of special tokens
    try tokens.append(tokenizer.bos_token);
    try tokens.append(32); // space
    try tokens.append(tokenizer.eos_token);
    try tokens.append(tokenizer.pad_token);

    const decoded = try tokenizer.decode(tokens);
    defer allocator.free(decoded);

    try testing.expectEqualStrings(" ", decoded);
}
