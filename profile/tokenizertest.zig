const std = @import("std");
const testing = std.testing;
const Tokenizer = @import("tokenizer.zig").Tokenizer;

// Test helpers
fn printTokensDetailed(tokenizer: *Tokenizer, tokens: std.ArrayList(u32)) !void {
    const writer = std.io.getStdOut().writer();

    // Print token IDs
    try writer.print("\nToken IDs ({} total): [", .{tokens.items.len});
    for (tokens.items, 0..) |token, i| {
        try writer.print("{}", .{token});
        if (i < tokens.items.len - 1) try writer.print(", ", .{});
    }
    try writer.print("]\n", .{});

    // Print detailed breakdown
    try writer.writeAll("\nDetailed token breakdown:\n");
    for (tokens.items) |token_id| {
        // Decode single token
        var single_token = std.ArrayList(u32).init(tokenizer.allocator);
        defer single_token.deinit();
        try single_token.append(token_id);

        const token_str = try tokenizer.decode(single_token);
        defer tokenizer.allocator.free(token_str);

        // Print token details
        try writer.print("Token ID: {d:5} → ", .{token_id});

        // Print string representation with special characters visible
        try writer.writeAll("'");
        for (token_str) |byte| {
            switch (byte) {
                ' ' => try writer.writeAll("␣"), // Space
                '\n' => try writer.writeAll("⏎"), // Newline
                '\r' => try writer.writeAll("⏎"), // Carriage return
                '\t' => try writer.writeAll("⇥"), // Tab
                else => {
                    if (std.ascii.isPrint(byte)) {
                        try writer.writeByte(byte);
                    } else {
                        try writer.print("\\x{X:0>2}", .{byte});
                    }
                },
            }
        }
        try writer.writeAll("'\n");
    }
}

fn testEncodeDecode(tokenizer: *Tokenizer, text: []const u8) !bool {
    const tokens = try tokenizer.encode(text);
    defer tokens.deinit();
    const decoded = try tokenizer.decode(tokens);
    defer tokenizer.allocator.free(decoded);
    return std.mem.eql(u8, text, decoded);
}

test "Basic tokenization" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var tokenizer = try Tokenizer.fromFile("../tokenizer.bin", allocator);
    defer tokenizer.deinit();

    // Test simple words
    {
        const text = "Hello world";
        const tokens = try tokenizer.encode(text);
        defer tokens.deinit();
        try testing.expect(tokens.items.len > 0);
    }

    // Test special characters
    {
        const text = "Hello\nworld";
        const tokens = try tokenizer.encode(text);
        defer tokens.deinit();
        try testing.expect(tokens.items.len > 0);
    }
}

test "Encode-decode roundtrip" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var tokenizer = try Tokenizer.fromFile("../tokenizer.bin", allocator);
    defer tokenizer.deinit();

    const test_cases = [_][]const u8{
        "Hello world",
        "This is a test.",
        "Multiple\nline\ntext",
        "Special characters: !@#$%^&*()",
        "Numbers 123456789",
        "Mixed case TeXt",
        "", // Empty string
        " ", // Single space
        "\n", // Single newline
        "  ", // Multiple spaces
        "\n\n", // Multiple newlines
    };

    for (test_cases) |text| {
        std.debug.print("test case : {any}", .{text});
        try testing.expect(try testEncodeDecode(&tokenizer, text));
    }
}

test "Special tokens" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var tokenizer = try Tokenizer.fromFile("../tokenizer.bin", allocator);
    defer tokenizer.deinit();

    // Test BOS/EOS tokens
    {
        const text = "Start End";
        var tokens = try tokenizer.encode(text);
        defer tokens.deinit();

        // Insert BOS/EOS tokens
        try tokens.insert(0, tokenizer.bos_token);
        try tokens.append(tokenizer.eos_token);

        // Decode should ignore special tokens
        const decoded = try tokenizer.decode(tokens);
        defer allocator.free(decoded);
        try testing.expectEqualStrings(text, decoded);
    }
}

test "UTF-8 handling" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var tokenizer = try Tokenizer.fromFile("../tokenizer.bin", allocator);
    defer tokenizer.deinit();

    const test_cases = [_][]const u8{"oogabooga"};

    for (test_cases) |text| {
        try testing.expect(try testEncodeDecode(&tokenizer, text));
    }
}

// Main function for interactive testing
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const stdout = std.io.getStdOut().writer();
    const stdin = std.io.getStdIn().reader();

    // // Initialize tokenizer
    // var tokenizer = try Tokenizer.fromFile("../../tokenizer.bin", allocator);
    // defer tokenizer.deinit();

    var tokenizer = try Tokenizer.fromJson("../tokenizer.json", allocator);
    defer tokenizer.deinit();
    // Buffer for reading input
    var buffer: [1024]u8 = undefined;

    try stdout.writeAll("\nGPT-2 Style Tokenizer Interactive Tester\n");
    try stdout.writeAll("Enter 'q' to quit, 'test' to run test suite\n");
    try stdout.writeAll("Or enter any text to tokenize:\n");

    while (true) {
        try stdout.writeAll("\n> ");
        const input = try stdin.readUntilDelimiter(&buffer, '\n');

        // Tokenize input
        const tokens = try tokenizer.encode(input);
        defer tokens.deinit();

        // Print detailed token information
        try stdout.writeAll("\n=== Token Analysis ===\n");
        try printTokensDetailed(&tokenizer, tokens);

        // Decode and print result
        const decoded = try tokenizer.decode(tokens);
        defer allocator.free(decoded);

        // Print original and decoded text with special characters visible
        try stdout.writeAll("\nOriginal text: '");
        for (input) |byte| {
            switch (byte) {
                ' ' => try stdout.writeAll("␣"),
                '\n' => try stdout.writeAll("⏎"),
                '\r' => try stdout.writeAll("⏎"),
                '\t' => try stdout.writeAll("⇥"),
                else => {
                    if (std.ascii.isPrint(byte)) {
                        try stdout.writeByte(byte);
                    } else {
                        try stdout.print("\\x{X:0>2}", .{byte});
                    }
                },
            }
        }
        try stdout.writeAll("'\n");

        try stdout.writeAll("Decoded text: '");
        for (decoded) |byte| {
            switch (byte) {
                ' ' => try stdout.writeAll("␣"),
                '\n' => try stdout.writeAll("⏎"),
                '\r' => try stdout.writeAll("⏎"),
                '\t' => try stdout.writeAll("⇥"),
                else => {
                    if (std.ascii.isPrint(byte)) {
                        try stdout.writeByte(byte);
                    } else {
                        try stdout.print("\\x{X:0>2}", .{byte});
                    }
                },
            }
        }
        try stdout.writeAll("'\n");

        // Print comparison
        try stdout.print("\nInput matches decoded: {}\n", .{std.mem.eql(u8, input, decoded)});
    }
}
