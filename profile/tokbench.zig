const std = @import("std");
const time = std.time;
const Timer = time.Timer;
const print = std.debug.print;
const Tokenizer = @import("tokenizer.zig").Tokenizer;
const Allocator = std.mem.Allocator;

pub fn benchmark(tokenizer: *const Tokenizer, filepath: []const u8, allocator: Allocator) !void {
    // Start the timer
    var timer = try Timer.start();
    const start = timer.lap();

    // Open and get file size
    const file = try std.fs.cwd().openFile(filepath, .{});
    defer file.close();
    const file_size = try file.getEndPos();

    // Read file content
    const content = try file.readToEndAlloc(allocator, file_size);
    defer allocator.free(content);

    // Tokenization phase timing
    const tokenize_start = timer.lap();
    var tokens = try tokenizer.encode(content);
    defer tokens.deinit();
    const tokenize_end = timer.lap();

    // Calculate timing metrics
    const total_time_ns = tokenize_end - start;
    const tokenize_time_ns = tokenize_end - tokenize_start;

    // Convert to milliseconds for readable output
    const total_time_ms = @as(f64, @floatFromInt(total_time_ns)) / time.ns_per_ms;
    const tokenize_time_ms = @as(f64, @floatFromInt(tokenize_time_ns)) / time.ns_per_ms;

    // Calculate tokens per second
    const tokens_per_second = @as(f64, @floatFromInt(tokens.items.len)) / (@as(f64, @floatFromInt(tokenize_time_ns)) / time.ns_per_s);

    // Calculate MB per second
    const mb_per_second = (@as(f64, @floatFromInt(file_size)) / 1_048_576.0) / (@as(f64, @floatFromInt(total_time_ns)) / time.ns_per_s);

    // Print benchmark results
    print("\n=== Tokenizer Benchmark Results ===\n", .{});
    print("File size: {d:.2} MB\n", .{@as(f64, @floatFromInt(file_size)) / 1_048_576.0});
    print("Total tokens: {d}\n", .{tokens.items.len});
    print("Total time: {d:.2} ms\n", .{total_time_ms});
    print("Tokenization time: {d:.2} ms\n", .{tokenize_time_ms});
    print("Throughput: {d:.2} tokens/second\n", .{tokens_per_second});
    print("Processing speed: {d:.2} MB/second\n", .{mb_per_second});
    print("Average tokens per byte: {d:.3}\n", .{@as(f64, @floatFromInt(tokens.items.len)) / @as(f64, @floatFromInt(file_size))});
    print("==============================\n\n", .{});

    // Optional: Sample of first few tokens
    print("First 10 tokens: ", .{});
    const sample_size = @min(10, tokens.items.len);
    for (tokens.items[0..sample_size]) |token| {
        print("{d} ", .{token});
    }
    print("\n", .{});

    // Create new mutable ArrayList for sampling
    var sample_tokens = try std.ArrayList(u32).initCapacity(allocator, sample_size);
    defer sample_tokens.deinit();

    // Copy the tokens we want to sample
    const sample_items = tokens.items[0..sample_size];
    try sample_tokens.appendSlice(sample_items[0..]);

    const sample_decoded = try tokenizer.decode(sample_tokens);
    defer allocator.free(sample_decoded);
    print("Sample decoded text: {s}\n", .{sample_decoded});
}

pub fn main() !void {
    // Initialize memory allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize tokenizer
    print("Loading tokenizer...\n", .{});
    var tokenizer = try Tokenizer.fromFile("../tokenizer.bin", allocator);
    defer tokenizer.deinit();
    print("Tokenizer loaded successfully.\n", .{});

    // Run benchmark multiple times
    const num_runs = 3;
    print("\nRunning {d} benchmark iterations...\n", .{num_runs});

    for (0..num_runs) |i| {
        print("\nBenchmark run {d}/{d}\n", .{ i + 1, num_runs });
        try benchmark(&tokenizer, "shakespeare.txt", allocator);
    }
}
