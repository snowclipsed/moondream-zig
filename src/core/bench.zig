const std = @import("std");
const Allocator = std.mem.Allocator;
const print = std.debug.print;
const time = std.time;
const math = std.math;

// Import the implementations
const sgemm = @import("sgemm.zig");
const hgemm = @import("hgemm.zig");
const hgemm_trans = @import("hgemm_trans.zig");
const layerNorm = @import("ops.zig").layerNorm;
const layerNormYoloInner = @import("ops.zig").layerNormYoloInner;

// Import common dependency
const Tensor = @import("tensor.zig").Tensor;

// ANSI color codes for terminal output
pub const Color = struct {
    pub const reset = "\x1b[0m";
    pub const bold = "\x1b[1m";
    pub const red = "\x1b[31m";
    pub const green = "\x1b[32m";
    pub const yellow = "\x1b[33m";
    pub const blue = "\x1b[34m";
    pub const magenta = "\x1b[35m";
    pub const cyan = "\x1b[36m";
    pub const white = "\x1b[37m";
    pub const bright_red = "\x1b[91m";
    pub const bright_green = "\x1b[92m";
    pub const bright_yellow = "\x1b[93m";
    pub const bright_blue = "\x1b[94m";
    pub const bright_magenta = "\x1b[95m";
    pub const bright_cyan = "\x1b[96m";
    pub const bright_white = "\x1b[97m";
};

// Data structures
pub const BenchmarkResult = struct {
    avg_time_ns: u64,
    min_time_ns: u64,
    max_time_ns: u64,
    avg_gflops: f64,
    max_gflops: f64,
    avg_bandwidth_gbps: f64,
};

const MatrixSize = struct {
    m: usize,
    n: usize,
    k: usize,
    desc: []const u8,
};

const ComparativeResult = struct {
    size: MatrixSize,
    sgemm: BenchmarkResult,
    hgemm: BenchmarkResult,
    hgemm_trans: BenchmarkResult,
};

// Benchmark functions
fn benchmarkSgemm(allocator: Allocator, M: usize, N: usize, K: usize, num_runs: usize) !BenchmarkResult {
    // Create tensors
    var a = try Tensor(f32).init(allocator, &[_]usize{ M, K });
    defer a.deinit();
    var b = try Tensor(f32).init(allocator, &[_]usize{ K, N });
    defer b.deinit();

    // Initialize with random data
    var rng = std.rand.DefaultPrng.init(42);
    const random = rng.random();
    for (a.data) |*val| {
        val.* = random.float(f32) * 0.1 - 0.05;
    }
    for (b.data) |*val| {
        val.* = random.float(f32) * 0.1 - 0.05;
    }

    // Warmup run
    var warmup = try sgemm.matmul(f32, a, b, allocator);
    warmup.deinit();

    // Benchmark runs
    var total_time: u64 = 0;
    var min_time: u64 = std.math.maxInt(u64);
    var max_time: u64 = 0;
    var max_gflops: f64 = 0;

    for (0..num_runs) |_| {
        var timer = try time.Timer.start();
        var result = try sgemm.matmul(f32, a, b, allocator);
        const elapsed = timer.read();
        result.deinit();

        total_time += elapsed;
        min_time = @min(min_time, elapsed);
        max_time = @max(max_time, elapsed);

        // Calculate GFLOPS for this run
        const seconds = @as(f64, @floatFromInt(elapsed)) / 1e9;
        const ops = 2 * M * N * K; // multiply-add is 2 operations
        const gflops = @as(f64, @floatFromInt(ops)) / seconds / 1e9;
        max_gflops = @max(max_gflops, gflops);
    }

    const avg_time = total_time / num_runs;
    const avg_seconds = @as(f64, @floatFromInt(avg_time)) / 1e9;
    const ops = 2 * M * N * K;
    const avg_gflops = @as(f64, @floatFromInt(ops)) / avg_seconds / 1e9;

    // Calculate memory bandwidth (in GB/s)
    const bytes_accessed = @as(f64, @floatFromInt((M * K + K * N + M * N) * @sizeOf(f32)));
    const avg_bandwidth = bytes_accessed / avg_seconds / 1e9; // Convert to GB/s

    return BenchmarkResult{
        .avg_time_ns = avg_time,
        .min_time_ns = min_time,
        .max_time_ns = max_time,
        .avg_gflops = avg_gflops,
        .max_gflops = max_gflops,
        .avg_bandwidth_gbps = avg_bandwidth,
    };
}

fn benchmarkHgemm(allocator: Allocator, M: usize, N: usize, K: usize, num_runs: usize) !BenchmarkResult {
    // Create tensors
    var a = try Tensor(f16).init(allocator, &[_]usize{ M, K });
    defer a.deinit();
    var b = try Tensor(f16).init(allocator, &[_]usize{ K, N });
    defer b.deinit();

    // Initialize with random data
    var rng = std.rand.DefaultPrng.init(42);
    const random = rng.random();
    for (a.data) |*val| {
        val.* = @floatCast(random.float(f32) * 0.1 - 0.05);
    }
    for (b.data) |*val| {
        val.* = @floatCast(random.float(f32) * 0.1 - 0.05);
    }

    // Warmup run
    var warmup = try hgemm.matmul(a, b, allocator);
    warmup.deinit();

    // Benchmark runs
    var total_time: u64 = 0;
    var min_time: u64 = std.math.maxInt(u64);
    var max_time: u64 = 0;
    var max_gflops: f64 = 0;

    for (0..num_runs) |_| {
        var timer = try time.Timer.start();
        var result = try hgemm.matmul(a, b, allocator);
        const elapsed = timer.read();
        result.deinit();

        total_time += elapsed;
        min_time = @min(min_time, elapsed);
        max_time = @max(max_time, elapsed);

        // Calculate GFLOPS for this run
        const seconds = @as(f64, @floatFromInt(elapsed)) / 1e9;
        const ops = 2 * M * N * K;
        const gflops = @as(f64, @floatFromInt(ops)) / seconds / 1e9;
        max_gflops = @max(max_gflops, gflops);
    }

    const avg_time = total_time / num_runs;
    const avg_seconds = @as(f64, @floatFromInt(avg_time)) / 1e9;
    const ops = 2 * M * N * K;
    const avg_gflops = @as(f64, @floatFromInt(ops)) / avg_seconds / 1e9;

    // Calculate memory bandwidth
    const bytes_accessed = @as(f64, @floatFromInt((M * K + K * N + M * N) * @sizeOf(f16)));
    const avg_bandwidth = bytes_accessed / avg_seconds / 1e9;

    return BenchmarkResult{
        .avg_time_ns = avg_time,
        .min_time_ns = min_time,
        .max_time_ns = max_time,
        .avg_gflops = avg_gflops,
        .max_gflops = max_gflops,
        .avg_bandwidth_gbps = avg_bandwidth,
    };
}

fn benchmarkHgemmTrans(allocator: Allocator, M: usize, N: usize, K: usize, num_runs: usize) !BenchmarkResult {
    // Create tensors
    var a = try Tensor(f16).init(allocator, &[_]usize{ M, K });
    defer a.deinit();
    var b = try Tensor(f16).init(allocator, &[_]usize{ K, N });
    defer b.deinit();

    // Initialize with random data
    var rng = std.rand.DefaultPrng.init(42);
    const random = rng.random();
    for (a.data) |*val| {
        val.* = @floatCast(random.float(f32) * 0.1 - 0.05);
    }
    for (b.data) |*val| {
        val.* = @floatCast(random.float(f32) * 0.1 - 0.05);
    }

    // Pre-transpose B for HGEMM_TRANS
    var b_t = try hgemm_trans.transpose(allocator, b);
    defer b_t.deinit();

    // Warmup run
    var warmup = try hgemm_trans.matmul(a, b_t, allocator);
    warmup.deinit();

    // Benchmark runs
    var total_time: u64 = 0;
    var min_time: u64 = std.math.maxInt(u64);
    var max_time: u64 = 0;
    var max_gflops: f64 = 0;

    for (0..num_runs) |_| {
        var timer = try time.Timer.start();
        var result = try hgemm_trans.matmul(a, b_t, allocator);
        const elapsed = timer.read();
        result.deinit();

        total_time += elapsed;
        min_time = @min(min_time, elapsed);
        max_time = @max(max_time, elapsed);

        // Calculate GFLOPS for this run
        const seconds = @as(f64, @floatFromInt(elapsed)) / 1e9;
        const ops = 2 * M * N * K;
        const gflops = @as(f64, @floatFromInt(ops)) / seconds / 1e9;
        max_gflops = @max(max_gflops, gflops);
    }

    const avg_time = total_time / num_runs;
    const avg_seconds = @as(f64, @floatFromInt(avg_time)) / 1e9;
    const ops = 2 * M * N * K;
    const avg_gflops = @as(f64, @floatFromInt(ops)) / avg_seconds / 1e9;

    // Calculate memory bandwidth
    const bytes_accessed = @as(f64, @floatFromInt((M * K + K * N + M * N) * @sizeOf(f16)));
    const avg_bandwidth = bytes_accessed / avg_seconds / 1e9;

    return BenchmarkResult{
        .avg_time_ns = avg_time,
        .min_time_ns = min_time,
        .max_time_ns = max_time,
        .avg_gflops = avg_gflops,
        .max_gflops = max_gflops,
        .avg_bandwidth_gbps = avg_bandwidth,
    };
}

// Utility functions for formatting and visualization
pub fn getColorForValue(value: f64, min: f64, max: f64) []const u8 {
    // Normalize value to 0-1 range
    const range = max - min;
    if (range <= 0.0001) return Color.yellow;

    const normalized = (value - min) / range;

    // Color gradient: red (0.0) -> yellow (0.5) -> green (1.0)
    if (normalized < 0.33) {
        return Color.red;
    } else if (normalized < 0.67) {
        return Color.yellow;
    } else {
        return Color.green;
    }
}

pub fn getInverseColorForValue(value: f64, min: f64, max: f64) []const u8 {
    // For metrics where lower is better (like execution time)
    // Normalize and invert value to 0-1 range
    const range = max - min;
    if (range <= 0.0001) return Color.yellow;

    const normalized = 1.0 - (value - min) / range;

    // Color gradient: red (0.0) -> yellow (0.5) -> green (1.0)
    if (normalized < 0.33) {
        return Color.red;
    } else if (normalized < 0.67) {
        return Color.yellow;
    } else {
        return Color.green;
    }
}

fn determineBestImpl(sgemm_gflops: f64, hgemm_gflops: f64, hgemm_trans_gflops: f64) []const u8 {
    if (sgemm_gflops >= hgemm_gflops and sgemm_gflops >= hgemm_trans_gflops) {
        return "SGEMM";
    } else if (hgemm_gflops >= sgemm_gflops and hgemm_gflops >= hgemm_trans_gflops) {
        return "HGEMM";
    } else {
        return "HGEMM_TRANS";
    }
}

pub fn printColoredValue(value: f64, color: []const u8, width: usize, precision: usize) void {
    // Use fixed formatting and then pad manually
    var buffer: [64]u8 = undefined;

    const formatted = switch (precision) {
        1 => std.fmt.bufPrint(&buffer, "{d:.1}", .{value}) catch unreachable,
        2 => std.fmt.bufPrint(&buffer, "{d:.2}", .{value}) catch unreachable,
        3 => std.fmt.bufPrint(&buffer, "{d:.3}", .{value}) catch unreachable,
        4 => std.fmt.bufPrint(&buffer, "{d:.4}", .{value}) catch unreachable,
        else => std.fmt.bufPrint(&buffer, "{d}", .{value}) catch unreachable,
    };

    // Apply manual right padding
    var padded_buffer: [64]u8 = undefined;
    const padding = if (width > formatted.len) width - formatted.len else 0;

    // Fill with spaces for padding
    for (0..padding) |i| {
        padded_buffer[i] = ' ';
    }

    // Copy the formatted string after padding
    @memcpy(padded_buffer[padding..], formatted);

    print("{s}{s}{s}", .{ color, padded_buffer[0 .. padding + formatted.len], Color.reset });
}

pub fn printColoredHeader(text: []const u8) void {
    print("\n{s}{s}{s}{s}\n", .{ Color.bold, Color.bright_cyan, text, Color.reset });
}

pub fn main() !void {
    // Initialize allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Benchmarking parameters
    const num_runs = 5;

    // Define matrix sizes to benchmark
    const sizes = [_]MatrixSize{
        // Common sizes from all three implementations
        .{ .m = 512, .n = 512, .k = 512, .desc = "Square (512)" },
        .{ .m = 1024, .n = 1024, .k = 1024, .desc = "Square (1024)" },
        .{ .m = 2048, .n = 2048, .k = 2048, .desc = "Square (2048)" },
        .{ .m = 768, .n = 512, .k = 256, .desc = "Rectangular (768x256)" },
        .{ .m = 2048, .n = 1024, .k = 512, .desc = "Rectangular (2048x512)" },
        .{ .m = 160, .n = 160, .k = 160, .desc = "Tile-sized (160)" },

        // Vector sizes (important for ML workloads)
        .{ .m = 1, .n = 2048, .k = 6144, .desc = "Vector-Matrix (large)" },
        .{ .m = 800, .n = 2048, .k = 6144, .desc = "Matrix-Matrix (large)" },
        .{ .m = 1, .n = 2048, .k = 2048, .desc = "Vector-Matrix (medium)" },

        // Additional square sizes
        .{ .m = 4096, .n = 4096, .k = 4096, .desc = "Square (4096)" },
    };

    // Print header with system information
    print("\n{s}{s}Matrix Multiplication Benchmark{s}\n", .{ Color.bold, Color.bright_green, Color.reset });
    print("╔══════════════════════════════════════════════════════════════╗\n", .{});
    print("║ {s}System Information{s}                                           ║\n", .{ Color.bright_yellow, Color.reset });
    print("╟──────────────────────────────────────────────────────────────╢\n", .{});
    print("║ CPU Threads:          {d:>10}                             ║\n", .{try std.Thread.getCpuCount()});
    print("║ SGEMM Tile Size:      {d:>10}                             ║\n", .{sgemm.Tile});
    print("║ HGEMM Tile Size:      {d:>10}                             ║\n", .{hgemm.T});
    print("║ HGEMM_TRANS Tile Size:{d:>10}                             ║\n", .{hgemm_trans.T});
    print("║ Iterations per test:  {d:>10}                             ║\n", .{num_runs});
    print("╚══════════════════════════════════════════════════════════════╝\n\n", .{});

    print("Running benchmarks...\n", .{});

    // Run all benchmarks and collect results
    var results = std.ArrayList(ComparativeResult).init(allocator);
    defer results.deinit();

    for (sizes) |size| {
        print("  Benchmarking {s} ({d}x{d}x{d})...\n", .{ size.desc, size.m, size.n, size.k });

        const sgemm_result = try benchmarkSgemm(allocator, size.m, size.n, size.k, num_runs);
        const hgemm_result = try benchmarkHgemm(allocator, size.m, size.n, size.k, num_runs);
        const hgemm_trans_result = try benchmarkHgemmTrans(allocator, size.m, size.n, size.k, num_runs);

        try results.append(.{
            .size = size,
            .sgemm = sgemm_result,
            .hgemm = hgemm_result,
            .hgemm_trans = hgemm_trans_result,
        });
    }

    // ===== PERFORMANCE COMPARISON TABLE =====

    // Find min/max GFLOPS for color normalization
    var min_gflops: f64 = std.math.inf(f64);
    var max_gflops: f64 = 0;

    for (results.items) |result| {
        min_gflops = @min(min_gflops, result.sgemm.avg_gflops);
        min_gflops = @min(min_gflops, result.hgemm.avg_gflops);
        min_gflops = @min(min_gflops, result.hgemm_trans.avg_gflops);

        max_gflops = @max(max_gflops, result.sgemm.avg_gflops);
        max_gflops = @max(max_gflops, result.hgemm.avg_gflops);
        max_gflops = @max(max_gflops, result.hgemm_trans.avg_gflops);
    }

    // Find min/max times for color normalization
    var min_time: f64 = std.math.inf(f64);
    var max_time: f64 = 0;

    for (results.items) |result| {
        const sgemm_ms = @as(f64, @floatFromInt(result.sgemm.min_time_ns)) / 1e6;
        const hgemm_ms = @as(f64, @floatFromInt(result.hgemm.min_time_ns)) / 1e6;
        const hgemm_trans_ms = @as(f64, @floatFromInt(result.hgemm_trans.min_time_ns)) / 1e6;

        min_time = @min(min_time, sgemm_ms);
        min_time = @min(min_time, hgemm_ms);
        min_time = @min(min_time, hgemm_trans_ms);

        max_time = @max(max_time, sgemm_ms);
        max_time = @max(max_time, hgemm_ms);
        max_time = @max(max_time, hgemm_trans_ms);
    }

    // Performance table header
    printColoredHeader("Performance Comparison (GFLOPS - higher is better)");

    // Print table header
    print("┌─────────────────────────┬──────────────────┬───────────────────┬───────────────────┬───────────────────┬───────────────────┬──────────────────┐\n", .{});
    print("│ {s}{s}Description{s}           │ {s}{s}Dimensions{s}       │ {s}{s}SGEMM{s}           │ {s}{s}HGEMM{s}           │ {s}{s}HGEMM_TRANS{s}     │ {s}{s}Best Time (ms){s}  │ {s}{s}Best Impl.{s}                 │\n", .{ Color.bold, Color.bright_white, Color.reset, Color.bold, Color.bright_white, Color.reset, Color.bold, Color.bright_white, Color.reset, Color.bold, Color.bright_white, Color.reset, Color.bold, Color.bright_white, Color.reset, Color.bold, Color.bright_white, Color.reset, Color.bold, Color.bright_white, Color.reset });
    print("├─────────────────────────┼──────────────────┼───────────────────┼───────────────────┼───────────────────┼───────────────────┼──────────────────┤\n", .{});

    // Print results
    for (results.items) |result| {
        const size = result.size;

        // Calculate times in milliseconds
        const sgemm_ms = @as(f64, @floatFromInt(result.sgemm.min_time_ns)) / 1e6;
        const hgemm_ms = @as(f64, @floatFromInt(result.hgemm.min_time_ns)) / 1e6;
        const hgemm_trans_ms = @as(f64, @floatFromInt(result.hgemm_trans.min_time_ns)) / 1e6;

        // Determine best implementation
        const best_impl = determineBestImpl(result.sgemm.avg_gflops, result.hgemm.avg_gflops, result.hgemm_trans.avg_gflops);

        // Get best time
        var best_time_ms: f64 = undefined;
        if (std.mem.eql(u8, best_impl, "SGEMM")) {
            best_time_ms = sgemm_ms;
        } else if (std.mem.eql(u8, best_impl, "HGEMM")) {
            best_time_ms = hgemm_ms;
        } else {
            best_time_ms = hgemm_trans_ms;
        }

        // Get colors based on performance
        const sgemm_color = getColorForValue(result.sgemm.avg_gflops, min_gflops, max_gflops);
        const hgemm_color = getColorForValue(result.hgemm.avg_gflops, min_gflops, max_gflops);
        const hgemm_trans_color = getColorForValue(result.hgemm_trans.avg_gflops, min_gflops, max_gflops);
        const time_color = getInverseColorForValue(best_time_ms, min_time, max_time); // Inverse because lower is better

        // Color for the best implementation
        const impl_color = if (std.mem.eql(u8, best_impl, "SGEMM"))
            Color.yellow
        else
            Color.green;

        // Print row
        print("│ {s:<23} │ {d:4}x{d:4}x{d:<4} │ ", .{ size.desc, size.m, size.n, size.k });

        // GFLOPS columns with color
        printColoredValue(result.sgemm.avg_gflops, sgemm_color, 15, 1);
        print(" │ ", .{});
        printColoredValue(result.hgemm.avg_gflops, hgemm_color, 15, 1);
        print(" │ ", .{});
        printColoredValue(result.hgemm_trans.avg_gflops, hgemm_trans_color, 15, 1);

        // Best time and implementation
        print(" │ ", .{});
        printColoredValue(best_time_ms, time_color, 15, 2);
        print(" │ {s}{s:^16}{s}           │\n", .{ impl_color, best_impl, Color.reset });
    }

    print("└─────────────────────────┴──────────────────┴───────────────────┴───────────────────┴───────────────────┴───────────────────┴──────────────────┘\n", .{});

    // ===== SPEEDUP COMPARISON TABLE =====

    // Find min/max speedups for color normalization
    var min_speedup: f64 = std.math.inf(f64);
    var max_speedup: f64 = 0;

    for (results.items) |result| {
        const hgemm_vs_sgemm = result.hgemm.avg_gflops / result.sgemm.avg_gflops;
        const hgemm_t_vs_sgemm = result.hgemm_trans.avg_gflops / result.sgemm.avg_gflops;
        const hgemm_t_vs_hgemm = result.hgemm_trans.avg_gflops / result.hgemm.avg_gflops;

        min_speedup = @min(min_speedup, hgemm_vs_sgemm);
        min_speedup = @min(min_speedup, hgemm_t_vs_sgemm);
        min_speedup = @min(min_speedup, hgemm_t_vs_hgemm);

        max_speedup = @max(max_speedup, hgemm_vs_sgemm);
        max_speedup = @max(max_speedup, hgemm_t_vs_sgemm);
        max_speedup = @max(max_speedup, hgemm_t_vs_hgemm);
    }

    // Speedup table header
    printColoredHeader("Relative Speedup Analysis (higher is better)");

    // Print table header
    print("┌─────────────────────────┬──────────────────┬───────────────────┬───────────────────┬───────────────────┐\n", .{});
    print("│ {s}{s}Description{s}           │ {s}{s}Dimensions{s}       │ {s}{s}HGEMM vs SGEMM{s}  │ {s}{s}HGEMM_T vs SGEMM{s}│ {s}{s}HGEMM_T vs HGEMM{s}│\n", .{ Color.bold, Color.bright_white, Color.reset, Color.bold, Color.bright_white, Color.reset, Color.bold, Color.bright_white, Color.reset, Color.bold, Color.bright_white, Color.reset, Color.bold, Color.bright_white, Color.reset });
    print("├─────────────────────────┼──────────────────┼───────────────────┼───────────────────┼───────────────────┤\n", .{});

    // Print results
    for (results.items) |result| {
        const size = result.size;

        // Calculate speedups
        const hgemm_vs_sgemm = result.hgemm.avg_gflops / result.sgemm.avg_gflops;
        const hgemm_t_vs_sgemm = result.hgemm_trans.avg_gflops / result.sgemm.avg_gflops;
        const hgemm_t_vs_hgemm = result.hgemm_trans.avg_gflops / result.hgemm.avg_gflops;

        // Get colors based on speedup
        const hgemm_vs_sgemm_color = getColorForValue(hgemm_vs_sgemm, min_speedup, max_speedup);
        const hgemm_t_vs_sgemm_color = getColorForValue(hgemm_t_vs_sgemm, min_speedup, max_speedup);
        const hgemm_t_vs_hgemm_color = getColorForValue(hgemm_t_vs_hgemm, min_speedup, max_speedup);

        // Print row
        print("│ {s:<23} │ {d:4}x{d:4}x{d:<4} │ ", .{ size.desc, size.m, size.n, size.k });

        // Speedup columns with color
        printColoredValue(hgemm_vs_sgemm, hgemm_vs_sgemm_color, 10, 2);
        print("x    │ ", .{});
        printColoredValue(hgemm_t_vs_sgemm, hgemm_t_vs_sgemm_color, 10, 2);
        print("x    │ ", .{});
        printColoredValue(hgemm_t_vs_hgemm, hgemm_t_vs_hgemm_color, 10, 2);
        print("x               │\n", .{});
    }

    print("└─────────────────────────┴──────────────────┴───────────────────┴───────────────────┴───────────────────┘\n", .{});

    // ===== MEMORY BANDWIDTH COMPARISON TABLE =====

    // Find min/max bandwidth for color normalization
    var min_bandwidth: f64 = std.math.inf(f64);
    var max_bandwidth: f64 = 0;

    for (results.items) |result| {
        min_bandwidth = @min(min_bandwidth, result.sgemm.avg_bandwidth_gbps);
        min_bandwidth = @min(min_bandwidth, result.hgemm.avg_bandwidth_gbps);
        min_bandwidth = @min(min_bandwidth, result.hgemm_trans.avg_bandwidth_gbps);

        max_bandwidth = @max(max_bandwidth, result.sgemm.avg_bandwidth_gbps);
        max_bandwidth = @max(max_bandwidth, result.hgemm.avg_bandwidth_gbps);
        max_bandwidth = @max(max_bandwidth, result.hgemm_trans.avg_bandwidth_gbps);
    }

    // Bandwidth table header
    printColoredHeader("Memory Bandwidth Comparison (GB/s - higher is better)");

    // Print table header
    print("┌─────────────────────────┬──────────────────┬───────────────────┬───────────────────┬───────────────────┐\n", .{});
    print("│ {s}{s}Description{s}           │ {s}{s}Dimensions{s}       │ {s}{s}SGEMM{s}           │ {s}{s}HGEMM{s}           │ {s}{s}HGEMM_TRANS{s}     │\n", .{ Color.bold, Color.bright_white, Color.reset, Color.bold, Color.bright_white, Color.reset, Color.bold, Color.bright_white, Color.reset, Color.bold, Color.bright_white, Color.reset, Color.bold, Color.bright_white, Color.reset });
    print("├─────────────────────────┼──────────────────┼───────────────────┼───────────────────┼───────────────────┤\n", .{});

    // Print results
    for (results.items) |result| {
        const size = result.size;

        // Get colors based on bandwidth
        const sgemm_bw_color = getColorForValue(result.sgemm.avg_bandwidth_gbps, min_bandwidth, max_bandwidth);
        const hgemm_bw_color = getColorForValue(result.hgemm.avg_bandwidth_gbps, min_bandwidth, max_bandwidth);
        const hgemm_trans_bw_color = getColorForValue(result.hgemm_trans.avg_bandwidth_gbps, min_bandwidth, max_bandwidth);

        // Print row
        print("│ {s:<23} │ {d:4}x{d:4}x{d:<4} │ ", .{ size.desc, size.m, size.n, size.k });

        // Bandwidth columns with color
        printColoredValue(result.sgemm.avg_bandwidth_gbps, sgemm_bw_color, 15, 2);
        print(" │ ", .{});
        printColoredValue(result.hgemm.avg_bandwidth_gbps, hgemm_bw_color, 15, 2);
        print(" │ ", .{});
        printColoredValue(result.hgemm_trans.avg_bandwidth_gbps, hgemm_trans_bw_color, 15, 2);
        print("         │\n", .{});
    }

    print("└─────────────────────────┴──────────────────┴───────────────────┴───────────────────┴───────────────────┘\n", .{});

    // ===== MEMORY USAGE ANALYSIS TABLE =====

    // Memory usage table header
    printColoredHeader("Memory Usage Analysis");

    // Print table header
    print("┌─────────────────────────┬──────────────────┬───────────────────┬───────────────────┬───────────────────┐\n", .{});
    print("│ {s}{s}Description{s}           │ {s}{s}Dimensions{s}       │ {s}{s}f32 Memory (MB){s} │ {s}{s}f16 Memory (MB){s} │ {s}{s}Memory Reduction{s}│\n", .{ Color.bold, Color.bright_white, Color.reset, Color.bold, Color.bright_white, Color.reset, Color.bold, Color.bright_white, Color.reset, Color.bold, Color.bright_white, Color.reset, Color.bold, Color.bright_white, Color.reset });
    print("├─────────────────────────┼──────────────────┼───────────────────┼───────────────────┼───────────────────┤\n", .{});

    // Print results
    for (results.items) |result| {
        const size = result.size;

        // Calculate memory usage (A + B + C matrices)
        const f32_memory_mb = @as(f64, @floatFromInt((size.m * size.k + size.k * size.n + size.m * size.n) * @sizeOf(f32))) / (1024 * 1024);
        const f16_memory_mb = @as(f64, @floatFromInt((size.m * size.k + size.k * size.n + size.m * size.n) * @sizeOf(f16))) / (1024 * 1024);
        const memory_reduction = 1.0 - (f16_memory_mb / f32_memory_mb);

        // Print row
        print("│ {s:<23} │ {d:4}x{d:4}x{d:<4} │ {d:15.2} │ {d:15.2} │ ", .{ size.desc, size.m, size.n, size.k, f32_memory_mb, f16_memory_mb });

        // Memory reduction with color (higher is better)
        const reduction_color = getColorForValue(memory_reduction, 0, 1);
        printColoredValue(memory_reduction * 100.0, reduction_color, 10, 2);
        print("%             │\n", .{});
    }

    print("└─────────────────────────┴──────────────────┴───────────────────┴───────────────────┴───────────────────┘\n", .{});

    // ===== PERFORMANCE VS MEMORY TRADEOFF ANALYSIS TABLE =====

    // Find min/max efficiency for color normalization
    var min_efficiency: f64 = std.math.inf(f64);
    var max_efficiency: f64 = 0;

    for (results.items) |result| {
        // Memory reduction
        const f32_memory = @as(f64, @floatFromInt((result.size.m * result.size.k + result.size.k * result.size.n + result.size.m * result.size.n) * @sizeOf(f32)));
        const f16_memory = @as(f64, @floatFromInt((result.size.m * result.size.k + result.size.k * result.size.n + result.size.m * result.size.n) * @sizeOf(f16)));
        const memory_reduction = 1.0 - (f16_memory / f32_memory);

        // Determine best implementation
        const best_impl = determineBestImpl(result.sgemm.avg_gflops, result.hgemm.avg_gflops, result.hgemm_trans.avg_gflops);

        // Calculate performance gain
        var perf_gain: f64 = undefined;
        if (std.mem.eql(u8, best_impl, "HGEMM")) {
            perf_gain = result.hgemm.avg_gflops / result.sgemm.avg_gflops;
        } else if (std.mem.eql(u8, best_impl, "HGEMM_TRANS")) {
            perf_gain = result.hgemm_trans.avg_gflops / result.sgemm.avg_gflops;
        } else {
            perf_gain = 1.0;
        }

        // Calculate overall efficiency
        const efficiency = if (!std.mem.eql(u8, best_impl, "SGEMM"))
            perf_gain * memory_reduction
        else
            0.0;

        min_efficiency = @min(min_efficiency, efficiency);
        max_efficiency = @max(max_efficiency, efficiency);
    }

    // Tradeoff analysis table header
    printColoredHeader("Performance vs Memory Tradeoff Analysis");

    // Print table header
    print("┌─────────────────────────┬──────────────────┬───────────────────┬───────────────────┬───────────────────┐\n", .{});
    print("│ {s}{s}Description{s}           │ {s}{s}Dimensions{s}       │ {s}{s}Best Impl.{s}      │ {s}{s}Memory Reduction{s}│ {s}{s}Overall Efficiency{s}│\n", .{ Color.bold, Color.bright_white, Color.reset, Color.bold, Color.bright_white, Color.reset, Color.bold, Color.bright_white, Color.reset, Color.bold, Color.bright_white, Color.reset, Color.bold, Color.bright_white, Color.reset });
    print("├─────────────────────────┼──────────────────┼───────────────────┼───────────────────┼───────────────────┤\n", .{});

    // Print results
    for (results.items) |result| {
        const size = result.size;

        // Memory reduction
        const f32_memory = @as(f64, @floatFromInt((size.m * size.k + size.k * size.n + size.m * size.n) * @sizeOf(f32)));
        const f16_memory = @as(f64, @floatFromInt((size.m * size.k + size.k * size.n + size.m * size.n) * @sizeOf(f16)));
        const memory_reduction = 1.0 - (f16_memory / f32_memory);

        // Determine best implementation
        const best_impl = determineBestImpl(result.sgemm.avg_gflops, result.hgemm.avg_gflops, result.hgemm_trans.avg_gflops);

        // Calculate performance gain
        var perf_gain: f64 = undefined;
        if (std.mem.eql(u8, best_impl, "HGEMM")) {
            perf_gain = result.hgemm.avg_gflops / result.sgemm.avg_gflops;
        } else if (std.mem.eql(u8, best_impl, "HGEMM_TRANS")) {
            perf_gain = result.hgemm_trans.avg_gflops / result.sgemm.avg_gflops;
        } else {
            perf_gain = 1.0;
        }

        // Calculate overall efficiency
        const efficiency = if (!std.mem.eql(u8, best_impl, "SGEMM"))
            perf_gain * memory_reduction
        else
            0.0;

        // Get colors
        const reduction_color = getColorForValue(memory_reduction, 0, 1);
        const efficiency_color = getColorForValue(efficiency, min_efficiency, max_efficiency);
        const impl_color = if (std.mem.eql(u8, best_impl, "SGEMM")) Color.red else Color.green;

        // Print row
        print("│ {s:<23} │ {d:4}x{d:4}x{d:<4} │ {s}{s:^15}{s} │ ", .{ size.desc, size.m, size.n, size.k, impl_color, best_impl, Color.reset });

        // Memory reduction with color
        printColoredValue(memory_reduction * 100.0, reduction_color, 10, 2);
        print("%    │ ", .{});

        // Overall efficiency with color
        printColoredValue(efficiency, efficiency_color, 15, 4);
        print("          │\n", .{});
    }

    print("└─────────────────────────┴──────────────────┴───────────────────┴───────────────────┴───────────────────┘\n", .{});

    print("\n{s}Benchmark completed!{s}\n", .{ Color.bright_green, Color.reset });
}
