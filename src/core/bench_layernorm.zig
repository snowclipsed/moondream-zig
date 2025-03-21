const std = @import("std");
const Allocator = std.mem.Allocator;
const print = std.debug.print;
const time = std.time;
const math = std.math;

// Import the implementations
const layerNormOldF64 = @import("ops.zig").layerNormOldF64;
const layerNormOld = @import("ops.zig").layerNormOld;
const layerNormInner = @import("ops.zig").layerNormInner;

// Import common dependency
const Tensor = @import("tensor.zig").Tensor;

// Import benchmark utilities
const printColoredHeader = @import("bench.zig").printColoredHeader;
const getColorForValue = @import("bench.zig").getColorForValue;
const getInverseColorForValue = @import("bench.zig").getInverseColorForValue;
const printColoredValue = @import("bench.zig").printColoredValue;
const Color = @import("bench.zig").Color;
// Updated BenchmarkResult struct that only tracks the best metrics
// This is a redefinition of what's in bench.zig to match the new approach
const BenchmarkResult = struct {
    min_time_ns: u64, // Best (minimum) execution time
    max_gflops: f64,  // Best (maximum) GFLOPS
    max_bandwidth_gbps: f64, // Best (maximum) bandwidth in GB/s
};

fn benchmarkSLayerNormYolo(allocator: Allocator, M: usize, K: usize, num_runs: usize) !BenchmarkResult {
    return benchmarkSLayerNormInner(10, 1, allocator, M, K, num_runs);
}

fn benchmarkSLayerNormInner(comptime T: type, comptime N_A: usize, comptime N_B: usize, comptime N_S2: usize, allocator: Allocator, M: usize, K: usize, num_runs: usize) !BenchmarkResult {
    // Create tensors
    var a = try Tensor(T).init(allocator, &[_]usize{ M, K });
    defer a.deinit();
    var g = try Tensor(T).init(allocator, &[_]usize{ K });
    defer g.deinit();
    var b = try Tensor(T).init(allocator, &[_]usize{ K });
    defer b.deinit();

    // Initialize with random data
    var rng = std.rand.DefaultPrng.init(42);
    const random = rng.random();
    for (a.data) |*val| {
        val.* = @floatCast(random.float(f32) * 10.0 - 5.0);
    }
    for (g.data) |*val| {
        val.* = @floatCast(random.float(f32) * 2.0 - 1.0);
    }
    for (b.data) |*val| {
        val.* = @floatCast(random.float(f32) * 2.0 - 1.0);
    }
    const eps = 0.00001;

    // Warmup run
    var warmup = try layerNormInner(T, N_A, N_B, N_S2, false, a, g, b, eps);
    warmup.deinit();

    // Benchmark runs
    var min_time: u64 = std.math.maxInt(u64);
    var max_gflops: f64 = 0;
    var max_bandwidth: f64 = 0;

    for (0..num_runs) |_| {
        var timer = try time.Timer.start();
        var result = try layerNormInner(T, N_A, N_B, N_S2, false, a, g, b, eps);
        const elapsed = timer.read();
        result.deinit();

        min_time = @min(min_time, elapsed);
    }

    const seconds = @as(f64, @floatFromInt(min_time)) / 1e9;
    const ops = 10 * M * K;
    const gflops = @as(f64, @floatFromInt(ops)) / seconds / 1e9;
    max_gflops = @max(max_gflops, gflops);
    const bytes_accessed = @as(f64, @floatFromInt((2 * M * K) * @sizeOf(T)));
    const bandwidth = bytes_accessed / seconds / 1e9;
    max_bandwidth = @max(max_bandwidth, bandwidth);

    return BenchmarkResult{
        .min_time_ns = min_time,
        .max_gflops = max_gflops,
        .max_bandwidth_gbps = max_bandwidth,
    };
}

fn benchmarkSLayerNormOld(T: type, allocator: Allocator, M: usize, K: usize, num_runs: usize) !BenchmarkResult {
    // Create tensors
    var a = try Tensor(T).init(allocator, &[_]usize{ M, K });
    defer a.deinit();
    var g = try Tensor(T).init(allocator, &[_]usize{ K });
    defer g.deinit();
    var b = try Tensor(T).init(allocator, &[_]usize{ K });
    defer b.deinit();

    // Initialize with random data
    var rng = std.rand.DefaultPrng.init(42);
    const random = rng.random();
    for (a.data) |*val| {
        val.* = @floatCast(random.float(f32) * 10.0 - 5.0);
    }
    for (g.data) |*val| {
        val.* = @floatCast(random.float(f32) * 2.0 - 1.0);
    }
    for (b.data) |*val| {
        val.* = @floatCast(random.float(f32) * 2.0 - 1.0);
    }
    const eps = 0.00001;

    // Warmup run
    var warmup = try layerNormOld(T, a, g, b, eps);
    warmup.deinit();

    // Benchmark runs
    var min_time: u64 = std.math.maxInt(u64);
    var max_gflops: f64 = 0;
    var max_bandwidth: f64 = 0;

    for (0..num_runs) |_| {
        var timer = try time.Timer.start();
        var result = try layerNormOld(T, a, g, b, eps);
        const elapsed = timer.read();
        result.deinit();

        min_time = @min(min_time, elapsed);
    }

    const seconds = @as(f64, @floatFromInt(min_time)) / 1e9;
    const ops = 10 * M * K;
    const gflops = @as(f64, @floatFromInt(ops)) / seconds / 1e9;
    max_gflops = gflops;
    const bytes_accessed = @as(f64, @floatFromInt((2 * M * K) * @sizeOf(T)));
    const bandwidth = bytes_accessed / seconds / 1e9;
    max_bandwidth = bandwidth;

    return BenchmarkResult{
        .min_time_ns = min_time,
        .max_gflops = max_gflops,
        .max_bandwidth_gbps = max_bandwidth,
    };
}

pub fn benchmarkLayerNorm() !void {
    // Initialize allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Benchmarking parameters
    const num_runs = 5;

    // Fixed M size (batch size) for all tests
    const M = 4096;

    // Use common embedding dimensions from known LLMs
    const embed_dims = [_]struct { dim: usize, desc: []const u8 }{
        .{ .dim = 768, .desc = "768" },
        .{ .dim = 1024, .desc = "1024" },
        .{ .dim = 2048, .desc = "2048" },
        .{ .dim = 3072, .desc = "3072" },
        .{ .dim = 4096, .desc = "4096" },
        .{ .dim = 5120, .desc = "5120" },
        .{ .dim = 8192, .desc = "8192" },
        .{ .dim = 12288, .desc = "12288" },
        .{ .dim = 14848, .desc = "14848" },
        .{ .dim = 16384, .desc = "16384" },
        .{ .dim = 25600, .desc = "25600" },
    };

    // Print header with system information
    print("\n{s}{s}Layer Normalization Benchmark{s}\n", .{ Color.bold, Color.bright_green, Color.reset });
    print("╔══════════════════════════════════════════════════════════════╗\n", .{});
    print("║ {s}System Information{s}                                           ║\n", .{ Color.bright_yellow, Color.reset });
    print("╟──────────────────────────────────────────────────────────────╢\n", .{});
    print("║ CPU Threads:          {d:>10}                             ║\n", .{try std.Thread.getCpuCount()});
    print("║ Batch Size (M):       {d:>10}                             ║\n", .{M});
    print("║ Iterations per test:  {d:>10}                             ║\n", .{num_runs});
    print("╚══════════════════════════════════════════════════════════════╝\n\n", .{});

    print("Running layer normalization benchmarks...\n", .{});

    // Define a struct to hold both benchmark results
    const LayerNormResults = struct {
        dim: usize,
        desc: []const u8,
        standard: BenchmarkResult,
        yolo: BenchmarkResult,
    };

    // Run all benchmarks and collect results
    var results = std.ArrayList(LayerNormResults).init(allocator);
    defer results.deinit();

    for (embed_dims) |dim_info| {
        print("  Benchmarking Layer Norm with embedding dimension: {d} ({s})...\n", .{ dim_info.dim, dim_info.desc });

        const standard_result = try benchmarkSLayerNormOld(allocator, M, dim_info.dim, num_runs);
        const yolo_result = try benchmarkSLayerNormYolo(allocator, M, dim_info.dim, num_runs);

        try results.append(.{
            .dim = dim_info.dim,
            .desc = dim_info.desc,
            .standard = standard_result,
            .yolo = yolo_result,
        });
    }

    // ===== PERFORMANCE COMPARISON TABLE =====

    // Find min/max GFLOPS for color normalization
    var min_gflops: f64 = std.math.inf(f64);
    var max_gflops: f64 = 0;

    for (results.items) |result| {
        min_gflops = @min(min_gflops, result.standard.avg_gflops);
        min_gflops = @min(min_gflops, result.yolo.avg_gflops);
        max_gflops = @max(max_gflops, result.standard.avg_gflops);
        max_gflops = @max(max_gflops, result.yolo.avg_gflops);
    }

    // Find min/max times for color normalization
    var min_time: f64 = std.math.inf(f64);
    var max_time: f64 = 0;

    for (results.items) |result| {
        const standard_ms = @as(f64, @floatFromInt(result.standard.min_time_ns)) / 1e6;
        const yolo_ms = @as(f64, @floatFromInt(result.yolo.min_time_ns)) / 1e6;

        min_time = @min(min_time, standard_ms);
        min_time = @min(min_time, yolo_ms);
        max_time = @max(max_time, standard_ms);
        max_time = @max(max_time, yolo_ms);
    }

    // Performance table header
    printColoredHeader("Performance Comparison (GFLOPS - higher is better)");

    // Print table header
    print("┌─────────────────────────┬──────────────────┬───────────────────┬───────────────────┬───────────────────┬───────────────────┐\n", .{});
    print("│ {s}{s}Embedding Dimension{s}   │ {s}{s}Size (M x K){s}    │ {s}{s}Standard LN{s}     │ {s}{s}Yolo LN{s}         │ {s}{s}Best Time (ms){s}  │ {s}{s}Best Impl.{s}      │\n", .{
        Color.bold, Color.bright_white, Color.reset,
        Color.bold, Color.bright_white, Color.reset,
        Color.bold, Color.bright_white, Color.reset,
        Color.bold, Color.bright_white, Color.reset,
        Color.bold, Color.bright_white, Color.reset,
        Color.bold, Color.bright_white, Color.reset
    });
    print("├─────────────────────────┼──────────────────┼───────────────────┼───────────────────┼───────────────────┼───────────────────┤\n", .{});

    // Print results
    for (results.items) |result| {
        // Calculate times in milliseconds
        const standard_ms = @as(f64, @floatFromInt(result.standard.min_time_ns)) / 1e6;
        const yolo_ms = @as(f64, @floatFromInt(result.yolo.min_time_ns)) / 1e6;

        // Determine best implementation
        const is_yolo_better = result.yolo.avg_gflops > result.standard.avg_gflops;
        const best_impl = if (is_yolo_better) "Yolo LN" else "Standard LN";
        const best_time_ms = if (is_yolo_better) yolo_ms else standard_ms;

        // Get colors based on performance
        const standard_color = getColorForValue(result.standard.avg_gflops, min_gflops, max_gflops);
        const yolo_color = getColorForValue(result.yolo.avg_gflops, min_gflops, max_gflops);
        const time_color = getInverseColorForValue(best_time_ms, min_time, max_time);
        const impl_color = if (is_yolo_better) Color.green else Color.blue;

        // Print row
        print("│ {s:<23} │ {d:6}x{d:<7} │ ", .{ result.desc, M, result.dim });

        // GFLOPS columns with color
        printColoredValue(result.standard.avg_gflops, standard_color, 15, 1);
        print(" │ ", .{});
        printColoredValue(result.yolo.avg_gflops, yolo_color, 15, 1);
        print(" │ ", .{});
        printColoredValue(best_time_ms, time_color, 15, 2);
        print(" │ {s}{s:^15}{s} │\n", .{ impl_color, best_impl, Color.reset });
    }

    print("└─────────────────────────┴──────────────────┴───────────────────┴───────────────────┴───────────────────┴───────────────────┘\n", .{});

    // ===== SPEEDUP COMPARISON TABLE =====

    // Speedup table header
    printColoredHeader("Yolo LN vs Standard LN Speedup Analysis");

    // Print table header
    print("┌─────────────────────────┬──────────────────┬───────────────────┬───────────────────┬───────────────────┐\n", .{});
    print("│ {s}{s}Embedding Dimension{s}   │ {s}{s}Size (M x K){s}    │ {s}{s}Speedup (GFLOPS){s}│ {s}{s}Time Reduction{s}  │ {s}{s}Bandwidth (GB/s){s}│\n", .{
        Color.bold, Color.bright_white, Color.reset,
        Color.bold, Color.bright_white, Color.reset,
        Color.bold, Color.bright_white, Color.reset,
        Color.bold, Color.bright_white, Color.reset,
        Color.bold, Color.bright_white, Color.reset
    });
    print("├─────────────────────────┼──────────────────┼───────────────────┼───────────────────┼───────────────────┤\n", .{});

    // Find min/max speedups for color normalization
    var min_speedup: f64 = std.math.inf(f64);
    var max_speedup: f64 = 0;
    var min_reduction: f64 = std.math.inf(f64);
    var max_reduction: f64 = 0;
    var min_bandwidth: f64 = std.math.inf(f64);
    var max_bandwidth: f64 = 0;

    for (results.items) |result| {
        const speedup = result.yolo.avg_gflops / result.standard.avg_gflops;
        const time_reduction = 1.0 - (@as(f64, @floatFromInt(result.yolo.avg_time_ns)) / @as(f64, @floatFromInt(result.standard.avg_time_ns)));

        min_speedup = @min(min_speedup, speedup);
        max_speedup = @max(max_speedup, speedup);
        min_reduction = @min(min_reduction, time_reduction);
        max_reduction = @max(max_reduction, time_reduction);
        min_bandwidth = @min(min_bandwidth, result.yolo.avg_bandwidth_gbps);
        max_bandwidth = @max(max_bandwidth, result.yolo.avg_bandwidth_gbps);
    }

    // Print results
    for (results.items) |result| {
        const speedup = result.yolo.avg_gflops / result.standard.avg_gflops;
        const time_reduction = 1.0 - (@as(f64, @floatFromInt(result.yolo.avg_time_ns)) / @as(f64, @floatFromInt(result.standard.avg_time_ns)));

        // Get colors
        const speedup_color = getColorForValue(speedup, min_speedup, max_speedup);
        const reduction_color = getColorForValue(time_reduction, min_reduction, max_reduction);
        const bandwidth_color = getColorForValue(result.yolo.avg_bandwidth_gbps, min_bandwidth, max_bandwidth);

        // Print row
        print("│ {s:<23} │ {d:6}x{d:<7} │ ", .{ result.desc, M, result.dim });

        // Speedup with color
        printColoredValue(speedup, speedup_color, 10, 2);
        print("x    │ ", .{});

        // Time reduction with color (as percentage)
        printColoredValue(time_reduction * 100.0, reduction_color, 10, 2);
        print("%    │ ", .{});

        // Bandwidth with color
        printColoredValue(result.yolo.avg_bandwidth_gbps, bandwidth_color, 15, 2);
        print(" │\n", .{});
    }

    print("└─────────────────────────┴──────────────────┴───────────────────┴───────────────────┴───────────────────┘\n", .{});

    // ===== ANALYSIS BY POWER-OF-2 ALIGNMENT =====

    // Group results by the largest power of 2 that divides the embedding dimension
    const PowerOfTwoAnalysis = struct {
        power: u32,
        factor: usize,
        avg_speedup: f64,
        samples: usize,
        min_speedup: f64,
        max_speedup: f64,
    };

    var power_analysis = std.AutoHashMap(u32, PowerOfTwoAnalysis).init(allocator);
    defer power_analysis.deinit();

    for (results.items) |result| {
        // Find largest power of 2 that divides the dimension
        var dim = result.dim;
        var power: u6 = 0;

        while (dim % 2 == 0) {
            dim /= 2;
            power += 1;
        }

        const factor = @as(usize, 1) << power;
        const speedup = result.yolo.avg_gflops / result.standard.avg_gflops;

        if (power_analysis.get(power)) |analysis| {
            const new_avg = (analysis.avg_speedup * @as(f64, @floatFromInt(analysis.samples)) + speedup) /
                           @as(f64, @floatFromInt(analysis.samples + 1));

            try power_analysis.put(power, .{
                .power = power,
                .factor = factor,
                .avg_speedup = new_avg,
                .samples = analysis.samples + 1,
                .min_speedup = @min(analysis.min_speedup, speedup),
                .max_speedup = @max(analysis.max_speedup, speedup),
            });
        } else {
            try power_analysis.put(power, .{
                .power = power,
                .factor = factor,
                .avg_speedup = speedup,
                .samples = 1,
                .min_speedup = speedup,
                .max_speedup = speedup,
            });
        }
    }

    // Create a sorted list of powers for display
    var powers = std.ArrayList(u32).init(allocator);
    defer powers.deinit();

    var power_iterator = power_analysis.keyIterator();
    while (power_iterator.next()) |power| {
        try powers.append(power.*);
    }

    // Sort powers in ascending order
    std.sort.pdq(u32, powers.items, {}, comptime std.sort.asc(u32));

    // Power of 2 analysis table header
    printColoredHeader("Analysis by Power-of-2 Alignment");

    // Print table header
    print("┌─────────────────────────┬──────────────────┬───────────────────┬───────────────────┬───────────────────┐\n", .{});
    print("│ {s}{s}Power of 2{s}           │ {s}{s}Factor (2^n){s}   │ {s}{s}Avg Speedup{s}     │ {s}{s}Min Speedup{s}     │ {s}{s}Max Speedup{s}     │\n", .{
        Color.bold, Color.bright_white, Color.reset,
        Color.bold, Color.bright_white, Color.reset,
        Color.bold, Color.bright_white, Color.reset,
        Color.bold, Color.bright_white, Color.reset,
        Color.bold, Color.bright_white, Color.reset
    });
    print("├─────────────────────────┼──────────────────┼───────────────────┼───────────────────┼───────────────────┤\n", .{});

    // Find min/max for color normalization
    var min_avg_speedup: f64 = std.math.inf(f64);
    var max_avg_speedup: f64 = 0;

    for (powers.items) |power| {
        const analysis = power_analysis.get(power).?;
        min_avg_speedup = @min(min_avg_speedup, analysis.avg_speedup);
        max_avg_speedup = @max(max_avg_speedup, analysis.avg_speedup);
    }

    // Print results
    for (powers.items) |power| {
        const analysis = power_analysis.get(power).?;

        // Get color based on average speedup
        const avg_color = getColorForValue(analysis.avg_speedup, min_avg_speedup, max_avg_speedup);
        const min_color = getColorForValue(analysis.min_speedup, min_avg_speedup, max_avg_speedup);
        const max_color = getColorForValue(analysis.max_speedup, min_avg_speedup, max_avg_speedup);

        // Print row
        print("│ 2^{d:<21} │ {d:<14} │ ", .{ power, analysis.factor });

        // Speedup values with color
        printColoredValue(analysis.avg_speedup, avg_color, 10, 2);
        print("x    │ ", .{});
        printColoredValue(analysis.min_speedup, min_color, 10, 2);
        print("x    │ ", .{});
        printColoredValue(analysis.max_speedup, max_color, 10, 2);
        print("x    │\n", .{});
    }

    print("└─────────────────────────┴──────────────────┴───────────────────┴───────────────────┴───────────────────┘\n", .{});

    // ===== RECOMMENDATIONS TABLE =====

    printColoredHeader("Implementation Recommendations by Embedding Dimension");

    // Print table header
    print("┌─────────────────────────┬──────────────────┬───────────────────┬───────────────────┐\n", .{});
    print("│ {s}{s}Embedding Dimension{s}   │ {s}{s}Power of 2{s}      │ {s}{s}Recommended Impl{s}│ {s}{s}Expected Speedup{s}│\n", .{
        Color.bold, Color.bright_white, Color.reset,
        Color.bold, Color.bright_white, Color.reset,
        Color.bold, Color.bright_white, Color.reset,
        Color.bold, Color.bright_white, Color.reset
    });
    print("├─────────────────────────┼──────────────────┼───────────────────┼───────────────────┤\n", .{});

    for (embed_dims) |dim_info| {
        var dim = dim_info.dim;
        var power: u32 = 0;

        while (dim % 2 == 0) {
            dim /= 2;
            power += 1;
        }

        // Find this dimension in results
        var speedup: f64 = 0;
        for (results.items) |result| {
            if (result.dim == dim_info.dim) {
                speedup = result.yolo.avg_gflops / result.standard.avg_gflops;
                break;
            }
        }

        // Determine recommended implementation
        const is_yolo_better = speedup > 1.0;
        const recommended = if (is_yolo_better) "Yolo LN" else "Standard LN";
        const impl_color = if (is_yolo_better) Color.green else Color.blue;
        const speedup_color = if (is_yolo_better)
                               getColorForValue(speedup, 1.0, max_speedup)
                             else
                               getColorForValue(1.0 / speedup, 1.0, 1.0 / min_speedup);

        const display_speedup = if (is_yolo_better) speedup else 1.0 / speedup;
        const speedup_label = if (is_yolo_better) "x faster" else "x slower";

        // Print row
        print("│ {s:<23} │ 2^{d:<12} │ {s}{s:^15}{s} │ ", .{
            dim_info.desc,
            power,
            impl_color,
            recommended,
            Color.reset
        });

        // Speedup with color
        printColoredValue(display_speedup, speedup_color, 10, 2);
        print("{s} │\n", .{speedup_label});
    }

    print("└─────────────────────────┴──────────────────┴───────────────────┴───────────────────┘\n", .{});

    print("\n{s}Layer Normalization Benchmark completed!{s}\n", .{ Color.bright_green, Color.reset });
}

pub fn benchmarkLayerNormGrid(T: type) !void {
    // Initialize allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Benchmarking parameters
    const num_runs = 20; // Reduced from 5 due to the large number of configurations

    // Fixed M size (batch size) for all tests
    const M = 1024;

    // Use common embedding dimensions from known LLMs
    // We'll test a subset of dimensions to keep the grid search manageable
    const embed_dims = [_]struct { dim: usize, desc: []const u8 }{
        .{ .dim = 768, .desc = "768" },
        .{ .dim = 1024, .desc = "1024" },
        .{ .dim = 2048, .desc = "2048" },
        .{ .dim = 4096, .desc = "4096" },
        .{ .dim = 5120, .desc = "5120" },
        .{ .dim = 8192, .desc = "8192" },
        .{ .dim = 12288, .desc = "12288" },
        .{ .dim = 16384, .desc = "16384" },
    };

    // Print header with system information
    print("\n{s}{s}Layer Normalization Grid Search Benchmark{s}\n", .{ Color.bold, Color.bright_green, Color.reset });
    print("╔══════════════════════════════════════════════════════════════╗\n", .{});
    print("║ {s}System Information{s}                                           ║\n", .{ Color.bright_yellow, Color.reset });
    print("╟──────────────────────────────────────────────────────────────╢\n", .{});
    print("║ CPU Threads:          {d:>10}                             ║\n", .{try std.Thread.getCpuCount()});
    print("║ Batch Size (M):       {d:>10}                             ║\n", .{M});
    print("║ Iterations per test:  {d:>10}                             ║\n", .{num_runs});
    print("║ Grid Search:          Fixed configurations                  ║\n", .{});
    print("╚══════════════════════════════════════════════════════════════╝\n\n", .{});

    print("Running layer normalization grid search...\n", .{});

    // Define a struct to hold benchmark results for a specific configuration
    const YoloConfig = struct {
        N_A: usize,
        N_B: usize,
        N_S2: usize,
        result: BenchmarkResult,
    };

    // Define a struct to hold all benchmark results for a dimension
    const DimensionResults = struct {
        dim: usize,
        desc: []const u8,
        standard: BenchmarkResult, // Standard LayerNorm as baseline
        configs: std.ArrayList(YoloConfig),
    };

    // Create array to hold results for each dimension
    var all_results = std.ArrayList(DimensionResults).init(allocator);
    defer {
        for (all_results.items) |dim_result| {
            dim_result.configs.deinit();
        }
        all_results.deinit();
    }

    const N_A_values = [_]usize{  0, 1, 0, 1, 8, 2, 6 };
    const N_B_values = [_]usize{  1, 0, 1, 0, 8, 6, 2 };
    const N_S2_values = [_]usize{ 4, 4, 8, 8, 4, 8, 8 };

    // Benchmark each dimension
    for (embed_dims) |dim_info| {
        print("  Benchmarking Layer Norm for dimension: {d} ({s})...\n", .{ dim_info.dim, dim_info.desc });

        // First benchmark standard LayerNorm as baseline
        const standard_result = try benchmarkSLayerNormOld(T, allocator, M, dim_info.dim, num_runs);

        // Create configs list for this dimension
        var configs = std.ArrayList(YoloConfig).init(allocator);

        // Now run benchmarks for fixed configurations
        inline for (0..N_A_values.len) |idx| {
            const N_A = N_A_values[idx];
            const N_B = N_B_values[idx];
            const N_S2 = N_S2_values[idx];
            print("    Testing Yolo(N_A={d}, N_B={d}, N_S2={d})...\n", .{ N_A, N_B, N_S2 });

            // Benchmark YoloLN with these parameters
            const yolo_result = try benchmarkSLayerNormInner(
                T,
                N_A,
                N_B,
                N_S2,
                allocator,
                M,
                dim_info.dim,
                num_runs*10,
            );

            try configs.append(.{
                .N_A = N_A,
                .N_B = N_B,
                .N_S2 = N_S2,
                .result = yolo_result,
            });
        }

        // Add results for this dimension
        try all_results.append(.{
            .dim = dim_info.dim,
            .desc = dim_info.desc,
            .standard = standard_result,
            .configs = configs,
        });
    }

    // ===== BEST CONFIGURATIONS TABLE =====

    printColoredHeader("Best Configuration for Each Embedding Dimension");

    // Print table header
    print("┌─────────────────────────┬──────────────────┬───────────────────┬───────────────────┬───────────────────┬───────────────────┐\n", .{});
    print("│ {s}{s}Embedding Dimension{s}   │ {s}{s}Size (M x K){s}    │ {s}{s}Best Config{s}     │ {s}{s}Best GFLOPS{s}     │ {s}{s}Speedup vs Std{s}  │ {s}{s}Best Time (ms){s}  │\n", .{
        Color.bold, Color.bright_white, Color.reset,
        Color.bold, Color.bright_white, Color.reset,
        Color.bold, Color.bright_white, Color.reset,
        Color.bold, Color.bright_white, Color.reset,
        Color.bold, Color.bright_white, Color.reset,
        Color.bold, Color.bright_white, Color.reset
    });
    print("├─────────────────────────┼──────────────────┼───────────────────┼───────────────────┼───────────────────┼───────────────────┤\n", .{});

    // Find min/max for color normalization across all configs
    var min_gflops: f64 = std.math.inf(f64);
    var max_gflops: f64 = 0;
    var min_speedup: f64 = std.math.inf(f64);
    var max_speedup: f64 = 0;
    var min_time: f64 = std.math.inf(f64);
    var max_time: f64 = 0;
    var min_bandwidth: f64 = std.math.inf(f64);
    var max_bandwidth: f64 = 0;

    for (all_results.items) |dim_result| {
        // Check standard
        min_gflops = @min(min_gflops, dim_result.standard.max_gflops);
        max_gflops = @max(max_gflops, dim_result.standard.max_gflops);

        const standard_ms = @as(f64, @floatFromInt(dim_result.standard.min_time_ns)) / 1e6;
        min_time = @min(min_time, standard_ms);
        max_time = @max(max_time, standard_ms);
        min_bandwidth = @min(min_bandwidth, dim_result.standard.max_bandwidth_gbps);
        max_bandwidth = @max(max_bandwidth, dim_result.standard.max_bandwidth_gbps);

        // Check all configs
        for (dim_result.configs.items) |config| {
            min_gflops = @min(min_gflops, config.result.max_gflops);
            max_gflops = @max(max_gflops, config.result.max_gflops);
            min_bandwidth = @min(min_bandwidth, config.result.max_bandwidth_gbps);
            max_bandwidth = @max(max_bandwidth, config.result.max_bandwidth_gbps);

            const speedup = config.result.max_gflops / dim_result.standard.max_gflops;
            min_speedup = @min(min_speedup, speedup);
            max_speedup = @max(max_speedup, speedup);

            const config_ms = @as(f64, @floatFromInt(config.result.min_time_ns)) / 1e6;
            min_time = @min(min_time, config_ms);
            max_time = @max(max_time, config_ms);
        }
    }

    // Print best configuration for each dimension
    for (all_results.items) |dim_result| {
        // Find the best configuration for this dimension
        var best_config_idx: usize = 0;
        var best_gflops: f64 = 0;

        for (dim_result.configs.items, 0..) |config, i| {
            if (config.result.max_gflops > best_gflops) {
                best_gflops = config.result.max_gflops;
                best_config_idx = i;
            }
        }

        const best_config = dim_result.configs.items[best_config_idx];
        const speedup = best_config.result.max_gflops / dim_result.standard.max_gflops;
        const config_ms = @as(f64, @floatFromInt(best_config.result.min_time_ns)) / 1e6;

        // Get colors based on performance
        const gflops_color = getColorForValue(best_config.result.max_gflops, min_gflops, max_gflops);
        const speedup_color = getColorForValue(speedup, min_speedup, max_speedup);
        const time_color = getInverseColorForValue(config_ms, min_time, max_time);

        // Format the config string
        var config_buf: [32]u8 = undefined;
        const config_str = std.fmt.bufPrint(&config_buf, "Yolo({d},{d},{d})", .{
            best_config.N_A, best_config.N_B, best_config.N_S2
        }) catch unreachable;

        // Print row
        print("│ {s:<23} │ {d:6}x{d:<7} │ {s:^15} │ ", .{
            dim_result.desc,
            M,
            dim_result.dim,
            config_str,
        });

        // Performance columns with color
        printColoredValue(best_config.result.max_gflops, gflops_color, 15, 1);
        print(" │ ", .{});
        printColoredValue(speedup, speedup_color, 15, 2);
        print("x │ ", .{});
        printColoredValue(config_ms, time_color, 15, 2);
        print(" │\n", .{});
    }

    print("└─────────────────────────┴──────────────────┴───────────────────┴───────────────────┴───────────────────┴───────────────────┘\n", .{});

    // ===== DETAILED CONFIG COMPARISON FOR EACH DIMENSION =====

    for (all_results.items) |dim_result| {
        // Create header for this dimension
        var header_buf: [128]u8 = undefined;
        const header_str = std.fmt.bufPrint(
            &header_buf,
            "Configuration Comparison for {s} ({d}x{d})",
            .{dim_result.desc, M, dim_result.dim}
        ) catch unreachable;

        printColoredHeader(header_str);

        // Print table header
        print("┌───────────────────┬───────────────────┬───────────────────┬───────────────────┬───────────────────┐\n", .{});
        print("│ {s}{s}Configuration{s}    │ {s}{s}GFLOPS{s}          │ {s}{s}Time (ms){s}       │ {s}{s}Speedup vs Std{s}  │ {s}{s}Bandwidth (GB/s){s}│\n", .{
            Color.bold, Color.bright_white, Color.reset,
            Color.bold, Color.bright_white, Color.reset,
            Color.bold, Color.bright_white, Color.reset,
            Color.bold, Color.bright_white, Color.reset,
            Color.bold, Color.bright_white, Color.reset
        });
        print("├───────────────────┼───────────────────┼───────────────────┼───────────────────┼───────────────────┤\n", .{});

        // First print standard LN as baseline
        const standard_ms = @as(f64, @floatFromInt(dim_result.standard.min_time_ns)) / 1e6;

        print("│ {s}{s:^15}{s} │ ", .{ Color.blue, "Standard LN", Color.reset });
        printColoredValue(dim_result.standard.max_gflops, getColorForValue(dim_result.standard.max_gflops, min_gflops, max_gflops), 15, 1);
        print(" │ ", .{});
        printColoredValue(standard_ms, getInverseColorForValue(standard_ms, min_time, max_time), 15, 2);
        print(" │ {s}{s:^15}{s} │ ", .{ Color.white, "baseline", Color.reset });
        printColoredValue(dim_result.standard.max_bandwidth_gbps,
            getColorForValue(dim_result.standard.max_bandwidth_gbps, 0, max_bandwidth), 15, 2);
        print(" │\n", .{});

        // Then print all Yolo configurations
        const configs_sorted = try allocator.dupe(YoloConfig, dim_result.configs.items);
        defer allocator.free(configs_sorted);

        // Sort configs by N_A, then by N_B, then by N_S2
        std.sort.pdq(YoloConfig, configs_sorted, {}, struct {
            pub fn lessThan(_: void, a: YoloConfig, b: YoloConfig) bool {
                if (a.N_A == b.N_A) {
                    if (a.N_B == b.N_B) {
                        return a.N_S2 < b.N_S2;
                    }
                    return a.N_B < b.N_B;
                }
                return a.N_A < b.N_A;
            }
        }.lessThan);

        for (configs_sorted) |config| {
            const config_ms = @as(f64, @floatFromInt(config.result.min_time_ns)) / 1e6;
            const speedup = config.result.max_gflops / dim_result.standard.max_gflops;

            // Format the config string
            var config_buf: [32]u8 = undefined;
            const config_str = std.fmt.bufPrint(&config_buf, "Yolo({d},{d},{d})", .{
                config.N_A, config.N_B, config.N_S2
            }) catch unreachable;

            // Get colors
            const gflops_color = getColorForValue(config.result.max_gflops, min_gflops, max_gflops);
            const time_color = getInverseColorForValue(config_ms, min_time, max_time);
            const speedup_color = getColorForValue(speedup, min_speedup, max_speedup);
            const bandwidth_color = getColorForValue(config.result.max_bandwidth_gbps, 0, max_bandwidth);

            // Print row
            print("│ {s:^15} │ ", .{config_str});
            printColoredValue(config.result.max_gflops, gflops_color, 15, 1);
            print(" │ ", .{});
            printColoredValue(config_ms, time_color, 15, 2);
            print(" │ ", .{});
            printColoredValue(speedup, speedup_color, 15, 2);
            print("x │ ", .{});
            printColoredValue(config.result.max_bandwidth_gbps, bandwidth_color, 15, 2);
            print(" │\n", .{});
        }

        print("└───────────────────┴───────────────────┴───────────────────┴───────────────────┴───────────────────┘\n", .{});
    }

    // ===== POWER OF TWO ANALYSIS =====

    printColoredHeader("Best Configuration by Power-of-2 Alignment");

    // Group dimensions by power of 2 and find best config for each power
    const PowerOfTwoConfig = struct {
        power: u6,
        factor: usize,
        dimensions: std.ArrayList(usize),
        best_N_A: usize,
        best_N_B: usize,
        best_N_S2: usize,
        avg_speedup: f64,
    };

    var power_configs = std.AutoHashMap(u6, PowerOfTwoConfig).init(allocator);
    defer {
        var power_iter = power_configs.valueIterator();
        while (power_iter.next()) |config| {
            config.dimensions.deinit();
        }
        power_configs.deinit();
    }

    // Process each dimension
    for (all_results.items) |dim_result| {
        // Find largest power of 2 that divides the dimension
        var dim = dim_result.dim;
        var power: u6 = 0;

        while (dim % 2 == 0) {
            dim /= 2;
            power += 1;
        }

        // Find best config for this dimension
        var best_config_idx: usize = 0;
        var best_gflops: f64 = 0;

        for (dim_result.configs.items, 0..) |config, i| {
            if (config.result.max_gflops > best_gflops) {
                best_gflops = config.result.max_gflops;
                best_config_idx = i;
            }
        }

        const best_config = dim_result.configs.items[best_config_idx];
        const speedup = best_config.result.max_gflops / dim_result.standard.max_gflops;

        // Add to power_configs
        if (power_configs.getPtr(power)) |config| {
            try config.dimensions.append(dim_result.dim);

            // Update best config if this one is better
            if (speedup > config.avg_speedup) {
                config.best_N_A = best_config.N_A;
                config.best_N_B = best_config.N_B;
                config.best_N_S2 = best_config.N_S2;
                config.avg_speedup = speedup;
            }
        } else {
            var dimensions = std.ArrayList(usize).init(allocator);
            try dimensions.append(dim_result.dim);

            try power_configs.put(power, .{
                .power = power,
                .factor = @as(usize, 1) << power,
                .dimensions = dimensions,
                .best_N_A = best_config.N_A,
                .best_N_B = best_config.N_B,
                .best_N_S2 = best_config.N_S2,
                .avg_speedup = speedup,
            });
        }
    }

    // Create sorted list of powers
    var powers = std.ArrayList(u6).init(allocator);
    defer powers.deinit();

    var power_iter = power_configs.keyIterator();
    while (power_iter.next()) |power| {
        try powers.append(power.*);
    }

    // Sort powers
    std.sort.pdq(u6, powers.items, {}, comptime std.sort.asc(u6));

    // Print table header
    print("┌───────────────────┬───────────────────┬───────────────────┬───────────────────┬───────────────────┐\n", .{});
    print("│ {s}{s}Power of 2{s}      │ {s}{s}Factor (2^n){s}   │ {s}{s}Best Config{s}     │ {s}{s}Avg Speedup{s}    │ {s}{s}Dimensions{s}      │\n", .{
        Color.bold, Color.bright_white, Color.reset,
        Color.bold, Color.bright_white, Color.reset,
        Color.bold, Color.bright_white, Color.reset,
        Color.bold, Color.bright_white, Color.reset,
        Color.bold, Color.bright_white, Color.reset
    });
    print("├───────────────────┼───────────────────┼───────────────────┼───────────────────┼───────────────────┤\n", .{});

    // Print results
    for (powers.items) |power| {
        const config = power_configs.get(power).?;

        // Format dimensions string
        var dims_buf: [128]u8 = undefined;
        var dims_str: []u8 = undefined;
        var dims_len: usize = 0;

        if (config.dimensions.items.len <= 3) {
            // List all dimensions if there are 3 or fewer
            for (config.dimensions.items, 0..) |dim, i| {
                if (i > 0) {
                    dims_buf[dims_len] = ',';
                    dims_len += 1;
                    dims_buf[dims_len] = ' ';
                    dims_len += 1;
                }

                const dim_str_len = std.fmt.formatIntBuf(
                    dims_buf[dims_len..], dim, 10, .lower, .{}
                );
                dims_len += dim_str_len;
            }
            dims_str = dims_buf[0..dims_len];
        } else {
            // Otherwise just show count
            dims_str = std.fmt.bufPrint(
                &dims_buf,
                "{d} dimensions",
                .{config.dimensions.items.len}
            ) catch unreachable;
            dims_len = dims_str.len;
        }

        // Format config string
        var config_buf: [32]u8 = undefined;
        const config_str = std.fmt.bufPrint(
            &config_buf,
            "Yolo({d},{d},{d})",
            .{config.best_N_A, config.best_N_B, config.best_N_S2}
        ) catch unreachable;

        // Print row
        print("│ 2^{d:<12} │ {d:<15} │ {s:^15} │ ", .{
            power,
            config.factor,
            config_str,
        });

        // Speedup with color
        printColoredValue(config.avg_speedup, getColorForValue(config.avg_speedup, min_speedup, max_speedup), 15, 2);
        print("x │ {s:<15} │\n", .{dims_buf[0..dims_len]});
    }

    print("└───────────────────┴───────────────────┴───────────────────┴───────────────────┴───────────────────┘\n", .{});

// ===== CONFIGURATION ANALYSIS =====

    printColoredHeader("Performance by Full Configuration");

    // Create an analysis of how each full configuration affects performance
    const ConfigAnalysis = struct {
        N_A: usize,
        N_B: usize,
        N_S2: usize,
        sample_count: usize,
        avg_speedup: f64,
    };

    var config_analysis = std.ArrayList(ConfigAnalysis).init(allocator);
    defer config_analysis.deinit();

    // Create a hash map to store aggregated data for each unique configuration
    var config_map = std.AutoHashMap(struct { N_A: usize, N_B: usize, N_S2: usize }, struct {
        total_speedup: f64,
        count: usize,
    }).init(allocator);
    defer config_map.deinit();

    // Collect all configurations and their performance
    for (all_results.items) |dim_result| {
        for (dim_result.configs.items) |config| {
            const key = .{
                .N_A = config.N_A,
                .N_B = config.N_B,
                .N_S2 = config.N_S2,
            };

            const speedup = config.result.max_gflops / dim_result.standard.max_gflops;

            if (config_map.getPtr(key)) |entry| {
                entry.total_speedup += speedup;
                entry.count += 1;
            } else {
                try config_map.put(key, .{
                    .total_speedup = speedup,
                    .count = 1,
                });
            }
        }
    }

    // Convert map to array for sorting
    var map_it = config_map.iterator();
    while (map_it.next()) |entry| {
        const key = entry.key_ptr.*;
        const value = entry.value_ptr.*;

        try config_analysis.append(.{
            .N_A = key.N_A,
            .N_B = key.N_B,
            .N_S2 = key.N_S2,
            .sample_count = value.count,
            .avg_speedup = value.total_speedup / @as(f64, @floatFromInt(value.count)),
        });
    }

    // Sort configurations by average speedup (descending)
    std.sort.pdq(ConfigAnalysis, config_analysis.items, {}, struct {
        pub fn lessThan(_: void, a: ConfigAnalysis, b: ConfigAnalysis) bool {
            return a.avg_speedup > b.avg_speedup; // Note: descending order
        }
    }.lessThan);

    // Print table header
    print("┌───────────────────┬───────────────────┬───────────────────┬───────────────────┬───────────────────┐\n", .{});
    print("│ {s}{s}N_A{s}            │ {s}{s}N_B{s}            │ {s}{s}N_S2{s}           │ {s}{s}Sample Count{s}   │ {s}{s}Avg Speedup{s}    │\n", .{
        Color.bold, Color.bright_white, Color.reset,
        Color.bold, Color.bright_white, Color.reset,
        Color.bold, Color.bright_white, Color.reset,
        Color.bold, Color.bright_white, Color.reset,
        Color.bold, Color.bright_white, Color.reset
    });
    print("├───────────────────┼───────────────────┼───────────────────┼───────────────────┼───────────────────┤\n", .{});

    // Find min/max speedup for coloring
    var config_min_speedup: f64 = std.math.inf(f64);
    var config_max_speedup: f64 = 0;

    for (config_analysis.items) |analysis| {
        config_min_speedup = @min(config_min_speedup, analysis.avg_speedup);
        config_max_speedup = @max(config_max_speedup, analysis.avg_speedup);
    }

    // Print results
    for (config_analysis.items) |analysis| {
        // Format the config values
        print("│ {d:^15} │ {d:^15} │ {d:^15} │ {d:^15} │ ", .{
            analysis.N_A,
            analysis.N_B,
            analysis.N_S2,
            analysis.sample_count,
        });

        // Print speedup with color
        printColoredValue(analysis.avg_speedup, getColorForValue(analysis.avg_speedup, config_min_speedup, config_max_speedup), 15, 2);
        print("x │\n", .{});
    }

    print("└───────────────────┴───────────────────┴───────────────────┴───────────────────┴───────────────────┘\n", .{});

    print("\n{s}Layer Normalization Grid Search Benchmark completed!{s}\n", .{ Color.bright_green, Color.reset });
}

pub fn main() !void {
    return benchmarkLayerNormGrid(f32);
}