const std = @import("std");
const Allocator = std.mem.Allocator;
const print = std.debug.print;
const time = std.time;
const math = std.math;

// Import the implementations
const layerNormOldF64 = @import("ops.zig").layerNormOldF64;
const layerNormOld = @import("ops.zig").layerNormOld;
const layerNormInner = @import("ops.zig").layerNormInner;
const SlabReusingAllocator = @import("slab_reusing_allocator.zig").SlabReusingAllocator;

// Import common dependency
const Tensor = @import("tensor.zig").Tensor;

// Import benchmark utilities
const printColoredHeader = @import("bench.zig").printColoredHeader;
const getColorForValue = @import("bench.zig").getColorForValue;
const getInverseColorForValue = @import("bench.zig").getInverseColorForValue;
const printColoredValue = @import("bench.zig").printColoredValue;
const Color = @import("bench.zig").Color;
const BenchmarkResult = @import("bench.zig").BenchmarkResult;

fn benchmarkSLayerNormYolo(allocator: Allocator, M: usize, K: usize, num_runs: usize) !BenchmarkResult {
    return benchmarkSLayerNormInner(10, 1, allocator, M, K, num_runs);
}

fn benchmarkSLayerNormInner(comptime T: type, comptime large_unroll: usize, comptime stage2_unroll: usize, allocator: Allocator, M: usize, K: usize, num_runs: usize) !BenchmarkResult {
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
    var warmup = try layerNormInner(T, large_unroll, large_unroll, stage2_unroll, false, a, g, b, eps);
    warmup.deinit();

    // Benchmark runs
    var total_time: u64 = 0;
    var min_time: u64 = std.math.maxInt(u64);
    var max_time: u64 = 0;
    var max_gflops: f64 = 0;

    for (0..num_runs) |_| {
        var timer = try time.Timer.start();
        var result = try layerNormInner(T, large_unroll, large_unroll, stage2_unroll, false, a, g, b, eps);
        const elapsed = timer.read();
        result.deinit();

        total_time += elapsed;
        min_time = @min(min_time, elapsed);
        max_time = @max(max_time, elapsed);

        // Calculate GFLOPS for this run
        const seconds = @as(f64, @floatFromInt(elapsed)) / 1e9;
        const ops = 10 * M * K;
        const gflops = @as(f64, @floatFromInt(ops)) / seconds / 1e9;
        max_gflops = @max(max_gflops, gflops);
    }

    const avg_time = total_time / num_runs;
    const avg_seconds = @as(f64, @floatFromInt(min_time)) / 1e9;
    const ops = 10 * M * K;
    const avg_gflops = @as(f64, @floatFromInt(ops)) / avg_seconds / 1e9;

    // Calculate memory bandwidth
    const bytes_accessed = @as(f64, @floatFromInt((2 * M * K) * @sizeOf(T)));
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
    var total_time: u64 = 0;
    var min_time: u64 = std.math.maxInt(u64);
    var max_time: u64 = 0;
    var max_gflops: f64 = 0;

    for (0..num_runs) |_| {
        //_ = try layerNormOldF64(T, a, g, b, eps);
        var timer = try time.Timer.start();
        var result = try layerNormOld(T, a, g, b, eps);
        const elapsed = timer.read();
        result.deinit();

        total_time += elapsed;
        min_time = @min(min_time, elapsed);
        max_time = @max(max_time, elapsed);

        // Calculate GFLOPS for this run
        const seconds = @as(f64, @floatFromInt(elapsed)) / 1e9;
        const ops = 10 * M * K;
        const gflops = @as(f64, @floatFromInt(ops)) / seconds / 1e9;
        max_gflops = @max(max_gflops, gflops);
    }

    const avg_time = total_time / num_runs;
    const avg_seconds = @as(f64, @floatFromInt(avg_time)) / 1e9;
    const ops = 10 * M * K;
    const avg_gflops = @as(f64, @floatFromInt(ops)) / avg_seconds / 1e9;

    // Calculate memory bandwidth
    const bytes_accessed = @as(f64, @floatFromInt((2 * M * K) * @sizeOf(T)));
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

pub fn benchmarkLayerNormGrid(T: type) !void {
    // Initialize allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const gpa_allocator = gpa.allocator();
    var slab_reusing_allocator = SlabReusingAllocator(100).init(gpa_allocator);
    defer slab_reusing_allocator.deinit();
    const allocator = slab_reusing_allocator.allocator();

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
    print("║ Grid Search:          Unroll factors from 0 to 8           ║\n", .{});
    print("╚══════════════════════════════════════════════════════════════╝\n\n", .{});

    print("Running layer normalization grid search...\n", .{});

    // Define a struct to hold benchmark results for a specific configuration
    const YoloConfig = struct {
        first_unroll: u8,
        second_unroll: u8,
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

    // Benchmark each dimension
    for (embed_dims) |dim_info| {
        print("  Benchmarking Layer Norm for dimension: {d} ({s})...\n", .{ dim_info.dim, dim_info.desc });

        // First benchmark standard LayerNorm as baseline
        const standard_result = try benchmarkSLayerNormOld(T, allocator, M, dim_info.dim, num_runs);

        // Create configs list for this dimension
        var configs = std.ArrayList(YoloConfig).init(allocator);

        // Now run grid search over unroll factors (first_unroll, second_unroll)
        // Total unroll will range from 0 to 8, divided between first and second unroll
        inline for (8..9) |first_unroll| {
            print("    Testing first unroll factor {d}...\n", .{first_unroll});

            inline for (4..5) |second_unroll| {
                // Benchmark YoloLN with these unroll factors
                const yolo_result = try benchmarkSLayerNormInner(
                    T,
                    first_unroll,
                    second_unroll,
                    allocator,
                    M,
                    dim_info.dim,
                    num_runs*10,
                );

                try configs.append(.{
                    .first_unroll = @intCast(first_unroll),
                    .second_unroll = @intCast(second_unroll),
                    .result = yolo_result,
                });
            }
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
    var max_time: f64 = 0;    // Find min/max speedups for color normalization
    var min_bandwidth: f64 = std.math.inf(f64);
    var max_bandwidth: f64 = 0;

    for (all_results.items) |dim_result| {
        // Check standard
        min_gflops = @min(min_gflops, dim_result.standard.max_gflops);
        max_gflops = @max(max_gflops, dim_result.standard.max_gflops);

        const standard_ms = @as(f64, @floatFromInt(dim_result.standard.min_time_ns)) / 1e6;
        min_time = @min(min_time, standard_ms);
        max_time = @max(max_time, standard_ms);
        min_bandwidth = @min(min_bandwidth, dim_result.standard.avg_bandwidth_gbps);
        max_bandwidth = @max(max_bandwidth, dim_result.standard.avg_bandwidth_gbps);

        // Check all configs
        for (dim_result.configs.items) |config| {
            min_gflops = @min(min_gflops, config.result.max_gflops);
            max_gflops = @max(max_gflops, config.result.max_gflops);
            min_bandwidth = @min(min_bandwidth, config.result.avg_bandwidth_gbps);
            max_bandwidth = @max(max_bandwidth, config.result.avg_bandwidth_gbps);

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
        const config_str = std.fmt.bufPrint(&config_buf, "Yolo({d},{d})", .{
            best_config.first_unroll, best_config.second_unroll
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
        printColoredValue(dim_result.standard.avg_bandwidth_gbps,
            getColorForValue(dim_result.standard.avg_bandwidth_gbps, 0, max_bandwidth), 15, 2);
        print(" │\n", .{});

        // Then print all Yolo configurations, sorted by total unroll
        const configs_sorted = try allocator.dupe(YoloConfig, dim_result.configs.items);
        defer allocator.free(configs_sorted);

        // Sort configs by total unroll, then by first_unroll
        std.sort.pdq(YoloConfig, configs_sorted, {}, struct {
            pub fn lessThan(_: void, a: YoloConfig, b: YoloConfig) bool {
                const a_total = a.first_unroll + a.second_unroll;
                const b_total = b.first_unroll + b.second_unroll;
                if (a_total == b_total) {
                    return a.first_unroll < b.first_unroll;
                }
                return a_total < b_total;
            }
        }.lessThan);

        for (configs_sorted) |config| {
            const config_ms = @as(f64, @floatFromInt(config.result.min_time_ns)) / 1e6;
            const speedup = config.result.max_gflops / dim_result.standard.max_gflops;

            // Format the config string
            var config_buf: [32]u8 = undefined;
            const config_str = std.fmt.bufPrint(&config_buf, "Yolo({d},{d})", .{
                config.first_unroll, config.second_unroll
            }) catch unreachable;

            // Get colors
            const gflops_color = getColorForValue(config.result.max_gflops, min_gflops, max_gflops);
            const time_color = getInverseColorForValue(config_ms, min_time, max_time);
            const speedup_color = getColorForValue(speedup, min_speedup, max_speedup);
            const bandwidth_color = getColorForValue(config.result.avg_bandwidth_gbps, 0, max_bandwidth);

            // Print row
            print("│ {s:^15} │ ", .{config_str});
            printColoredValue(config.result.max_gflops, gflops_color, 15, 1);
            print(" │ ", .{});
            printColoredValue(config_ms, time_color, 15, 2);
            print(" │ ", .{});
            printColoredValue(speedup, speedup_color, 15, 2);
            print("x │ ", .{});
            printColoredValue(config.result.avg_bandwidth_gbps, bandwidth_color, 15, 2);
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
        best_first_unroll: u8,
        best_second_unroll: u8,
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
                config.best_first_unroll = best_config.first_unroll;
                config.best_second_unroll = best_config.second_unroll;
                config.avg_speedup = speedup;
            }
        } else {
            var dimensions = std.ArrayList(usize).init(allocator);
            try dimensions.append(dim_result.dim);

            try power_configs.put(power, .{
                .power = power,
                .factor = @as(usize, 1) << power,
                .dimensions = dimensions,
                .best_first_unroll = best_config.first_unroll,
                .best_second_unroll = best_config.second_unroll,
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
            "Yolo({d},{d})",
            .{config.best_first_unroll, config.best_second_unroll}
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

    // ===== UNROLL FACTOR ANALYSIS =====

    printColoredHeader("Performance by Unroll Factor Distribution");

    // Create a 2D grid of total_unroll vs first_unroll ratio
    const max_unroll = 20;
    const UnrollAnalysis = struct {
        total_unroll: u8,
        first_ratio: f64, // first_unroll / total_unroll
        sample_count: usize,
        avg_speedup: f64,
    };

    var unroll_grid = std.ArrayList(UnrollAnalysis).init(allocator);
    defer unroll_grid.deinit();

    // Collect data for the grid
    for (1..max_unroll + 1) |total_unroll| {
        for (0..total_unroll + 1) |first_unroll| {
            const first_ratio = if (total_unroll > 0)
                @as(f64, @floatFromInt(first_unroll)) / @as(f64, @floatFromInt(total_unroll))
            else
                0.0;

            var sample_count: usize = 0;
            var total_speedup: f64 = 0.0;

            // Collect all results with this unroll configuration
            for (all_results.items) |dim_result| {
                for (dim_result.configs.items) |config| {
                    if (config.first_unroll + config.second_unroll == total_unroll and
                        config.first_unroll == first_unroll) {
                        sample_count += 1;
                        total_speedup += config.result.max_gflops / dim_result.standard.max_gflops;
                    }
                }
            }

            if (sample_count > 0) {
                try unroll_grid.append(.{
                    .total_unroll = @intCast(total_unroll),
                    .first_ratio = first_ratio,
                    .sample_count = sample_count,
                    .avg_speedup = total_speedup / @as(f64, @floatFromInt(sample_count)),
                });
            }
        }
    }

    // Print table header
    print("┌───────────────────┬───────────────────┬───────────────────┬───────────────────┬───────────────────┐\n", .{});
    print("│ {s}{s}Total Unroll{s}    │ {s}{s}First Unroll{s}    │ {s}{s}Second Unroll{s}   │ {s}{s}First Ratio{s}     │ {s}{s}Avg Speedup{s}    │\n", .{
        Color.bold, Color.bright_white, Color.reset,
        Color.bold, Color.bright_white, Color.reset,
        Color.bold, Color.bright_white, Color.reset,
        Color.bold, Color.bright_white, Color.reset,
        Color.bold, Color.bright_white, Color.reset
    });
    print("├───────────────────┼───────────────────┼───────────────────┼───────────────────┼───────────────────┤\n", .{});

    // Find min/max speedup for coloring
    var unroll_min_speedup: f64 = std.math.inf(f64);
    var unroll_max_speedup: f64 = 0;

    for (unroll_grid.items) |analysis| {
        unroll_min_speedup = @min(unroll_min_speedup, analysis.avg_speedup);
        unroll_max_speedup = @max(unroll_max_speedup, analysis.avg_speedup);
    }

    // Sort by total_unroll, then by first_ratio
    std.sort.pdq(UnrollAnalysis, unroll_grid.items, {}, struct {
        pub fn lessThan(_: void, a: UnrollAnalysis, b: UnrollAnalysis) bool {
            if (a.total_unroll == b.total_unroll) {
                return a.first_ratio < b.first_ratio;
            }
            return a.total_unroll < b.total_unroll;
        }
    }.lessThan);

    // Print results
    for (unroll_grid.items) |analysis| {
        const first_unroll = @as(usize, @intFromFloat(@as(f64, @floatFromInt(analysis.total_unroll)) * analysis.first_ratio));
        const second_unroll = analysis.total_unroll - @as(u8, @intCast(first_unroll));

        print("│ {d:^15} │ {d:^15} │ {d:^15} │ ", .{
            analysis.total_unroll,
            first_unroll,
            second_unroll,
        });

        // Print first_ratio as percentage
        printColoredValue(analysis.first_ratio * 100.0, Color.white, 12, 1);
        print("% │ ", .{});

        // Print speedup with color
        printColoredValue(analysis.avg_speedup, getColorForValue(analysis.avg_speedup, unroll_min_speedup, unroll_max_speedup), 15, 2);
        print("x │\n", .{});
    }

    print("└───────────────────┴───────────────────┴───────────────────┴───────────────────┴───────────────────┘\n", .{});

    print("\n{s}Layer Normalization Grid Search Benchmark completed!{s}\n", .{ Color.bright_green, Color.reset });
}

pub fn main() !void {
    return benchmarkLayerNormGrid(f32);
}