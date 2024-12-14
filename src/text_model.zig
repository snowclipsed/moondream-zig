const std = @import("std");
const print = std.debug.print;
const Allocator = std.mem.Allocator;
const Weights = @import("weights.zig").Weights;
const Config = @import("config.zig").Config;
const Tensor = @import("tensor.zig").Tensor;
const ops = @import("ops.zig");

pub const TextModel = struct {
    const Self = @This();
    config: Config,
    weights: Weights,
    allocator: Allocator,

    pub fn init(config: Config, weights: Weights, allocator: Allocator) TextModel {
        const textmodel = TextModel{
            .config = config,
            .weights = weights,
            .allocator = allocator,
        };
        return textmodel;
    }

    pub fn deinit(self: *Self) void {
        self.weights.deinit();

        self.* = undefined;
    }

    // TODO: Implement deinit
    pub fn text_decoder(self: Self, input_embeds: Tensor(f32)) !void {
        const dim = self.config.dim;
        const eps = 1e-5; // TODO move to config
        var hidden_BTC = try input_embeds.copy();

        print("hidden_BTC shape: {d}\n", .{hidden_BTC.shape});
        print("dim: {d}\n", .{dim});

        for (0..self.config.n_layers) |layer| {
            print("Layer: {d}\n", .{layer});

            var layer_ln_w = try self.weights.t_ln_w.getDimensionSlice(0, layer);
            var layer_ln_b = try self.weights.t_ln_b.getDimensionSlice(0, layer);
            var ln_in = try ops.layerNorm(
                f32,
                hidden_BTC,
                layer_ln_w,
                layer_ln_b,
                eps,
            );

            // frees
            defer layer_ln_w.deinit();
            defer layer_ln_b.deinit();
            ln_in.deinit();
        }

        defer hidden_BTC.deinit();
    }

    // fn attention_block(self: Self) !Tensor(f32) {
    //     const dim = self.config.dim;
    //     const n_heads = self.config.n_heads;
    //     const head_dim = self.config.head_dim;
    // }
};
