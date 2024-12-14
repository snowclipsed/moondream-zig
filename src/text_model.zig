const std = @import("std");
const Allocator = std.mem.Allocator;
const Weights = @import("weights.zig").Weights;
const Config = @import("config.zig").Config;
const Tensor = @import("tensor.zig").Tensor;

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
        var hidden_BTC = try input_embeds.copy();

        std.debug.print("hidden_BTC shape: {d}\n", .{hidden_BTC.shape});
        std.debug.print("dim: {d}\n", .{dim});

        defer hidden_BTC.deinit();
    }
};
