const std = @import("std");
const Allocator = std.mem.Allocator;
const Tensor = @import("tensor.zig").Tensor;
const Weights = @import("weights.zig").Weights;
pub const TextPreSlicedWeights = struct {
    // Layer norm weights
    t_ln_w: []Tensor(f16),
    t_ln_b: []Tensor(f16),

    // Attention weights
    t_Wqkv_w: []Tensor(f16),
    t_Wqkv_b: []Tensor(f16),
    t_out_proj_w: []Tensor(f16),
    t_out_proj_b: []Tensor(f16),

    // MLP weights
    t_fc1_w: []Tensor(f16),
    t_fc1_b: []Tensor(f16),
    t_fc2_w: []Tensor(f16),
    t_fc2_b: []Tensor(f16),

    // Original weights reference (kept for non-sliced weights)
    original_weights: Weights,
    allocator: Allocator,

    pub fn init(allocator: Allocator, weights: Weights, n_layers: usize) !TextPreSlicedWeights {
        // Allocate arrays for each set of weights
        var t_ln_w = try allocator.alloc(Tensor(f16), n_layers);
        errdefer allocator.free(t_ln_w);
        var t_ln_b = try allocator.alloc(Tensor(f16), n_layers);
        errdefer allocator.free(t_ln_b);

        var t_Wqkv_w = try allocator.alloc(Tensor(f16), n_layers);
        errdefer allocator.free(t_Wqkv_w);
        var t_Wqkv_b = try allocator.alloc(Tensor(f16), n_layers);
        errdefer allocator.free(t_Wqkv_b);
        var t_out_proj_w = try allocator.alloc(Tensor(f16), n_layers);
        errdefer allocator.free(t_out_proj_w);
        var t_out_proj_b = try allocator.alloc(Tensor(f16), n_layers);
        errdefer allocator.free(t_out_proj_b);

        var t_fc1_w = try allocator.alloc(Tensor(f16), n_layers);
        errdefer allocator.free(t_fc1_w);
        var t_fc1_b = try allocator.alloc(Tensor(f16), n_layers);
        errdefer allocator.free(t_fc1_b);
        var t_fc2_w = try allocator.alloc(Tensor(f16), n_layers);
        errdefer allocator.free(t_fc2_w);
        var t_fc2_b = try allocator.alloc(Tensor(f16), n_layers);
        errdefer allocator.free(t_fc2_b);

        // Pre-slice all weights
        for (0..n_layers) |i| {
            // Layer norms
            t_ln_w[i] = try weights.t_ln_w.getDimensionSlice(0, i);
            errdefer t_ln_w[i].deinit();
            t_ln_b[i] = try weights.t_ln_b.getDimensionSlice(0, i);
            errdefer t_ln_b[i].deinit();

            // Attention
            t_Wqkv_w[i] = try weights.t_Wqkv_w.getDimensionSlice(0, i);
            errdefer t_Wqkv_w[i].deinit();
            t_Wqkv_b[i] = try weights.t_Wqkv_b.getDimensionSlice(0, i);
            errdefer t_Wqkv_b[i].deinit();
            t_out_proj_w[i] = try weights.t_out_proj_w.getDimensionSlice(0, i);
            errdefer t_out_proj_w[i].deinit();
            t_out_proj_b[i] = try weights.t_out_proj_bias.getDimensionSlice(0, i);
            errdefer t_out_proj_b[i].deinit();

            // MLP
            t_fc1_w[i] = try weights.t_fc1_w.getDimensionSlice(0, i);
            errdefer t_fc1_w[i].deinit();
            t_fc1_b[i] = try weights.t_fc1_b.getDimensionSlice(0, i);
            errdefer t_fc1_b[i].deinit();
            t_fc2_w[i] = try weights.t_fc2_w.getDimensionSlice(0, i);
            errdefer t_fc2_w[i].deinit();
            t_fc2_b[i] = try weights.t_fc2_b.getDimensionSlice(0, i);
            errdefer t_fc2_b[i].deinit();
        }

        return TextPreSlicedWeights{
            .t_ln_w = t_ln_w,
            .t_ln_b = t_ln_b,
            .t_Wqkv_w = t_Wqkv_w,
            .t_Wqkv_b = t_Wqkv_b,
            .t_out_proj_w = t_out_proj_w,
            .t_out_proj_b = t_out_proj_b,
            .t_fc1_w = t_fc1_w,
            .t_fc1_b = t_fc1_b,
            .t_fc2_w = t_fc2_w,
            .t_fc2_b = t_fc2_b,
            .original_weights = weights,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *TextPreSlicedWeights) void {
        for (self.t_ln_w) |*tensor| tensor.deinit();
        for (self.t_ln_b) |*tensor| tensor.deinit();

        for (self.t_Wqkv_w) |*tensor| tensor.deinit();
        for (self.t_Wqkv_b) |*tensor| tensor.deinit();
        for (self.t_out_proj_w) |*tensor| tensor.deinit();
        for (self.t_out_proj_b) |*tensor| tensor.deinit();

        for (self.t_fc1_w) |*tensor| tensor.deinit();
        for (self.t_fc1_b) |*tensor| tensor.deinit();
        for (self.t_fc2_w) |*tensor| tensor.deinit();
        for (self.t_fc2_b) |*tensor| tensor.deinit();

        // Free the arrays
        self.allocator.free(self.t_ln_w);
        self.allocator.free(self.t_ln_b);
        self.allocator.free(self.t_Wqkv_w);
        self.allocator.free(self.t_Wqkv_b);
        self.allocator.free(self.t_out_proj_w);
        self.allocator.free(self.t_out_proj_b);
        self.allocator.free(self.t_fc1_w);
        self.allocator.free(self.t_fc1_b);
        self.allocator.free(self.t_fc2_w);
        self.allocator.free(self.t_fc2_b);
    }
};
