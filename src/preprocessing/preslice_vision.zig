const std = @import("std");
const Allocator = std.mem.Allocator;
const Tensor = @import("../core/tensor.zig").Tensor;
const Weights = @import("../model/weights.zig").Weights;

pub const visionPreSlicedWeights = struct {
    // Transformer block weights
    v_norm1_w: []Tensor(f16),
    v_norm1_b: []Tensor(f16),
    v_norm2_w: []Tensor(f16),
    v_norm2_b: []Tensor(f16),

    // Attention weights
    v_Wqkv_w: []Tensor(f16),
    v_Wqkv_b: []Tensor(f16),
    v_out_proj_w: []Tensor(f16),
    v_out_proj_b: []Tensor(f16),

    // MLP weights
    v_fc1_w: []Tensor(f16),
    v_fc1_b: []Tensor(f16),
    v_fc2_w: []Tensor(f16),
    v_fc2_b: []Tensor(f16),

    // Original weights reference (kept for non-sliced weights)
    original_weights: Weights,
    allocator: Allocator,

    pub fn init(allocator: Allocator, weights: Weights, n_layers: usize) !visionPreSlicedWeights {
        // Allocate arrays for each set of weights
        var v_norm1_w = try allocator.alloc(Tensor(f16), n_layers);
        errdefer allocator.free(v_norm1_w);
        var v_norm1_b = try allocator.alloc(Tensor(f16), n_layers);
        errdefer allocator.free(v_norm1_b);
        var v_norm2_w = try allocator.alloc(Tensor(f16), n_layers);
        errdefer allocator.free(v_norm2_w);
        var v_norm2_b = try allocator.alloc(Tensor(f16), n_layers);
        errdefer allocator.free(v_norm2_b);

        var v_Wqkv_w = try allocator.alloc(Tensor(f16), n_layers);
        errdefer allocator.free(v_Wqkv_w);
        var v_Wqkv_b = try allocator.alloc(Tensor(f16), n_layers);
        errdefer allocator.free(v_Wqkv_b);
        var v_out_proj_w = try allocator.alloc(Tensor(f16), n_layers);
        errdefer allocator.free(v_out_proj_w);
        var v_out_proj_b = try allocator.alloc(Tensor(f16), n_layers);
        errdefer allocator.free(v_out_proj_b);

        var v_fc1_w = try allocator.alloc(Tensor(f16), n_layers);
        errdefer allocator.free(v_fc1_w);
        var v_fc1_b = try allocator.alloc(Tensor(f16), n_layers);
        errdefer allocator.free(v_fc1_b);
        var v_fc2_w = try allocator.alloc(Tensor(f16), n_layers);
        errdefer allocator.free(v_fc2_w);
        var v_fc2_b = try allocator.alloc(Tensor(f16), n_layers);
        errdefer allocator.free(v_fc2_b);

        // Pre-slice all weights
        for (0..n_layers) |i| {
            // Layer norms
            v_norm1_w[i] = try weights.v_norm1_w.getDimensionSlice(0, i);
            errdefer v_norm1_w[i].deinit();
            v_norm1_b[i] = try weights.v_norm1_b.getDimensionSlice(0, i);
            errdefer v_norm1_b[i].deinit();
            v_norm2_w[i] = try weights.v_norm2_w.getDimensionSlice(0, i);
            errdefer v_norm2_w[i].deinit();
            v_norm2_b[i] = try weights.v_norm2_b.getDimensionSlice(0, i);
            errdefer v_norm2_b[i].deinit();

            // Attention
            v_Wqkv_w[i] = try weights.v_Wqkv_w.getDimensionSlice(0, i);
            errdefer v_Wqkv_w[i].deinit();
            v_Wqkv_b[i] = try weights.v_Wqkv_b.getDimensionSlice(0, i);
            errdefer v_Wqkv_b[i].deinit();
            v_out_proj_w[i] = try weights.v_out_proj_w.getDimensionSlice(0, i);
            errdefer v_out_proj_w[i].deinit();
            v_out_proj_b[i] = try weights.v_out_proj_b.getDimensionSlice(0, i);
            errdefer v_out_proj_b[i].deinit();

            // MLP
            v_fc1_w[i] = try weights.v_fc1_w.getDimensionSlice(0, i);
            errdefer v_fc1_w[i].deinit();
            v_fc1_b[i] = try weights.v_fc1_b.getDimensionSlice(0, i);
            errdefer v_fc1_b[i].deinit();
            v_fc2_w[i] = try weights.v_fc2_w.getDimensionSlice(0, i);
            errdefer v_fc2_w[i].deinit();
            v_fc2_b[i] = try weights.v_fc2_b.getDimensionSlice(0, i);
            errdefer v_fc2_b[i].deinit();
        }

        return visionPreSlicedWeights{
            .v_norm1_w = v_norm1_w,
            .v_norm1_b = v_norm1_b,
            .v_norm2_w = v_norm2_w,
            .v_norm2_b = v_norm2_b,
            .v_Wqkv_w = v_Wqkv_w,
            .v_Wqkv_b = v_Wqkv_b,
            .v_out_proj_w = v_out_proj_w,
            .v_out_proj_b = v_out_proj_b,
            .v_fc1_w = v_fc1_w,
            .v_fc1_b = v_fc1_b,
            .v_fc2_w = v_fc2_w,
            .v_fc2_b = v_fc2_b,
            .original_weights = weights,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *visionPreSlicedWeights) void {
        // Free all pre-sliced tensors
        for (self.v_norm1_w) |*tensor| tensor.deinit();
        for (self.v_norm1_b) |*tensor| tensor.deinit();
        for (self.v_norm2_w) |*tensor| tensor.deinit();
        for (self.v_norm2_b) |*tensor| tensor.deinit();

        for (self.v_Wqkv_w) |*tensor| tensor.deinit();
        for (self.v_Wqkv_b) |*tensor| tensor.deinit();
        for (self.v_out_proj_w) |*tensor| tensor.deinit();
        for (self.v_out_proj_b) |*tensor| tensor.deinit();

        for (self.v_fc1_w) |*tensor| tensor.deinit();
        for (self.v_fc1_b) |*tensor| tensor.deinit();
        for (self.v_fc2_w) |*tensor| tensor.deinit();
        for (self.v_fc2_b) |*tensor| tensor.deinit();

        // Free the arrays themselves
        self.allocator.free(self.v_norm1_w);
        self.allocator.free(self.v_norm1_b);
        self.allocator.free(self.v_norm2_w);
        self.allocator.free(self.v_norm2_b);
        self.allocator.free(self.v_Wqkv_w);
        self.allocator.free(self.v_Wqkv_b);
        self.allocator.free(self.v_out_proj_w);
        self.allocator.free(self.v_out_proj_b);
        self.allocator.free(self.v_fc1_w);
        self.allocator.free(self.v_fc1_b);
        self.allocator.free(self.v_fc2_w);
        self.allocator.free(self.v_fc2_b);
    }
};
