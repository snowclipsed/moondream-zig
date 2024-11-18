const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;
const T: usize = 64; // Tile size (adjust as necessary)
const V: usize = 32; // Vector size (adjust as necessary)
const simd_align = @alignOf(@Vector(V, f32));
const Config = @import("config.zig").Config;

const RunState = struct {
    const Self = @This();

    img: []align(simd_align) f32,
    patches: []align(simd_align) f32,
    patch_emb: []align(simd_align) f32,
    final_emb: []align(simd_align) f32,
    projection: []align(simd_align) f32,
    v_x: []align(simd_align) f32,
    v_xb: []align(simd_align) f32,
    v_xb2: []align(simd_align) f32,
    v_xb3: []align(simd_align) f32,
    v_qkv: []align(simd_align) f32,
    v_q: []align(simd_align) f32,
    v_k: []align(simd_align) f32,
    v_v: []align(simd_align) f32,
    v_attn: []align(simd_align) f32,
    v_output: []align(simd_align) f32,
    v_proj: []align(simd_align) f32,
    k_cache: []align(simd_align) f32,
    v_cache: []align(simd_align) f32,
    cos_cache: []align(simd_align) f32,
    sin_cache: []align(simd_align) f32,
    // emb: []align(simd_align) f32,
    // ln_in: []align(simd_align) f32,
    // attn_in: []align(simd_align) f32,
    // x: []align(simd_align) f32,
    // xb: []align(simd_align) f32,
    // mlp_in: []align(simd_align) f32,
    // t_qkv: []align(simd_align) f32, // a buffer that holds the combined kqv
    // q: []align(simd_align) f32,
    // k: []align(simd_align) f32,
    // v: []align(simd_align) f32,
    // attn: []align(simd_align) f32,
    // output: []align(simd_align) f32,
    // inv_freq: []align(simd_align) f32,
    // logits: []align(simd_align) f32,

    fn init(allocator: Allocator, config: Config) !Self {
        return Self{
            .img = try allocator.alignedAlloc(f32, simd_align, config.img_dim * config.img_dim * config.img_channels),
            .patches = try allocator.alignedAlloc(f32, simd_align, config.img_dim * config.img_dim * config.img_channels),
            .patch_emb = try allocator.alignedAlloc(f32, simd_align, config.num_patches * config.vit_dim),
            .final_emb = try allocator.alignedAlloc(f32, simd_align, config.num_patches * 2 * config.vit_dim),
            .projection = try allocator.alignedAlloc(f32, simd_align, config.num_patches * config.dim),
            .v_x = try allocator.alignedAlloc(f32, simd_align, config.num_patches * config.vit_dim),
            .v_xb = try allocator.alignedAlloc(f32, simd_align, config.num_patches * config.hidden_features),
            .v_xb2 = try allocator.alignedAlloc(f32, simd_align, config.num_patches * config.vit_dim),
            .v_xb3 = try allocator.alignedAlloc(f32, simd_align, config.num_patches * config.hidden_dim),
            .v_qkv = try allocator.alignedAlloc(f32, simd_align, config.num_patches * config.vit_dim * 3),
            .v_q = try allocator.alignedAlloc(f32, simd_align, config.num_patches * config.vit_dim),
            .v_k = try allocator.alignedAlloc(f32, simd_align, config.num_patches * config.vit_dim),
            .v_v = try allocator.alignedAlloc(f32, simd_align, config.num_patches * config.vit_dim),
            .v_attn = try allocator.alignedAlloc(f32, simd_align, config.num_patches * config.num_patches * config.vit_dim),
            .v_output = try allocator.alignedAlloc(f32, simd_align, config.num_patches * config.vit_dim),
            .v_proj = try allocator.alignedAlloc(f32, simd_align, config.num_patches * config.vit_dim),
            .k_cache = try allocator.alignedAlloc(f32, simd_align, config.n_layers * config.seq_len * config.dim),
            .v_cache = try allocator.alignedAlloc(f32, simd_align, config.n_layers * config.seq_len * config.dim),
            .cos_cache = try allocator.alignedAlloc(f32, simd_align, config.dim * config.head_dim / 2),
            .sin_cache = try allocator.alignedAlloc(f32, simd_align, config.dim * config.head_dim / 2), // hardcoding the partial rotary factor as 1/2 here
            // .emb = try allocator.alignedAlloc(f32, simd_align, config.dim),
            // .ln_in = try allocator.alignedAlloc(f32, simd_align, config.dim),
            // .attn_in = try allocator.alignedAlloc(f32, simd_align, config.dim),
            // .x = try allocator.alignedAlloc(f32, simd_align, config.hidden_dim),
            // .xb = try allocator.alignedAlloc(f32, simd_align, config.dim),
            // .mlp_in = try allocator.alignedAlloc(f32, simd_align, config.dim),
            // .t_qkv = try allocator.alignedAlloc(f32, simd_align, config.dim * 3),
            // .q = try allocator.alignedAlloc(f32, simd_align, config.n_heads * config.head_dim),
            // .k = try allocator.alignedAlloc(f32, simd_align, config.n_heads * config.head_dim),
            // .v = try allocator.alignedAlloc(f32, simd_align, config.n_heads * config.head_dim),
            // .attn = try allocator.alignedAlloc(f32, simd_align, config.n_heads * config.seq_len),
            // .output = try allocator.alignedAlloc(f32, simd_align, config.dim),
            // .inv_freq = try allocator.alignedAlloc(f32, simd_align, config.dim / 2),
            // .logits = try allocator.alignedAlloc(f32, simd_align, config.vocab),
        };
    }

    fn deinit(self: *Self, allocator: Allocator) void {
        inline for (std.meta.fields(Self)) |f| {
            allocator.free(@field(self, f.name));
        }
        self.* = undefined;
    }
};
