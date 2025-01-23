const std = @import("std");

pub const ConfigReader = extern struct {
    const Self = @This();

    // Text model
    vocab: i32, // vocabulary size, usually 256 (byte-level)
    dim: i32, // transformer dimension
    hidden_dim: i32, // for ffn layers
    n_layers: i32, // number of transformer layers, 24 for text model
    n_heads: i32, // number of attn heads per layer
    head_dim: i32, // size of attn heads per layer
    seq_len: i32, // max sequence length
    rope_theta: f32,
    max_pos_embeddings: i32,
    partial_rotary_factor: f32,

    // vision
    img_channels: i32, // number of channels per patch, RGB has 3
    img_dim: i32, // dimension of the the image, 378x378 default
    patch_size: i32, // size of patch, 14x14 default
    vit_embed_len: i32, // vision embed len
    vit_dim: i32, // width of each patch embedding created from linear patch embedding layer, 1152 default
    n_vit_layers: i32,
    n_vit_heads: i32,
    vit_head_dim: i32,
    hidden_features: i32, // the number of hidden features, equivalent to hidden_dim in text model, 4304 default

    pub fn config(self: Self) Config {
        return Config{
            .vocab = @intCast(self.vocab),
            .dim = @intCast(self.dim),
            .hidden_dim = @intCast(self.hidden_dim),
            .n_layers = @intCast(self.n_layers),
            .n_heads = @intCast(self.n_heads),
            .head_dim = @intCast(self.head_dim),
            .seq_len = @intCast(self.seq_len),
            .rope_theta = self.rope_theta,
            .max_pos_embeddings = @intCast(self.max_pos_embeddings),
            .partial_rotary_factor = self.partial_rotary_factor,
            .img_channels = @intCast(self.img_channels),
            .img_dim = @intCast(self.img_dim),
            .patch_size = @intCast(self.patch_size),
            .num_patches = @intCast(@divTrunc(self.img_dim, self.patch_size) * @divTrunc(self.img_dim, self.patch_size)),
            .vit_embed_len = @intCast(self.vit_embed_len),
            .vit_dim = @intCast(self.vit_dim),
            .n_vit_layers = @intCast(self.n_vit_layers),
            .n_vit_heads = @intCast(self.n_vit_heads),
            .vit_head_dim = @intCast(self.vit_head_dim),
            .hidden_features = @intCast(self.hidden_features),
        };
    }
};

pub const Config = struct {

    // Text Model
    vocab: usize,
    dim: usize, //text transformer dim, 2048
    hidden_dim: usize, // hidden fc dim
    n_layers: usize, //number of transformer layers, 24 for text model
    n_heads: usize, //number of attn heads per layer
    head_dim: usize, //size of attn heads
    seq_len: usize, // sequence length, 2048
    rope_theta: f32,
    max_pos_embeddings: usize,
    partial_rotary_factor: f32,

    // Vision Model
    img_channels: usize, // number of channels per patch, RGB has 3
    img_dim: usize, // dimension of the the image, 378x378 default
    patch_size: usize, // size of patch, 14x14 default
    num_patches: usize,
    vit_embed_len: usize,
    vit_dim: usize, // width of each patch embedding created from linear patch embedding layer, 1152 default
    n_vit_layers: usize, // number of ViT layers, 27 default for the vision model
    n_vit_heads: usize, // number of attn heads for each attn layer, 16 default
    vit_head_dim: usize, // size of each attn head, 72 default
    hidden_features: usize, // size of hidden features in ViT fc layers, 4304 in length
};
