import numpy as np
from safetensors import safe_open
import struct

# Magic number and version constants
WEIGHTS_MAGIC = 0x4D4F4F4E  # "MOON" in ASCII
WEIGHTS_VERSION = 2  # Version 2 for tensor format




def write_tensor(f, name, tensor, transpose=False):
    """Write a single tensor with its header and data."""
    # Convert to numpy array if it's not already
    np_array = tensor.numpy() if hasattr(tensor, 'numpy') else tensor
    
    if np_array.dtype != np.float32:
        np_array = np_array.astype(np.float32)
    
    # Get sizes
    name_bytes = name.encode('utf-8')
    data_size = np.prod(np_array.shape)
    
    # Write header
    header = struct.pack('<III', 
        len(name_bytes),      # name length
        len(np_array.shape),  # shape length 
        data_size            # data length
    )
    f.write(header)
    
    # Write name
    f.write(name_bytes)
    
    # Write shape with explicit 64-bit integers
    shape_bytes = struct.pack(f'<{len(np_array.shape)}Q', *np_array.shape)
    f.write(shape_bytes)
    
    # Write tensor data as float32
    data_bytes = np_array.astype(np.float32).tobytes()
    f.write(data_bytes)
    
    # Debug info
    current_pos = f.tell()
    print(f"Wrote tensor {name}")
    print(f"  Shape: {np_array.shape}")
    print(f"  Header size: {len(header)}")
    print(f"  Name size: {len(name_bytes)}")
    print(f"  Shape info size: {len(shape_bytes)}")
    print(f"  Data size: {len(data_bytes)}")
    print(f"  Current file position: {current_pos}\n")

def collect_and_stack_layer_tensors(f, n_layers, prefix_template, components):
    """Collect tensors across layers and stack them."""
    # Initialize lists to collect tensors
    collected = {name: [] for name in components.keys()}
    
    # Collect tensors from each layer
    for layer in range(n_layers):
        prefix = prefix_template.format(layer)
        for comp_name, (comp_path, should_transpose) in components.items():
            full_path = f"{prefix}.{comp_path}"
            tensor = f.get_tensor(full_path).numpy()
            # Transpose weight matrices before stacking if needed
            if should_transpose:
                tensor = tensor.T
            collected[comp_name].append(tensor)
    
    # Stack tensors along a new first dimension
    stacked = {name: np.stack(tensors) for name, tensors in collected.items()}
    return stacked

def convert_safetensors_to_binary(input_file, output_file):
    with safe_open(input_file, framework="pt", device="cpu") as f:
        with open(output_file, 'wb') as out_f:
            # Write magic number and version
            out_f.write(struct.pack('<II', WEIGHTS_MAGIC, WEIGHTS_VERSION))
            
            # Write word embedding (no transpose)
            write_tensor(out_f, "word_token_embedding", 
                        f.get_tensor("text_model.transformer.embd.wte.weight"))

            # Collect and write text transformer layer tensors
            text_components = {
                # (path, should_transpose)
                "t_ln_w": ("ln.weight", False),
                "t_ln_b": ("ln.bias", False),
                "t_Wqkv_w": ("mixer.Wqkv.weight", True),
                "t_Wqkv_b": ("mixer.Wqkv.bias", False),
                "t_out_proj_w": ("mixer.out_proj.weight", True),
                "t_out_proj_bias": ("mixer.out_proj.bias", False),
                "t_fc1_w": ("mlp.fc1.weight", True),
                "t_fc1_b": ("mlp.fc1.bias", False),
                "t_fc2_w": ("mlp.fc2.weight", True),
                "t_fc2_b": ("mlp.fc2.bias", False),
            }
            
            text_stacked = collect_and_stack_layer_tensors(
                f, 24, "text_model.transformer.h.{}", text_components
            )
            
            # Write stacked text tensors
            for name, tensor in text_stacked.items():
                write_tensor(out_f, name, tensor)

            # Write text model end layers (with transpose for weights)
            write_tensor(out_f, "t_linear_w", 
                        f.get_tensor("text_model.lm_head.linear.weight"), transpose=True)
            write_tensor(out_f, "t_linear_b", 
                        f.get_tensor("text_model.lm_head.linear.bias"))
            write_tensor(out_f, "t_ln_out_w", 
                        f.get_tensor("text_model.lm_head.ln.weight"))
            write_tensor(out_f, "t_ln_out_b", 
                        f.get_tensor("text_model.lm_head.ln.bias"))

            # Vision model start layers
            write_tensor(out_f, "v_patch_embedding_linear_w",
                        f.get_tensor("vision_encoder.encoder.model.visual.patch_embed.linear.weight"), transpose=True)
            write_tensor(out_f, "v_patch_embedding_linear_b",
                        f.get_tensor("vision_encoder.encoder.model.visual.patch_embed.linear.bias"))
            write_tensor(out_f, "v_pos_embedding",
                        f.get_tensor("vision_encoder.encoder.model.visual.pos_embed"))

            # Collect and write vision transformer layer tensors
            vision_components = {
                # (path, should_transpose)
                "v_Wqkv_w": ("attn.qkv.weight", True),
                "v_Wqkv_b": ("attn.qkv.bias", False),
                "v_out_proj_w": ("attn.proj.weight", True),
                "v_out_proj_b": ("attn.proj.bias", False),
                "v_fc1_w": ("mlp.fc1.weight", True),
                "v_fc1_b": ("mlp.fc1.bias", False),
                "v_fc2_w": ("mlp.fc2.weight", True),
                "v_fc2_b": ("mlp.fc2.bias", False),
                "v_norm1_w": ("norm1.weight", False),
                "v_norm1_b": ("norm1.bias", False),
                "v_norm2_w": ("norm2.weight", False),
                "v_norm2_b": ("norm2.bias", False),
            }
            
            vision_stacked = collect_and_stack_layer_tensors(
                f, 27, "vision_encoder.encoder.model.visual.blocks.{}", vision_components
            )
            
            # Write stacked vision tensors
            for name, tensor in vision_stacked.items():
                write_tensor(out_f, name, tensor)

            # Vision end layers
            write_tensor(out_f, "v_norm_out_w",
                        f.get_tensor("vision_encoder.encoder.model.visual.norm.weight"))
            write_tensor(out_f, "v_norm_out_b",
                        f.get_tensor("vision_encoder.encoder.model.visual.norm.bias"))

            # Projection layers
            write_tensor(out_f, "v_proj_fc1_w",
                        f.get_tensor("vision_encoder.projection.mlp.fc1.weight"), transpose=True)
            write_tensor(out_f, "v_proj_fc1_b",
                        f.get_tensor("vision_encoder.projection.mlp.fc1.bias"))
            write_tensor(out_f, "v_proj_fc2_w",
                        f.get_tensor("vision_encoder.projection.mlp.fc2.weight"), transpose=True)
            write_tensor(out_f, "v_proj_fc2_b",
                        f.get_tensor("vision_encoder.projection.mlp.fc2.bias"))
if __name__ == "__main__":
    input_file = 'model.safetensors'
    output_file = 'moondream_f32.bin'
    convert_safetensors_to_binary(input_file, output_file)