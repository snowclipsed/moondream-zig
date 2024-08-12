import numpy as np
from safetensors import safe_open
import struct

def convert_safetensors_to_custom_binary(input_file, output_file):
    total_float_count = 0
    text_repeat_counts = {}
    vision_repeat_counts = {}

    def process_tensor(tensor_name, tensor, out_f):
        nonlocal total_float_count
        np_array = tensor.numpy()
        
        if np_array.dtype != np.float16:
            np_array = np_array.astype(np.float16)
        
        float_count = np_array.size
        total_float_count += float_count
        
        print(f"{tensor_name}: {np_array.shape}, Floats: {float_count:,}")
        
        out_f.write(np_array.tobytes())
        return float_count

    def update_repeat_count(counts_dict, suffix, count):
        if suffix not in counts_dict:
            counts_dict[suffix] = 0
        counts_dict[suffix] += count

    with safe_open(input_file, framework="pt", device="cpu") as f:
        text_start_layers = [
            "text_model.transformer.embd.wte.weight"
        ]
        text_repeat_layers = [
            ".ln.weight",
            ".ln.bias",
            ".mixer.Wqkv.weight",
            ".mixer.Wqkv.bias",
            ".mixer.out_proj.weight",
            ".mixer.out_proj.bias",
            ".mlp.fc1.weight",
            ".mlp.fc1.bias",
            ".mlp.fc2.weight",
            ".mlp.fc2.bias",
        ]
        text_end_layers = [
            "text_model.lm_head.linear.weight",
            "text_model.lm_head.linear.bias",
            "text_model.lm_head.ln.weight",
            "text_model.lm_head.ln.bias"
        ]
        vision_start_layers = [
            "vision_encoder.encoder.model.visual.patch_embed.linear.weight",
            "vision_encoder.encoder.model.visual.patch_embed.linear.bias",
            "vision_encoder.encoder.model.visual.pos_embed"
        ]
        vision_repeat_layers = [
            ".attn.qkv.weight",
            ".attn.qkv.bias",
            ".attn.proj.weight",
            ".attn.proj.bias",
            ".mlp.fc1.weight",
            ".mlp.fc1.bias",
            ".mlp.fc2.weight",
            ".mlp.fc2.bias",
            ".norm1.weight",
            ".norm1.bias",
            ".norm2.weight",
            ".norm2.bias",
        ]
        vision_end_layers = [
            "vision_encoder.encoder.model.visual.norm.weight",
            "vision_encoder.encoder.model.visual.norm.bias"
        ]
        projection_layers = [
            "vision_encoder.projection.mlp.fc1.weight",
            "vision_encoder.projection.mlp.fc1.bias",
            "vision_encoder.projection.mlp.fc2.weight",
            "vision_encoder.projection.mlp.fc2.bias"
        ]
        
        text_model_prefix = "text_model.transformer.h."
        vision_model_prefix = "vision_encoder.encoder.model.visual.blocks."
        transformer_layers = 24
        vision_layers = 27

        with open(output_file, 'wb') as out_f:
            print("Text start layers:")
            for tensor_name in text_start_layers:
                process_tensor(tensor_name, f.get_tensor(tensor_name), out_f)

            for layer in range(transformer_layers):
                for tensor_suffix in text_repeat_layers:
                    tensor_name = f"{text_model_prefix}{layer}{tensor_suffix}"
                    if tensor_name in f.keys():
                        count = process_tensor(tensor_name, f.get_tensor(tensor_name), out_f)
                        update_repeat_count(text_repeat_counts, tensor_suffix, count)

            print("\nText repeat layers summary:")
            for suffix, count in text_repeat_counts.items():
                print(f"  {suffix}: {count:,} floats")

            print("\nText end layers:")
            for tensor_name in text_end_layers:
                if tensor_name in f.keys():
                    process_tensor(tensor_name, f.get_tensor(tensor_name), out_f)
                else:
                    print(f"Warning: Tensor {tensor_name} not found")

            print("\nVision start layers:")
            for tensor_name in vision_start_layers:
                if tensor_name in f.keys():
                    process_tensor(tensor_name, f.get_tensor(tensor_name), out_f)
                else:
                    print(f"Warning: Tensor {tensor_name} not found")

            for layer in range(vision_layers):
                for tensor_suffix in vision_repeat_layers:
                    tensor_name = f"{vision_model_prefix}{layer}{tensor_suffix}"
                    if tensor_name in f.keys():
                        count = process_tensor(tensor_name, f.get_tensor(tensor_name), out_f)
                        update_repeat_count(vision_repeat_counts, tensor_suffix, count)

            print("\nVision repeat layers summary:")
            for suffix, count in vision_repeat_counts.items():
                print(f"  {suffix}: {count:,} floats")

            print("\nVision end layers:")
            for tensor_name in vision_end_layers:
                if tensor_name in f.keys():
                    process_tensor(tensor_name, f.get_tensor(tensor_name), out_f)
                else:
                    print(f"Warning: Tensor {tensor_name} not found")

            print("\nProjection layers:")
            for tensor_name in projection_layers:
                if tensor_name in f.keys():
                    process_tensor(tensor_name, f.get_tensor(tensor_name), out_f)
                else:
                    print(f"Warning: Tensor {tensor_name} not found")

            print(f"\nTotal number of floats: {total_float_count:,}")

# Usage
input_file = 'model.safetensors'
output_file = 'moondream.bin'
convert_safetensors_to_custom_binary(input_file, output_file)