import numpy as np
from safetensors import safe_open
import struct

def convert_safetensors_to_custom_binary(input_file, output_file):
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

        vision_start_layers=[
            "vision_encoder.encoder.model.visual.patch_embed.linear.weight",
            "vision_encoder.encoder.model.visual.patch_embed.linear.bias",
            "vision_encoder.encoder.model.visual.pos_embed"
        ]

        vision_repeat_layers = [
            ".attn.proj.bias",
            ".attn.proj.weight",
            ".attn.qkv.bias",
            ".attn.qkv.weight",
            ".mlp.fc1.bias",
            ".mlp.fc1.weight",
            ".mlp.fc2.bias",
            ".mlp.fc2.weight",
            ".norm1.bias",
            ".norm1.weight",
            ".norm2.bias",
            ".norm2.weight"
        ]

        vision_end_layers=[
            "vision_encoder.encoder.model.visual.norm.weight",
            "vision_encoder.encoder.model.visual.norm.bias"
        ]

        projection_layers = [
            "vision_encoder.projection.mlp.fc1.weight",
            "vision_encoder.projection.mlp.fc1.bias",
            "vision_encoder.projection.mlp.fc2.weight",
            "vision_encoder.projection.mlp.fc2.bias"
        ]
        
        text_model_prefix = "text_model.transformer.h." # + number
        vision_model_prefix = "vision_encoder.encoder.model.visual.blocks."
        transformer_layers = 24
        vision_layers = 27

        with open(output_file, 'wb') as out_f:
            

            # Text model start layers
            for tensor_name in text_start_layers:
                tensor = f.get_tensor(tensor_name)
                np_array = tensor.numpy()
                
                # Ensure the data is in float16 format
                if np_array.dtype != np.float16:
                    np_array = np_array.astype(np.float16)
                
                # Write the tensor data directly to the file
                out_f.write(np_array.tobytes())

            # Text model repeat layers for the repeating transformer blocks
            for tensor_suffix in text_repeat_layers:
                for layer in range(transformer_layers): # iterating over number of transformer layers that repeat
                    tensor_name = text_model_prefix + str(layer) + tensor_suffix
                    if tensor_name in f.keys():
                        tensor = f.get_tensor(tensor_name)
                        np_array = tensor.numpy()
                        

                        if np_array.dtype != np.float16:
                            np_array = np_array.astype(np.float16)
                        
                        out_f.write(np_array.tobytes())
                    else:
                        print(f"Warning: Tensor {tensor_name} not found in the safetensors file.")

            # Text model end layers -> end layer norm and linear layers
            for tensor_name in text_end_layers:
                if tensor_name in f.keys():
                    tensor = f.get_tensor(tensor_name)
                    np_array = tensor.numpy()
                    
                    # Ensure the data is in float16 format
                    if np_array.dtype != np.float16:
                        np_array = np_array.astype(np.float16)
                    
                    # Write the tensor data directly to the file
                    out_f.write(np_array.tobytes())
                else:
                    print(f"Warning: Tensor {tensor_name} not found in the safetensors file.")            

            # Vision start layers
            for tensor_name in vision_start_layers:
                if tensor_name in f.keys():
                    tensor = f.get_tensor(tensor_name)
                    np_array = tensor.numpy()
                    
                    # Ensure the data is in float16 format
                    if np_array.dtype != np.float16:
                        np_array = np_array.astype(np.float16)
                    
                    # Write the tensor data directly to the file
                    out_f.write(np_array.tobytes())
                else:
                    print(f"Warning: Tensor {tensor_name} not found in the safetensors file.")

            # Vision repeat layers
            for tensor_suffix in vision_repeat_layers:
                for layer in range(vision_layers):
                    tensor_name = vision_model_prefix + str(layer) + tensor_suffix
                    if tensor_name in f.keys():
                        tensor = f.get_tensor(tensor_name)
                        np_array = tensor.numpy()
                        
                        # Ensure the data is in float16 format
                        if np_array.dtype != np.float16:
                            np_array = np_array.astype(np.float16)
                        
                        # Write the tensor data directly to the file
                        out_f.write(np_array.tobytes())
                    else:
                        print(f"Warning: Tensor {tensor_name} not found in the safetensors file.")


            # Vision end layers
            for tensor_name in vision_end_layers:
                if tensor_name in f.keys():
                    tensor = f.get_tensor(tensor_name)
                    np_array = tensor.numpy()
                    
                    # Ensure the data is in float16 format
                    if np_array.dtype != np.float16:
                        np_array = np_array.astype(np.float16)
                    
                    # Write the tensor data directly to the file
                    out_f.write(np_array.tobytes())
                else:
                    print(f"Warning: Tensor {tensor_name} not found in the safetensors file.")
            
            

# # Usage
input_file = 'model.safetensors'
output_file = 'moondream.bin'
convert_safetensors_to_custom_binary(input_file, output_file)
