import numpy as np
from safetensors import safe_open
import struct
import os
import requests
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import torch

# Magic number and version constants
WEIGHTS_MAGIC = 0x4D4F4F4E  # "MOON" in ASCII
WEIGHTS_VERSION = 3  # Version 3 for quantized tensor format

def download_file(url, filename):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
            desc=filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
    
    print(f"Downloaded {filename}")

def download_from_hf(repo_id, filename, local_path=None):
    """Download file from Hugging Face"""
    if local_path is None:
        local_path = filename
    
    print(f"Downloading {filename} from {repo_id}...")
    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=".", local_dir_use_symlinks=False)
    
    # Some files might be in subdirectories, so check and move if needed
    if not os.path.exists(local_path) and os.path.exists(os.path.join(repo_id.split('/')[-1], filename)):
        os.rename(os.path.join(repo_id.split('/')[-1], filename), local_path)
        
    print(f"Downloaded {filename}")

def ensure_model_files():
    """Ensure model.safetensors and tokenizer.json files exist"""
    # Define the repository ID for Moondream model
    repo_id = "vikhyatk/moondream2"
    
    # Check and download model.safetensors
    if not os.path.exists("model.safetensors"):
        try:
            download_from_hf(repo_id, "model.safetensors")
        except Exception as e:
            print(f"Error downloading from Hugging Face: {e}")
            # Fallback to direct URL if needed
            model_url = "https://huggingface.co/vikhyatk/moondream2/resolve/main/model.safetensors"
            download_file(model_url, "model.safetensors")
    else:
        print("model.safetensors already exists")
    
    # Check and download tokenizer.json
    if not os.path.exists("tokenizer.json"):
        try:
            download_from_hf(repo_id, "tokenizer.json")
        except Exception as e:
            print(f"Error downloading from Hugging Face: {e}")
            # Fallback to direct URL if needed
            tokenizer_url = "https://huggingface.co/vikhyatk/moondream2/resolve/main/tokenizer.json"
            download_file(tokenizer_url, "tokenizer.json")
    else:
        print("tokenizer.json already exists")

@torch.inference_mode()
def optimize_weights_proximal(
    tensor,
    scale,
    zero,
    min_max,
    axis=1,  # Changed to 1 to match our quantization approach
    device="cuda",  # Default to CPU for broader compatibility
    opt_params={"lp_norm": 0.7, "beta": 1e1, "kappa": 1.01, "iters": 200},
    verbose=False,
):
    """Optimize quantization parameters using half quadratic quantization."""
    lp_norm, beta, kappa, iters = (
        opt_params["lp_norm"],
        opt_params["beta"],
        opt_params["kappa"],
        opt_params["iters"],
    )

    # Use CPU by default for broader compatibility
    dtype = torch.float16
    if device in ["cuda", "mps"] and torch.cuda.is_available():
        dtype = torch.float16
        device = "cuda"
    
    W_f = tensor.to(dtype).to(device)
    scale = scale.to(dtype).to(device)
    zero = zero.to(dtype).to(device)

    # Define shrinkage operator based on Lp norm
    if lp_norm == 1:
        shrink_op = lambda x, beta: torch.sign(x) * torch.nn.functional.relu(
            torch.abs(x) - 1.0 / beta
        )
    else:
        shrink_op = lambda x, beta, p=lp_norm: torch.sign(x) * torch.nn.functional.relu(
            torch.abs(x) - (1.0 / beta) * torch.pow(torch.abs(x), p - 1)
        )

    # Iterative optimization
    best_error = 1e4
    for i in range(iters):
        W_q = torch.round(W_f * scale + zero).clamp(min_max[0], min_max[1])
        W_r = (W_q - zero) / scale
        W_e = shrink_op(W_f - W_r, beta)
        zero = torch.mean(W_q - (W_f - W_e) * scale, axis=axis, keepdim=True)
        beta *= kappa

        current_error = float(torch.abs(W_f - W_r).mean())
        if verbose and i % 20 == 0:
            print(f"Iteration {i}, error: {current_error:.6f}")
            
        if current_error < best_error:
            best_error = current_error
        else:
            if verbose:
                print(f"Stopping at iteration {i}, error: {current_error:.6f}")
            break

    # Move results back to the original device
    scale = scale.to("cpu")
    zero = zero.to("cpu")

    return scale, zero

def quantize_tensor(tensor, nbits=8, optimize=True, verbose=False):
    """Quantize a tensor to nbits using half quadratic quantization."""
    # Convert to torch tensor if it's not already
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.from_numpy(tensor)
    
    # Work with float tensors
    W = tensor.float()
    
    # Get min/max values per output channel (axis=1 for weights)
    axis = 1  # Use axis=1 for per-channel quantization
    _min = W.min(dim=axis, keepdim=True)[0]
    _max = W.max(dim=axis, keepdim=True)[0]
    
    max_v = 2**nbits - 1
    min_v = 0
    min_max = [min_v, max_v]
    
    # Calculate scale and zero point
    # Note: Here we calculate inverse of scale for quantization, will invert it later
    scale = (max_v / (_max - _min)).clamp(max=2e4)
    
    # Handle case where range is zero
    min_max_axis = _max - _min
    if (min_max_axis == 0).sum().item() > 0:
        min_max_axis[min_max_axis == 0] = max_v
        scale = (max_v / min_max_axis).clamp(max=2e4)
    
    zero = -_min * scale
    zero = torch.round(zero)
    
    # Apply optimization if requested
    if optimize:
        if verbose:
            print(f"Optimizing quantization parameters...")
        scale, zero = optimize_weights_proximal(
            tensor=W, 
            scale=scale, 
            zero=zero, 
            min_max=min_max, 
            axis=axis,
            verbose=verbose
        )
    
    # Quantize
    W_q = torch.round(W * scale + zero).clamp(min_v, max_v).to(torch.uint8)
    
    # Invert scale for dequantization
    scale = 1.0 / scale
    
    return {
        'weight': W_q.cpu().numpy(),
        'scale': scale.cpu().numpy().astype(np.float16),
        'zero': zero.cpu().numpy().astype(np.uint8)
    }

def debug_f16_bits(name: str, value: np.float16):
    bits = value.view(np.uint16)
    sign = (bits & 0x8000) >> 15
    exp = (bits & 0x7C00) >> 10 
    mantissa = bits & 0x03FF
    print(f"{name}: value={value}, bits=0x{bits:04X}")
    print(f"  sign={sign}, exp={exp}, mantissa=0x{mantissa:03X}")

def write_tensor(f, name, tensor, transpose=False):
    """Write a single tensor with its header and data."""
    # Convert to numpy array
    np_array = tensor.numpy() if hasattr(tensor, 'numpy') else tensor
    
    if transpose:
        original_shape = np_array.shape
        np_array = np_array.T
        print(f"Transposing {name} from {original_shape} to {np_array.shape}")
    
    # Get sizes for header
    name_bytes = name.encode('utf-8')
    data_size = np.prod(np_array.shape)
    
    # Write header
    header = struct.pack('<III', 
        len(name_bytes),      
        len(np_array.shape),  
        data_size            
    )
    f.write(header)
    f.write(name_bytes)
    
    # Write shape
    shape_bytes = struct.pack(f'<{len(np_array.shape)}Q', *np_array.shape)
    f.write(shape_bytes)
    
    # Write data
    data_bytes = np_array.tobytes()
    f.write(data_bytes)
    
    current_pos = f.tell()
    print(f"Wrote tensor {name}")
    print(f"  Shape: {np_array.shape}")
    print(f"  dtype: {np_array.dtype}")
    print(f"  Header size: {len(header)}")
    print(f"  Name size: {len(name_bytes)}")
    print(f"  Shape info size: {len(shape_bytes)}")
    print(f"  Data size: {len(data_bytes)}")
    print(f"  Current file position: {current_pos}\n")

def write_quantized_tensor(f, name, quantized_data):
    """Write a quantized tensor with its header, weights, scale, and zero point."""
    # Unpack the quantized data
    weight = quantized_data['weight']
    scale = quantized_data['scale']
    zero = quantized_data['zero']
    
    # Write the quantized weight tensor
    weight_name = f"{name}_q"
    weight_bytes = weight.tobytes()
    
    # Write weight header
    name_bytes = weight_name.encode('utf-8')
    header = struct.pack('<III', 
        len(name_bytes),      
        len(weight.shape),  
        np.prod(weight.shape)            
    )
    f.write(header)
    f.write(name_bytes)
    
    # Write weight shape
    shape_bytes = struct.pack(f'<{len(weight.shape)}Q', *weight.shape)
    f.write(shape_bytes)
    
    # Write weight data
    f.write(weight_bytes)
    
    # Write scale tensor
    scale_name = f"{name}_s"
    write_tensor(f, scale_name, scale.astype(np.float16))
    
    # Write zero point tensor
    zero_name = f"{name}_z"
    write_tensor(f, zero_name, zero.astype(np.uint8))
    
    print(f"Wrote quantized tensor {name}")
    print(f"  Weight shape: {weight.shape}, dtype: {weight.dtype}")
    print(f"  Scale shape: {scale.shape}, dtype: {scale.dtype}")
    print(f"  Zero shape: {zero.shape}, dtype: {zero.dtype}\n")

def collect_and_stack_layer_tensors(f, n_layers, prefix_template, components, quantize=False, verbose=False):
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
    
    # Quantize if requested
    if quantize:
        quantized = {}
        for name, tensor in stacked.items():
            if name.endswith('_w') and not name.startswith('t_ln_') and not name.startswith('v_norm'):
                print(f"Quantizing {name}")
                quantized[name] = quantize_tensor(tensor, nbits=8, optimize=True, verbose=verbose)
            else:
                quantized[name] = tensor
        return quantized
    
    return stacked

def convert_safetensors_to_binary(input_file, output_file, quantize_weights=True, verbose=False):
    with safe_open(input_file, framework="pt", device="cpu") as f:
        with open(output_file, 'wb') as out_f:
            # Write magic number and version
            out_f.write(struct.pack('<II', WEIGHTS_MAGIC, WEIGHTS_VERSION))
            
            # Write word embedding (no transpose, no quantization)
            write_tensor(out_f, "word_token_embedding", 
                        f.get_tensor("model.text.wte"))

            # Collect and write text transformer layer tensors
            text_components = {
                # (path, should_transpose)
                "t_ln_w": ("ln.weight", False),
                "t_ln_b": ("ln.bias", False),
                "t_Wqkv_w": ("attn.qkv.weight", False),
                "t_Wqkv_b": ("attn.qkv.bias", False),
                "t_out_proj_w": ("attn.proj.weight", False),
                "t_out_proj_bias": ("attn.proj.bias", False),
                "t_fc1_w": ("mlp.fc1.weight", False),
                "t_fc1_b": ("mlp.fc1.bias", False),
                "t_fc2_w": ("mlp.fc2.weight", False),
                "t_fc2_b": ("mlp.fc2.bias", False),
            }
            
            print("Processing text transformer layers...")
            text_stacked = collect_and_stack_layer_tensors(
                f, 24, "model.text.blocks.{}", text_components, quantize=quantize_weights, verbose=verbose
            )
            
            # Write stacked text tensors
            # Write stacked text tensors
            for name, tensor in text_stacked.items():
                if isinstance(tensor, dict) and 'weight' in tensor:  # This is a quantized tensor
                    write_quantized_tensor(out_f, name, tensor)
                else:
                    write_tensor(out_f, name, tensor)

            # Write text model end layers
            if quantize_weights:
                print("Quantizing LM head...")
                lm_head = quantize_tensor(
                    f.get_tensor("model.text.lm_head.weight").numpy(), 
                    nbits=8, 
                    optimize=True,
                    verbose=verbose
                )
                write_quantized_tensor(out_f, "t_linear_w", lm_head)
            else:
                write_tensor(out_f, "t_linear_w", 
                            f.get_tensor("model.text.lm_head.weight"), transpose=False)
                
            write_tensor(out_f, "t_linear_b", 
                        f.get_tensor("model.text.lm_head.bias"))
            write_tensor(out_f, "t_ln_out_w", 
                        f.get_tensor("model.text.post_ln.weight"))
            write_tensor(out_f, "t_ln_out_b", 
                        f.get_tensor("model.text.post_ln.bias"))

            # Vision model start layers
            if quantize_weights:
                print("Quantizing patch embedding...")
                patch_emb = quantize_tensor(
                    f.get_tensor("model.vision.patch_emb.weight").T.numpy(), 
                    nbits=8, 
                    optimize=True,
                    verbose=verbose
                )
                write_quantized_tensor(out_f, "v_patch_embedding_linear_w", patch_emb)
            else:
                write_tensor(out_f, "v_patch_embedding_linear_w",
                            f.get_tensor("model.vision.patch_emb.weight"), transpose=True)
                
            write_tensor(out_f, "v_patch_embedding_linear_b",
                        f.get_tensor("model.vision.patch_emb.bias"))
            write_tensor(out_f, "v_pos_embedding",
                        f.get_tensor("model.vision.pos_emb"))

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
                "v_norm1_w": ("ln1.weight", False),
                "v_norm1_b": ("ln1.bias", False),
                "v_norm2_w": ("ln2.weight", False),
                "v_norm2_b": ("ln2.bias", False),
            }
            
            print("Processing vision transformer layers...")
            vision_stacked = collect_and_stack_layer_tensors(
                f, 27, "model.vision.blocks.{}", vision_components, quantize=quantize_weights, verbose=verbose
            )
            
            # Write stacked vision tensors
            for name, tensor in vision_stacked.items():
                if isinstance(tensor, dict) and 'weight' in tensor:  # This is a quantized tensor
                    write_quantized_tensor(out_f, name, tensor)
                else:
                    write_tensor(out_f, name, tensor)

            # Vision end layers - not quantizing norm layers
            write_tensor(out_f, "v_norm_out_w",
                        f.get_tensor("model.vision.post_ln.weight"))
            write_tensor(out_f, "v_norm_out_b",
                        f.get_tensor("model.vision.post_ln.bias"))

            # Projection layers
            if quantize_weights:
                print("Quantizing vision projection layers...")
                proj_fc1 = quantize_tensor(
                    f.get_tensor("model.vision.proj_mlp.fc1.weight").T.numpy(), 
                    nbits=8, 
                    optimize=True,
                    verbose=verbose
                )
                write_quantized_tensor(out_f, "v_proj_fc1_w", proj_fc1)
                
                proj_fc2 = quantize_tensor(
                    f.get_tensor("model.vision.proj_mlp.fc2.weight").T.numpy(), 
                    nbits=8, 
                    optimize=True,
                    verbose=verbose
                )
                write_quantized_tensor(out_f, "v_proj_fc2_w", proj_fc2)
            else:
                write_tensor(out_f, "v_proj_fc1_w",
                            f.get_tensor("model.vision.proj_mlp.fc1.weight"), transpose=True)
                write_tensor(out_f, "v_proj_fc2_w",
                            f.get_tensor("model.vision.proj_mlp.fc2.weight"), transpose=True)
                
            write_tensor(out_f, "v_proj_fc1_b",
                        f.get_tensor("model.vision.proj_mlp.fc1.bias"))
            write_tensor(out_f, "v_proj_fc2_b",
                        f.get_tensor("model.vision.proj_mlp.fc2.bias"))

def delete_file(file_path):
    """Delete a file if it exists and return success status"""
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            return True
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
            return False
    else:
        print(f"File {file_path} does not exist")
        return False

if __name__ == "__main__":
    # First ensure we have the model files
    print("Checking for required model files...")
    ensure_model_files()
    
    # Now run the conversion with quantization
    input_file = 'model.safetensors'
    output_file = 'moondream_int8.bin'
    print(f"\nConverting {input_file} to {output_file} with INT8 quantization...")
    
    # Ask about verbose output
    verbose = input("Enable verbose output for optimization? (y/n): ").lower() == 'y'
    
    # Run the conversion
    convert_safetensors_to_binary(input_file, output_file, quantize_weights=True, verbose=verbose)
    
    print(f"\nConversion complete! Output saved to {output_file}")
    
    # Optionally delete the original safetensors file to save space
    print("\nDo you want to delete the original safetensors file to save disk space? (y/n)")
    if input().lower() == 'y':
        if delete_file(input_file):
            print(f"Successfully deleted {input_file}")
    
    print("\nProcess complete! The quantized model is available as moondream_int8.bin and tokenizer.json is preserved.")