import numpy as np
from PIL import Image
from torchvision.transforms.v2.functional import InterpolationMode, resize as tv_resize

def get_pytorch_output(image_path: str, size: tuple) -> np.ndarray:
    img = Image.open(image_path)
    resized = tv_resize(img, size, InterpolationMode.BICUBIC)
    arr = np.array(resized)
    print("PyTorch output shape:", arr.shape)
    print("PyTorch first few values:", arr.flatten()[:10])
    return arr

def load_and_compare(pytorch_arr: np.ndarray, zig_path: str):
    zig_arr = np.fromfile(zig_path, dtype=np.uint8)
    print("Zig output shape:", zig_arr.shape)
    print("Zig first few values:", zig_arr[:10])

    diff = np.abs(pytorch_arr.flatten() - zig_arr)
    print(f"Max difference: {np.max(diff)}")
    print(f"Mean difference: {np.mean(diff)}")
    print(f"Median difference: {np.median(diff)}")

test_image = "/home/snow/projects/moondream-zig/core/images/demo-1.jpg"
size = (378, 378)
pytorch_result = get_pytorch_output(test_image, size)
load_and_compare(pytorch_result, "/home/snow/projects/moondream-zig/core/zig_output.bin")