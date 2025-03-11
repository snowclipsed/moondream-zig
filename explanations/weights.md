# Building a Custom Weight Loader in Zig

I needed a way to load the Moondream2 weights in my Zig implementation. Rather than trying to parse the safetensors format directly (which would have been pretty complicated), I decided to create my own custom format. This gave me a chance to learn more about binary file formats and freed me from being constrained by the original format's structure. Plus, I could organize the weights in a way that would make indexing much more efficient, and this pays off later on.

Let me walk you through how I approached this and why certain design decisions made a big difference in performance.

## Setting up for Weight Parsing
When designing the binary format, I wanted to keep things relatively simple while still being efficient. Here's what I came up with:

```zig
// Magic constants for format validation
pub const WEIGHTS_MAGIC = 0x4D4F4F4E; // "MOON" in ASCII
pub const WEIGHTS_VERSION = 2; // Incrementing version for tensor format

// Header structure for each tensor
const WeightHeader = struct {
    name_length: u32,
    shape_length: u32,
    data_length: u32,
};
```

The format starts with a magic number ("MOON" in ASCII) and a version number which vikhyatk put in his original python implementation. After the header scan, each tensor follows a simple structure:

1. A header containing the name length, shape dimensions count, and data length
2. The tensor name as a UTF-8 string
3. The tensor shape as an array of dimensions
4. The raw f16 data

When loading, we first verify the magic number and version:

```zig
var magic_buf: [@sizeOf(u32)]u8 = undefined;
const magic_bytes = try file.read(&magic_buf);
if (magic_bytes != @sizeOf(u32)) return error.InvalidRead;
const magic = std.mem.readInt(u32, &magic_buf, .little);
if (magic != WEIGHTS_MAGIC) {
    return error.InvalidMagicNumber;
}
```

This simple structure sets up our weight parsing to be straightforward, but the real magic is in how we organize the weights themselves.

## Weight Organization Strategy

This is where things get really interesting. The original Moondream weights are organized in a way that makes sense for PyTorch, but not for my purposes.

In the original safetensors format, weights are grouped by layer first:

```
model.text.blocks.0.attn.proj.bias
model.text.blocks.0.attn.proj.weight
model.text.blocks.0.attn.qkv.bias
model.text.blocks.0.attn.qkv.weight
...and so on for each layer (0, 1, 2...)
```

This meant that if I wanted to access all the QKV weights across all layers, I'd need to load 24 separate tensors and deal with them individually. That's a lot of unnecessary overhead.

Instead, I completely reorganized the weights to group them by type across all layers:

```
t_ln_w [24, 2048] - all layer normalization weights stacked together
t_Wqkv_w [24, 6144, 2048] - all QKV weights stacked together
t_out_proj_w [24, 2048, 2048] - all projection weights stacked together
```

Why does this matter? There are several significant benefits:

1. **Contiguous memory access**: Our Tensor struct stores data in row-major order, so having similar weights next to each other in memory means better cache utilization. When the CPU fetches one weight, it's likely to fetch nearby weights that we'll need soon too.

2. **Simplified indexing**: Instead of managing 24 separate tensors for the QKV weights, we can just slice into the appropriate part of a single tensor. This is not just cleaner code—it's fundamentally more efficient.

3. **Better vectorization opportunities**: Modern CPUs love working on contiguous chunks of similar data. By grouping weights by type, we enable better SIMD utilization in our ops and GEMM kernels.

4. **Reduced memory fragmentation**: Loading fewer, larger tensors instead of many small ones reduces memory fragmentation and management overhead, this is useful even though zig's allocator gives us a lot of control over it already.

The conversion process in `weights.py` is where this reorganization happens:

```python
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
```

This takes each type of weight from each layer, optionally transposes it (more on that in a moment), and stacks them together into a single larger tensor. The result is a much more cache-friendly and indexing-friendly organization.

In practice, this means that during inference, I can do something like:

```zig
// Get the QKV weights for a specific layer
var layer3_qkv = weights.t_Wqkv_w.getDimensionSlice(0, layer);
```

Instead of having to juggle 24 separate tensors, I just index into the right spot in a single tensor. It's cleaner, faster, and uses memory more efficiently.

## The Matrix Transposition Puzzle

I spent way more time than I'd like to admit figuring out the matrix transposition issues. It was frustrating at first, but ended up leading to one of the biggest performance improvements.

Here's the thing: in the original safetensors file, most weights are already stored in a transposed form to match PyTorch's expectations. PyTorch's GEMM (General Matrix Multiplication) functions expect weight matrices in a certain orientation.

But our tensor library is row-major and has different expectations. We have two different GEMM kernels:

1. `hgemm.zig` - Expects both matrices in their "original", non-transposed form
2. `hgemm_trans.zig` - Expects one of the input matrices (the weight) to be transposed

The existence of two different kernels isn't just for fun—there's a real performance reason. The `hgemm_trans.zig` kernel is significantly faster for a specific pattern: when you're multiplying a single token vector with a weight matrix. This is exactly what happens during token generation, where we process one token at a time.

So I had to make careful decisions about which weights to transpose when loading them in `weights.py`:

```python
# Vision model components - transpose most weights
vision_components = {
    # (path, should_transpose)
    "v_Wqkv_w": ("attn.qkv.weight", True),  # Transpose this one
    "v_Wqkv_b": ("attn.qkv.bias", False),
    "v_out_proj_w": ("attn.proj.weight", True),  # Transpose this one too
    # .. so on
}

# Text model components - keep many weights transposed
text_components = {
    # (path, should_transpose)
    "t_ln_w": ("ln.weight", False),
    "t_ln_b": ("ln.bias", False),
    "t_Wqkv_w": ("attn.qkv.weight", False),  # Note: NOT transposed!
    # ... and other weights
}
```

For the vision model, I generally transposed weights during loading to get them in the "original" orientation. This makes sense because:

1. We always process full images in a single forward pass
2. Image dimensions are fixed (we resize inputs)
3. The standard `hgemm.zig` kernel works best for these larger, non-skinny matrices

But for the text model, I took a different approach. I deliberately kept many weights in their "transposed" form, especially those involved in token generation. Why? Because:

1. During token generation, we're repeatedly multiplying with "skinny" vectors (single token embeddings)
2. The `hgemm_trans.zig` kernel is much faster for this pattern
3. The small penalty during prompt processing is worth the massive gain during token generation

Let me show you what this looks like in the inference code:

```zig
// For vision (using standard HGEMM with non-transposed weights):
var out = try hgemm(
    allocator,
    false, false,  // No transposition needed during computation
    batch_size * seq_len, hidden_dim, dim,
    1.0,
    input, weights.v_fc1_w.sliceLayer(layer),
    0.0, null
);

// For text token generation (using HGEMM_Trans with transposed weights):
var out = try hgemm_trans(
    allocator,
    false, false,  // Weights are ALREADY transposed, so no flag needed
    1, hidden_dim, dim,  // Note the batch size of 1 for a single token
    1.0,
    input, weights.t_fc1_w.sliceLayer(layer),
    0.0, null
);
```

The results were amazing. Token generation speed improved by 5-10x with this approach, with only a tiny hit to prompt processing speed. Since most of the time in actual use is spent generating tokens one by one, this was a huge win overall.

I can't emphasize enough how much time I spent debugging matrix dimension mismatches before I figured this all out. If you're implementing something similar, pay very close attention to how your matrices are oriented!


## How to Read Weights from the Binary

We utilize a very straightforward reading method for our weights using a reading buffer. The heart of the weight loading is in the `init` function of the `Weights` struct:

```zig
pub fn init(config: Config, filename: []const u8, allocator: Allocator) !Self {
    const file = try std.fs.cwd().openFile(filename, .{});
    defer file.close();

    var current_pos: u64 = 0;
    
    // Read and verify magic number and version...
```

First, we open the file and set up position tracking. This position tracking is critical—we use it to verify our reading process is staying aligned with the file structure.

Before reading any tensors, we calculate the expected shapes for all tensors based on the model configuration:

```zig
const expected_shapes = try calculateShapes(allocator, config);
defer {
    inline for (@typeInfo(@TypeOf(expected_shapes)).Struct.fields) |field| {
        allocator.free(@field(expected_shapes, field.name));
    }
}
```

This `calculateShapes` function creates a struct containing all the expected tensor shapes. For example, for the QKV weights in the text model, we expect a shape of `[config.n_layers, config.n_heads * config.head_dim * 3, config.dim]`.

We initialize an empty `Weights` struct with all tensors marked as `undefined`:

```zig
var self = Self{
    .allocator = allocator,
    .word_token_embedding = undefined,
    .t_ln_w = undefined,
    // ... many more fields ...
};
```

Now comes the core reading loop that processes each tensor in the file. We read the name and shape of the tensor stored as an integer in little endian format.

```zig
while (true) {
    var header_buf: [@sizeOf(WeightHeader)]u8 = undefined;
    const header_bytes = try file.read(&header_buf);
    if (header_bytes == 0) break;  // End of file
    if (header_bytes != @sizeOf(WeightHeader)) return error.InvalidHeader;

    const name_length = std.mem.readInt(u32, header_buf[0..4], .little);
    const shape_length = std.mem.readInt(u32, header_buf[4..8], .little);

    // Safety checks on dimensions
    if (name_length > 256) {
        std.debug.print("Invalid name length encountered: {d}\n", .{name_length});
        return error.InvalidNameLength;
    }
    if (shape_length > 4) return error.InvalidShapeLength;

    current_pos += @sizeOf(WeightHeader);
```

First, we read the header that tells us how big the tensor name is, how many dimensions its shape has, and how much data it contains. We also do some basic validation—tensor names shouldn't be enormous, and the number of shape dimensions should be reasonable.

Next, we read the tensor name:

```zig
var name_buf = try allocator.alloc(u8, name_length);
defer allocator.free(name_buf);

const name_bytes = try file.read(name_buf);
if (name_bytes != name_length) return error.InvalidRead;
current_pos += name_length;

const name = name_buf[0..name_length];
```

The name is critical because it tells us which tensor we're dealing with—is this the word embedding matrix, a QKV weight tensor, etc.?

After the name comes the shape information:

```zig
var shape = try allocator.alloc(usize, shape_length);
defer allocator.free(shape);

for (0..shape_length) |i| {
    var dim_bytes: [8]u8 = undefined;
    const bytes_read = try file.read(&dim_bytes);
    if (bytes_read != 8) return error.InvalidRead;

    const dim = std.mem.readInt(u64, &dim_bytes, .little);
    shape[i] = @intCast(dim);
    current_pos += 8;
}
```

Each dimension is stored as a 64-bit integer. For example, a matrix might have shape `[2048, 2048]`.

Now comes an important validation step—we verify the shape we read matches what we expected for this tensor:

```zig
const expected_shape = try getExpectedShape(name, expected_shapes);
if (!std.mem.eql(usize, shape, expected_shape)) {
    printShapeMismatchError(name, shape, expected_shape);
    return error.ShapeMismatch;
}
```

If the shape doesn't match what we expected (e.g., if the model file is corrupted or incompatible), we print a detailed error message and abort. This is crucial for preventing subtle bugs that could arise from shape mismatches.

Once we've validated the shape, we allocate the tensor and read the raw data:

```zig
const tensor = try Tensor(f16).init(allocator, shape);
// Calculate total data size in bytes, accounting for f16 size
const data_size_bytes = tensor.data.len * @sizeOf(f16);

// Read raw bytes into the f16 tensor data buffer
const data_bytes = try file.read(std.mem.sliceAsBytes(tensor.data));
if (data_bytes != data_size_bytes) {
    std.debug.print("Data read mismatch! Expected {d} bytes, got {d}\n", .{ data_size_bytes, data_bytes });
    return error.InvalidRead;
}
current_pos += data_size_bytes;
```

This reads the raw f16 values directly into the tensor's data buffer. We're using half-precision floating point (f16) for all weights to save memory and improve computation speed.

As a final check, we verify our position tracking matches the actual file position:

```zig
const actual_pos = try file.getPos();
if (actual_pos != current_pos) {
    std.debug.print("Position mismatch after tensor {s}!\n", .{name});
    std.debug.print("Expected: {d}, Actual: {d}\n", .{ current_pos, actual_pos });
    return error.InvalidPosition;
}
```

This is extremely useful for catching subtle bugs in the reading process. If our position tracking doesn't match the file's actual position, something has gone wrong, and we need to investigate.

Finally, we store the tensor in the appropriate field of the Weights struct:

```zig
try self.storeTensor(name, tensor);
```

The `storeTensor` function uses the tensor name to determine which field it belongs to:

```zig
fn storeTensor(self: *Self, name: []const u8, tensor: Tensor(f16)) !void {
    if (std.mem.eql(u8, name, "word_token_embedding")) {
        self.word_token_embedding = tensor;
    } else if (std.mem.eql(u8, name, "t_ln_w")) {
        self.t_ln_w = tensor;
    } 
    // ... many more cases ...
    else {
        return error.UnknownTensorName;
    }
}
```

This might seem verbose, but it's actually a very efficient way to handle mapping between tensor names and their storage locations. It also helps catch typos or inconsistencies between the converter and loader.


## Memory Management

For a model of this size, memory efficiency is absolutely critical. Even small inefficiencies can add up quickly. The weight loader carefully manages memory in several ways:

```zig
pub fn deinit(self: *Self) void {
    // Text model tensors
    self.word_token_embedding.deinit();
    self.t_ln_w.deinit();
    self.t_ln_b.deinit();
    // ... more deallocation ...
    
    // Set self to undefined after cleaning up
    self.* = undefined;
}
```

First, we allocate tensors with exactly the shapes we need based on the model configuration. There's no unnecessary padding or wasted space. We also use temporary buffers during parsing that get freed immediately after use.

The Weights struct has a clear ownership model—it owns all the tensors it contains and is responsible for freeing them when deinit() is called. This prevents memory leaks and makes the memory management pattern very clear.

## Using the Weight Loader

Using the loader in your code is pretty straightforward. We call it in the moondream.zig and moonchat.zig files.

```zig
var weights = try Weights.init(config, "moondream.bin", allocator);
defer weights.deinit();
```

The weights file is generated using the Python conversion script (`weights.py`), which converts from the HuggingFace safetensors format to our custom binary format.