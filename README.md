# mooondream-zig

Moondream inference in zig!

## Setup Instructions

### Install zig

On Windows:

```bash
choco install zig
```

On Linux:

```bash
sudo apt install zig
```

On MacOS:

```bash
brew install zig
```

### Initialize python environment & install dependencies

From root directory

On Linux and MacOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Download model weights & tokenizer

```bash
# On Windows
curl.exe -L https://huggingface.co/vikhyatk/moondream2/blob/main/tokenizer.json -o src/tokenizer.json && curl.exe -L https://huggingface.co/vikhyatk/moondream2/blob/main/model.safetensors -o model.safetensors
# On Linux and MacOS
curl -L https://huggingface.co/vikhyatk/moondream2/blob/main/tokenizer.json -o src/tokenizer.json && curl -L https://huggingface.co/vikhyatk/moondream2/blob/main/model.safetensors -o model.safetensors
```

### Convert model weights

```bash
python weights.py
```

### Build Zig with optimizations

```bash
cd src/
zig build run -Doptimize=ReleaseFast
```
