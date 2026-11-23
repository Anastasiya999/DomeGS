# Panorama outpainting

This folder completes a partial 360° equirectangular (ERP) panorama, so that the full surrounding environment can be used in the 3DGS pipeline. It uses [FLUX.2 [klein] base 9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) with the [360-ERP outpainting LoRA](https://huggingface.co/nomadoor/flux-2-klein-9B-360-erp-outpaint-lora) through 🤗 `diffusers`.

```
panorama_outpainting/
├── outpaint_panorama.py   # main script (CLI)
├── requirements.txt
└── README.md
```

## How it works

1. **Canvas.** The input panorama is resized to 2048×1024. Every pixel that is white in the mask is painted pure green `(0, 255, 0)`, which is the marker the LoRA was trained to fill.
2. **Pass 1: outpainting.** FLUX.2 klein + LoRA generates the green regions.
3. **Restore known pixels.** The original pixels are pasted back wherever the mask is black, with an optional small feather.
4. **Pass 2: wrap-seam fix.** The panorama is rolled by half its width, so the 0°/360° boundary moves to the centre. A 256 px vertical strip there is inpainted again, then the image is rolled back. This removes the visible left/right seam. Skip it with `--no-seam-fix`.

## Hardware requirements

| | Minimum | Recommended |
|---|---|---|
| GPU | NVIDIA, **≥ 24 GB VRAM** with `--cpu-offload` (slower) | NVIDIA, **≥ 48 GB VRAM** (e.g. RTX A6000, L40S, A100, H100, GH200) |
| System RAM | 64 GB with `--cpu-offload` | 32–40 GB |
| Disk | ~40 GB free for the Hugging Face cache (model + LoRA weights) | same, on an SSD |
| CUDA | 12.x driver; GPU with bfloat16 support (Ampere or newer) | CUDA 12.6 |
| OS | Linux (x86_64 or aarch64) | |

Without offloading, the full pipeline is held on the GPU in bfloat16: a ~9B transformer, a Qwen3-based text encoder and the VAE, which together take roughly 35 GB. Running at 2048×1024 adds activation memory on top of that. CPU-only and Apple Silicon (MPS) are not supported.

## Software requirements

- Python ≥ 3.10 (tested with 3.13)
- PyTorch with CUDA
- The packages in [`requirements.txt`](requirements.txt): `diffusers` (a version that provides `Flux2KleinPipeline`), `transformers`, `accelerate`, `huggingface_hub`, `peft`, `safetensors`, `pillow` and `numpy`.

## Installation

```bash
cd panorama_outpainting
python -m venv venv
source venv/bin/activate
pip install --upgrade pip

# 1) PyTorch matching your CUDA version (here CUDA 12.6; see https://pytorch.org/get-started/locally/)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 2) Remaining dependencies
pip install -r requirements.txt
```

Check the install:

```bash
python -c "import torch, diffusers; from diffusers import Flux2KleinPipeline; print(torch.cuda.is_available(), diffusers.__version__)"
```

### Hugging Face access

The weights are downloaded from the Hugging Face Hub on the first run and cached in `~/.cache/huggingface`. Set `HF_HOME` to use another location. If the model page asks you to accept a license, accept it with your account, then log in:

```bash
huggingface-cli login        # or: export HF_TOKEN=<your token>
```

## Usage

### Inputs

| Argument | Description |
|---|---|
| `--image` | Partial ERP panorama (any size; resized to `--width`×`--height`, default 2048×1024, 2:1). |
| `--mask` | Grayscale mask with the same aspect ratio. **White = generate, black = keep.** |
| `--output` | Output PNG path. |

### Basic run

```bash
python outpaint_panorama.py \
    --image  panoramas/garden.png \
    --mask   panoramas/garden_mask.png \
    --output results/garden_filled.png
```

On a GPU with ~24 GB of memory, add `--cpu-offload`:

```bash
python outpaint_panorama.py --image in.png --mask mask.png --output out.png --cpu-offload
```

To use a custom prompt and save the intermediate images:

```bash
python outpaint_panorama.py --image in.png --mask mask.png --output out.png \
    --prompt "Fill the green spaces according to the image. Outpaint as a seamless 360 equirectangular panorama (2:1). Sunny, slightly cloudy sky." \
    --debug-dir debug/
```

### Options

| Option | Default | Meaning |
|---|---|---|
| `--prompt` | ERP outpainting prompt | Text prompt; you can append scene hints |
| `--width` / `--height` | 2048 / 1024 | Working resolution (keep 2:1) |
| `--steps` / `--guidance` / `--seed` | 28 / 3.0 / 5 | Pass-1 sampling |
| `--feather` | 2 | Blur (px) when pasting known pixels back |
| `--mask-offset` | 0 | Shift the mask boundary down (+) / up (−) before pasting back |
| `--no-seam-fix` | off | Skip pass 2 |
| `--seam-strip-width` | 256 | Width of the re-inpainted seam strip (try 192–384) |
| `--seam-feather` | 12 | Blur (px) of the seam strip mask |
| `--seam-steps` / `--seam-guidance` / `--seam-seed` | 28 / 2.5 / 7 | Pass-2 sampling |
| `--cpu-offload` | off | Model CPU offload for ~24 GB GPUs |
| `--debug-dir DIR` | – | Save the canvas, pass-1 result and seam masks |
| `--model` / `--lora` | see above | Hugging Face model / LoRA ids |
| `--device` | `cuda` | Torch device |

Run `python outpaint_panorama.py --help` for the full list.

## Licenses

FLUX.2 [klein] 9B and the LoRA are released under their own licenses (see the model pages). Check them before any commercial use.
