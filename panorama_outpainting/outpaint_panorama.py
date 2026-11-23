#!/usr/bin/env python3
"""
Outpaint a partial 360° equirectangular (ERP) panorama with FLUX.2 [klein] 9B
and the 360-ERP outpainting LoRA.

Pipeline
--------
1. Build a "green canvas": the input panorama resized to WIDTH x HEIGHT,
   with every pixel that is WHITE in the mask replaced by pure green (0,255,0).
2. Pass 1 - outpaint the green regions with FLUX.2 klein + 360-ERP LoRA.
3. Restore the original (known) pixels in the BLACK mask region.
4. Pass 2 (seam fix) - roll the panorama by half its width so the left/right
   wrap boundary lands in the centre, inpaint a narrow vertical strip there,
   and roll back. This removes the visible seam at 0°/360°.

Mask convention
---------------
    white (255) = region to generate
    black (0)   = region to keep from the input image

Example
-------
    python outpaint_panorama.py \
        --image panoramas/garden.png \
        --mask  panoramas/garden_mask.png \
        --output results/garden_filled.png
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

DEFAULT_MODEL = "black-forest-labs/FLUX.2-klein-base-9B"
DEFAULT_LORA = "nomadoor/flux-2-klein-9B-360-erp-outpaint-lora"
DEFAULT_PROMPT = (
    "Fill the green spaces according to the image. "
    "Outpaint as a seamless 360 equirectangular panorama (2:1). "
    "Match left and right edges. Keep the horizon level. "
    "Seamless left-right wrap boundary."
)
GREEN = (0, 255, 0)


# --------------------------------------------------------------------------- #
# Image / mask helpers
# --------------------------------------------------------------------------- #
def load_image_and_mask(image_path, mask_path, size):
    """Load the panorama and its mask, resized to `size` (w, h)."""
    img = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if img.size != size:
        print(f"Resizing image {img.size} -> {size}")
        img = img.resize(size, Image.LANCZOS)
    if mask.size != size:
        print(f"Resizing mask {mask.size} -> {size}")
        mask = mask.resize(size, Image.NEAREST)  # nearest keeps the mask binary
    return img, mask


def make_green_canvas(img, mask):
    """White mask -> green, black mask -> keep the image."""
    green = Image.new("RGB", img.size, GREEN)
    return Image.composite(green, img.convert("RGB"), mask.convert("L"))


def offset_mask_boundary(mask, offset_px):
    """
    Shift the mask content down (positive) or up (negative) by `offset_px`.
    Assumes a roughly horizontal white(top)/black(bottom) layout. Rows vacated
    at the top are filled white; rows vacated at the bottom are filled black.
    """
    if offset_px == 0:
        return mask
    arr = np.roll(np.array(mask), offset_px, axis=0)
    if offset_px > 0:
        arr[:offset_px, :] = 255
    else:
        arr[offset_px:, :] = 0
    return Image.fromarray(arr)


def paste_original(original, mask, result, feather_px=0, mask_offset_px=0):
    """
    Put the known pixels back: black mask -> original, white mask -> generated.
    `feather_px` > 0 gives a soft transition instead of a hard cut.
    """
    if mask_offset_px:
        mask = offset_mask_boundary(mask, mask_offset_px)
    if feather_px > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_px))
    out = Image.composite(result.convert("RGB"), original.convert("RGB"), mask)
    restored = 1.0 - (np.asarray(mask) / 255.0).mean()
    print(f"Restored original pixels in ~{restored:.0%} of the panorama")
    return out


def make_center_strip_mask(size, strip_w, feather_px):
    """Vertical strip in the image centre: white = inpaint, black = keep."""
    w, h = size
    mask = Image.new("L", (w, h), 0)
    x0 = max(0, w // 2 - strip_w // 2)
    x1 = min(w, w // 2 + strip_w // 2)
    mask.paste(255, (x0, 0, x1, h))
    if feather_px > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_px))
    return mask


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def load_pipeline(model_id, lora_id, device, cpu_offload):
    from diffusers import Flux2KleinPipeline

    print(f"Loading {model_id} ...")
    pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    print(f"Loading LoRA {lora_id} ...")
    pipe.load_lora_weights(lora_id)

    if cpu_offload:
        # Lower VRAM use (~24 GB GPUs), slower; needs more system RAM.
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    # Decode large panoramas in tiles to save memory.
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    if hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    return pipe


def run_pipe(pipe, canvas, prompt, steps, guidance, seed, device):
    w, h = canvas.size
    return pipe(
        prompt=prompt,
        image=[canvas],
        height=h,
        width=w,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=torch.Generator(device).manual_seed(seed),
    ).images[0]


def fix_wrap_seam(pipe, img, prompt, strip_w, feather_px, steps, guidance,
                  seed, device, debug_dir=None):
    """Re-inpaint the 0°/360° wrap boundary so left and right edges match."""
    w, h = img.size
    # Move the left|right join to the centre of the image.
    rolled = Image.fromarray(np.roll(np.array(img), w // 2, axis=1))
    strip_mask = make_center_strip_mask((w, h), strip_w, feather_px)
    canvas = make_green_canvas(rolled, strip_mask)

    if debug_dir:
        strip_mask.save(debug_dir / "seam_strip_mask.png")
        canvas.save(debug_dir / "seam_canvas.png")

    fixed = run_pipe(pipe, canvas, prompt, steps, guidance, seed, device)
    if fixed.size != (w, h):
        fixed = fixed.resize((w, h), Image.LANCZOS)
    # Keep everything outside the strip exactly as it was.
    fixed = Image.composite(fixed.convert("RGB"), rolled, strip_mask)
    return Image.fromarray(np.roll(np.array(fixed), -(w // 2), axis=1))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="Outpaint a partial 360° equirectangular panorama with FLUX.2 klein.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    io = p.add_argument_group("input / output")
    io.add_argument("--image", required=True, help="Input (partial) ERP panorama.")
    io.add_argument("--mask", required=True,
                    help="Mask: white = generate, black = keep.")
    io.add_argument("--output", required=True, help="Output PNG path.")
    io.add_argument("--debug-dir", default=None,
                    help="If set, save intermediate canvases/masks here.")

    gen = p.add_argument_group("generation")
    gen.add_argument("--prompt", default=DEFAULT_PROMPT)
    gen.add_argument("--width", type=int, default=2048)
    gen.add_argument("--height", type=int, default=1024)
    gen.add_argument("--steps", type=int, default=28)
    gen.add_argument("--guidance", type=float, default=3.0)
    gen.add_argument("--seed", type=int, default=5)
    gen.add_argument("--feather", type=int, default=2,
                     help="Blur radius (px) when pasting the original pixels back.")
    gen.add_argument("--mask-offset", type=int, default=0,
                     help="Shift the white/black mask boundary down (+) or up (-) "
                          "before pasting the original back.")

    seam = p.add_argument_group("seam fix (second pass)")
    seam.add_argument("--no-seam-fix", action="store_true",
                      help="Skip the left/right wrap-seam pass.")
    seam.add_argument("--seam-strip-width", type=int, default=256,
                      help="Width of the re-inpainted strip; try 192-384.")
    seam.add_argument("--seam-feather", type=int, default=12)
    seam.add_argument("--seam-steps", type=int, default=28)
    seam.add_argument("--seam-guidance", type=float, default=2.5)
    seam.add_argument("--seam-seed", type=int, default=7)

    model = p.add_argument_group("model")
    model.add_argument("--model", default=DEFAULT_MODEL)
    model.add_argument("--lora", default=DEFAULT_LORA)
    model.add_argument("--device", default="cuda")
    model.add_argument("--cpu-offload", action="store_true",
                       help="Offload idle sub-models to CPU (for ~24 GB GPUs).")
    return p.parse_args()


def main():
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available - this script needs an NVIDIA GPU.")
    if args.width != 2 * args.height:
        print(f"[warn] {args.width}x{args.height} is not 2:1; ERP panoramas should be 2:1.")

    debug_dir = Path(args.debug_dir) if args.debug_dir else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    size = (args.width, args.height)
    original, mask = load_image_and_mask(args.image, args.mask, size)
    canvas = make_green_canvas(original, mask)
    if debug_dir:
        canvas.save(debug_dir / "canvas.png")

    pipe = load_pipeline(args.model, args.lora, args.device, args.cpu_offload)

    print("Pass 1: outpainting ...")
    result = run_pipe(pipe, canvas, args.prompt, args.steps, args.guidance,
                      args.seed, args.device)
    if result.size != size:
        result = result.resize(size, Image.LANCZOS)
    result = paste_original(original, mask, result,
                            feather_px=args.feather, mask_offset_px=args.mask_offset)
    if debug_dir:
        result.save(debug_dir / "pass1.png")

    if not args.no_seam_fix:
        print("Pass 2: fixing the left/right wrap seam ...")
        result = fix_wrap_seam(
            pipe, result, args.prompt,
            strip_w=args.seam_strip_width, feather_px=args.seam_feather,
            steps=args.seam_steps, guidance=args.seam_guidance,
            seed=args.seam_seed, device=args.device, debug_dir=debug_dir,
        )

    result.save(out_path, format="PNG")
    print(f"Saved {out_path} ({result.size[0]}x{result.size[1]})")


if __name__ == "__main__":
    main()
