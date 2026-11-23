"""Panorama image + sidecar I/O."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as tf
from PIL import Image


def load_panorama_cuda(path: str) -> torch.Tensor:
    """Load equirectangular panorama and tile 3× horizontally on GPU."""
    img = Image.open(path).convert("RGB")
    t = tf.to_tensor(img).cuda()  # (3, H, W)
    return torch.cat([t, t, t], dim=2)  # (3, H, 3W)


def flip_panorama_horizontal(panorama_tiled: torch.Tensor) -> torch.Tensor:
    """
    Mirror an equirect left↔right.

    ``load_panorama_cuda`` returns a 3× horizontally tiled texture (C, H, 3W).
    Flip the single-period center strip, then re-tile so seams stay valid.
    """
    if panorama_tiled.dim() != 3:
        raise ValueError(
            f"Expected (C,H,W) panorama, got {tuple(panorama_tiled.shape)}"
        )
    _c, _h, w3 = panorama_tiled.shape
    if w3 % 3 != 0:
        return torch.flip(panorama_tiled, dims=[-1])
    w = w3 // 3
    center = panorama_tiled[:, :, w:2 * w]
    flipped = torch.flip(center, dims=[-1])
    return torch.cat([flipped, flipped, flipped], dim=2)


def panorama_sidecar_path(pano_path: str | Path) -> Path:
    p = Path(pano_path)
    return Path(p.with_suffix("").as_posix() + "_eye.json")


def load_panorama_sidecar(pano_path: str | Path) -> dict:
    """
    Load render_panorama.py sidecar next to the panorama image.

    Returns dict with keys: eye, target, R_align, aligned, meta.

    ``eye`` and ``target`` are in the *aligned* scene frame when ``aligned``
    is true. When alignment was disabled they remain in the original frame
    (and are rotated into the aligned frame here if R_align ≠ I).
    """
    sc = panorama_sidecar_path(pano_path)
    if not sc.exists():
        raise FileNotFoundError(
            f"Panorama sidecar not found: {sc}\n"
            f"Run render_panorama.py first — it writes <panorama_stem>_eye.json."
        )
    with open(sc) as f:
        meta = json.load(f)

    R_align = np.asarray(meta.get("R_align", np.eye(3).tolist()), dtype=np.float64)
    eye = np.asarray(meta["eye"], dtype=np.float64)
    target = np.asarray(meta.get("target", eye), dtype=np.float64)

    if not meta.get("aligned", False) and not np.allclose(R_align, np.eye(3), atol=1e-8):
        eye = R_align @ eye
        target = R_align @ target

    print(f"  [sidecar] {sc.name}")
    print(f"    eye    = {eye.tolist()}")
    print(f"    target = {target.tolist()}")
    print(f"    aligned flag = {meta.get('aligned', False)}")
    ang = math.degrees(
        math.acos(np.clip((np.trace(R_align) - 1.0) / 2.0, -1.0, 1.0))
    )
    print(f"    R_align rotation ≈ {ang:.2f}°")

    return {
        "eye": eye,
        "target": target,
        "R_align": R_align,
        "aligned": bool(meta.get("aligned", False)),
        "meta": meta,
    }
