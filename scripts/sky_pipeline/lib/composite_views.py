"""Composite view rendering and export-frame helpers."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as tf
from tqdm import tqdm

from lib.alignment import unalign_c2w
from lib.gaussians import make_render_camera
from lib.panorama_sphere import composite_panorama_sphere, panorama_sphere_background
from lib.transforms_json import split_records_train_test_with_prefix
from renderer.gaussian_renderer import render
from scene import GaussianModel


def get_scaled_cam_data(cam_data: dict, max_dim: int = 1600) -> dict:
    """Scale camera dimensions and intrinsics if they exceed max_dim."""
    orig_w, orig_h = cam_data["width"], cam_data["height"]
    max_side = max(orig_w, orig_h)
    if max_side <= max_dim:
        return cam_data

    scale = max_dim / float(max_side)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    scale_x = new_w / float(orig_w)
    scale_y = new_h / float(orig_h)

    scaled = cam_data.copy()
    scaled["width"] = new_w
    scaled["height"] = new_h
    scaled["fx"] = cam_data["fx"] * scale_x
    scaled["fy"] = cam_data["fy"] * scale_y
    if "cx" in cam_data:
        scaled["cx"] = cam_data["cx"] * scale_x
    if "cy" in cam_data:
        scaled["cy"] = cam_data["cy"] * scale_y
    return scaled


def records_to_export_frame(
    records: list,
    R_align: np.ndarray,
    export_frame: str,
) -> list:
    """Return records whose c2w matches ``export_frame`` (aligned|original)."""
    if export_frame == "aligned" or np.allclose(R_align, np.eye(3), atol=1e-8):
        return records
    if export_frame != "original":
        raise ValueError(
            f"export_frame must be 'aligned' or 'original', got {export_frame!r}"
        )
    out = []
    for r in records:
        rr = dict(r)
        rr["c2w"] = unalign_c2w(r["c2w"], R_align)
        out.append(rr)
    print(f"  Unaligned {len(out)} transform c2w → original frame for export.")
    return out


def render_composite_views(
    c2w_matrices: list,
    loaded_cams: list,
    gaussians: GaussianModel,
    pipeline,
    panorama_tiled: torch.Tensor,
    pano_eye: np.ndarray,
    sphere_radius: float,
    panorama_yaw_deg: float,
    out_root: Path,
    test_ratio: float = 0.1,
    split_seed: int = 42,
) -> list:
    """
    Render one PNG per camera into ``train/`` or ``test/``.

    Per-camera ``render_mode`` (optional on loaded_cams entries):
      • ``composite`` (default) — Gaussians over panorama sphere
      • ``gs_only`` — Gaussians only (black background)
      • ``bg_only`` — panorama sphere only
    """
    train_dir = out_root / "train"
    test_dir = out_root / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    split_records = split_records_train_test_with_prefix(
        c2w_matrices, test_ratio=test_ratio, seed=split_seed, prefix="r",
    )
    n_train = sum(1 for r in split_records if r["split"] == "train")
    n_test = sum(1 for r in split_records if r["split"] == "test")
    print(
        f"  Dataset split: train={n_train}  test={n_test}  "
        f"(test_ratio={test_ratio}, seed={split_seed})"
    )

    R_identity = np.eye(3, dtype=np.float64)
    bg_black = torch.zeros(3, device="cuda")
    records = []

    with torch.no_grad():
        for i, c2w in enumerate(tqdm(
            c2w_matrices, desc="Rendering", total=len(c2w_matrices),
        )):
            raw_cam_data = loaded_cams[i]
            cam_data = get_scaled_cam_data(raw_cam_data, max_dim=1600)
            cam_fov_x = 2.0 * math.atan(cam_data["width"] / (2.0 * cam_data["fx"]))
            cam_w, cam_h = cam_data["width"], cam_data["height"]
            mode = cam_data.get("render_mode", "composite")
            rec = split_records[i]

            if mode == "bg_only":
                rendered = panorama_sphere_background(
                    panorama_tiled, c2w, pano_eye, sphere_radius,
                    R_identity, cam_fov_x, cam_w, cam_h,
                    yaw_deg=panorama_yaw_deg,
                ).clamp(0.0, 1.0)
            elif mode == "gs_only":
                cam = make_render_camera(cam_data, c2w)
                try:
                    rendered = render(
                        cam, gaussians, pipeline, bg_black,
                    )["render"].clamp(0.0, 1.0)
                finally:
                    del cam
            else:
                cam = make_render_camera(cam_data, c2w)
                try:
                    rendered = composite_panorama_sphere(
                        cam, c2w, gaussians, pipeline, panorama_tiled,
                        pano_origin=pano_eye,
                        sphere_radius=sphere_radius,
                        R_align=R_identity,
                        fov_x_rad=cam_fov_x,
                        img_w=cam_w,
                        img_h=cam_h,
                        yaw_deg=panorama_yaw_deg,
                    )
                finally:
                    del cam

            # Match train_sky temporary convention: dataset PNGs are vertically
            # flipped; training flips the rasterizer output the same way before loss.
            rendered = torch.flip(rendered, dims=[1])

            out_dir = train_dir if rec["split"] == "train" else test_dir
            out_name = f"{rec['prefix']}_{rec['idx']}.png"
            tf.to_pil_image(rendered.cpu()).save(out_dir / out_name)
            del rendered
            records.append({
                "split": rec["split"],
                "idx": rec["idx"],
                "c2w": c2w,
                "prefix": rec["prefix"],
                "render_mode": mode,
                "pitch_deg": cam_data.get("pitch_deg"),
                "ring_tag": cam_data.get("ring_tag"),
                "look_back": cam_data.get("look_back", False),
            })

            if (i + 1) % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

    return records
