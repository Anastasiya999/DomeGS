
"""Remove probe-visible Gaussians and save a training PLY."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from lib.alignment import apply_alignment_to_gaussians
from lib.gaussians import make_render_camera, _subset_gaussians
from renderer.gaussian_renderer import render
from scene import GaussianModel


# def save_training_ply_removing_visible(
#     gaussians: GaussianModel,
#     probe_cams: list,
#     pipeline,
#     out_path: Path,
#     R_align: np.ndarray,
#     export_frame: str = "original",
#     label: str = "remove probes",
#     world_up: np.ndarray | None = None,
#     remove_above_height: float | None = None,
#     remove_image_sky_frac: float | None = None,
#     remove_sky_image_side: str = "top",
#     remove_min_elev_deg: float | None = None,
# ) -> dict:
#     """
#     Remove Gaussians visible from ``probe_cams`` and save the remainder.

#     Visibility is tested in the *aligned* render frame.
#     The saved PLY is written in ``export_frame`` (default original).
#     Returns ``removed`` = kept Gaussians still in the aligned frame.
#     """
#     if not probe_cams:
#         print("  [training_ply] no probe cameras — skip.")
#         return {"skipped": True}

#     n_total = int(gaussians._xyz.shape[0])
#     visible = torch.zeros(n_total, dtype=torch.bool, device="cuda")
#     bg_black = torch.zeros(3, device="cuda")

#     height_ok = None
#     if remove_above_height is not None:
#         if world_up is None:
#             raise ValueError("remove_above_height requires world_up")
#         heights = _gaussian_heights_along_up(gaussians, world_up)
#         height_ok = heights >= float(remove_above_height)
#         print(
#             f"  [training_ply] height gate: xyz·up >= {float(remove_above_height):.4f}  "
#             f"candidates={int(height_ok.sum())}/{n_total}"
#         )

#     min_elev = None
#     if remove_min_elev_deg is not None:
#         if world_up is None:
#             raise ValueError("remove_min_elev_deg requires world_up")
#         min_elev = float(remove_min_elev_deg)
#         print(
#             f"  [training_ply] elevation gate: elev_from_cam >= {min_elev:.2f}° "
#             f"(0°=horizon, +90°=zenith)"
#         )

#     sky_frac = None
#     if remove_image_sky_frac is not None:
#         sky_frac = float(remove_image_sky_frac)
#         if not (0.0 < sky_frac <= 1.0):
#             raise ValueError(
#                 f"remove_image_sky_frac must be in (0,1], got {sky_frac}"
#             )
#         side = str(remove_sky_image_side or "top").lower()
#         if side not in ("top", "bottom"):
#             raise ValueError("remove_sky_image_side must be 'top' or 'bottom'")
#         print(
#             f"  [training_ply] image sky gate: {side} {100.0 * sky_frac:.0f}% "
#             f"of rasterizer frame"
#         )

#     print(
#         f"\n  [training_ply] scanning {len(probe_cams)} {label} "
#         f"for visible Gaussians (to remove)..."
#     )
#     xyz = gaussians.get_xyz.detach()
#     with torch.no_grad():
#         for k, cam_data in enumerate(tqdm(probe_cams, desc="Remove visibility")):
#             c2w = cam_data["c2w"]
#             cam = make_render_camera(cam_data, c2w)
#             try:
#                 pkg = render(cam, gaussians, pipeline, bg_black)
#                 newly = pkg["radii"] > 0
#                 if height_ok is not None:
#                     newly = newly & height_ok
#                 if min_elev is not None:
#                     cam_pos = torch.as_tensor(
#                         c2w[:3, 3], dtype=xyz.dtype, device=xyz.device,
#                     )
#                     elev = _elevation_above_horizon_deg(xyz, cam_pos, world_up)
#                     newly = newly & (elev >= min_elev)
#                 if sky_frac is not None:
#                     y_pix, proj_ok = _project_gaussians_pixel_y(xyz, cam)
#                     h = float(cam.image_height)
#                     side = str(remove_sky_image_side or "top").lower()
#                     if side == "top":
#                         in_sky = y_pix < (sky_frac * h)
#                     else:
#                         in_sky = y_pix >= ((1.0 - sky_frac) * h)
#                     newly = newly & proj_ok & in_sky
#                 visible |= newly
#             finally:
#                 del cam
#             if (k + 1) % max(1, len(probe_cams) // 10) == 0 or k == 0:
#                 print(
#                     f"    probe {k + 1}/{len(probe_cams)}  "
#                     f"visible_so_far={int(visible.sum())}"
#                 )
#             if (k + 1) % 20 == 0 and torch.cuda.is_available():
#                 torch.cuda.empty_cache()

#     n_visible = int(visible.sum())
#     keep = ~visible
#     n_keep = int(keep.sum())
#     print(
#         f"  [training_ply] total={n_total}  remove_visible={n_visible}  "
#         f"keep={n_keep} ({100.0 * n_keep / max(n_total, 1):.1f}%)"
#     )

#     kept = subset_gaussians(gaussians, keep)
#     kept_aligned = subset_gaussians(gaussians, keep)

#     if export_frame == "original" and not np.allclose(R_align, np.eye(3), atol=1e-8):
#         apply_alignment_to_gaussians(kept, R_align.T)
#         print("  [training_ply] unaligned kept Gaussians → original frame")

#     out_path = Path(out_path)
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     kept.save_ply(str(out_path))
#     print(f"  ✓ Saved {out_path}  ({n_keep} Gaussians)")

#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()

#     return {
#         "skipped": False,
#         "n_total": n_total,
#         "n_removed_visible": n_visible,
#         "n_kept": n_keep,
#         "n_probe_cams": len(probe_cams),
#         "path": str(out_path),
#         "export_frame": export_frame,
#         "label": label,
#         "remove_above_height": (
#             float(remove_above_height) if remove_above_height is not None else None
#         ),
#         "remove_min_elev_deg": min_elev,
#         "remove_image_sky_frac": sky_frac,
#         "remove_sky_image_side": (
#             str(remove_sky_image_side) if sky_frac is not None else None
#         ),
#         "removed": kept_aligned,
#     }

def save_training_ply_removing_visible(
    gaussians: GaussianModel,
    probe_cams: list,
    pipeline,
    out_path: Path,
    R_align: np.ndarray,
    export_frame: str = "original",
    label: str = "remove probes",
    # world_up: np.ndarray | None = None,
    # remove_sky_image_side: str = "top",
) -> dict:
    """
    Remove Gaussians visible from ``probe_cams`` and save the remainder as
    ``gaussians_for_training.ply``.

    Visibility is tested in the *aligned* render frame.
    The saved PLY is written in ``export_frame`` (default original).
    Returns ``removed`` = kept Gaussians still in the aligned frame (for optional
    continued rendering).

    Optional sky-only gates (important for upward pitch probes that still see
    ground in the lower frustum):

      • ``remove_min_elev_deg`` — only remove if elevation from the *probe
        camera* to the Gaussian is ≥ this (0° = horizon, +90° = zenith).
        This is the main guard against deleting ground / facades.
      • ``remove_above_height`` — only remove Gaussians with world height
        ``xyz·up >= threshold``.
      • ``remove_image_sky_frac`` — only count visibility in this fraction of
        the rasterizer image on the sky side (``top`` = small y / zenith).
    """
    if not probe_cams:
        print("  [training_ply] no probe cameras — skip.")
        return {"skipped": True}

    n_total = int(gaussians._xyz.shape[0])
    visible = torch.zeros(n_total, dtype=torch.bool, device="cuda")
    bg_black = torch.zeros(3, device="cuda")

    # height_ok = None
    # min_elev = None
    # sky_frac = None

    print(f"\n  [training_ply] scanning {len(probe_cams)} {label} "
          f"for visible Gaussians (to remove)...")
    # xyz = gaussians.get_xyz.detach()
    with torch.no_grad():
        for k, cam_data in enumerate(tqdm(probe_cams, desc="Remove visibility")):
            c2w = cam_data["c2w"]
            cam = make_render_camera(cam_data, c2w)
            try:
                pkg = render(cam, gaussians, pipeline, bg_black)
                newly = pkg["radii"] > 0
                visible |= newly
            finally:
                del cam
            if (k + 1) % max(1, len(probe_cams) // 10) == 0 or k == 0:
                print(f"    probe {k+1}/{len(probe_cams)}  "
                      f"visible_so_far={int(visible.sum())}")
            if (k + 1) % 20 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

    n_visible = int(visible.sum())
    keep = ~visible
    n_keep = int(keep.sum())
    print(f"  [training_ply] total={n_total}  remove_visible={n_visible}  "
          f"keep={n_keep} ({100.0 * n_keep / max(n_total, 1):.1f}%)")

    kept = _subset_gaussians(gaussians, keep)
    kept_aligned = _subset_gaussians(gaussians, keep)
    

    if export_frame == "original" and not np.allclose(R_align, np.eye(3), atol=1e-8):
        apply_alignment_to_gaussians(kept, R_align.T)
        print("  [training_ply] unaligned kept Gaussians → original frame")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept.save_ply(str(out_path))
    print(f"  ✓ Saved {out_path}  ({n_keep} Gaussians)")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "skipped": False,
        "n_total": n_total,
        "n_removed_visible": n_visible,
        "n_kept": n_keep,
        "n_probe_cams": len(probe_cams),
        "path": str(out_path),
        "export_frame": export_frame,
        "label": label,
        # "remove_above_height": (
        #      None
        # ),
        # "remove_min_elev_deg": min_elev,
        # "remove_image_sky_frac": sky_frac,
        # "remove_sky_image_side": (
        #     str(remove_sky_image_side) if sky_frac is not None else None
        # ),
        "removed": kept_aligned,
    }