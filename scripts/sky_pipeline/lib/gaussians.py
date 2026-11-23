"""Gaussian PLY resolution, filtering, and render-camera helpers."""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from scene import GaussianModel
from scene.cameras import Camera


def resolve_ply(
    model_path: str | None,
    iteration: int,
    ply_path: str | None = None,
) -> Path:
    """
    Resolve the Gaussian PLY to load.

    Priority:
      1. ``ply_path`` if given
      2. ``model_path`` if it is already a .ply file
      3. ``model_path/point_cloud/iteration_<N>/point_cloud.ply``
      4. latest iteration under ``model_path/point_cloud/``
      5. ``model_path/point_cloud.ply``
    """
    if ply_path is not None:
        p = Path(ply_path)
        if not p.is_file():
            raise FileNotFoundError(f"--ply_path not found: {p}")
        print(f"  Checkpoint (direct): {p}")
        return p

    if model_path is None:
        raise ValueError("Provide --model_path or --ply_path")

    root = Path(model_path)
    if root.is_file() and root.suffix.lower() == ".ply":
        print(f"  Checkpoint (direct): {root}")
        return root

    candidate = root / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if candidate.exists():
        print(f"  Checkpoint: {candidate}")
        return candidate

    pc_dir = root / "point_cloud"
    if pc_dir.exists():
        available = sorted(
            [
                d for d in pc_dir.iterdir()
                if d.is_dir() and (d / "point_cloud.ply").exists()
            ],
            key=lambda d: int(d.name.split("_")[-1])
            if d.name.split("_")[-1].isdigit()
            else 0,
        )
        if available:
            chosen = available[-1] / "point_cloud.ply"
            print(f"  iteration_{iteration} not found - using: {chosen}")
            return chosen

    flat = root / "point_cloud.ply"
    if flat.exists():
        print(f"  Checkpoint (flat): {flat}")
        return flat

    raise FileNotFoundError(
        f"Could not find point_cloud.ply under: {root}\n"
        f"Pass --ply_path /path/to/point_cloud.ply to load a file directly."
    )


# Aliases used by older call sites
resolve_ply_path = resolve_ply
_resolve_ply = resolve_ply


def pipeline():
    """Minimal PipelineParams stand-in for the rasterizer."""
    return type("Pipeline", (), {
        "convert_SHs_python": False,
        "compute_cov3D_python": False,
        "antialiasing": False,
        "debug": False,
    })()


_pipeline = pipeline


def gaussians_to_cuda(gaussians: GaussianModel) -> None:
    for attr in (
        "_xyz", "_features_dc", "_features_rest",
        "_opacity", "_scaling", "_rotation",
    ):
        t = getattr(gaussians, attr, None)
        if t is not None and not t.is_cuda:
            setattr(gaussians, attr, t.cuda())


_gaussians_to_cuda = gaussians_to_cuda


def subset_gaussians(gaussians, keep: torch.Tensor):
    """
    Return a new GaussianModel containing only Gaussians selected by ``keep``.
    The original model is never modified.
    """
    n_total = int(gaussians._xyz.shape[0])
    g = GaussianModel(sh_degree=gaussians.max_sh_degree)

    def _copy(param):
        return nn.Parameter(param[keep].detach().clone().requires_grad_(False))

    g._xyz = _copy(gaussians._xyz)
    g._features_dc = _copy(gaussians._features_dc)
    g._features_rest = _copy(gaussians._features_rest)
    g._opacity = _copy(gaussians._opacity)
    g._scaling = _copy(gaussians._scaling)
    g._rotation = _copy(gaussians._rotation)

    for aux in ("max_radii2D", "xyz_gradient_accum", "denom"):
        t = getattr(gaussians, aux, None)
        if t is not None and t.numel() == n_total:
            setattr(g, aux, t[keep].clone())

    return g


_subset_gaussians = subset_gaussians


def c2w_to_R_T(c2w: np.ndarray) -> tuple:
    R_c2w = c2w[:3, :3]
    t_c2w = c2w[:3, 3]
    return R_c2w.astype(np.float32), (-R_c2w.T @ t_c2w).astype(np.float32)


def make_gs_camera(
    cam_data: dict,
    R,
    T,
    fov_x_cam: float,
    fov_y_cam: float,
    blank: torch.Tensor,
):
    """
    Construct scene.cameras.Camera across vanilla 3DGS and forks.

    Only pass kwargs present in Camera.__init__.
    """
    params = inspect.signature(Camera.__init__).parameters
    cam_w = int(blank.shape[2])
    cam_h = int(blank.shape[1])

    kwargs = {
        "colmap_id": cam_data["id"],
        "R": R,
        "T": T,
        "FoVx": fov_x_cam,
        "FoVy": fov_y_cam,
        "image": blank,
        "image_name": cam_data["img_name"],
        "uid": cam_data["id"],
        "gt_alpha_mask": None,
        "bg": torch.zeros(3, dtype=torch.float32),
        "image_width": cam_w,
        "image_height": cam_h,
        "image_path": "",
        "timestep": 0,
    }
    kwargs = {k: v for k, v in kwargs.items() if k in params}
    cam = Camera(**kwargs)

    for attr in (
        "world_view_transform", "projection_matrix",
        "full_proj_transform", "camera_center",
    ):
        t = getattr(cam, attr, None)
        if torch.is_tensor(t) and t.device.type != "cuda":
            setattr(cam, attr, t.cuda())
    return cam


_make_gs_camera = make_gs_camera


def make_render_camera(cam_data: dict, c2w: np.ndarray):
    """
    Build one GS Camera for a single view.

    Uses a 1×1 placeholder image, then overrides image_width/height so the
    rasterizer renders at the real resolution without allocating HxW on CUDA.
    """
    cam_w, cam_h = int(cam_data["width"]), int(cam_data["height"])
    fov_x_cam = 2.0 * math.atan(cam_w / (2.0 * cam_data["fx"]))
    fov_y_cam = 2.0 * math.atan(cam_h / (2.0 * cam_data["fy"]))
    R, T = c2w_to_R_T(c2w)
    blank = torch.zeros((3, 1, 1), dtype=torch.float32)
    cam = make_gs_camera(cam_data, R, T, fov_x_cam, fov_y_cam, blank)
    cam.image_width = cam_w
    cam.image_height = cam_h
    if hasattr(cam, "original_image") and cam.original_image is not None:
        del cam.original_image
        cam.original_image = None
    return cam


def target_from_opacity_centroid(
    gaussians,
    opacity_thresh: float = 0.5,
) -> np.ndarray:
    with torch.no_grad():
        xyz = gaussians._xyz.float()
        opacity = torch.sigmoid(gaussians._opacity.float().squeeze(-1))

    mask = opacity > opacity_thresh
    n_kept = int(mask.sum())
    if n_kept < 10:
        print(
            f"  Only {n_kept} Gaussians above threshold "
            f"{opacity_thresh} - lowering to 0.1"
        )
        mask = opacity > 0.1
        n_kept = int(mask.sum())

    centroid = (
        (xyz[mask] * opacity[mask].unsqueeze(1)).sum(0)
        / opacity[mask].sum()
    )
    target = centroid.detach().cpu().numpy().astype(np.float64)
    print(
        f"  Target (opacity centroid, {n_kept}/{len(opacity)} "
        f"Gaussians): {target.tolist()}"
    )
    return target

# def compute_exclusion_radius(
#     gaussians,
#     eye_position,
#     c2w_list=None,
#     scene_frac: float = 0.1,     # radius = 10% of median eye->scene distance (fallback)
#     cam_factor: float = 1.2,     # radius = 1.2 x camera spread (preferred)
#     max_scene_frac: float = 0.5, # never exceed 50% of median scene distance
#     min_opacity: float = 0.1,
#     percentile: float = 50.0,
#     max_samples: int = 1_000_000,
# ) -> float:
#     """Pick an exclusion radius proportional to the scene's own scale."""
#     with torch.no_grad():
#         xyz = gaussians._xyz.float()
#         opacity = torch.sigmoid(gaussians._opacity.float().squeeze(-1))
#         eye_t = torch.as_tensor(eye_position, dtype=torch.float32, device=xyz.device)

#         pts = xyz[opacity > min_opacity]
#         if pts.shape[0] < 100:          # degenerate scene -> use everything
#             pts = xyz
#         d = torch.norm(pts - eye_t[None], dim=1)

#         # torch.quantile fails on very large tensors -> subsample
#         if d.numel() > max_samples:
#             idx = torch.randint(0, d.numel(), (max_samples,), device=d.device)
#             d = d[idx]
#         scene_dist = torch.quantile(d, percentile / 100.0).item()

#     r_scene = scene_frac * scene_dist
#     r_max = max_scene_frac * scene_dist

#     r_cam = None
#     if c2w_list is not None and len(c2w_list) > 1:
#         centers = np.stack([np.asarray(c)[:3, 3] for c in c2w_list])
#         cam_spread = float(np.linalg.norm(centers - np.asarray(eye_position), axis=1).max())
#         if cam_spread > 1e-6:
#             r_cam = cam_factor * cam_spread

#     radius = min(r_cam, r_max) if r_cam is not None else r_scene

#     print(
#         f"  [Radius] median scene dist={scene_dist:.3f}, "
#         f"cam radius={r_cam if r_cam is None else round(r_cam, 3)}, "
#         f"scene radius={r_scene:.3f} -> exclusion_radius={radius:.3f}"
#     )
#     return radius

def compute_exclusion_radius(gaussians, eye_position, c2w_list) -> float:
    """Half the distance from the eye to the nearest real geometry (2nd percentile)."""
    with torch.no_grad():
        xyz = gaussians._xyz.float()
        opacity = torch.sigmoid(gaussians._opacity.float().squeeze(-1))
        eye_t = torch.as_tensor(eye_position, dtype=torch.float32, device=xyz.device)
        d = torch.norm(xyz[opacity > 0.1] - eye_t[None], dim=1)
        d = d[torch.randperm(d.numel(), device=d.device)[:1_000_000]]  # quantile size limit
        centers = np.stack([np.asarray(c)[:3, 3] for c in c2w_list])
        radius = 0.5 * np.linalg.norm(centers - eye_position, axis=1).max()
    print(f"  [Radius] exclusion_radius={radius:.3f}")
    return radius

def filter_gaussians(
    gaussians,
    eye_position=None,
    exclusion_radius: float = 1.5,
) -> None:
    """
    In-place drop floaters near the panorama eye and oversized / bright splats.

    Fixed: ``eye_position is None`` no longer references ``opacity_mask``
    before it is assigned.
    """
    with torch.no_grad():
        opacity = torch.sigmoid(gaussians._opacity.float().squeeze(-1))
        scales = torch.exp(gaussians._scaling.float())
        max_scale = scales.max(dim=1).values
        xyz = gaussians._xyz.float()
    

        if eye_position is not None:
            eye_t = torch.tensor(
                eye_position, dtype=torch.float32, device=xyz.device,
            )
            dist = torch.norm(xyz - eye_t[None], dim=1)
            exclusion_mask = dist > (exclusion_radius + max_scale)

            by_radius = int((dist <= exclusion_radius).sum())
            by_scale = int(((dist > exclusion_radius) & (dist <= exclusion_radius + max_scale)).sum())
            print(f"  [Filter] removed by radius only: {by_radius}")
            print(f"  [Filter] removed only because of max_scale: {by_scale}")
            print(f"  [Filter] max_scale p50/p99/max: "
            f"{max_scale.median().item():.3f} / "
            f"{torch.quantile(max_scale[:1_000_000], 0.99).item():.3f} / "
            f"{max_scale.max().item():.3f}")
        else:
            exclusion_mask = torch.ones(
                opacity.shape[0], dtype=torch.bool, device=opacity.device,
            )

        mask = exclusion_mask

        n_total = int(opacity.shape[0])
        n_kept = int(mask.sum())

       
        print(
            f"  [Filter] kept {n_kept}/{n_total} "
            f"({100.0 * n_kept / max(n_total, 1):.1f}%)"
        )

        if n_kept == n_total:
            print("  [Filter] nothing removed.")
            return

        for attr in (
            "_xyz", "_features_dc", "_features_rest",
            "_opacity", "_scaling", "_rotation",
        ):
            tensor = getattr(gaussians, attr, None)
            if tensor is not None:
                setattr(gaussians, attr, tensor[mask])
