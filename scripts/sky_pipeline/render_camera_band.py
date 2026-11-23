from __future__ import annotations

import argparse
import json
import math
import sys
import torch
from tqdm import tqdm
from pathlib import Path

import torchvision.transforms.functional as tf

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

from lib.panorama_io import (
    flip_panorama_horizontal,
    load_panorama_cuda,
    load_panorama_sidecar,
    panorama_sidecar_path,
)


from lib.cameras import (
    load_cameras_from_json,
    default_cameras_json,
    look_at_c2w,
    regularize_cameras_on_circle,
    subsample_cameras,
    save_loaded_cameras_json
)

from lib.band import (
    expand_band_cameras,
    frustum_overlap_deg,
    fov_y_deg_from_cam,
    assign_band_roles,
    cam_with_lookback_pitch,
    apply_base_pitch_json,
    parse_float_list
)

from lib.panorama_sphere import (
    composite_panorama_sphere,
    panorama_sphere_background
)

from lib.training_ply import save_training_ply_removing_visible

from lib.composite_views import records_to_export_frame

from lib.transforms_json import build_transforms_from_records

from lib.alignment import align_loaded_cameras, apply_alignment_to_gaussians, unalign_point

from lib.gaussians import (
    pipeline as make_pipeline,
    resolve_ply, 
    _gaussians_to_cuda, 
    make_render_camera, 
    _subset_gaussians
    )


from scene import GaussianModel

from renderer.gaussian_renderer import render
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

CANONICAL_UP = np.array([0.0, 1.0, 0.0], dtype=np.float64)

CUBE_DIRS = {
    "+x": np.array([1, 0, 0.], dtype=np.float32),
    "-x": np.array([-1, 0, 0.], dtype=np.float32),
    "+y": np.array([0, 1, 0.], dtype=np.float32),
    "-y": np.array([0, -1, 0.], dtype=np.float32),
    "+z": np.array([0, 0, 1.], dtype=np.float32),
    "-z": np.array([0, 0, -1.], dtype=np.float32),
}

def _pick_up(forward, world_up=np.array([0., 1., 0.], dtype=np.float32)):
    return (
        np.array([0., 0., 1.], dtype=np.float32)
        if abs(np.dot(forward, world_up)) > 0.99
        else world_up
    )

def _world_view_transform(eye, forward, up):
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    true_up = np.cross(right, forward)

    R = np.stack([right, true_up, -forward], axis=0)

    T = -R @ eye

    Rt = np.eye(4, dtype=np.float32)
    Rt[:3, :3] = R
    Rt[:3, 3] = T

    return Rt

def _proj_matrix(fovx, fovy, znear, zfar):
    tanHalfFovY = np.tan(fovy / 2)
    tanHalfFovX = np.tan(fovx / 2)

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = np.zeros((4, 4), dtype=np.float32)

    P[0, 0] = 2 * znear / (right - left)
    P[1, 1] = 2 * znear / (top - bottom)

    P[3, 2] = 1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)

    return P

def get_visible_gaussians(gaussians, cam, image_size):

    settings = GaussianRasterizationSettings(
        image_height=image_size,
        image_width=image_size,
        tanfovx=np.tan(cam.FoVx / 2),
        tanfovy=np.tan(cam.FoVy / 2),
        bg=torch.zeros(3, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=cam.world_view_transform,
        projmatrix=cam.full_proj_transform,
        sh_degree=0,
        campos=cam.camera_center,
        prefiltered=False,
        debug=False,
        antialiasing=False,
    )

    rasterizer = GaussianRasterizer(
        raster_settings=settings
    )

    means3D = gaussians.get_xyz

    means2D = torch.zeros_like(means3D)

    dummy_color = torch.ones_like(means3D)

    _, radii, _ = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=None,
        colors_precomp=dummy_color,
        opacities=gaussians.get_opacity,
        scales=gaussians.get_scaling,
        rotations=gaussians.get_rotation,
    )

    return radii > 0

def split_records_train_test_with_prefix(
    c2w_matrices: list,
    test_ratio: float = 0.1,
    seed: int = 42,
    prefix: str = "r",
) -> list:
    """Deterministic train/test split with per-split local indices and prefix."""
    n = len(c2w_matrices)
    rng = np.random.RandomState(seed)
    order = np.arange(n)
    rng.shuffle(order)
    n_test = max(1, int(round(n * test_ratio))) if n > 1 else 0
    test_set = set(order[:n_test].tolist())

    train_i, test_i = 0, 0
    records = []
    for i, c2w in enumerate(c2w_matrices):
        if i in test_set:
            records.append({"split": "test",  "idx": test_i,  "c2w": c2w, "prefix": prefix})
            test_i += 1
        else:
            records.append({"split": "train", "idx": train_i, "c2w": c2w, "prefix": prefix})
            train_i += 1
    return records

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

class MiniCam:
    def __init__(
        self,
        width,
        height,
        fovx,
        fovy,
        znear,
        zfar,
        world_view_np,
    ):
        self.image_width = width
        self.image_height = height

        self.FoVx = fovx
        self.FoVy = fovy

        self.world_view_transform = torch.tensor(
            world_view_np,
            dtype=torch.float32,
            device="cuda",
        ).transpose(0, 1)

        proj = torch.tensor(
            _proj_matrix(fovx, fovy, znear, zfar),
            dtype=torch.float32,
            device="cuda",
        ).transpose(0, 1)

        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0)
            .bmm(proj.unsqueeze(0))
            .squeeze(0)
        )

        self.camera_center = (
            torch.inverse(self.world_view_transform)[3, :3]
        )


def build_band_from_base( 
    base_cams: list,
    target_pt: np.ndarray, 
    world_up: np.ndarray, 
    pitches: str,
    height_offsets: str | None, 
    role_tags: bool = True, 
    base_pitch_deg: float = 0.0, 
    sky_pitch_sign: float = -1.0, 
    ) -> tuple:
    """Return (band_cams, pitches_deg, seams, fov_y, heights, roles)."""

    fov_y = fov_y_deg_from_cam(base_cams[0])

    if fov_y <= 1e-6:
        raise ValueError(f"FoVy must be positive, got {fov_y}")

    heights = parse_float_list(height_offsets, default=[0.0])
    pitch_values = parse_float_list(pitches)
    if not pitch_values: 
        raise ValueError("--pitches is required")

    pitches_deg = unique_pitches(pitch_values)

    roles = assign_band_roles( pitches_deg, float(base_pitch_deg), float(sky_pitch_sign), )

    # Calculate actual overlap between neighboring frustums. 
    seams = []

    for i in range(len(pitches_deg) - 1):
        a, b = pitches_deg[i], pitches_deg[i + 1]
        ov = frustum_overlap_deg(a, b, fov_y)
        seams.append({
            "from_pitch_deg": a,
            "to_pitch_deg": b,
            "look_axis_sep_deg": abs(b - a),
            "frustum_overlap_deg": ov,
            "relation": (
                "overlap" if ov > 1e-3 else ("kiss" if abs(ov) <= 1e-3 else "gap")
            ),
        })

    band_cams, _rows = expand_band_cameras(
        base_cams, target_pt, world_up, pitches_deg,
        height_offsets=heights,
        role_tags=role_tags,
        band_roles=roles,
    )
    return band_cams, pitches_deg, seams, fov_y, heights, roles

def unique_pitches(pitches: list[float]) -> list[float]:
    """Return pitches with duplicates removed, preserving caller order."""
    out = []
    seen = set()

    for p in pitches:
        p = float(p)
        key = round(p, 6)

        if key not in seen:
            seen.add(key)
            out.append(p)

    if not out :
        raise ValueError("At least one pitch is required")

    return out

def assign_render_modes(
    band_cams: list,
    scene_mode: str = "composite",
    sky_mode: str = "composite",
    down_mode: str | None = None,
) -> list:
    """Attach render_mode + ring_tag from band_role."""
    down_m = scene_mode if down_mode is None else down_mode
    out = []
    for cam in band_cams:
        c = dict(cam)
        role = c.get("band_role", "scene")
        if is_sky_role(role):
            mode = sky_mode
        elif is_down_role(role):
            mode = down_m
        else:
            mode = scene_mode
        c["render_mode"] = mode
        c["ring_tag"] = str(role)
        out.append(c)
    return out

# def resolve_remove_lookback_pitch(
#     *,
#     lookback_pitch_deg: float | None,
#     base_cams: list,
#     pitch_pos_deg: float | None = None,
#     outpaint_gap_deg: float = 0.0,
# ) -> tuple:
#     """
#     Resolve look-back pitch for removal probes.

#     Priority: ``--remove_pitch_json`` → ``--remove_lookback_pitch_deg`` →
#     kiss formula from ``pitch_pos`` (default FoVy/2) → fallback -150.
#     Returns (pitch_deg, meta_dict).
#     """
#     meta = {}

#     if lookback_pitch_deg is not None:
#         pitch = float(lookback_pitch_deg)
#         meta = {"source": "cli", "lookback_pitch_deg": pitch}
#         return pitch, meta

#     fov_y = fov_y_deg_from_cam(base_cams[0])
#     pos = float(pitch_pos_deg) if pitch_pos_deg is not None else 0.5 * fov_y
#     try:
#         pitch = lookback_pitch_from_pos(pos, fov_y, float(outpaint_gap_deg))
#         meta = {
#             "source": "kiss_formula",
#             "pitch_pos_deg": pos,
#             "fov_y_deg": fov_y,
#             "outpaint_gap_deg": float(outpaint_gap_deg),
#             "lookback_pitch_deg": pitch,
#         }
#         return pitch, meta
#     except ValueError as e:
#         pitch = -150.0
#         meta = {
#             "source": "fallback_-150",
#             "error": str(e),
#             "lookback_pitch_deg": pitch,
#         }
#         print(f"  ⚠ lookback formula failed ({e}); using {pitch}°")
#         return pitch, meta

def build_lookback_remove_probes(
    base_circle_cams: list,
    target_pt: np.ndarray,
    world_up: np.ndarray,
    lookback_pitch_deg: float,
) -> list:
    """Regularized-circle cameras looking outward, then pitched."""
    out = []
    for i, src in enumerate(base_circle_cams):
        out.append(cam_with_lookback_pitch(
            src, target_pt, world_up, float(lookback_pitch_deg),
            render_mode="probe", tag="remove_lookback", cam_id=i,
        ))
    print(
        f"  [remove lookback] {len(out)} probes on regularized circle  "
        f"lookback_pitch={float(lookback_pitch_deg):+.2f}°"
    )
    return out

def is_sky_role(role: str | None) -> bool:
    """True for sky outpaint pitches (not the scene/down anchors)."""
    r = str(role or "")
    return r == "sky_top" or r.startswith("sky_")


def is_down_role(role: str | None) -> bool:
    r = str(role or "")
    return r == "down" or r == "down_bottom" or r.startswith("down_")

def estimate_panorama_radius_rasterized(
    gaussians,
    eye_position,
    image_size=512,
    percentile=97.0,
    margin=1.0,
    znear=0.01,
    zfar=15000.0,
):

    eye = np.asarray(eye_position, dtype=np.float32)

    fov = np.pi / 2

    visible_global = torch.zeros(
        gaussians.get_xyz.shape[0],
        dtype=torch.bool,
        device="cuda",
    )

    with torch.no_grad():

        for fwd in CUBE_DIRS.values():

            up = _pick_up(fwd)

            wv = _world_view_transform(
                eye,
                fwd,
                up,
            )

            cam = MiniCam(
                image_size,
                image_size,
                fov,
                fov,
                znear,
                zfar,
                wv,
            )

            visible = get_visible_gaussians(
                gaussians,
                cam,
                image_size,
            )

            visible_global |= visible


    xyz = gaussians.get_xyz[visible_global]

    cam_center = torch.tensor(
        eye,
        dtype=torch.float32,
        device="cuda",
    )

    distances = torch.norm(
        xyz - cam_center,
        dim=1,
    )

    pct_dist = torch.quantile(
        distances,
        percentile / 100.0,
    ).item()

    radius = pct_dist * margin

    print(
        f"[sphere_radius] visible={visible_global.sum().item()} "
        f"{percentile:.1f}%={pct_dist:.2f} "
        f"radius={radius:.2f}"
    )
    print("distance statistics:")
    print("min:", distances.min().item())
    print("50%:", torch.quantile(distances, 0.50).item())
    print("90%:", torch.quantile(distances, 0.90).item())
    print("95%:", torch.quantile(distances, 0.95).item())
    print("99%:", torch.quantile(distances, 0.99).item())
    print("99.9%:", torch.quantile(distances, 0.999).item())
    print("max:", distances.max().item())

    return radius



# def save_training_ply_removing_visible(
#     gaussians: GaussianModel,
#     probe_cams: list,
#     pipeline,
#     out_path: Path,
#     R_align: np.ndarray,
#     export_frame: str = "original",
#     label: str = "remove probes",
#     world_up: np.ndarray | None = None,
#     remove_sky_image_side: str = "top",
# ) -> dict:
#     """
#     Remove Gaussians visible from ``probe_cams`` and save the remainder as
#     ``gaussians_for_training.ply``.

#     Visibility is tested in the *aligned* render frame.
#     The saved PLY is written in ``export_frame`` (default original).
#     Returns ``removed`` = kept Gaussians still in the aligned frame (for optional
#     continued rendering).

#     Optional sky-only gates (important for upward pitch probes that still see
#     ground in the lower frustum):

#       • ``remove_min_elev_deg`` — only remove if elevation from the *probe
#         camera* to the Gaussian is ≥ this (0° = horizon, +90° = zenith).
#         This is the main guard against deleting ground / facades.
#       • ``remove_above_height`` — only remove Gaussians with world height
#         ``xyz·up >= threshold``.
#       • ``remove_image_sky_frac`` — only count visibility in this fraction of
#         the rasterizer image on the sky side (``top`` = small y / zenith).
#     """
#     if not probe_cams:
#         print("  [training_ply] no probe cameras — skip.")
#         return {"skipped": True}

#     n_total = int(gaussians._xyz.shape[0])
#     visible = torch.zeros(n_total, dtype=torch.bool, device="cuda")
#     bg_black = torch.zeros(3, device="cuda")

#     height_ok = None
#     min_elev = None
#     sky_frac = None

#     print(f"\n  [training_ply] scanning {len(probe_cams)} {label} "
#           f"for visible Gaussians (to remove)...")
#     xyz = gaussians.get_xyz.detach()
#     with torch.no_grad():
#         for k, cam_data in enumerate(tqdm(probe_cams, desc="Remove visibility")):
#             c2w = cam_data["c2w"]
#             cam = make_render_camera(cam_data, c2w)
#             try:
#                 pkg = render(cam, gaussians, pipeline, bg_black)
#                 newly = pkg["radii"] > 0
#                 visible |= newly
#             finally:
#                 del cam
#             if (k + 1) % max(1, len(probe_cams) // 10) == 0 or k == 0:
#                 print(f"    probe {k+1}/{len(probe_cams)}  "
#                       f"visible_so_far={int(visible.sum())}")
#             if (k + 1) % 20 == 0 and torch.cuda.is_available():
#                 torch.cuda.empty_cache()

#     n_visible = int(visible.sum())
#     keep = ~visible
#     n_keep = int(keep.sum())
#     print(f"  [training_ply] total={n_total}  remove_visible={n_visible}  "
#           f"keep={n_keep} ({100.0 * n_keep / max(n_total, 1):.1f}%)")

#     kept = _subset_gaussians(gaussians, keep)
#     kept_aligned = _subset_gaussians(gaussians, keep)
    

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
#              None
#         ),
#         "remove_min_elev_deg": min_elev,
#         "remove_image_sky_frac": sky_frac,
#         "remove_sky_image_side": (
#             str(remove_sky_image_side) if sky_frac is not None else None
#         ),
#         "removed": kept_aligned,
#     }

def render_composite_views(
    c2w_matrices: list,
    loaded_cams: list,
    gaussians: GaussianModel,
    gaussians_removed: GaussianModel,
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

    Cameras are built lazily (one at a time) to avoid OOM. Per-camera
    ``render_mode`` (optional on loaded_cams entries):
      • ``composite`` (default) — Gaussians over panorama sphere
      • ``gs_only`` — Gaussians only (black background, no panorama)
      • ``bg_only`` — panorama sphere only (100% background, no Gaussians)
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
    print(f"  Dataset split: train={n_train}  test={n_test}  "
          f"(test_ratio={test_ratio}, seed={split_seed})")

    R_identity = np.eye(3, dtype=np.float64)
    bg_black = torch.zeros(3, device="cuda")
    records = []

    with torch.no_grad():
        for i, c2w in enumerate(tqdm(
            c2w_matrices,
            desc="Rendering",
            total=len(c2w_matrices),
        )):
            #cam_data = loaded_cams[i]
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
                    rendered = render(cam, gaussians, pipeline, bg_black)["render"].clamp(0.0, 1.0)
                finally:
                    del cam
            else:
                cam = make_render_camera(cam_data, c2w)
                try:
                    rendered = composite_panorama_sphere(
                        cam, c2w, gaussians_removed, pipeline, panorama_tiled,
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




def run(
    model_path: str | None = None,
    ply_path: str | None = None,
    bg_panorama: str | None = None,
    cameras_json: str | None = None,
    output_path: str | None = None,
    iteration: int = 30_000,
    regularize_circle: bool = True,
    circle_radius: float | None = None,
    circle_height: float | None = None,
    circle_start_azimuth_deg: float | None = None,
    base_pitch_deg: float = 0.0,
    base_pitch_json: str | None = None,
    pitch_pos_offset_deg: float = 5.0,
    seam_overlap_deg: float = 5.0,
    n_sky_steps: int = 2,
    sky_pitch_sign: float = -1.0,
    height_offsets: str | None = None,
    fov_y_deg: float | None = None,
    scene_render_mode: str = "composite",
    sky_render_mode: str = "composite",
    down_render_mode: str | None = "gs_only",
    remove_mode: str = "lookback",
    remove_roles: str = "sky",
    save_training_ply: bool = True,
    sphere_radius: float | None = None,
    panorama_yaw_deg: float = 0.0,
    panorama_flip_horizontal: bool = False,
    img_w: int | None = None,
    img_h: int | None = None,
    max_views: int | None = None,
    test_ratio: float = 0.1,
    split_seed: int = 42,
    role_tags: bool = True,
):
    export_frame = "original"

    if model_path is None and ply_path is None:
        raise ValueError("Provide at least one of --model_path or --ply_path")
    if cameras_json is None:
        raise ValueError("--cameras_json is required")
    if bg_panorama is None:
        raise ValueError("--bg_panorama is required")

    for mode, name in (
        (scene_render_mode, "scene_render_mode"),
        (sky_render_mode, "sky_render_mode"),
    ):
        if mode not in ("composite", "gs_only", "bg_only"):
            raise ValueError(f"--{name} must be composite|gs_only|bg_only, got {mode!r}")
    if down_render_mode is not None and down_render_mode not in (
        "composite", "gs_only", "bg_only",
    ):
        raise ValueError(
            f"--down_render_mode must be composite|gs_only|bg_only, got {down_render_mode!r}"
        )

    bp = apply_base_pitch_json(
        base_pitch_json=base_pitch_json,
        fov_y_deg=fov_y_deg,
        base_pitch_deg=base_pitch_deg,
        seam_overlap_deg=seam_overlap_deg,
        n_sky_steps=n_sky_steps,
        pitch_pos_offset_deg=pitch_pos_offset_deg,
    )

    pitches = bp["pitches"]
    fov_y_deg = bp["fov_y_deg"]
    base_pitch_deg = bp["base_pitch_deg"]
    remove_lookback_pitch_deg = bp["remove_lookback_pitch_deg"] + 2.0
    base_pitch_meta = bp["base_pitch_meta"]

    model_root = Path(model_path).resolve() if model_path else Path(ply_path).resolve().parent
    cam_json_path = Path(cameras_json).resolve()
    pano_path = Path(bg_panorama).resolve()
    out_root = (
        Path(output_path).resolve()
        if output_path
        else pano_path.parent / "sky_band"
    )
    if not cam_json_path.is_file():
        raise FileNotFoundError(f"cameras.json not found: {cam_json_path}")
    if not pano_path.is_file():
        raise FileNotFoundError(f"Panorama not found: {pano_path}")

    out_root.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("  Camera-band sky compositing")
    print(f"  model   : {model_root}")
    print(f"  pano    : {pano_path}")
    print(f"  cameras : {cam_json_path}")
    print(f"  output  : {out_root}")
    print(f"  seam_overlap: {seam_overlap_deg}°  n_sky_steps: {n_sky_steps}  "
          f"sky_pitch_sign={sky_pitch_sign:+.0f}")
    print(f"  scene_mode={scene_render_mode}  sky_mode={sky_render_mode}")
    print(f"  remove_mode={remove_mode}  remove_roles={remove_roles}  "
          f"export_frame={export_frame}")
    print(f"{'=' * 60}\n")

    # ── 1. Panorama + sidecar ─────────────────────────────────────────────
    print("Step 1 — load panorama + sidecar")
    sidecar = load_panorama_sidecar(pano_path)
    R_align = sidecar["R_align"]
    pano_eye = sidecar["eye"]
    pano_target = sidecar["target"]
    panorama_tiled = load_panorama_cuda(str(pano_path))
    if panorama_flip_horizontal:
        panorama_tiled = flip_panorama_horizontal(panorama_tiled)
        print("  Applied horizontal flip to panorama.")
    print(f"  Panorama tiled shape: {tuple(panorama_tiled.shape)}")

    # ── 2. Cameras → align → circle → band ────────────────────────────────
    print("\nStep 2 — load / align cameras, build pitch band")
    raw_cams = load_cameras_from_json(
        str(cam_json_path), img_w_override=img_w, img_h_override=img_h,
    )
    aligned_cams = align_loaded_cameras(raw_cams, R_align)

    if regularize_circle:
        base_cams = regularize_cameras_on_circle(
            aligned_cams,
            pano_target,
            CANONICAL_UP,
            radius=circle_radius,
            height=circle_height,
            start_azimuth_deg=circle_start_azimuth_deg,
        )
    else:
        print("  Keeping positions; re-aiming at pano_target…")
        base_cams = []
        for src in aligned_cams:
            pos = src["c2w"][:3, 3].astype(np.float64)
            c2w = look_at_c2w(pos, pano_target, world_up=CANONICAL_UP)
            base_cams.append({**src, "c2w": c2w})

    base_cams = subsample_cameras(base_cams, max_views)

    band_cams, pitches_deg, seams, fov_y, heights, roles = build_band_from_base(
        base_cams,
        pano_target,
        CANONICAL_UP,
        base_pitch_deg=base_pitch_deg,
        pitches=pitches,
        height_offsets=height_offsets,
        role_tags=role_tags,
        sky_pitch_sign=sky_pitch_sign,
    )

    
    band_cams = assign_render_modes(
        band_cams,
        scene_mode=scene_render_mode,
        sky_mode=sky_render_mode,
        down_mode=down_render_mode,
    )

    print(f"  FoVy={fov_y:.3f}°  pitches={[round(p, 3) for p in pitches_deg]}")
    print(f"  roles={roles}")
    for s in seams:
        print(
            f"    seam {s['from_pitch_deg']:+.2f}° → {s['to_pitch_deg']:+.2f}°  "
            f"overlap={s['frustum_overlap_deg']:+.2f}°  [{s['relation']}]"
        )
    n_sky = sum(1 for c in band_cams if is_sky_role(c.get("band_role")))
    n_down = sum(1 for c in band_cams if is_down_role(c.get("band_role")))
    n_scene = len(band_cams) - n_sky - n_down
    print(f"  Band views: {len(band_cams)}  "
          f"(down={n_down}, scene={n_scene}, sky={n_sky})")

    band_json = out_root / "cameras_band_aligned.json"
    save_loaded_cameras_json(band_cams, str(band_json))

    # ── 3. Load + align Gaussians ─────────────────────────────────────────
    print("\nStep 3 — load and align Gaussians")
    gaussians = GaussianModel(sh_degree=3)
    ckpt_path = resolve_ply(model_path, iteration, ply_path)
    gaussians.load_ply(str(ckpt_path))
    _gaussians_to_cuda(gaussians)
    apply_alignment_to_gaussians(gaussians, R_align)
    pipeline = make_pipeline()

    # ── 4. Remove sky-visible artefact Gaussians → training PLY ───────────
    training_ply_meta = None
    lookback_meta = None
    mode = (remove_mode or "lookback").strip().lower()
    if mode not in ("lookback", "band", "none", "off"):
        raise ValueError(
            f"--remove_mode must be lookback|band|none, got {remove_mode!r}"
        )

    if mode in ("none", "off"):
        probe_cams = []
    elif mode == "lookback":
        if base_pitch_meta is not None:
            lb_pitch = float(base_pitch_meta["lookback_pitch_deg"])
            lookback_meta = {
                "source": "base_pitch_json",
                "path": base_pitch_meta.get("path"),
                "pitch_pos_deg": base_pitch_meta.get("pitch_pos_deg"),
                "lookback_pitch_deg": lb_pitch,
                "outpaint_gap_deg": base_pitch_meta.get("outpaint_gap_deg"),
            }
        else:
            raise ValueError(
            f"--base_pitch_meta is required"
        )
        probe_cams = build_lookback_remove_probes(
            base_cams, pano_target, CANONICAL_UP, lb_pitch,
        )
    else:
        print("no lookback removal")

    if save_training_ply and probe_cams:
        if mode == "lookback":
     
            print(f"\nStep 4 — remove artefact Gaussians from {len(probe_cams)} "
              f"probes (mode={mode})")
        if mode == "lookback" and lookback_meta:
            print(f"  lookback pitch = {lookback_meta.get('lookback_pitch_deg'):+.2f}°  "
                  f"(source={lookback_meta.get('source')})")
            print("  gates: none (unconstrained lookback visibility)")
        

        label = (
            f"lookback_pitch={lookback_meta.get('lookback_pitch_deg'):+.1f}"
            if mode == "lookback" and lookback_meta
            else f"band remove_roles={remove_roles}"
        )
        training_ply_meta = save_training_ply_removing_visible(
            gaussians,
            probe_cams,
            pipeline,
            out_root / "gaussians_for_training.ply",
            R_align=R_align,
            export_frame=export_frame,
            label=label,
        )
        if training_ply_meta and not training_ply_meta.get("skipped"):
            gaussians_removed = training_ply_meta["removed"]
            
            # if not panorama:
            #     print("not panorama")
            # gaussians = training_ply_meta["removed"]
            training_ply_meta["remove_mode"] = mode
            training_ply_meta["remove_roles"] = remove_roles if mode == "band" else None
            training_ply_meta["lookback"] = lookback_meta
            training_ply_meta["n_probe_cams"] = len(probe_cams)
    elif save_training_ply:
        print("\nStep 4 — skipped (no probe cameras; check --remove_mode / --remove_roles)")
    else:
        print("\nStep 4 — skipped (--no_save_training_ply)")

    
    
    # ── 5. Sphere radius + render composites ──────────────────────────────
    if sphere_radius is None:
        sphere_radius = estimate_panorama_radius_rasterized(
            gaussians, pano_eye,
        )
        print(sphere_radius)
       
    else:
        sphere_radius = float(sphere_radius)
    
    print(f"\n  Panorama sphere origin: {pano_eye.tolist()}")
    print(f"  Panorama sphere radius: {sphere_radius:.2f}")

    print("\nStep 5 — render Gaussians over projected panorama")
    c2w_mats = [c["c2w"] for c in band_cams]
    fov_x_rad = float(np.median(
        [2.0 * math.atan(c["width"] / (2.0 * c["fx"])) for c in band_cams]
    ))
    # TODO: check below
    records = render_composite_views(
        c2w_mats, band_cams, gaussians, gaussians_removed, pipeline,
        panorama_tiled, pano_eye, sphere_radius, panorama_yaw_deg, out_root,
        test_ratio=test_ratio, split_seed=split_seed,
    )
    # Annotate band metadata onto records (render_composite_views already
    # copies pitch/ring_tag/render_mode when present).
    for i, rec in enumerate(records):
        cam = band_cams[i]
        rec["band_role"] = cam.get("band_role")
        rec["band_pitch_index"] = cam.get("band_pitch_index")

    # ── 6. Export transforms in requested frame ───────────────────────────
    print(f"\nStep 6 — write transforms/meta in export_frame={export_frame}")
    export_records = records_to_export_frame(records, R_align, export_frame)

    up_aligned = CANONICAL_UP / (np.linalg.norm(CANONICAL_UP) + 1e-12)
    if export_frame == "original":
        export_target = unalign_point(pano_target, R_align)
        export_eye = unalign_point(pano_eye, R_align)
        export_up = unalign_point(up_aligned, R_align)
        export_up = export_up / (np.linalg.norm(export_up) + 1e-12)
    else:
        export_target = np.asarray(pano_target, dtype=np.float64).reshape(3)
        export_eye = np.asarray(pano_eye, dtype=np.float64).reshape(3)
        export_up = up_aligned.copy()

    panorama_eye_up_offset = float(np.dot(export_eye - export_target, export_up))
    train_records = [r for r in export_records if r["split"] == "train"]
    test_records = [r for r in export_records if r["split"] == "test"]

    def _annotate_frames(transforms: dict, split_recs: list) -> dict:
        for frame, rec in zip(transforms["frames"], split_recs):
            for key in (
                "render_mode", "pitch_deg", "ring_tag", "look_back",
                "band_role", "band_pitch_index", "height_offset",
            ):
                if rec.get(key) is not None:
                    frame[key] = rec[key]
        return transforms

    transforms_train = _annotate_frames(
        build_transforms_from_records(train_records, fov_x_rad), train_records,
    )
    transforms_test = _annotate_frames(
        build_transforms_from_records(test_records, fov_x_rad), test_records,
    )
    train_json = out_root / "transforms_train.json"
    test_json = out_root / "transforms_test.json"
    with open(train_json, "w") as f:
        json.dump(transforms_train, f, indent=2)
    with open(test_json, "w") as f:
        json.dump(transforms_test, f, indent=2)
    print(f"  ✓ {train_json}  ({len(transforms_train['frames'])} frames)")
    print(f"  ✓ {test_json}  ({len(transforms_test['frames'])} frames)")

    meta = {
        "strategy": "camera_band_scene_preserve_seam_overlap",
        "model_path": str(model_root),
        "checkpoint_ply": str(ckpt_path),
        "panorama": str(pano_path),
        "sidecar": str(panorama_sidecar_path(pano_path)),
        "export_frame": export_frame,
        "target_pt": export_target.tolist(),
        "v_up_scene": export_up.tolist(),
        "panorama_eye_up_offset": panorama_eye_up_offset,
        "sphere_radius": sphere_radius,
        "pano_eye": export_eye.tolist(),
        "pano_target": export_target.tolist(),
        "pano_eye_aligned": pano_eye.tolist(),
        "pano_target_aligned": pano_target.tolist(),
        "v_up_scene_aligned": up_aligned.tolist(),
        "R_align": R_align.tolist(),
        "panorama_yaw_deg": panorama_yaw_deg,
        "panorama_flip_horizontal": panorama_flip_horizontal,
        "fov_y_deg": fov_y,
        "base_pitch_deg": float(base_pitch_deg),
        "base_pitch_json": base_pitch_json,
        "pitch_pos_offset_deg": float(pitch_pos_offset_deg),
        "seam_overlap_deg": float(seam_overlap_deg),
        "n_sky_steps": int(n_sky_steps),
        "sky_pitch_sign": float(sky_pitch_sign),
        "pitches_deg": pitches_deg,
        "band_roles": roles,
        "height_offsets": heights,
        "seams": seams,
        "scene_render_mode": scene_render_mode,
        "sky_render_mode": sky_render_mode,
        "down_render_mode": down_render_mode,
        "remove_mode": remove_mode,
        "remove_roles": remove_roles,
        "remove_lookback_pitch_deg": remove_lookback_pitch_deg,
        "base_pitch_meta": (
            {k: v for k, v in base_pitch_meta.items() if k != "raw"}
            if base_pitch_meta is not None else None
        ),
        "regularize_circle": bool(regularize_circle),
        "circle_radius": circle_radius,
        "circle_height": circle_height,
        "max_views": max_views,
        "n_views": len(c2w_mats),
        "n_train": len(train_records),
        "n_test": len(test_records),
        "test_ratio": test_ratio,
        "split_seed": split_seed,
        "cameras_band_json": str(band_json),
        "training_ply": (
            {k: v for k, v in training_ply_meta.items() if k != "removed"}
            if training_ply_meta is not None else None
        ),
        "sidecar_meta": sidecar["meta"],
    }
    meta_path = out_root / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  ✓ {meta_path}")

    print(
        f"\nDone — {len(train_records)} train + {len(test_records)} test "
        f"in {out_root}\n"
    )

  
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--ply_path", type=str, default=None)
    parser.add_argument("--bg_panorama", type=str, required=True)
    parser.add_argument("--cameras_json", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--iteration", type=int, default=30_000)

    parser.add_argument(
        "--regularize_circle",
        action="store_true",
        default=True,
        help="Snap cameras to a circle around pano target (default: on).",
    )
    parser.add_argument("--circle_radius", type=float, default=None)
    parser.add_argument("--circle_height", type=float, default=None)
    parser.add_argument("--circle_start_azimuth_deg", type=float, default=None)

    parser.add_argument("--base_pitch_deg", type=float, default=0.0,
                        help="Scene-preserving first pitch (default 0).")
    parser.add_argument(
        "--sky_pitch_sign",
        type=float,
        default=-1.0,
        help="Sky step direction: -1 (default) = negative pitches are sky; "
             "+1 = positive pitches are sky. Matches flipped-PNG convention.",
    )
    parser.add_argument("--seam_overlap_deg", type=float, default=5.0,
                        help="Shared meridian strip between consecutive pitches.")
    parser.add_argument("--n_sky_steps", type=int, default=2,
                        help="Sky steps from base pitch (default 2).")
   
    parser.add_argument("--height_offsets", type=str, default=None,
                        help="Optional vertical offsets, e.g. '0,1.5,-1.5'.")
    parser.add_argument("--fov_y_deg", type=float, default=None)


    parser.add_argument(
        "--scene_render_mode",
        type=str,
        default="bg_only",
        choices=["composite", "gs_only", "bg_only"],
        help="Render mode for scene pitch. Default: bg_only.",
    )
    parser.add_argument(
        "--sky_render_mode",
        type=str,
        default="bg_only",
        choices=["composite", "gs_only", "bg_only"],
        help="Render mode for sky pitches. Default: bg_only.",
    )
    parser.add_argument(
        "--down_render_mode",
        type=str,
        default="gs_only",
        choices=["composite", "gs_only", "bg_only"],
        help="Render mode for down pitches (default: gs_only.",
    )
    parser.add_argument(
        "--remove_mode",
        type=str,
        default="lookback",
        choices=["lookback", "band", "none"],
        help="Artefact removal probes: 'lookback' (default) = regularized "
             "circle looking outward at --remove_lookback_pitch_deg (≈-150); "
             "'band' = sky-role band pitches; 'none' = skip removal.",
    )

    parser.add_argument(
        "--remove_roles",
        type=str,
        default="sky",
        help="For --remove_mode band: which roles probe removal — "
             "'sky' (default), 'all', 'none', or e.g. 'sky_1,sky_top'.",
    )
    parser.add_argument(
        "--save_training_ply",
        action="store_true",
        default=True,
        help="Drop Gaussians visible from remove probes and save "
             "gaussians_for_training.ply (default: on).",
    )

    parser.add_argument("--sphere_radius", type=float, default=None)
    parser.add_argument("--panorama_yaw_deg", type=float, default=0.0)
    parser.add_argument("--panorama_flip_horizontal", action="store_true")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height_px", type=int, default=None)
    parser.add_argument("--max_views", type=int, default=50)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=42)

    parser.add_argument(
        "--role_tags",
        action="store_true",
        default=True,
    )

    parser.add_argument(
        "--base_pitch_json", type=str, default=None,
        help="base_pitches.json from estimate_base_pitches.py "
             "(default: next to --bg_panorama).",
    )

    parser.add_argument(
        "--pitch_pos_offset_deg", type=float, default=5.0,
        help="Added to pitch_pos for the first (down) pitch when using "
             "--base_pitch_json (default 5).",
    )
    
    args = parser.parse_args()
    if not args.model_path and not args.ply_path:
        parser.error("Provide at least one of --model_path or --ply_path")
    if args.n_sky_steps < 0:
        parser.error("--n_sky_steps must be >= 0")

    if args.base_pitch_json is None:
        args.base_pitch_json = str(Path(args.bg_panorama).parent / "base_pitches.json")
    if not Path(args.base_pitch_json).is_file():
        parser.error(
            f"base pitches not found: {args.base_pitch_json} "
            "(run estimate_base_pitches.py first, or pass --base_pitch_json)"
        )

    args.cameras_json = default_cameras_json(args.model_path, args.cameras_json)
    if args.base_pitch_json is None:
        candidate = Path(args.bg_panorama).parent / "base_pitches.json"
        if candidate.is_file():
            args.base_pitch_json = str(candidate)
    
   
    run(
        model_path=args.model_path,
        ply_path=args.ply_path,
        bg_panorama=args.bg_panorama,
        cameras_json=args.cameras_json,
        output_path=args.output_path,
        iteration=args.iteration,
        regularize_circle=args.regularize_circle,
        circle_radius=args.circle_radius,
        circle_height=args.circle_height,
        circle_start_azimuth_deg=args.circle_start_azimuth_deg,
        base_pitch_deg=args.base_pitch_deg,
        base_pitch_json=args.base_pitch_json,
        pitch_pos_offset_deg=args.pitch_pos_offset_deg,
        seam_overlap_deg=args.seam_overlap_deg,
        n_sky_steps=args.n_sky_steps,
        sky_pitch_sign=args.sky_pitch_sign,
        height_offsets=args.height_offsets,
        fov_y_deg=args.fov_y_deg,
        scene_render_mode=args.scene_render_mode,
        sky_render_mode=args.sky_render_mode,
        down_render_mode=args.down_render_mode,
        remove_mode=args.remove_mode,
        remove_roles=args.remove_roles,
        save_training_ply=args.save_training_ply,
        sphere_radius=args.sphere_radius,
        panorama_yaw_deg=args.panorama_yaw_deg,
        panorama_flip_horizontal=args.panorama_flip_horizontal,
        img_w=args.width,
        img_h=args.height_px,
        max_views=args.max_views,
        test_ratio=args.test_ratio,
        split_seed=args.split_seed,
        role_tags=args.role_tags,
    )


if __name__ == "__main__":
    main()


    



