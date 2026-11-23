from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

from lib.alignment import ( 
    CANONICAL_UP,
    align_loaded_cameras,
    apply_alignment_to_gaussians,
)
from lib.band import fov_y_deg_from_cam  
from lib.cameras import (  
    load_cameras_from_json, default_cameras_json,
    regularize_cameras_on_circle,
    subsample_cameras,
)
from lib.gaussians import gaussians_to_cuda, pipeline, resolve_ply 
from lib.panorama_io import load_panorama_sidecar 
from lib.pitch_search import estimate_scene_aware_base_pitches 
from scene import GaussianModel  


def build_pitch_json_payload(
    result: dict,
    *,
    ply_path: str,
    cameras_json: str,
    bg_panorama: str,
    circle_radius: float | None,
    circle_height: float | None,
    max_views: int | None,
    n_base_cams: int,
    pano_target: list,
) -> dict:
    """Flatten estimate result into a reviewable / reloadable JSON."""
    pitch_pos = float(result["pitch_pos_deg"])
    lookback = float(result["lookback_pitch_deg"])
    gap = float(result.get("outpaint_gap_deg", 0.0))
    fov_y = float(result["fov_y_deg"])

    return {
        # --- fields consumed by render_camera_band.py ---
        "pitch_pos_deg": pitch_pos,
        "lookback_pitch_deg": lookback,
        "outpaint_gap_deg": gap,
        "fov_y_deg": fov_y,
        # --- kiss check (option 2 / up-behind) ---
        "kiss_lhs_deg": abs(lookback),
        "kiss_rhs_deg": pitch_pos + 180.0 - fov_y - gap,
        "base_neg_mode": "lookback",
        "method": "scene",
        # --- diagnostics ---
        "fallback": bool(result.get("fallback", False)),
        "best_score": result.get("best_score"),
        "scores": result.get("scores", []),
        "n_probe_cams": result.get("n_probe_cams"),
        "scene_dist_percentile": result.get("scene_dist_percentile"),
        "opacity_thresh": result.get("opacity_thresh"),
        "junk_weight": result.get("junk_weight"),
        # --- provenance ---
        "inputs": {
            "ply_path": ply_path,
            "cameras_json": cameras_json,
            "bg_panorama": bg_panorama,
            "circle_radius": circle_radius,
            "circle_height": circle_height,
            "max_views": max_views,
            "n_base_cams": n_base_cams,
            "pano_target_aligned": pano_target,
        },
    }


def run_estimate(
    ply_path: str = None,
    model_path: str = None,
    bg_panorama: str = None,
    cameras_json: str = None,
    output_json: str = None,
    iteration: int = 30_000,
    circle_radius: float = None,
    circle_height: float = None,
    circle_start_azimuth_deg: float = None,
    max_views: int = None,
    img_w: int = None,
    img_h: int = None,
    outpaint_gap_deg: float = 0.0,
    pitch_search_cams: int = 6,
    pitch_step_deg: float = 1.0,
    scene_dist_percentile: float = 95.0,
    opacity_thresh: float = 0.05,
    pitch_junk_weight: float = 0.3,
    render_scale: float = 0.25,
) -> dict:
    if cameras_json is None:
        raise ValueError("--cameras_json is required")
    if bg_panorama is None:
        raise ValueError("--bg_panorama is required (for sidecar eye/target + R_align)")
    if model_path is None and ply_path is None:
        raise ValueError("Provide --ply_path or --model_path")

    cam_json_path = Path(cameras_json).resolve()
    pano_path = Path(bg_panorama).resolve()
    out_path = (
        Path(output_json).resolve()
        if output_json
        else pano_path.parent / "base_pitches.json"
    )

    print(f"\n{'=' * 60}")
    print("  Estimate scene-aware base pitches")
    print(f"  panorama : {pano_path}")
    print(f"  cameras  : {cam_json_path}")
    print(f"  output   : {out_path}")
    print(f"{'=' * 60}\n")

    print("Step 1 — panorama sidecar")
    sidecar = load_panorama_sidecar(pano_path)
    R_align = sidecar["R_align"]
    pano_target = sidecar["target"]

    print("\nStep 2 — cameras → align → circle")
    raw_cams = load_cameras_from_json(
        str(cam_json_path), img_w_override=img_w, img_h_override=img_h,
    )
    aligned_cams = align_loaded_cameras(raw_cams, R_align)
    circle_cams = regularize_cameras_on_circle(
        aligned_cams,
        pano_target,
        CANONICAL_UP,
        radius=circle_radius,
        height=circle_height,
        start_azimuth_deg=circle_start_azimuth_deg,
    )
    circle_cams = subsample_cameras(circle_cams, max_views)
    base_circle_cams = [dict(c) for c in circle_cams]
    fov_y = fov_y_deg_from_cam(base_circle_cams[0])
    print(f"  FoVy={fov_y:.2f}°  n_base={len(base_circle_cams)}")

    print("\nStep 3 — load + align Gaussians")
    gaussians = GaussianModel(sh_degree=3)
    ckpt_path = resolve_ply(model_path, iteration, ply_path)
    gaussians.load_ply(str(ckpt_path))
    gaussians_to_cuda(gaussians)
    apply_alignment_to_gaussians(gaussians, R_align)

    print("\nStep 4 — scene-aware pitch search")
    pipe = pipeline()
    result = estimate_scene_aware_base_pitches(
        gaussians,
        base_circle_cams,
        pano_target,
        CANONICAL_UP,
        pipe,
        fov_y_deg=fov_y,
        outpaint_gap_deg=float(outpaint_gap_deg),
        pitch_step_deg=float(pitch_step_deg),
        n_probe_cams=int(pitch_search_cams),
        scene_dist_percentile=float(scene_dist_percentile),
        opacity_thresh=float(opacity_thresh),
        junk_weight=float(pitch_junk_weight),
        render_scale=float(render_scale),
    )

    payload = build_pitch_json_payload(
        result,
        ply_path=str(ckpt_path),
        cameras_json=str(cam_json_path),
        bg_panorama=str(pano_path),
        circle_radius=circle_radius,
        circle_height=circle_height,
        max_views=max_views,
        n_base_cams=len(base_circle_cams),
        pano_target=np.asarray(pano_target, dtype=np.float64).reshape(3).tolist(),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  pitch_pos_deg      = {payload['pitch_pos_deg']:.4f}")
    print(f"  lookback_pitch_deg = {payload['lookback_pitch_deg']:.4f}")
    print(f"  outpaint_gap_deg   = {payload['outpaint_gap_deg']:.4f}")
    print(f"  FoVy               = {payload['fov_y_deg']:.4f}")
    print(
        f"  kiss |lookback|    = {payload['kiss_lhs_deg']:.4f} "
        f"== {payload['kiss_rhs_deg']:.4f} ? "
        f"{abs(payload['kiss_lhs_deg'] - payload['kiss_rhs_deg']) < 1e-3}"
    )
    print(f"  ✓ wrote {out_path}")
    print(f"{'=' * 60}\n")
    print("Inspect the JSON, then pass it to render:")
    print(
        f"  python vr_sky/utils/render_camera_band.py \\\n"
        f"      ... \\\n"
        f"      --base_pitch_json {out_path} \\\n"
        f"      --remove_mode lookback \\\n"
        f"      --down_render_mode gs_only \\\n"
        f"      --sky_render_mode composite"
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return payload


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--ply_path", type=str, default=None)
    parser.add_argument(
        "--bg_panorama", type=str, required=True,
        help="Equirect + sidecar <stem>_eye.json (for R_align / target).",
    )
    parser.add_argument("--cameras_json", type=str, default=None)
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Where to write base_pitches.json (default: next to cameras.json).",
    )
    parser.add_argument("--iteration", type=int, default=30_000)
    parser.add_argument("--circle_radius", type=float, default=None)
    parser.add_argument("--circle_height", type=float, default=None)
    parser.add_argument("--circle_start_azimuth_deg", type=float, default=None)
    parser.add_argument(
       "--max_views", type=int, default=16,
        help="Subsample base circle (same as render).",
    )
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height_px", type=int, default=None)
    parser.add_argument("--outpaint_gap_deg", type=float, default=0.0)
    parser.add_argument("--pitch_search_cams", type=int, default=6)
    parser.add_argument("--pitch_step_deg", type=float, default=1.0)
    parser.add_argument("--scene_dist_percentile", type=float, default=95.0)
    parser.add_argument("--opacity_thresh", type=float, default=0.05)
    parser.add_argument("--pitch_junk_weight", type=float, default=0.3)
    parser.add_argument(
        "--render_scale",
        type=float,
        default=0.25,
        help="Probe render resolution scale (FoV preserved). Default 0.25.",
    )
    args = parser.parse_args()

    if not args.model_path and not args.ply_path:
        parser.error("Provide at least one of --model_path or --ply_path")
    
    args.cameras_json = default_cameras_json(args.model_path, args.cameras_json)

    run_estimate(
        ply_path=args.ply_path,
        model_path=args.model_path,
        bg_panorama=args.bg_panorama,
        cameras_json=args.cameras_json,
        output_json=args.output_json,
        iteration=args.iteration,
        circle_radius=args.circle_radius,
        circle_height=args.circle_height,
        circle_start_azimuth_deg=args.circle_start_azimuth_deg,
        max_views=args.max_views,
        img_w=args.width,
        img_h=args.height_px,
        outpaint_gap_deg=args.outpaint_gap_deg,
        pitch_search_cams=args.pitch_search_cams,
        pitch_step_deg=args.pitch_step_deg,
        scene_dist_percentile=args.scene_dist_percentile,
        opacity_thresh=args.opacity_thresh,
        pitch_junk_weight=args.pitch_junk_weight,
        render_scale=args.render_scale,
    )


if __name__ == "__main__":
    main()
