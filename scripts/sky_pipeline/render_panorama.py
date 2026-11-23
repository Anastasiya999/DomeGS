from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

from lib.alignment import (
    apply_alignment_to_cameras,
    apply_alignment_to_gaussians,
    compute_scene_alignment,
)
from lib.cameras import load_cameras_json, default_cameras_json, target_from_camera_centroid
from lib.equirect import render_equirect_panorama

from lib.gaussians import (
    filter_gaussians,
    compute_exclusion_radius,
    gaussians_to_cuda,
    pipeline,
    resolve_ply,
    target_from_opacity_centroid,
)
from scene.gaussian_model import GaussianModel


def main(args):
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("\nLoading 3DGS model...")
    gaussians = GaussianModel(sh_degree=3)
    ckpt_path = resolve_ply(
        args.model_path, args.iteration, ply_path=args.ply_path,
    )
    gaussians.load_ply(str(ckpt_path))
    gaussians_to_cuda(gaussians)

    print("\nLoading cameras...")
    c2w_list = load_cameras_json(args.cameras_json)

    print("\nFiltering Gaussians...")
   
    eye = target_from_camera_centroid(c2w_list)

    exclusion_radius = compute_exclusion_radius(gaussians, eye, c2w_list)
    filter_gaussians(
        gaussians,
        eye_position=eye.astype(np.float32),
        exclusion_radius=exclusion_radius,
    )

    R_align = np.eye(3)

    print("\nComputing scene alignment...")
    R_align = compute_scene_alignment(ckpt_path)
    apply_alignment_to_gaussians(gaussians, R_align)

    # Cameras must also be rotated into the aligned frame for target
    # recomputation. We DO NOT save these aligned cameras.
    c2w_list = apply_alignment_to_cameras(c2w_list, R_align)

    # Eye was computed in the original frame; move into aligned frame.
    eye = R_align @ eye

    # Pre-offset target (panorama center) in the same frame as the scene.
    pano_target = eye.copy()

    print("\nComputing panorama eye position...")
   
    eye_sidecar = {
        "exclusion_radius": float(exclusion_radius),
        "eye": [float(v) for v in eye],
        "target": [float(v) for v in pano_target],
        "target_mode": "cameras",
        "eye_up_offset": 0.0,
        "up_axis": args.up_axis,
        "aligned": True,
        "R_align": np.asarray(R_align, dtype=np.float64).tolist(),
        "model_path": str(args.model_path) if args.model_path else str(ckpt_path),
        "ply_path": str(ckpt_path),
        "cameras_json": str(args.cameras_json),
    }
    eye_json_path = Path(out_path).with_suffix("").as_posix() + "_eye.json"
    with open(eye_json_path, "w") as f:
        json.dump(eye_sidecar, f, indent=2)
    print(f"  Saved panorama eye sidecar: {eye_json_path}")
    print(
        f"eye={eye.tolist()}  (target={pano_target.tolist()}, "
        f"axis={args.up_axis})"
    )

    if args.bg_color == "white":
        bg = torch.tensor([1.0, 1.0, 1.0], device="cuda")
    else:
        bg = torch.tensor([0.0, 0.0, 0.0], device="cuda")

    pipe = pipeline()

    # Image.fromarray(render_black).save(out_path)
    print(f"\nSaved panorama: {out_path}")


    print("\nRendering panorama...")
    equirect = render_equirect_panorama(
        gaussians=gaussians,
        pipeline=pipe,
        eye=eye.astype(np.float32),
        background=bg,
        face_size=args.face_size,
        out_width=args.out_width,
        znear=args.znear,
        zfar=args.zfar,
    )
    equirect = np.ascontiguousarray(np.fliplr(equirect))
    Image.fromarray(equirect).save(out_path)
    print(
        f"\nSaved panorama: {out_path} "
        f"({equirect.shape[1]}x{equirect.shape[0]})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="3DGS model folder (or a .ply file).",
    )
    parser.add_argument(
        "--ply_path", type=str, default=None,
        help="Direct path to point_cloud.ply (overrides --model_path lookup).",
    )
    parser.add_argument("--cameras_json", type=str, default=None,
                        help="Default: <model_path>/cameras.json")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Default: <model_path>/panorama.png")
    parser.add_argument("--iteration", type=int, default=30000)
 
    parser.add_argument(
        "--up_axis", type=str, default="y", choices=["x", "y", "z"],
    )

    parser.add_argument("--face_size", type=int, default=1024)
    parser.add_argument("--out_width", type=int, default=8192)
    parser.add_argument("--znear", type=float, default=0.05)
    parser.add_argument("--zfar", type=float, default=100.0)
    parser.add_argument(
        "--bg_color", type=str, default="black", choices=["black", "white"],
    )


    args = parser.parse_args()
    if not args.model_path and not args.ply_path:
        parser.error("Provide at least one of --model_path or --ply_path")
    args.cameras_json = default_cameras_json(args.model_path, args.cameras_json)
    if args.output_path is None:
        args.output_path = str(Path(args.model_path) / "panorama.png")
    main(args)
