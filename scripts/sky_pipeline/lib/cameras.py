"""Camera loading, construction, and circle regularization."""

from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np

def default_cameras_json(model_path, cameras_json):
    """<model_path>/cameras.json unless given explicitly."""
    if cameras_json:
        return cameras_json
    if not model_path:
        raise ValueError("Provide --cameras_json or --model_path")
    return str(Path(model_path) / "cameras.json")


def load_cameras_from_json(
    json_path: str,
    img_w_override: int | None = None,
    img_h_override: int | None = None,
) -> list:
    """Load camera poses from a 3DGS cameras.json file into dicts with ``c2w``."""
    with open(json_path) as f:
        data = json.load(f)

    cameras = []
    for cam in data:
        R = np.array(cam["rotation"], dtype=np.float64)
        pos = np.array(cam["position"], dtype=np.float64)
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = R
        c2w[:3, 3] = pos

        src_w, src_h = int(cam["width"]), int(cam["height"])
        fx, fy = float(cam["fx"]), float(cam["fy"])
        out_w = img_w_override if img_w_override is not None else src_w
        out_h = img_h_override if img_h_override is not None else src_h
        if out_w != src_w or out_h != src_h:
            fx = fx * out_w / src_w
            fy = fy * out_h / src_h

        cameras.append(dict(
            c2w=c2w, fx=fx, fy=fy, width=out_w, height=out_h,
            img_name=cam["img_name"], id=cam["id"],
        ))

    print(f"  Loaded {len(cameras)} cameras from {json_path}")
    return cameras


def load_cameras_json(json_path: str) -> list:
    """Load cameras.json as a list of 4×4 c2w matrices (panorama CLI)."""
    with open(json_path) as f:
        data = json.load(f)

    c2w_list = []
    for cam in data:
        R = np.array(cam["rotation"], dtype=np.float64)
        pos = np.array(cam["position"], dtype=np.float64)
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = R
        c2w[:3, 3] = pos
        c2w_list.append(c2w)

    print(f"  Loaded {len(c2w_list)} cameras from {json_path}")
    return c2w_list


def save_loaded_cameras_json(loaded_cams: list, json_path: str) -> None:
    """Write loaded camera dicts back to a 3DGS-style cameras.json."""
    records = []
    for cam in loaded_cams:
        c2w = cam["c2w"]
        records.append({
            "id": cam["id"],
            "img_name": cam["img_name"],
            "width": cam["width"],
            "height": cam["height"],
            "fx": float(cam["fx"]),
            "fy": float(cam["fy"]),
            "position": c2w[:3, 3].tolist(),
            "rotation": c2w[:3, :3].tolist(),
        })
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"  ✓ Saved {len(records)} cameras → {json_path}")


def look_at_c2w(
    eye: np.ndarray,
    target: np.ndarray,
    world_up: np.ndarray = np.array([0.0, 1.0, 0.0]),
) -> np.ndarray:
    """
    Build a COLMAP / 3DGS camera-to-world matrix looking from eye toward target.

    Convention (matches cameras.json + scene.cameras.Camera):
      +X right, +Y down, +Z forward (toward the look-at point).
    """
    forward = target - eye
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-8:
        raise ValueError(f"eye and target are the same point: {eye}")
    forward = forward / forward_norm

    w_up = np.asarray(world_up, dtype=np.float64).reshape(3)
    w_up = w_up / (np.linalg.norm(w_up) + 1e-12)
    if abs(np.dot(forward, w_up)) > 0.999:
        w_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    # OpenGL-style orthonormal frame first, then convert to COLMAP by flipping Y,Z.
    right = np.cross(forward, w_up)
    right /= np.linalg.norm(right) + 1e-12
    cam_up = np.cross(right, forward)
    cam_up /= np.linalg.norm(cam_up) + 1e-12

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = right
    c2w[:3, 1] = -cam_up   # COLMAP Y-down
    c2w[:3, 2] = forward   # COLMAP +Z look
    c2w[:3, 3] = eye
    return c2w


def pitch_c2w_toward_world_up(
    c2w: np.ndarray,
    world_up: np.ndarray,
    pitch_deg: float,
) -> np.ndarray:
    """Pitch a COLMAP c2w about the right axis toward ``world_up``."""
    if abs(pitch_deg) < 1e-6:
        return c2w.copy()

    pos = c2w[:3, 3].astype(np.float64)
    w_up = np.asarray(world_up, dtype=np.float64).reshape(3)
    w_up = w_up / (np.linalg.norm(w_up) + 1e-12)
    Rm = c2w[:3, :3].astype(np.float64)

    # COLMAP: look direction is +Z column
    fwd = Rm[:, 2]
    fn = np.linalg.norm(fwd)
    if fn < 1e-8:
        return c2w.copy()
    fwd = fwd / fn

    right = np.cross(fwd, w_up)
    rn = np.linalg.norm(right)
    right = right / rn if rn >= 1e-8 else np.array([1.0, 0.0, 0.0])

    rad = math.radians(float(pitch_deg))
    c, s = math.cos(rad), math.sin(rad)
    fwd_new = (
        fwd * c
        + np.cross(right, fwd) * s
        + right * np.dot(right, fwd) * (1.0 - c)
    )
    fn2 = np.linalg.norm(fwd_new)
    if fn2 < 1e-8:
        return c2w.copy()
    return look_at_c2w(pos, pos + fwd_new / fn2, world_up=w_up)


def target_from_camera_centroid(c2w_list: list) -> np.ndarray:
    """Mean of camera positions from a list of 4×4 c2w matrices."""
    positions = np.stack([c2w[:3, 3] for c2w in c2w_list], axis=0)
    target = positions.mean(axis=0)
    print(f"  Target (camera centroid): {target.tolist()}")
    return target


def subsample_cameras(
    loaded_cams: list,
    max_views: int | None,
    seed: int = 42,
) -> list:
    """Keep at most max_views cameras, evenly spaced by index order."""
    del seed  # even spacing; kept for API compatibility
    if max_views is None or max_views <= 0 or len(loaded_cams) <= max_views:
        return loaded_cams

    n = len(loaded_cams)
    idxs = np.linspace(0, n - 1, num=int(max_views), dtype=int)
    out = [loaded_cams[int(i)] for i in idxs]
    for i, c in enumerate(out):
        c = dict(c)
        c["id"] = i
        out[i] = c
    print(f"  Subsampled cameras: {n} → {len(out)} (--max_views={max_views})")
    return out


def _plane_basis(world_up: np.ndarray) -> tuple:
    """Orthonormal (u, v, up) spanning the horizontal plane."""
    up = np.asarray(world_up, dtype=np.float64).reshape(3)
    up = up / (np.linalg.norm(up) + 1e-12)
    ref = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    u = np.cross(up, ref)
    u /= np.linalg.norm(u) + 1e-12
    v = np.cross(up, u)
    return u, v, up


def _azimuth_in_plane(rel: np.ndarray, u: np.ndarray, v: np.ndarray) -> float:
    return float(math.atan2(np.dot(rel, v), np.dot(rel, u)))


def regularize_cameras_on_circle(
    loaded_cams: list,
    target_pt: np.ndarray,
    world_up: np.ndarray,
    radius: float | None = None,
    height: float | None = None,
    start_azimuth_deg: float | None = None,
) -> list:
    """
    Replace camera positions with N evenly spaced points on a circle around
    target_pt (plane ⊥ world_up), all looking at target.

    Radius / height default to median horizontal distance / height of inputs.
    """
    if not loaded_cams:
        raise ValueError("regularize_cameras_on_circle: empty camera list")

    target = np.asarray(target_pt, dtype=np.float64).reshape(3)
    u, v, up = _plane_basis(world_up)
    n = len(loaded_cams)

    rel = np.stack([cam["c2w"][:3, 3] - target for cam in loaded_cams], axis=0)
    heights = rel @ up
    in_plane = rel - np.outer(heights, up)
    radii = np.linalg.norm(in_plane, axis=1)

    use_radius = float(np.median(radii)) if radius is None else float(radius)
    use_height = float(np.median(heights)) if height is None else float(height)

    azimuths = np.array([_azimuth_in_plane(rel[i], u, v) for i in range(n)])
    order = np.argsort(azimuths)
    if start_azimuth_deg is None:
        start_az = float(azimuths[order[0]])
    else:
        start_az = math.radians(float(start_azimuth_deg))

    print(
        f"\n  [circle] n={n}  radius={use_radius:.4f} "
        f"(input median, range [{radii.min():.4f}, {radii.max():.4f}])"
    )
    print(
        f"  [circle] height={use_height:.4f} "
        f"(input median, range [{heights.min():.4f}, {heights.max():.4f}])"
    )
    print(
        f"  [circle] start_azimuth={math.degrees(start_az):.2f}°  "
        f"step={360.0 / n:.2f}°"
    )

    out = []
    for i, src_idx in enumerate(order):
        theta = start_az + 2.0 * math.pi * i / n
        pos = (
            target
            + use_radius * (math.cos(theta) * u + math.sin(theta) * v)
            + use_height * up
        )
        c2w = look_at_c2w(pos, target, world_up=up)
        src = loaded_cams[src_idx]
        out.append(dict(
            c2w=c2w,
            fx=src["fx"],
            fy=src["fy"],
            width=src["width"],
            height=src["height"],
            img_name=src["img_name"],
            id=src["id"],
        ))
    return out
