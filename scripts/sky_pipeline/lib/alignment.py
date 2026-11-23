"""Scene / camera / Gaussian alignment helpers."""

from __future__ import annotations

import numpy as np
import torch
from plyfile import PlyData

CANONICAL_UP = np.array([0.0, 1.0, 0.0], dtype=np.float64)


def rotation_align_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rodrigues rotation that maps unit vector ``a`` onto unit vector ``b``."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a / (np.linalg.norm(a) + 1e-15)
    b = b / (np.linalg.norm(b) + 1e-15)

    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)

    if s < 1e-10:
        if c > 0:
            return np.eye(3)
        t = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(t) < 1e-6:
            t = np.cross(a, np.array([0.0, 1.0, 0.0]))
        t = t / np.linalg.norm(t)
        return -np.eye(3) + 2.0 * np.outer(t, t)

    vx = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def compute_scene_alignment(
    ply_path,
    target_up_vec: np.ndarray | None = None,
) -> np.ndarray:
    """
    PCA-based up alignment: map the thinnest principal axis onto ``target_up``.

    Returns a 3×3 rotation (identity if tilt < 2°).
    """
    if target_up_vec is None:
        target_up_vec = CANONICAL_UP.copy()

    tgt_up = np.asarray(target_up_vec, dtype=np.float64).reshape(3)
    tgt_up = tgt_up / (np.linalg.norm(tgt_up) + 1e-15)

    ply = PlyData.read(str(ply_path))
    v = ply["vertex"].data
    xyz = np.stack([
        v["x"].astype(np.float64),
        v["y"].astype(np.float64),
        v["z"].astype(np.float64),
    ], axis=1)

    Xc = xyz - xyz.mean(axis=0)
    cov = (Xc.T @ Xc) / max(xyz.shape[0] - 1, 1)
    _evals, evecs = np.linalg.eigh(cov)

    v_up = evecs[:, 0].copy()
    if np.dot(v_up, tgt_up) < 0:
        v_up = -v_up

    tilt_deg = float(np.degrees(
        np.arccos(np.clip(np.dot(v_up, tgt_up), -1.0, 1.0))
    ))
    print(f"  [Alignment] scene up: {v_up.tolist()}")
    print(f"  [Alignment] tilt: {tilt_deg:.3f} deg")

    if tilt_deg < 2.0:
        print("  [Alignment] no correction needed.")
        return np.eye(3)

    R_align = rotation_align_a_to_b(v_up, tgt_up)
    print("  [Alignment] applying correction.")
    return R_align


def apply_alignment_to_cameras(c2w_list: list, R_align: np.ndarray) -> list:
    """Rotate camera-to-world matrices into the aligned frame (in-memory only)."""
    if np.allclose(R_align, np.eye(3), atol=1e-8):
        return c2w_list

    aligned = []
    for c2w in c2w_list:
        new_c2w = c2w.copy()
        new_c2w[:3, 3] = R_align @ c2w[:3, 3]
        new_c2w[:3, :3] = R_align @ c2w[:3, :3]
        aligned.append(new_c2w)
    return aligned


def align_loaded_cameras(loaded_cams: list, R_align: np.ndarray) -> list:
    """Rotate loaded camera dicts (with ``c2w``) into the aligned frame."""
    if np.allclose(R_align, np.eye(3), atol=1e-8):
        return loaded_cams

    aligned = []
    for cam in loaded_cams:
        c2w = cam["c2w"].copy()
        c2w[:3, 3] = R_align @ c2w[:3, 3]
        c2w[:3, :3] = R_align @ c2w[:3, :3]
        aligned.append({**cam, "c2w": c2w})
    print(f"  Rotated {len(aligned)} cameras by R_align.")
    return aligned


def unalign_point(p: np.ndarray, R_align: np.ndarray) -> np.ndarray:
    """Map aligned-frame point → original frame (R_align: original→aligned)."""
    R = np.asarray(R_align, dtype=np.float64).reshape(3, 3)
    return R.T @ np.asarray(p, dtype=np.float64).reshape(3)


def unalign_c2w(c2w: np.ndarray, R_align: np.ndarray) -> np.ndarray:
    """Map aligned-frame c2w → original frame."""
    if np.allclose(R_align, np.eye(3), atol=1e-8):
        return c2w.copy()
    R = np.asarray(R_align, dtype=np.float64).reshape(3, 3)
    out = c2w.copy()
    out[:3, :3] = R.T @ c2w[:3, :3]
    out[:3, 3] = R.T @ c2w[:3, 3]
    return out


def _matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    m = R
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z], dtype=np.float32)
    return q / (np.linalg.norm(q) + 1e-12)


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


def apply_alignment_to_gaussians(gaussians, R_align: np.ndarray) -> None:
    """In-place rotate Gaussian means and orientations by ``R_align``."""
    if np.allclose(R_align, np.eye(3), atol=1e-8):
        print("  [Alignment] identity transform.")
        return

    print("  [Alignment] rotating Gaussian scene...")
    R_align_torch = torch.tensor(
        R_align, dtype=torch.float32, device=gaussians._xyz.device,
    )

    with torch.no_grad():
        gaussians._xyz = gaussians._xyz @ R_align_torch.T

        q_align = torch.tensor(
            _matrix_to_quaternion(R_align),
            dtype=torch.float32,
            device=gaussians._rotation.device,
        )
        q_align = q_align.unsqueeze(0).expand_as(gaussians._rotation)
        gaussians._rotation = _quat_mul(q_align, gaussians._rotation)
        gaussians._rotation = (
            gaussians._rotation
            / (gaussians._rotation.norm(dim=-1, keepdim=True) + 1e-12)
        )

    print("  [Alignment] scene rotated.")
