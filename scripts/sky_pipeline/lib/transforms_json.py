"""Train/test split and Nerfstudio-style transforms JSON builders."""

from __future__ import annotations

import numpy as np


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
            records.append({
                "split": "test", "idx": test_i, "c2w": c2w, "prefix": prefix,
            })
            test_i += 1
        else:
            records.append({
                "split": "train", "idx": train_i, "c2w": c2w, "prefix": prefix,
            })
            train_i += 1
    return records


def build_transforms_from_records(records: list, fov_x_rad: float) -> dict:
    """Build a Nerfstudio-style transforms dict (OpenGL Y/Z flip on c2w)."""
    if len(records) > 1:
        centers = np.stack(
            [r["c2w"][:3, 3] for r in records], axis=0,
        ).astype(np.float64)
        orbit_center = centers.mean(axis=0)
        rel = centers - orbit_center[None, :]
        rel_n = rel / (np.linalg.norm(rel, axis=1, keepdims=True) + 1e-12)
        rots = []
        for i in range(len(records)):
            j = (i + 1) % len(records)
            c = float(np.clip(np.dot(rel_n[i], rel_n[j]), -1.0, 1.0))
            rots.append(float(np.arccos(c)))
    else:
        rots = [0.0]

    frames = []
    for i, r in enumerate(records):
        c2w_json = r["c2w"].copy()
        c2w_json[:3, 1:3] *= -1  # COLMAP → OpenGL for transforms JSON
        prefix = r.get("prefix", "r")
        frames.append({
            "file_path": f"./{r['split']}/{prefix}_{r['idx']}",
            "rotation": rots[i],
            "transform_matrix": c2w_json.tolist(),
        })
    return {"camera_angle_x": float(fov_x_rad), "frames": frames}
