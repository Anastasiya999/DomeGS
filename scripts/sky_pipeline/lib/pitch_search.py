"""Scene-aware base-pitch search (Strategy 2) for look-back kiss pairing."""

from __future__ import annotations

import numpy as np
import torch

from lib.band import cam_with_pitch, lookback_pitch_from_pos
from lib.gaussians import make_render_camera
from renderer.gaussian_renderer import render
from scene import GaussianModel


def scene_gaussian_mask(
    gaussians: GaussianModel,
    target_pt: np.ndarray,
    dist_percentile: float = 95.0,
    opacity_thresh: float = 0.05,
) -> torch.Tensor:
    """Boolean mask of Gaussians treated as 'scene' (not far floaters)."""
    xyz = gaussians.get_xyz.detach()
    opac = gaussians.get_opacity.detach().reshape(-1)
    target = torch.as_tensor(
        target_pt, dtype=xyz.dtype, device=xyz.device,
    ).reshape(1, 3)
    dists = torch.norm(xyz - target, dim=1)
    r_max = torch.quantile(dists, float(dist_percentile) / 100.0)
    return (opac >= float(opacity_thresh)) & (dists <= r_max)


def probe_subset(cams: list, n_probe: int) -> list:
    if n_probe is None or n_probe <= 0 or len(cams) <= n_probe:
        return cams
    idxs = np.linspace(0, len(cams) - 1, num=int(n_probe), dtype=int)
    return [cams[int(i)] for i in idxs]


def score_pitch_pos_coverage(
    gaussians: GaussianModel,
    probe_cams: list,
    target_pt: np.ndarray,
    world_up: np.ndarray,
    pitch_pos_deg: float,
    pipeline,
    scene_mask: torch.Tensor,
    junk_weight: float = 0.3,
    render_scale: float = 0.25,
) -> dict:
    """
    Score an inward pitch by scene coverage minus junk fraction.

    Renders at ``render_scale`` of native resolution for speed.
    """
    n_scene = int(scene_mask.sum())
    if n_scene == 0:
        return {"score": -1e9, "coverage": 0.0, "junk": 0.0, "n_visible": 0}

    visible = torch.zeros(
        int(gaussians.get_xyz.shape[0]), dtype=torch.bool, device="cuda",
    )
    bg_black = torch.zeros(3, device="cuda")
    scale = max(0.05, min(1.0, float(render_scale)))

    with torch.no_grad():
        for src in probe_cams:
            cam_data = cam_with_pitch(
                src, target_pt, world_up, pitch_pos_deg,
                render_mode="probe", tag="score", cam_id=0,
            )
            cam_data = dict(cam_data)
            cam_data["width"] = max(32, int(round(cam_data["width"] * scale)))
            cam_data["height"] = max(32, int(round(cam_data["height"] * scale)))
            cam_data["fx"] = float(cam_data["fx"]) * scale
            cam_data["fy"] = float(cam_data["fy"]) * scale
            cam = make_render_camera(cam_data, cam_data["c2w"])
            try:
                visible |= render(cam, gaussians, pipeline, bg_black)["radii"] > 0
            finally:
                del cam

    n_vis = int(visible.sum())
    n_scene_vis = int((visible & scene_mask).sum())
    n_junk_vis = int((visible & ~scene_mask).sum())
    coverage = n_scene_vis / max(n_scene, 1)
    junk = n_junk_vis / max(n_vis, 1)
    score = coverage - float(junk_weight) * junk
    return {
        "score": float(score),
        "coverage": float(coverage),
        "junk": float(junk),
        "n_visible": n_vis,
        "n_scene_visible": n_scene_vis,
        "n_scene": n_scene,
    }


def estimate_scene_aware_base_pitches(
    gaussians: GaussianModel,
    base_circle_cams: list,
    target_pt: np.ndarray,
    world_up: np.ndarray,
    pipeline,
    fov_y_deg: float,
    outpaint_gap_deg: float = 0.0,
    pitch_step_deg: float = 1.0,
    n_probe_cams: int = 6,
    scene_dist_percentile: float = 95.0,
    opacity_thresh: float = 0.05,
    junk_weight: float = 0.3,
    render_scale: float = 0.25,
) -> dict:
    """
    Strategy 2: pick ``pitch_pos`` maximizing scene coverage, then lock
    look-back pitch via the option-2 kiss formula:

        lookback = (FoVy + gap) - 180 - pitch_pos
    """
    fov_y = float(fov_y_deg)
    half = 0.5 * fov_y
    gap = float(outpaint_gap_deg)

    # lookback ∈ (-180, 0)  ⇒  FoVy+G-180 < p < FoVy+G
    #TODO(12_09.26)
    p_lo_kiss = fov_y + gap - 180.0 + 1.0
    p_hi_kiss = fov_y + gap - 1.0
    p_lo = max(0.15 * fov_y, 1.0, p_lo_kiss)
    p_hi = min(0.85 * fov_y, p_hi_kiss)

    # p_lo = max(0.15 * fov_y, 1.0)
    # p_hi = min(0.85 * fov_y, fov_y + gap - 1.0)

    if p_hi <= p_lo:
        pitch_pos = half
        lookback = lookback_pitch_from_pos(pitch_pos, fov_y, gap)
        print(
            f"  [auto_pitch scene] search band empty — fallback "
            f"pitch_pos={pitch_pos:.2f} lookback={lookback:.2f}"
        )
        return {
            "pitch_pos_deg": pitch_pos,
            "lookback_pitch_deg": lookback,
            "outpaint_gap_deg": gap,
            "fov_y_deg": fov_y,
            "fallback": True,
            "scores": [],
        }

    scene_mask = scene_gaussian_mask(
        gaussians, target_pt,
        dist_percentile=scene_dist_percentile,
        opacity_thresh=opacity_thresh,
    )
    probes = probe_subset(base_circle_cams, n_probe_cams)
    step = max(0.25, float(pitch_step_deg))
    candidates = list(np.arange(p_lo, p_hi + 0.5 * step, step))
    if all(abs(c - half) > 0.25 for c in candidates) and p_lo <= half <= p_hi:
        candidates.append(half)
        candidates.sort()

    print(f"\n  [auto_pitch scene] FoVy={fov_y:.2f}°  gap={gap:.2f}°")
    print(
        f"    scene mask: {int(scene_mask.sum())}/"
        f"{int(gaussians.get_xyz.shape[0])} "
        f"(opacity≥{opacity_thresh}, dist≤p{scene_dist_percentile:.0f})"
    )
    print(
        f"    probes={len(probes)}  search [{p_lo:.2f}, {p_hi:.2f}] "
        f"step={step}°"
    )

    scores = []
    best = None
    for p in candidates:
        s = score_pitch_pos_coverage(
            gaussians, probes, target_pt, world_up, float(p), pipeline,
            scene_mask, junk_weight=junk_weight, render_scale=render_scale,
        )
        s["pitch_pos_deg"] = float(p)
        scores.append(s)
        if best is None or s["score"] > best["score"]:
            best = s
        print(
            f"    p={p:+6.2f}°  score={s['score']:+.4f}  "
            f"cov={100 * s['coverage']:.1f}%  junk={100 * s['junk']:.1f}%"
        )

    refine_step = max(0.25, step * 0.5)
    refine_lo = max(p_lo, best["pitch_pos_deg"] - step)
    refine_hi = min(p_hi, best["pitch_pos_deg"] + step)
    for p in np.arange(refine_lo, refine_hi + 0.5 * refine_step, refine_step):
        if any(abs(p - s["pitch_pos_deg"]) < 0.1 for s in scores):
            continue
        s = score_pitch_pos_coverage(
            gaussians, probes, target_pt, world_up, float(p), pipeline,
            scene_mask, junk_weight=junk_weight, render_scale=render_scale,
        )
        s["pitch_pos_deg"] = float(p)
        scores.append(s)
        if s["score"] > best["score"]:
            best = s
            print(
                f"    p={p:+6.2f}°  score={s['score']:+.4f}  "
                f"cov={100 * s['coverage']:.1f}%  "
                f"junk={100 * s['junk']:.1f}%  ← refine"
            )

    pitch_pos = float(best["pitch_pos_deg"])
    lookback = lookback_pitch_from_pos(pitch_pos, fov_y, gap)
    kiss_lhs = abs(lookback)
    kiss_rhs = pitch_pos + 180.0 - fov_y - gap
    print(
        f"  [auto_pitch scene] chosen pitch_pos={pitch_pos:.2f}°  "
        f"lookback={lookback:.2f}°  "
        f"(|lookback|={kiss_lhs:.2f} vs pitch_pos+180-FoVy-gap={kiss_rhs:.2f})"
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "pitch_pos_deg": pitch_pos,
        "lookback_pitch_deg": lookback,
        "outpaint_gap_deg": gap,
        "fov_y_deg": fov_y,
        "fallback": False,
        "best_score": best,
        "scores": scores,
        "n_probe_cams": len(probes),
        "scene_dist_percentile": scene_dist_percentile,
        "opacity_thresh": opacity_thresh,
        "junk_weight": junk_weight,
    }
