"""Sky-outpaint camera band: pitch ladder, roles, expand, lookback probes."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from lib.cameras import look_at_c2w, pitch_c2w_toward_world_up


DEFAULT_UP = np.array([0.0, 1.0, 0.0], dtype=np.float64)


def fov_y_deg_from_cam(cam: dict) -> float:
    return math.degrees(2.0 * math.atan(cam["height"] / (2.0 * cam["fy"])))


def parse_float_list(s: str | None, default: list | None = None) -> list:
    if s is None or not str(s).strip():
        return list(default) if default is not None else []
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]

def build_pitches_from_pitch_pos(
    pitch_pos_deg: float,
    fov_y_deg: float,
    *,
    offset_deg: float = 5.0,
    seam_overlap_deg: float = 5.0,
    n_sky_steps: int = 2,
) -> list:
    """
    Band pitches from ``estimate_base_pitches`` ``pitch_pos``:
      p0 = pitch_pos + offset          # down
      p1 = p0 - FoVy                   # scene (kisses p0; no overlap)
      p2 = p1 - FoVy + overlap         # sky_1
      p3 = p2 - FoVy + overlap         # sky_2
      …
    Returns ``[down, scene, sky…]``. Role assignment uses ``p1`` as base.
    """
    fov_y = float(fov_y_deg)
    if fov_y <= 1e-6:
        raise ValueError(f"FoVy must be positive, got {fov_y}")
    overlap = float(seam_overlap_deg)
    if overlap >= fov_y - 1e-6:
        raise ValueError(
            f"seam_overlap ({overlap:.3f}°) must be < FoVy ({fov_y:.3f}°)"
        )
    p0 = float(pitch_pos_deg) + float(offset_deg)
    p1 = p0 - fov_y
    pitches = [p0, p1]
    p = p1
    for _ in range(max(0, int(n_sky_steps))):
        p = p - fov_y + overlap
        pitches.append(float(p))
    return [float(x) for x in pitches]


def build_pitch_ladder(
    fov_y_deg: float,
    base_pitch_deg: float,
    n_sky_steps: int,
    seam_overlap_deg: float,
    pitches_override: list | None = None,
    top_pitch_deg: float | None = None,
    sky_pitch_sign: float = -1.0,
    n_down_steps: int = 1,
) -> list:
    """
    Build pitch angles for the band.

    Spacing magnitude: |step| = FoVy − seam_overlap.
    ``sky_pitch_sign``: -1 → negative pitches are sky (dataset convention).
    ``n_down_steps`` adds the opposite side. Order: ``[down …, scene, sky …]``.
    """
    if pitches_override:
        out, seen = [], set()
        for p in pitches_override:
            key = round(float(p), 6)
            if key not in seen:
                seen.add(key)
                out.append(float(p))
        return out

    fov_y = float(fov_y_deg)
    if fov_y <= 1e-6:
        raise ValueError(f"FoVy must be positive, got {fov_y}")

    overlap = float(seam_overlap_deg)
    step_mag = fov_y - overlap
    if step_mag <= 1e-6:
        raise ValueError(
            f"Pitch step non-positive ({step_mag:.3f}°): "
            f"FoVy={fov_y:.3f}° seam_overlap={overlap:.3f}°. "
            f"Need seam_overlap < FoVy."
        )

    sign = float(sky_pitch_sign)
    if abs(sign) < 1e-12:
        raise ValueError("sky_pitch_sign must be non-zero (use -1 or +1)")
    sign = -1.0 if sign < 0 else 1.0
    sky_step = sign * step_mag
    down_step = -sky_step

    base = float(base_pitch_deg)
    n_down = max(0, int(n_down_steps))
    down_pitches = [base + k * down_step for k in range(n_down, 0, -1)]

    sky_pitches: list = []
    if top_pitch_deg is not None:
        top = float(top_pitch_deg)
        if (top - base) * sign < -1e-9:
            raise ValueError(
                f"--top_pitch_deg ({top}) is on the wrong side of "
                f"--base_pitch_deg ({base}) for sky_pitch_sign={sign:.0f}. "
                f"For sign=-1, top should be more negative (e.g. -80)."
            )
        p = base + sky_step
        while (p - top) * sign <= 1e-6:
            sky_pitches.append(float(p))
            if abs(p - top) < 1e-6:
                break
            p += sky_step
        if (
            sky_pitches
            and abs(sky_pitches[-1] - top) > 1e-3
            and (top - sky_pitches[-1]) * sign > 0
        ) or (not sky_pitches and abs(top - base) > 1e-6):
            if not sky_pitches or abs(sky_pitches[-1] - top) > 1e-3:
                sky_pitches.append(top)
    else:
        n = max(0, int(n_sky_steps))
        for k in range(1, n + 1):
            sky_pitches.append(base + k * sky_step)

    pitches = list(down_pitches) + [base] + list(sky_pitches)
    out, seen = [], set()
    for p in pitches:
        key = round(float(p), 6)
        if key not in seen:
            seen.add(key)
            out.append(float(p))
    return out


def assign_band_roles(
    pitches_deg: list,
    base_pitch_deg: float,
    sky_pitch_sign: float = -1.0,
) -> list:
    """Role per pitch: ``down*`` | ``scene`` | ``sky*``."""
    sign = -1.0 if float(sky_pitch_sign) < 0 else 1.0
    base = float(base_pitch_deg)
    pitches = [float(p) for p in pitches_deg]

    roles = ["scene"] * len(pitches)
    sky_idxs, down_idxs = [], []
    for i, p in enumerate(pitches):
        d = p - base
        if abs(d) < 1e-6:
            roles[i] = "scene"
        elif d * sign > 0:
            sky_idxs.append(i)
        else:
            down_idxs.append(i)

    for j, i in enumerate(sky_idxs):
        roles[i] = "sky_top" if j == len(sky_idxs) - 1 else f"sky_{j + 1}"
    for j, i in enumerate(down_idxs):
        if len(down_idxs) == 1:
            roles[i] = "down"
        elif j == len(down_idxs) - 1:
            roles[i] = "down_bottom"
        else:
            roles[i] = f"down_{j + 1}"
    return roles


def pitch_role(index: int, n: int) -> str:
    """Deprecated index-only roles; prefer ``assign_band_roles``."""
    if n <= 1:
        return "scene"
    if index == 0:
        return "scene"
    if index == n - 1:
        return "sky_top"
    return f"sky_{index}"


def frustum_overlap_deg(pitch_a: float, pitch_b: float, fov_y_deg: float) -> float:
    """Shared meridian strip between two look-at pitches (degrees)."""
    sep = abs(float(pitch_a) - float(pitch_b))
    return float(fov_y_deg) - sep


def expand_band_cameras(
    base_cams: list,
    target_pt: np.ndarray,
    world_up: np.ndarray,
    pitches_deg: list,
    height_offsets: list | None = None,
    role_tags: bool = True,
    band_roles: list | None = None,
) -> tuple:
    """
    For each (height_offset, pitch), place a look-at-target camera at every
    base-ring position and pitch toward world up.

    Returns (cameras, band_meta_rows).
    """
    up = np.asarray(world_up, dtype=np.float64).reshape(3)
    up = up / (np.linalg.norm(up) + 1e-12)
    target = np.asarray(target_pt, dtype=np.float64).reshape(3)

    heights = [float(h) for h in (height_offsets or [0.0])]
    if not heights:
        heights = [0.0]
    seen_h, uniq_h = set(), []
    for h in heights:
        key = round(h, 6)
        if key not in seen_h:
            seen_h.add(key)
            uniq_h.append(h)
    heights = uniq_h

    pitches = [float(p) for p in pitches_deg]
    n_p = len(pitches)
    if band_roles is None:
        roles = [pitch_role(i, n_p) for i in range(n_p)]
    else:
        if len(band_roles) != n_p:
            raise ValueError(
                f"band_roles length {len(band_roles)} != n_pitches {n_p}"
            )
        roles = [str(r) for r in band_roles]

    out = []
    rows = []
    cam_id = 0

    for h_off in heights:
        h_tag = "" if abs(h_off) < 1e-9 else f"_h{h_off:+.2f}".replace(".", "p")
        for pi, pitch in enumerate(pitches):
            role = roles[pi]
            p_tag = f"_p{pitch:+.1f}".replace(".", "p")
            role_tag = f"_{role}" if role_tags else ""

            for src in base_cams:
                pos = src["c2w"][:3, 3].astype(np.float64) + up * h_off
                c2w = look_at_c2w(pos, target, world_up=up)
                c2w = pitch_c2w_toward_world_up(c2w, up, pitch)

                name = f"{src['img_name']}{h_tag}{role_tag}{p_tag}"
                out.append(dict(
                    c2w=c2w,
                    fx=src["fx"],
                    fy=src["fy"],
                    width=src["width"],
                    height=src["height"],
                    img_name=name,
                    id=cam_id,
                    pitch_deg=pitch,
                    height_offset=h_off,
                    band_role=role,
                    band_pitch_index=pi,
                ))
                rows.append({
                    "id": cam_id,
                    "img_name": name,
                    "pitch_deg": pitch,
                    "height_offset": h_off,
                    "band_role": role,
                    "band_pitch_index": pi,
                    "src_img_name": src["img_name"],
                })
                cam_id += 1

    return out, rows


def save_band_meta(path: Path, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  ✓ Band meta → {path}")


def is_sky_role(role: str | None) -> bool:
    r = str(role or "")
    return r == "sky_top" or r.startswith("sky_")


def is_down_role(role: str | None) -> bool:
    r = str(role or "")
    return r == "down" or r == "down_bottom" or r.startswith("down_")


def cam_with_pitch(
    src: dict,
    target_pt: np.ndarray,
    world_up: np.ndarray,
    pitch_deg: float,
    render_mode: str,
    tag: str,
    cam_id: int,
) -> dict:
    """Look-at target from ``src`` position, apply pitch toward world up."""
    up = np.asarray(world_up, dtype=np.float64).reshape(3)
    up = up / (np.linalg.norm(up) + 1e-12)
    target = np.asarray(target_pt, dtype=np.float64).reshape(3)
    pos = src["c2w"][:3, 3].astype(np.float64)
    c2w = look_at_c2w(pos, target, world_up=up)
    c2w = pitch_c2w_toward_world_up(c2w, up, float(pitch_deg))
    return dict(
        c2w=c2w,
        fx=src["fx"],
        fy=src["fy"],
        width=src["width"],
        height=src["height"],
        img_name=f"{src['img_name']}_{tag}_p{float(pitch_deg):+.1f}",
        id=cam_id,
        render_mode=render_mode,
        pitch_deg=float(pitch_deg),
        ring_tag=tag,
        look_back=False,
    )


def cam_with_lookback_pitch(
    src: dict,
    target_pt: np.ndarray,
    world_up: np.ndarray,
    lookback_pitch_deg: float,
    render_mode: str,
    tag: str,
    cam_id: int,
) -> dict:
    """
    Look *away* from target (180° yaw flip), then pitch toward world up.

    ``lookback_pitch_deg`` is applied on the look-back base pose.
    Typically ≈ -150 for the flipped-render sky complement.
    """
    up = np.asarray(world_up, dtype=np.float64).reshape(3)
    up = up / (np.linalg.norm(up) + 1e-12)
    target = np.asarray(target_pt, dtype=np.float64).reshape(3)
    pos = src["c2w"][:3, 3].astype(np.float64)
    away = pos + (pos - target)
    c2w = look_at_c2w(pos, away, world_up=up)
    c2w = pitch_c2w_toward_world_up(c2w, up, float(lookback_pitch_deg))
    return dict(
        c2w=c2w,
        fx=src["fx"],
        fy=src["fy"],
        width=src["width"],
        height=src["height"],
        img_name=f"{src['img_name']}_{tag}_lb{float(lookback_pitch_deg):+.1f}",
        id=cam_id,
        render_mode=render_mode,
        pitch_deg=float(lookback_pitch_deg),
        ring_tag=tag,
        look_back=True,
    )


def lookback_pitch_from_pos(
    pitch_pos_deg: float,
    fov_y_deg: float,
    outpaint_gap_deg: float = 0.0,
) -> float:
    """
    Look-back pitch for the flipped-render "up behind" complement:

        lookback = (FoVy + gap) - 180 - pitch_pos
    """
    alpha = (
        float(fov_y_deg)
        + float(outpaint_gap_deg)
        - 180.0
        - float(pitch_pos_deg)
    )
    if alpha >= 0.0:
        raise ValueError(
            f"look-back pitch non-negative ({alpha:.2f}°): "
            f"pitch_pos={pitch_pos_deg:.2f} FoVy={fov_y_deg:.2f} "
            f"gap={outpaint_gap_deg:.2f}. "
            f"Need pitch_pos > FoVy + gap - 180."
        )
    if alpha <= -180.0:
        raise ValueError(
            f"look-back pitch too negative ({alpha:.2f}°): "
            f"pitch_pos={pitch_pos_deg:.2f} FoVy={fov_y_deg:.2f} "
            f"gap={outpaint_gap_deg:.2f}."
        )
    return float(alpha)


def load_base_pitch_json(path: str | Path) -> dict:
    """Load pitches written by ``estimate_base_pitches.py``."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"--base_pitch_json / --remove_pitch_json not found: {p}")
    with open(p) as f:
        data = json.load(f)
    if "pitch_pos_deg" not in data or "lookback_pitch_deg" not in data:
        raise ValueError(
            f"{p} must contain 'pitch_pos_deg' and 'lookback_pitch_deg' "
            f"(got keys: {sorted(data.keys())})"
        )
    pitch_pos = float(data["pitch_pos_deg"])
    lookback = float(data["lookback_pitch_deg"])
    gap = float(data.get("outpaint_gap_deg", 0.0))
    print(f"  [base_pitch_json] {p}")
    print(f"    pitch_pos_deg      = {pitch_pos:.4f}")
    print(f"    lookback_pitch_deg = {lookback:.4f}")
    print(f"    outpaint_gap_deg   = {gap:.4f}")
    if "fov_y_deg" in data:
        fov_y = float(data["fov_y_deg"])
        lhs = abs(lookback)
        rhs = pitch_pos + 180.0 - fov_y - gap
        print(
            f"    kiss |lookback|    = {lhs:.4f} vs {rhs:.4f} "
            f"(Δ={abs(lhs - rhs):.4f})"
        )
    return {
        "pitch_pos_deg": pitch_pos,
        "lookback_pitch_deg": lookback,
        "outpaint_gap_deg": gap,
        "fov_y_deg": float(data["fov_y_deg"]) if "fov_y_deg" in data else None,
        "path": str(p.resolve()),
        "raw": data,
    }


def apply_base_pitch_json(
    *,
    base_pitch_json: str | None,
    fov_y_deg: float | None,
    base_pitch_deg: float,
    seam_overlap_deg: float,
    n_sky_steps: int,
    pitch_pos_offset_deg: float,
) -> dict:
    """
    If ``base_pitch_json`` is set, build the pitch-pos band and lookback.

    Returns updated kwargs (pitches string, base_pitch, fov_y, remove_*).
    """
    out = {
        "fov_y_deg": fov_y_deg,
        "base_pitch_deg": float(base_pitch_deg),
        "base_pitch_meta": None,
    }
    if not base_pitch_json:
        return out

    data = load_base_pitch_json(base_pitch_json)
    out["base_pitch_meta"] = data
    fov_y = (
        float(fov_y_deg)
        if fov_y_deg is not None
        else data.get("fov_y_deg")
    )
    if fov_y is None:
        raise ValueError(
            f"{base_pitch_json} has no fov_y_deg; pass --fov_y_deg"
        )
    out["fov_y_deg"] = float(fov_y)

    band = build_pitches_from_pitch_pos(
        data["pitch_pos_deg"],
        float(fov_y),
        offset_deg=float(pitch_pos_offset_deg),
        seam_overlap_deg=float(seam_overlap_deg),
        n_sky_steps=int(n_sky_steps),
    )
    # Second pitch is the scene / role base (p0 is down).
    out["pitches"] = ",".join(f"{p:.10g}" for p in band)
    out["base_pitch_deg"] = float(band[1]) if len(band) > 1 else float(band[0])
    print(
        f"  [base_pitch_json] band pitches from pitch_pos+"
        f"{float(pitch_pos_offset_deg):.1f}°: "
        f"{[round(p, 3) for p in band]}"
    )
    print(f"  [base_pitch_json] role base (scene) = {out['base_pitch_deg']:+.3f}°")
    
    out["remove_lookback_pitch_deg"] = float(data["lookback_pitch_deg"])
    print(
            f"  [base_pitch_json] remove_lookback_pitch_deg = "
            f"{out['remove_lookback_pitch_deg']:+.4f}°"
        )
    return out
