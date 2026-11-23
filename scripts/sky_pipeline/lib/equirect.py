"""Cubemap → equirectangular panorama render + dual-bg inpaint mask."""

from __future__ import annotations

import math

import cv2
import numpy as np
import torch

from renderer.gaussian_renderer import render
from scene.cameras import MiniCam
from utils.graphics_utils import getProjectionMatrix, getWorld2View2


def build_inpaint_mask_from_dual_render(
    black_img,
    white_img,
    overlap_pixels: int = 0,
    diff_threshold: int = 30,
):
    """
    Inpainting mask from black-bg vs white-bg renders.

    Black = unchangeable (object), white = changeable (background / holes),
    with optional overlap expansion of the black region into white.
    """
    black_gray = cv2.cvtColor(black_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    white_gray = cv2.cvtColor(white_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    diff = cv2.absdiff(black_gray, white_gray)

    _, object_mask = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
    object_mask = object_mask.astype(np.uint8)

    final_mask = cv2.bitwise_not(object_mask)

    if overlap_pixels > 0:
        kernel = np.ones((overlap_pixels, overlap_pixels), np.uint8)
        inv = cv2.bitwise_not(final_mask)
        inv = cv2.dilate(inv, kernel, iterations=1)
        final_mask = cv2.bitwise_not(inv)

    return final_mask


def make_minicam(eye, forward, up, face_size, znear, zfar):
    """90° FoV MiniCam for one cubemap face (COLMAP / 3DGS convention)."""
    forward = np.asarray(forward, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)
    eye = np.asarray(eye, dtype=np.float32)

    forward = forward / (np.linalg.norm(forward) + 1e-8)
    right = np.cross(up, forward)
    right /= np.linalg.norm(right) + 1e-8
    true_up = np.cross(forward, right)
    true_up /= np.linalg.norm(true_up) + 1e-8

    c2w_rot = np.stack([right, true_up, forward], axis=1).astype(np.float32)
    t = (-c2w_rot.T @ eye).astype(np.float32)
    fov = math.pi / 2.0

    world_view_transform = torch.tensor(
        getWorld2View2(c2w_rot, t), dtype=torch.float32, device="cuda",
    ).transpose(0, 1)

    projection_matrix = getProjectionMatrix(
        znear=znear, zfar=zfar, fovX=fov, fovY=fov,
    ).transpose(0, 1).cuda()

    full_proj_transform = (
        world_view_transform.unsqueeze(0)
        .bmm(projection_matrix.unsqueeze(0))
        .squeeze(0)
    )

    return MiniCam(
        face_size, face_size, fov, fov, znear, zfar,
        world_view_transform, full_proj_transform,
    )


def cubemap_face_specs():
    f = np.float32
    return [
        ("nx", np.array([1.0, 0.0, 0.0], dtype=f), np.array([0.0, -1.0, 0.0], dtype=f)),
        ("px", np.array([-1.0, 0.0, 0.0], dtype=f), np.array([0.0, -1.0, 0.0], dtype=f)),
        ("py", np.array([0.0, 1.0, 0.0], dtype=f), np.array([0.0, 0.0, 1.0], dtype=f)),
        ("ny", np.array([0.0, -1.0, 0.0], dtype=f), np.array([0.0, 0.0, -1.0], dtype=f)),
        ("pz", np.array([0.0, 0.0, 1.0], dtype=f), np.array([0.0, -1.0, 0.0], dtype=f)),
        ("nz", np.array([0.0, 0.0, -1.0], dtype=f), np.array([0.0, -1.0, 0.0], dtype=f)),
    ]


def sample_cubemap_vectorized(face_pixels, face_size, dx, dy, dz):
    abs_x = np.abs(dx)
    abs_y = np.abs(dy)
    abs_z = np.abs(dz)

    face_idx = np.zeros(dx.shape, dtype=np.int32)
    sc = np.zeros(dx.shape, dtype=np.float32)
    tc = np.zeros(dx.shape, dtype=np.float32)
    ma = np.zeros(dx.shape, dtype=np.float32)

    mask_x = (abs_x >= abs_y) & (abs_x >= abs_z)

    m = mask_x & (dx > 0)
    face_idx[m] = 0
    ma[m] = abs_x[m]
    sc[m] = -dz[m]
    tc[m] = -dy[m]

    m = mask_x & (dx <= 0)
    face_idx[m] = 1
    ma[m] = abs_x[m]
    sc[m] = dz[m]
    tc[m] = -dy[m]

    mask_y = (~mask_x) & (abs_y >= abs_z)

    m = mask_y & (dy > 0)
    face_idx[m] = 2
    ma[m] = abs_y[m]
    sc[m] = dx[m]
    tc[m] = dz[m]

    m = mask_y & (dy <= 0)
    face_idx[m] = 3
    ma[m] = abs_y[m]
    sc[m] = dx[m]
    tc[m] = -dz[m]

    mask_z = (~mask_x) & (~mask_y)

    m = mask_z & (dz > 0)
    face_idx[m] = 4
    ma[m] = abs_z[m]
    sc[m] = dx[m]
    tc[m] = -dy[m]

    m = mask_z & (dz <= 0)
    face_idx[m] = 5
    ma[m] = abs_z[m]
    sc[m] = -dx[m]
    tc[m] = -dy[m]

    u = np.clip(
        np.floor(0.5 * (sc / (ma + 1e-8) + 1.0) * face_size).astype(np.int32),
        0, face_size - 1,
    )
    v = np.clip(
        np.floor(0.5 * (tc / (ma + 1e-8) + 1.0) * face_size).astype(np.int32),
        0, face_size - 1,
    )

    H, W = dx.shape
    out = np.empty((H, W, 3), dtype=np.uint8)
    out_f = out.reshape(-1, 3)
    fi_flat = face_idx.ravel()
    u_flat = u.ravel()
    v_flat = v.ravel()

    for k, key in enumerate(["px", "nx", "py", "ny", "pz", "nz"]):
        m = fi_flat == k
        if m.any():
            out_f[m] = face_pixels[key][v_flat[m], u_flat[m]]

    return out


def stitch_equirect(face_pixels, out_width: int) -> np.ndarray:
    H = out_width // 2
    W = out_width
    face_size = next(iter(face_pixels.values())).shape[0]

    yy, xx = np.meshgrid(
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing="ij",
    )
    phi = (yy / H) * math.pi
    theta = (xx / W) * 2.0 * math.pi
    sin_phi = np.sin(phi).astype(np.float32)

    dx = (sin_phi * np.sin(theta)).astype(np.float32)
    dy = np.cos(phi).astype(np.float32)
    dz = (sin_phi * np.cos(theta)).astype(np.float32)

    # 3DGS MiniCam / rasterizer uses COLMAP Y-down image space, while the
    # cubemap→equirect sampler assumes OpenGL-style face V. Without this flip
    # the panorama comes out vertically inverted (sky at bottom).
    out = sample_cubemap_vectorized(face_pixels, face_size, dx, dy, dz)
    return np.ascontiguousarray(np.flipud(out))


def render_equirect_panorama(
    gaussians,
    pipeline,
    eye,
    background,
    face_size: int = 512,
    out_width: int = 4096,
    znear: float = 0.15,
    zfar: float = 100.0,
) -> np.ndarray:
    """Render six cubemap faces and stitch to an equirectangular panorama."""
    face_imgs = {}
    with torch.no_grad():
        for face_key, forward, up in cubemap_face_specs():
            cam = make_minicam(eye, forward, up, face_size, znear, zfar)
            cam.znear = znear
            cam.zfar = zfar
            pkg = render(cam, gaussians, pipeline, background)
            img = (
                pkg["render"].clamp(0.0, 1.0)
                .permute(1, 2, 0).cpu().numpy()
            )
            face_imgs[face_key] = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    return stitch_equirect(face_imgs, out_width)
