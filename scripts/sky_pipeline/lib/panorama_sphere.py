"""Panorama sphere projection and compositing over Gaussians."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from renderer.gaussian_renderer import render


def _pick_up(forward, world_up=np.array([0.0, 1.0, 0.0], dtype=np.float32)):
    return (
        np.array([0.0, 0.0, 1.0], dtype=np.float32)
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


class _RadiusMiniCam:
    """Minimal camera for sphere-radius visibility rasterization."""

    def __init__(self, width, height, fovx, fovy, znear, zfar, world_view_np):
        self.image_width = width
        self.image_height = height
        self.FoVx = fovx
        self.FoVy = fovy

        self.world_view_transform = torch.tensor(
            world_view_np, dtype=torch.float32, device="cuda",
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
        self.camera_center = torch.inverse(self.world_view_transform)[3, :3]


_CUBE_DIRS = {
    "+x": np.array([1, 0, 0.0], dtype=np.float32),
    "-x": np.array([-1, 0, 0.0], dtype=np.float32),
    "+y": np.array([0, 1, 0.0], dtype=np.float32),
    "-y": np.array([0, -1, 0.0], dtype=np.float32),
    "+z": np.array([0, 0, 1.0], dtype=np.float32),
    "-z": np.array([0, 0, -1.0], dtype=np.float32),
}


def _get_visible_gaussians(gaussians, cam, image_size):
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

    rasterizer = GaussianRasterizer(raster_settings=settings)
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


def pinhole_rays_scene(
    c2w: np.ndarray,
    fov_x: float,
    img_w: int,
    img_h: int,
    c2w_is_opengl: bool = False,
) -> torch.Tensor:
    """
    Pinhole rays in scene/world space (not unit-length).

    OpenGL dirs_cam; COLMAP c2w needs Y/Z column flip.
    """
    fov_y = 2.0 * math.atan(math.tan(fov_x * 0.5) * img_h / img_w)
    xs = torch.linspace(-1.0, 1.0, img_w, device="cuda")
    ys = torch.linspace(-1.0, 1.0, img_h, device="cuda")
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    dirs_cam = torch.stack([
        grid_x * math.tan(fov_x * 0.5),
        -grid_y * math.tan(fov_y * 0.5),
        -torch.ones_like(grid_x),
    ], dim=-1)

    R_c2w = torch.tensor(c2w[:3, :3], dtype=torch.float32, device="cuda")
    if not c2w_is_opengl:
        R_opengl = R_c2w.clone()
        R_opengl[:, 1] = -R_c2w[:, 1]
        R_opengl[:, 2] = -R_c2w[:, 2]
    else:
        R_opengl = R_c2w
    return dirs_cam @ R_opengl.T


def panorama_sphere_background(
    panorama_tiled: torch.Tensor,
    c2w: np.ndarray,
    pano_origin: np.ndarray,
    sphere_radius: float,
    R_align: np.ndarray,
    fov_x: float,
    img_w: int,
    img_h: int,
    yaw_deg: float = 0.0,
    c2w_is_opengl: bool = False,
) -> torch.Tensor:
    """
    Intersect camera rays with |P - O| = R, take the far hit, map (hit - O) to
    canonical via R_align, then sample equirect UV (tiled seam handling).
    """
    dirs_scene = pinhole_rays_scene(c2w, fov_x, img_w, img_h, c2w_is_opengl)
    dirs = F.normalize(dirs_scene, dim=-1)

    cam_pos = torch.tensor(c2w[:3, 3], dtype=torch.float32, device="cuda")
    center = torch.tensor(
        np.asarray(pano_origin, dtype=np.float64),
        dtype=torch.float32, device="cuda",
    )
    oc = cam_pos - center
    oc = oc.view(1, 1, 3).expand_as(dirs)
    b = (oc * dirs).sum(dim=-1)

    R2 = float(sphere_radius) ** 2
    c_coeff = (oc * oc).sum(dim=-1) - R2
    disc = (b * b - c_coeff).clamp(min=0.0)
    t_far = (-b + torch.sqrt(disc)).clamp(min=1e-6)
    hit = cam_pos.view(1, 1, 3) + dirs * t_far.unsqueeze(-1)

    R_align_t = torch.tensor(R_align, dtype=torch.float32, device="cuda")
    dirs_from_center = F.normalize(hit - center.view(1, 1, 3), dim=-1)
    dirs_canon = F.normalize(dirs_from_center @ R_align_t.T, dim=-1)

    dx, dy, dz = dirs_canon[..., 0], dirs_canon[..., 1], dirs_canon[..., 2]
    lon = torch.atan2(dx, dz)
    lat = torch.asin(dy.clamp(-1.0 + 1e-6, 1.0 - 1e-6))
    u = (lon / (2.0 * math.pi)) % 1.0
    v = 0.5 + lat / math.pi

    if yaw_deg != 0.0:
        u = (u + yaw_deg / 360.0) % 1.0

    u_tiled = (u + 1.0) / 3.0
    gx = u_tiled * 2.0 - 1.0
    gy = v * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)

    return F.grid_sample(
        panorama_tiled.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).squeeze(0)


def composite_panorama_sphere(
    cam_cuda,
    c2w: np.ndarray,
    gaussians,
    pipeline,
    panorama_tiled: torch.Tensor,
    pano_origin: np.ndarray,
    sphere_radius: float,
    R_align: np.ndarray,
    fov_x_rad: float,
    img_w: int,
    img_h: int,
    yaw_deg: float = 0.0,
    c2w_is_opengl: bool = False,
) -> torch.Tensor:
    """Composite Gaussians over the projected panorama sphere."""
    del c2w_is_opengl  # reserved for callers that already baked OpenGL c2w
    bg_black = torch.zeros(3, device="cuda")
    with torch.no_grad():
        pkg = render(cam_cuda, gaussians, pipeline, bg_black)
    r_black = pkg["render"]

    # Prefer rasterizer alpha; fall back to black/white double-render.
    alpha = pkg.get("alpha", pkg.get("opacity_accum", None))
    if alpha is None:
        bg_white = torch.ones(3, device="cuda")
        with torch.no_grad():
            r_white = render(cam_cuda, gaussians, pipeline, bg_white)["render"]
        diff = (r_white - r_black).clamp(0, 1)
        if diff.max().item() < 1e-4:
            h, w = r_black.shape[1], r_black.shape[2]
            alpha = torch.zeros(1, h, w, device="cuda")
        else:
            alpha = (1.0 - diff.mean(0, keepdim=True)).clamp(0, 1)

    bg_img = panorama_sphere_background(
        panorama_tiled, c2w, pano_origin, sphere_radius,
        R_align, fov_x_rad, img_w, img_h, yaw_deg=yaw_deg,
    )
    return (r_black + bg_img * (1.0 - alpha)).clamp(0.0, 1.0)


def estimate_panorama_sphere_radius(
    gaussians,
    eye_position: np.ndarray,
    image_size: int = 512,
    percentile: float = 98.0,
    margin: float = 1.0,
    znear: float = 0.01,
    zfar: float = 1000.0,
) -> float:
    """
    Sphere radius from distances of Gaussians visible in a 6-face cubemap
    rasterized from ``eye_position`` (``percentile`` × ``margin``).
    """
    eye = np.asarray(eye_position, dtype=np.float32)
    fov = np.pi / 2

    visible_global = torch.zeros(
        gaussians.get_xyz.shape[0],
        dtype=torch.bool,
        device="cuda",
    )

    with torch.no_grad():
        for fwd in _CUBE_DIRS.values():
            up = _pick_up(fwd)
            wv = _world_view_transform(eye, fwd, up)
            cam = _RadiusMiniCam(
                image_size, image_size, fov, fov, znear, zfar, wv,
            )
            visible_global |= _get_visible_gaussians(gaussians, cam, image_size)

        xyz = gaussians.get_xyz[visible_global]
        if xyz.shape[0] == 0:
            raise RuntimeError(
                "No Gaussians visible from panorama eye across cubemap faces; "
                "cannot estimate sphere radius."
            )

        cam_center = torch.tensor(eye, dtype=torch.float32, device="cuda")
        distances = torch.norm(xyz - cam_center, dim=1)
        pct_dist = torch.quantile(distances, percentile / 100.0).item()
        radius = pct_dist * margin

    print(
        f"  [sphere_radius] visible={visible_global.sum().item()} "
        f"{percentile:.1f}%={pct_dist:.2f} "
        f"radius={radius:.2f}"
    )
    print("  distance statistics:")
    print(f"    min: {distances.min().item():.2f}")
    print(f"    50%: {torch.quantile(distances, 0.50).item():.2f}")
    print(f"    90%: {torch.quantile(distances, 0.90).item():.2f}")
    print(f"    95%: {torch.quantile(distances, 0.95).item():.2f}")
    print(f"    99%: {torch.quantile(distances, 0.99).item():.2f}")
    print(f"    99.9%: {torch.quantile(distances, 0.999).item():.2f}")
    print(f"    max: {distances.max().item():.2f}")

    return float(radius)
