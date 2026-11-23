#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from utils.general_utils import build_rotation
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def envmap_shell_loss(
    xyz: torch.Tensor,
    center: torch.Tensor,
    r_inner: float,
    r_outer: float,
) -> torch.Tensor:
    """
    EnvMap-GS / two-shell outdoor GS, Eq. (1).

    Soft barrier keeping Gaussian centres inside the spherical shell
    [R_i, R_o] centred at O.  Penalises squared violation distance.
    """
    center = center.reshape(1, 3)
    dist = torch.norm(xyz - center, dim=1)
    ri = float(r_inner)
    ro = float(r_outer)
    violation = torch.clamp(dist - ro, min=0.0) + torch.clamp(ri - dist, min=0.0)
    return (violation ** 2).mean()


def envmap_planarity_loss(
    xyz: torch.Tensor,
    scaling: torch.Tensor,
    rotation: torch.Tensor,
    center: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    EnvMap-GS / two-shell outdoor GS, Eq. (2).

    Discourages radial spikes by aligning each Gaussian's shortest axis with
    the radial direction from the shell centre.  Highly anisotropic splats
    are penalised more; nearly spherical ones are mostly unaffected.
    """
    center = center.reshape(1, 3)
    radial = xyz - center
    radial = radial / torch.norm(radial, dim=1, keepdim=True).clamp(min=eps)

    min_idx = scaling.argmin(dim=1)
    a_local = torch.zeros(
        xyz.shape[0], 3, device=xyz.device, dtype=xyz.dtype,
    )
    a_local[torch.arange(xyz.shape[0], device=xyz.device), min_idx] = 1.0

    rot_m = build_rotation(rotation)
    axis_global = torch.bmm(rot_m, a_local.unsqueeze(-1)).squeeze(-1)
    axis_global = axis_global / torch.norm(
        axis_global, dim=1, keepdim=True,
    ).clamp(min=eps)

    alignment = (axis_global * radial).sum(dim=1).abs()
    s_max = scaling.max(dim=1).values
    s_min = scaling.min(dim=1).values
    penalty = (1.0 - alignment) * (s_max / (s_min + eps))
    return penalty.mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

