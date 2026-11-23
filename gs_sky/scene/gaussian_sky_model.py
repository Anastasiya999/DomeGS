import json

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.gaussian_model import GaussianModel

class GaussianSkyModel(GaussianModel):

    def __init__(self, sh_degree: int):
        super().__init__(sh_degree)         

        self.freeze_mask = None
        self.freeze_center = None
        self.freeze_radius = -1.0
        self.seed_mask = None
        self.seed_prunable = False         
        self.pretrained_mask = None
        self.pretrained_prunable = False    

        self.seed_up_dir = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device="cuda")
        self.seed_shell_center = None        
        self.seed_shell_r_min = None
        self.seed_shell_r_max = None
        self._freeze_hook_handles = []


    def capture(self):
        return super().capture() + (self.freeze_mask, self.seed_mask, self.pretrained_mask)


    def restore(self, model_args, training_args):
        super().restore(model_args[:12], training_args)   # vanilla unpacks exactly 12 fields
        extra = model_args[12:]
        if len(extra) >= 1: self.freeze_mask = extra[0]
        if len(extra) >= 2: self.seed_mask = extra[1]
        if len(extra) >= 3: self.pretrained_mask = extra[2]
        self._register_freeze_hooks()

    def _protected_mask(self, include_seed=True):
        protect = torch.zeros(self._xyz.shape[0], dtype=torch.bool, device="cuda")
        if self.freeze_mask is not None:
            protect |= self.freeze_mask
        if self.pretrained_mask is not None and not self.pretrained_prunable:
            protect |= self.pretrained_mask
        if include_seed and self.seed_mask is not None and not self.seed_prunable:
            protect |= self.seed_mask
        return protect


    def _clear_freeze_hooks(self):
        for h in self._freeze_hook_handles:
            h.remove()
        self._freeze_hook_handles = []

    def seed_shell_regularization_loss(
        self,
        lambda_shell: float = 0.0,
        lambda_planarity: float = 0.0,
        planarity_eps: float = 1e-6,
    ):
        """
        EnvMap-GS geometric regularisers applied only to ``seed_mask`` Gaussians.

        Returns (total_loss, details_dict) where details has optional 'shell' /
        'planarity' scalar values for logging.
        """
        from utils.loss_utils import envmap_planarity_loss, envmap_shell_loss

        details = {}
        if lambda_shell <= 0.0 and lambda_planarity <= 0.0:
            return None, details
        if self.seed_mask is None or not bool(self.seed_mask.any()):
            return None, details
        if self.seed_shell_center is None:
            return None, details

        xyz = self.get_xyz[self.seed_mask]
        scaling = self.get_scaling[self.seed_mask]
        rotation = self.get_rotation[self.seed_mask]
        center = self.seed_shell_center

        terms = []
        if lambda_shell > 0.0:
            l_shell = envmap_shell_loss(
                xyz, center, self.seed_shell_r_min, self.seed_shell_r_max,
            )
            terms.append(lambda_shell * l_shell)
            details["shell"] = float(l_shell.detach().item())
        if lambda_planarity > 0.0:
            l_plan = envmap_planarity_loss(
                xyz, scaling, rotation, center, eps=planarity_eps,
            )
            terms.append(lambda_planarity * l_plan)
            details["planarity"] = float(l_plan.detach().item())

        if not terms:
            return None, details
        return sum(terms), details

    def _register_freeze_hooks(self):
        self._clear_freeze_hooks()
        if self.freeze_mask is None:
            return

        def get_hook(mask):
            def hook(grad):
                if grad.shape[0] != mask.shape[0]:
                    return grad
                expanded_mask = mask.view(mask.shape[0], *([1] * (grad.dim() - 1)))
                return grad * (~expanded_mask).float()
            return hook

        m = self.freeze_mask
        self._freeze_hook_handles.append(self._xyz.register_hook(get_hook(m)))
        self._freeze_hook_handles.append(self._features_dc.register_hook(get_hook(m)))
        self._freeze_hook_handles.append(self._features_rest.register_hook(get_hook(m)))
        self._freeze_hook_handles.append(self._opacity.register_hook(get_hook(m)))
        self._freeze_hook_handles.append(self._scaling.register_hook(get_hook(m)))
        self._freeze_hook_handles.append(self._rotation.register_hook(get_hook(m)))



    def _inside_protected_region_mask(self, xyz):
        """Return mask of points inside current protected region.

        Priority:
        1) radius around pretrained center (derived from freeze_radius)

        Returns all-False when freeze_radius <= 0 (i.e. when the workflow
        does not use spatial freezing at all). This is independent of, and
        does not substitute for, `pretrained_mask` identity protection.
        """

        if (self.freeze_radius is not None and float(self.freeze_radius) > 0.0
                and self.freeze_center is not None):
            return torch.norm(xyz - self.freeze_center, dim=1) <= float(self.freeze_radius)

        return torch.zeros(xyz.shape[0], dtype=torch.bool, device=xyz.device)

    def _infer_seed_up_direction(self, center):
        """Infer scene-aware "up" direction for seed placement.

        Preference:
        1) OBB local +Y mapped to world.
        2) PCA plane-normal candidate from scene geometry.
        Sign is chosen so the "up" side is sparser (likely sky).
        """
        xyz = self._xyz.detach()
        if xyz.shape[0] < 64:
            return torch.tensor([0.0, 1.0, 0.0], device="cuda"), "fallback"

        rel = xyz - center

        def normalize(v):
            return v / (torch.norm(v) + 1e-8)

        candidates = []

        cov = torch.matmul(rel.T, rel) / max(rel.shape[0] - 1, 1)
        _, evecs = torch.linalg.eigh(cov)
        pca_up = normalize(evecs[:, 0])  # small-variance direction ~= dominant plane normal
        candidates.append(("pca", pca_up))

        best_name = "fallback"
        best_dir = torch.tensor([0.0, 1.0, 0.0], device="cuda")
        best_score = -float("inf")

        for name, cand in candidates:
            proj = torch.matmul(rel, cand)
            q20 = torch.quantile(proj, 0.20)
            q80 = torch.quantile(proj, 0.80)
            top_count = torch.sum(proj >= q80).float()
            bot_count = torch.sum(proj <= q20).float()

            # Keep sign where positive side is sparser (outdoor sky heuristic).
            if top_count > bot_count:
                cand = -cand
                top_count, bot_count = bot_count, top_count

            # Prefer stronger asymmetry.
            score = (bot_count - top_count) / (top_count + bot_count + 1e-8)
            if score > best_score:
                best_score = score
                best_name = name
                best_dir = cand

        return normalize(best_dir), best_name

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        protect = self._protected_mask(include_seed=True)
        if protect.any():
            opacities_new[protect] = self._opacity.data[protect]
        self._opacity = self.replace_tensor_to_optimizer(opacities_new, "opacity")["opacity"]
        self._register_freeze_hooks()
        self.zero_frozen_gradients()


    def init_from_pretrained(self, pretrained_path, spatial_lr_scale, seed_center, seed_up_offset, up_direction, freeze_radius=0.0,
                             seed_num_points=0, seed_radius=0.0,
                             seed_scale=0.8, seed_opacity_init=0.5,
                             seed_mode="upper_ground_mixture"):
    
        self.load_ply(pretrained_path)
        self.spatial_lr_scale = spatial_lr_scale
        num_pretrained = self._xyz.shape[0]

        center = seed_center
        scene_spread = torch.norm(self._xyz.detach() - center, dim=1)

        print(f"[pretrained] scene center: {center.cpu().numpy()}")
        print(f"[pretrained] scene spread: median={scene_spread.median().item():.2f}, "
              f"max={scene_spread.max().item():.2f}")
        print(f"[pretrained] cameras_extent (spatial_lr_scale): {spatial_lr_scale:.2f}")

        self.freeze_center = center
        self.freeze_radius = float(freeze_radius)
        self.freeze_mask = self._inside_protected_region_mask(self._xyz)
       
        self.pretrained_mask = torch.ones(num_pretrained, dtype=torch.bool, device="cuda")

        print(f"[pretrained-final] Loaded {num_pretrained} pretrained Gaussians, "
              f"{int(self.freeze_mask.sum().item())} hard-frozen (inside freeze_radius), "
              f"{num_pretrained} identity-protected via pretrained_mask, "
              f"center={center.cpu().numpy().tolist()}, "
              f"freeze_radius={self.freeze_radius:.2f}")

        # Pretrained rows are never seeds.
        self.seed_mask = torch.zeros(num_pretrained, dtype=torch.bool, device="cuda")
        sky_shell_center = seed_center + self.seed_up_dir * seed_up_offset
        seed_r_max = float(seed_radius) if seed_radius > 0 else float(scene_spread.max().item())
        seed_r_min = seed_r_max * 0.999
        self.seed_shell_center = sky_shell_center
        self.seed_shell_r_min = seed_r_min
        self.seed_shell_r_max = seed_r_max

        self.seed_up_dir = up_direction

        if seed_num_points > 0:
            seed_r_max = float(seed_radius) if seed_radius > 0 else float(scene_spread.max().item())
            seed_r_min = seed_r_max * 0.999

    

            if seed_mode == "upper_ground_mixture":
                seed_xyz, seed_scales, seed_rots = self._create_seed_cloud_upper_ground_mixture(
                    seed_r_max, seed_num_points, center, seed_scale,
                    min_radius=seed_r_min, up_dir=self.seed_up_dir
                )
            elif seed_mode == "sky_shell":
                seed_xyz, seed_scales, seed_rots = self._create_seed_cloud_sky_shell(
                    seed_r_max, seed_num_points, seed_center, seed_scale,
                    min_radius=seed_r_min, up_dir=self.seed_up_dir
                )
           
            num_seed = seed_xyz.shape[0]
            num_sh_rest = (self.max_sh_degree + 1) ** 2 - 1
            seed_color = torch.tensor([0.5, 0.5, 0.5], device="cuda")
            seed_sh = RGB2SH(seed_color)
            seed_features_dc = seed_sh.view(1, 1, 3).repeat(num_seed, 1, 1)
            seed_features_rest = torch.zeros(num_seed, num_sh_rest, 3, device="cuda")
            seed_opacity = inverse_sigmoid(
                seed_opacity_init * torch.ones(num_seed, 1, device="cuda")
            )

            self._xyz = nn.Parameter(torch.cat([self._xyz.data, seed_xyz], 0).requires_grad_(True))
            self._features_dc = nn.Parameter(torch.cat([self._features_dc.data, seed_features_dc], 0).requires_grad_(True))
            self._features_rest = nn.Parameter(torch.cat([self._features_rest.data, seed_features_rest], 0).requires_grad_(True))
            self._scaling = nn.Parameter(torch.cat([self._scaling.data, seed_scales], 0).requires_grad_(True))
            self._rotation = nn.Parameter(torch.cat([self._rotation.data, seed_rots], 0).requires_grad_(True))
            self._opacity = nn.Parameter(torch.cat([self._opacity.data, seed_opacity], 0).requires_grad_(True))

            self.freeze_mask = torch.cat([
                self.freeze_mask,
                torch.zeros(num_seed, dtype=torch.bool, device="cuda")
            ])
            self.seed_mask = torch.cat([
                self.seed_mask,
                torch.ones(num_seed, dtype=torch.bool, device="cuda")
            ])
            # FIX (2): seed points are synthesized, never pretrained.
            self.pretrained_mask = torch.cat([
                self.pretrained_mask,
                torch.zeros(num_seed, dtype=torch.bool, device="cuda")
            ])

            print(f"Added {num_seed} seed Gaussians (radius={seed_r_max:.2f}, "
                  f"min_radius={seed_r_min:.2f}, scale={seed_scale}, "
                  f"opacity_init={seed_opacity_init}, mode={seed_mode})")

        self.max_radii2D = torch.zeros(self._xyz.shape[0], device="cuda")
        print(f"Total: {self._xyz.shape[0]} Gaussians "
              f"({int(self.freeze_mask.sum().item())} hard-frozen, "
              f"{int(self.pretrained_mask.sum().item())} pretrained-protected, "
              f"{int(self.seed_mask.sum().item())} seeds)")

    @torch.no_grad()
    def enforce_max_points(self, max_points: int):

        if max_points <= 0:
            return
        n = self.get_xyz.shape[0]
        if n <= max_points:
            return
        n_frozen = int(self.freeze_mask.sum().item()) if self.freeze_mask is not None else 0
        if max_points < n_frozen:
            return

        device = self.get_xyz.device
        excess = n - max_points

        trainable = torch.ones(n, dtype=torch.bool, device=device)
        if self.freeze_mask is not None:
            trainable = trainable & ~self.freeze_mask
        
        if self.seed_mask is not None and not self.seed_prunable:
            trainable = trainable & ~self.seed_mask
        if self.pretrained_mask is not None and not self.pretrained_prunable:
            trainable = trainable & ~self.pretrained_mask
        n_prunable = int(trainable.sum().item())
        if n_prunable <= 0:
            return

        if self.xyz_gradient_accum.numel() == n and self.denom.numel() == n:
            avg_grad = (self.xyz_gradient_accum.squeeze() / (self.denom.squeeze() + 1e-8))
        else:
            avg_grad = torch.zeros(n, device=device)

        opacity = self.get_opacity.squeeze()
        max_scale = self.get_scaling.max(dim=1).values
        extent = max(float(max_scale.max().item()), 1e-6)

        score = (
            0.65 * opacity
            + 0.25 * torch.clamp(avg_grad * 200.0, 0.0, 1.0)
            + 0.10 * torch.clamp(max_scale / (0.2 * extent), 0.0, 1.0)
            - 0.05 * torch.clamp(max_scale / extent, 0.0, 2.0)
        )

        selected = torch.zeros(n, dtype=torch.bool, device=device)

        # Stage 0: explicitly target "dead" points first.
        dead = trainable & (opacity < 0.01) & (avg_grad < 1e-5)
        k0 = min(excess, int(dead.sum().item()))
        idx0 = torch.empty(0, dtype=torch.long, device=device)
        if k0 > 0:
            s0 = score.clone()
            s0[~dead] = float("inf")
            _, idx0 = torch.topk(s0, k0, largest=False)
            selected[idx0] = True

        # Stage 1: remaining weak points across all trainable points (no region split).
        remaining = excess - int(k0)
        idx1 = torch.empty(0, dtype=torch.long, device=device)
        if remaining > 0:
            candidates = trainable & ~selected
            k1 = min(remaining, int(candidates.sum().item()))
            if k1 > 0:
                s1 = score.clone()
                s1[~candidates] = float("inf")
                _, idx1 = torch.topk(s1, k1, largest=False)

        idx = torch.cat([idx0, idx1], dim=0)
        if idx.numel() == 0:
            return
        prune_mask = torch.zeros(n, dtype=torch.bool, device=device)
        prune_mask[idx] = True
        self.prune_points(prune_mask)


    @staticmethod
    def _create_seed_cloud_sky_shell(radius, num_points, center, scale, min_radius=0.0, up_dir=None):
       
        r_min = max(float(min_radius), 0.0)
        r_max = float(radius)
        r_surface = 0.5 * (r_min + r_max) if 0.0 < r_min < r_max else r_max

        if up_dir is None:
            up_dir = torch.tensor([0.0, 1.0, 0.0], device="cuda")
        up_dir = up_dir / (torch.norm(up_dir) + 1e-8)

        ref = torch.tensor([1.0, 0.0, 0.0], device="cuda")
        if torch.abs(torch.dot(up_dir, ref)) > 0.95:
            ref = torch.tensor([0.0, 0.0, 1.0], device="cuda")
        u_axis = ref - torch.dot(ref, up_dir) * up_dir
        u_axis = u_axis / (torch.norm(u_axis) + 1e-8)
        v_axis = torch.cross(up_dir, u_axis, dim=0)
        v_axis = v_axis / (torch.norm(v_axis) + 1e-8)

        fraction = 0.5
        cos_min = 1.0 - 2.0 * fraction
        u_elev = torch.rand(num_points, device="cuda")
        cos_phi = cos_min + (1.0 - cos_min) * u_elev
        phi = torch.acos(cos_phi.clamp(-1.0 + 1e-6, 1.0 - 1e-6))
        sin_phi = torch.sin(phi)
        theta = 2.0 * np.pi * torch.rand(num_points, device="cuda")

        local_dirs = (
            sin_phi.unsqueeze(1) * torch.cos(theta).unsqueeze(1) * u_axis.unsqueeze(0) +
            torch.cos(phi).unsqueeze(1) * up_dir.unsqueeze(0) +
            sin_phi.unsqueeze(1) * torch.sin(theta).unsqueeze(1) * v_axis.unsqueeze(0)
        )

        jitter = (torch.rand(num_points, 1, device="cuda") * 2.0 - 1.0) * 0.005
        r = r_surface * (1.0 + jitter)
        xyz = center + local_dirs * r

        # Calculate cap area and exact point spacing
        cap_area = 4.0 * np.pi * (r_surface ** 2) * fraction
        spacing = np.sqrt(cap_area / max(num_points, 1))

        # Set base scale to half the inter-point spacing
        base = spacing * 0.5
        scales = torch.log(torch.full((num_points, 3), base, device="cuda"))
        rots = torch.zeros(num_points, 4, device="cuda")
        rots[:, 0] = 1

        print(f"[sky_shell] surface r={r_surface:.2f} (band [{r_min:.2f},{r_max:.2f}]), "
              f"n={num_points}, spacing={spacing:.3f}, base_scale={base:.3f}")
        return xyz, scales, rots


    def _create_seed_cloud_upper_ground_mixture(
        self, radius, num_points, center, scale, min_radius=0.0, up_dir=None
    ):
        """Mixture seeds: upper shell + ground annulus + small isotropic fallback."""
        n_upper = int(round(num_points * 0.65))
        n_ground = int(round(num_points * 0.30))
        n_fallback = max(0, num_points - n_upper - n_ground)

        r_min = max(float(min_radius), 0.0)
        r_max = float(radius)
        if r_min >= r_max:
            r_min = 0.0

        if up_dir is None:
            up_dir = torch.tensor([0.0, 1.0, 0.0], device="cuda")
        up_dir = up_dir / (torch.norm(up_dir) + 1e-8)

        ref = torch.tensor([1.0, 0.0, 0.0], device="cuda")
        if torch.abs(torch.dot(up_dir, ref)) > 0.95:
            ref = torch.tensor([0.0, 0.0, 1.0], device="cuda")
        u_axis = ref - torch.dot(ref, up_dir) * up_dir
        u_axis = u_axis / (torch.norm(u_axis) + 1e-8)
        v_axis = torch.cross(up_dir, u_axis, dim=0)
        v_axis = v_axis / (torch.norm(v_axis) + 1e-8)

        def sample_shell_r(n):
            u = torch.rand(n, 1, device="cuda")
            return (u * r_max**3 + (1 - u) * r_min**3).pow(1.0 / 3.0)

        pts = []

        if n_upper > 0:
            dirs = torch.randn(n_upper, 3, device="cuda")
            dots = torch.matmul(dirs, up_dir)
            dirs[dots < 0.0] *= -1.0
            dirs = dirs + 0.15 * up_dir.unsqueeze(0)
            dirs = dirs / (torch.norm(dirs, dim=1, keepdim=True) + 1e-8)
            r = sample_shell_r(n_upper)
            pts.append(center + dirs * r)

        if n_ground > 0:
            rel = self._xyz.detach() - center
            h_vals = torch.matmul(rel, up_dir)
            h_ground = torch.quantile(h_vals, 0.05)
            h_span = torch.quantile(h_vals, 0.95) - h_ground
            h_jitter = torch.clamp(h_span * 0.02, min=1e-3)
            theta = 2.0 * np.pi * torch.rand(n_ground, 1, device="cuda")
            u = torch.rand(n_ground, 1, device="cuda")
            r_xy = torch.sqrt(u * r_max * r_max + (1.0 - u) * r_min * r_min)
            planar = (
                (r_xy * torch.cos(theta)) * u_axis.unsqueeze(0) +
                (r_xy * torch.sin(theta)) * v_axis.unsqueeze(0)
            )
            h = h_ground + (torch.rand(n_ground, 1, device="cuda") * 2.0 - 1.0) * h_jitter
            pts.append(center + planar + h * up_dir.unsqueeze(0))

        if n_fallback > 0:
            dirs = torch.randn(n_fallback, 3, device="cuda")
            dirs = dirs / (torch.norm(dirs, dim=1, keepdim=True) + 1e-8)
            r = sample_shell_r(n_fallback)
            pts.append(center + dirs * r)

        xyz = torch.cat(pts, dim=0) if len(pts) > 0 else torch.empty(0, 3, device="cuda")
        shell_thickness = r_max - r_min if r_min > 0 else r_max
        base = max(shell_thickness * 0.03, 1e-4) * scale
        scales = torch.log(torch.full((num_points, 3), base, device="cuda"))
        rots = torch.zeros(num_points, 4, device="cuda")
        rots[:, 0] = 1
        return xyz, scales, rots

    def zero_frozen_gradients(self):
        
        if self.freeze_mask is None or self.freeze_mask.sum() == 0:
            return

        for param in [self._xyz, self._features_dc, self._features_rest,
                      self._opacity, self._scaling, self._rotation]:
            if param.grad is not None:
                param.grad[self.freeze_mask] = 0

        for group in self.optimizer.param_groups:
            p = group['params'][0]
            state = self.optimizer.state.get(p, None)
            if state is None:
                continue
            if 'exp_avg' in state:
                state['exp_avg'][self.freeze_mask] = 0.0
            if 'exp_avg_sq' in state:
                state['exp_avg_sq'][self.freeze_mask] = 0.0



    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
  
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        n_size_pruned = 0

        n_low_op = int((self.get_opacity < min_opacity).sum().item())
       
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        if self.freeze_mask is not None:
            selected_pts_mask = selected_pts_mask & ~self.freeze_mask

        if self.pretrained_mask is not None:
            selected_pts_mask = selected_pts_mask & ~self.pretrained_mask

        inside_protected = self._inside_protected_region_mask(self.get_xyz)
        selected_pts_mask = selected_pts_mask & ~inside_protected


        parent_is_seed = self.seed_mask[selected_pts_mask] if self.seed_mask is not None else None
        new_seed_flags = parent_is_seed.repeat(N) if parent_is_seed is not None else None

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)


        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        new_N = new_xyz.shape[0]
        if new_N > 0:
            prune_new_inside = self._inside_protected_region_mask(new_xyz)
        else:
            prune_new_inside = torch.zeros(0, device="cuda", dtype=torch.bool)
        prune_filter = torch.cat((selected_pts_mask, prune_new_inside))
        self.prune_points(prune_filter, ignore_seed_mask=False)

    
    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                            torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        if self.freeze_mask is not None:
            selected_pts_mask = selected_pts_mask & ~self.freeze_mask

        if self.pretrained_mask is not None:
            selected_pts_mask = selected_pts_mask & ~self.pretrained_mask

        inside_protected = self._inside_protected_region_mask(self.get_xyz)
        selected_pts_mask = selected_pts_mask & ~inside_protected

        if self.seed_mask is not None:
            selected_pts_mask = selected_pts_mask & ~self.seed_mask

        parent_is_seed = self.seed_mask[selected_pts_mask] if self.seed_mask is not None else None
        new_seed_flags = parent_is_seed  # clones are 1:1, no repeat needed unlike split

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest,
                                new_opacities, new_scaling, new_rotation)

 

    def prune_points(self, mask, ignore_seed_mask=False):
        mask = mask & ~self._protected_mask(include_seed=not ignore_seed_mask)
        valid = ~mask
        super().prune_points(mask)           # tensors, optimizer state, accum, denom, radii
        if self.freeze_mask is not None:
            self.freeze_mask = self.freeze_mask[valid]
        if self.seed_mask is not None:
            self.seed_mask = self.seed_mask[valid]
        if self.pretrained_mask is not None:
            self.pretrained_mask = self.pretrained_mask[valid]
        self._register_freeze_hooks()     


    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities,
                          new_scaling, new_rotation, new_seed_flags=None):
        super().densification_postfix(new_xyz, new_features_dc, new_features_rest,
                                    new_opacities, new_scaling, new_rotation)
        zeros = torch.zeros(new_xyz.shape[0], dtype=torch.bool, device="cuda")
        if self.freeze_mask is not None:
            self.freeze_mask = torch.cat([self.freeze_mask, zeros])
        if self.seed_mask is not None:
            self.seed_mask = torch.cat([self.seed_mask, new_seed_flags if new_seed_flags is not None else zeros])
        if self.pretrained_mask is not None:
            self.pretrained_mask = torch.cat([self.pretrained_mask, zeros])
        self._register_freeze_hooks()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
      
        if self.freeze_mask is not None and self.freeze_mask.any():
            active = update_filter & ~self.freeze_mask
        else:
            active = update_filter

        self.xyz_gradient_accum[active] += torch.norm(
            viewspace_point_tensor.grad[active, :2], dim=-1, keepdim=True
        )
        self.denom[active] += 1

    @torch.no_grad()
    def prune_floaters_near_center(
        self,
        center: torch.Tensor,
        radius: float,
        max_opacity: float = 0.3,
        min_scale: float = 0.1,   # prune Gaussians larger than this
    ):
        if self._xyz.shape[0] == 0:
            return 0

        xyz = self._xyz.detach()
        center = center.to(xyz.device).view(1, 3)

        in_sphere = torch.norm(xyz - center, dim=1) < float(radius)
        low_opacity = self.get_opacity.squeeze() < float(max_opacity)
        large = self.get_scaling.max(dim=1).values > float(min_scale)

        mask = in_sphere & large

        if self.seed_mask is not None:
            mask = mask & ~self.seed_mask

        n = int(mask.sum().item())
        if n > 0:
            self.prune_points(mask, ignore_freeze_mask=True)

        print(
            f"[prune_floaters_near_center] radius={radius:.2f}, "
            f"max_opacity={max_opacity}, min_scale={min_scale}, "
            f"in_sphere={int(in_sphere.sum())}, pruned={n}"
        )
        return n

    