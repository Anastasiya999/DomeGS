# works for bear
import json
import os

import torch
from torchvision.utils import save_image
from random import randint
from utils.loss_utils import l1_loss, ssim
from renderer.gaussian_renderer import render, network_gui
import sys
from scene import Scene
from gs_sky import GaussianSkyModel, OptimizationParamsSky

from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams
from utils.train_utils import prepare_output_and_logger, training_report

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def _camera_center_cuda(cam):
    c = cam.camera_center
    if torch.is_tensor(c):
        return c.detach().to(device="cuda", dtype=torch.float32).reshape(3)
    return torch.tensor(c, device="cuda", dtype=torch.float32).reshape(3)

FREEZE_RADIUS = 2.0
MAX_POINTS_BUDGET = 3500000


def training( dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint,
             debug_from, save_xyz,
             seed_num_points=12000, 
             seed_opacity_init=0.5, 
             seed_mode="sky_shell",
             save_renders_every=0):

    
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianSkyModel(dataset.sh_degree)

    scene = Scene(dataset, gaussians, shuffle=False)
    train_cams = list(scene.getTrainCameras())

    print(f"Cached {len(train_cams)} train cameras")

    cam_centers = torch.stack([_camera_center_cuda(c) for c in train_cams])
    seed_center = cam_centers.mean(dim=0)
    print(f"Seed center {seed_center}")

    if not dataset.source_path:
        raise ValueError("Error: --dataset_path must be specified.")

    meta_path = os.path.join(dataset.source_path, "meta.json")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Error: meta.json not found at {meta_path}")

    with open(meta_path, "r") as f:
        meta_data = json.load(f)

    sphere_radius = meta_data.get("sphere_radius") 
    pano_eye_offset = meta_data.get("panorama_eye_up_offset")
    seed_center = torch.tensor(meta_data.get("target_pt"), device="cuda", dtype=torch.float32)
    up_direction = -torch.tensor(meta_data.get("v_up_scene"), device="cuda", dtype=torch.float32)
    if sphere_radius is None:
        raise ValueError(f"Error: 'sphere_radius' key is missing from {meta_path}")

    pretrained_ply = os.path.join(dataset.source_path, "gaussians_for_training.ply")
    if not os.path.exists(pretrained_ply):
        raise FileNotFoundError(f"Error: Required training PLY not found at {pretrained_ply}")

    print(f"Successfully loaded metadata from: {meta_path}")
    print(f"  Using pretrained PLY: {pretrained_ply}")
    print(f"  Extracted sphere radius: {sphere_radius}")


    gaussians.init_from_pretrained(
        pretrained_ply,
        scene.cameras_extent,
        freeze_radius=FREEZE_RADIUS,
        seed_num_points=seed_num_points,
        seed_radius=sphere_radius,
        seed_opacity_init=seed_opacity_init,
        seed_mode=seed_mode,
        seed_center=seed_center,
        seed_up_offset=pano_eye_offset,
        up_direction=up_direction,
    )

    gaussians.training_setup(opt)

    cam_to_center_dists = torch.norm(cam_centers - seed_center, dim=1)
    cam_ring_radius = cam_to_center_dists.median().item()
    print(f"[floater cleanup] camera ring radius (median): {cam_ring_radius:.2f}")

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    fixed_cameras = [
        train_cams[min(32, len(train_cams) - 1)],
        train_cams[min(100, len(train_cams) - 1)],
    ]

    for iteration in range(first_iter, opt.iterations + 1):
        
        if save_xyz and (iteration % 5000 == 1 or iteration == opt.iterations):
            os.makedirs(f"{scene.model_path}/xyz", exist_ok=True)
            torch.save(gaussians.get_xyz, f"{scene.model_path}/xyz/{iteration}.pt")

        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if save_renders_every > 0 and iteration % save_renders_every == 0:
            out_dir = os.path.join(scene.model_path, "renders", f"iter_{iteration:06d}")
            os.makedirs(out_dir, exist_ok=True)
            with torch.no_grad():
                strips = []
                for cam in fixed_cameras:
                    rendered = render(cam, gaussians, pipe, background)["render"]
                    rendered = torch.clamp(rendered, 0.0, 1.0)
                    rendered = torch.flip(rendered, dims=[1])  # match GT PNGs

                    gt = cam.original_image.cuda().clamp(0.0, 1.0)
                    if gt.shape != rendered.shape:
                        gt = torch.nn.functional.interpolate(
                            gt.unsqueeze(0),
                            size=(rendered.shape[1], rendered.shape[2]),
                            mode="bilinear", align_corners=False,
                        ).squeeze(0)

                    diff = (rendered - gt).abs()
                    diff_amplified = (diff * 5.0).clamp(0.0, 1.0)
                    divider = torch.ones(3, rendered.shape[1], 2, device="cuda")
                    strip = torch.cat(
                        [gt, divider, rendered, divider, diff_amplified],
                        dim=2,
                    )
                    strips.append(strip)

                h_div = torch.ones(3, 2, strips[0].shape[2], device="cuda")
                combined = strips[0]
                for s in strips[1:]:
                    combined = torch.cat([combined, h_div, s], dim=1)

                save_image(combined, f"{out_dir}/gt_vs_render.png")
                print(f"  [iter {iteration}] saved gt_vs_render.png  "
                      f"(left=GT, mid=render, right=diff×5)")

        # Shallow-copy the cached list only — do NOT call getTrainCameras() again.
        if not viewpoint_stack:
            viewpoint_stack = train_cams.copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if getattr(opt, "random_background", False) else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"], render_pkg["viewspace_points"],
            render_pkg["visibility_filter"], render_pkg["radii"],
        )

        image = torch.flip(image, dims=[1])  # match flipped sky_sample PNGs

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        lambda_shell = float(getattr(opt, "lambda_seed_shell", 1.0))
        lambda_plan = float(getattr(opt, "lambda_seed_planarity", 0.1))
        shell_until = int(getattr(opt, "seed_shell_loss_until_iter", 30000))
        shell_active = (
            (lambda_shell > 0.0 or lambda_plan > 0.0)
            and (iteration <= shell_until and iteration >= 0)
        )
        L_shell_geom = None
        shell_geom_details = {}
        if shell_active:
            L_shell_geom, shell_geom_details = gaussians.seed_shell_regularization_loss(
                lambda_shell=lambda_shell,
                lambda_planarity=lambda_plan,
            )
            if L_shell_geom is not None:
                if iteration % 1000 == 0:
                    print(f"Shell loss: {L_shell_geom.item():.6f} ({shell_geom_details})")
                loss = loss + L_shell_geom

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            training_report(tb_writer, iteration, Ll1, loss, l1_loss,
                            iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render, (pipe, background),
                            train_cams=train_cams, flip_render=True)

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if True:
                if iteration < opt.densify_until_iter:
                    gaussians.max_radii2D[visibility_filter] = torch.max(
                        gaussians.max_radii2D[visibility_filter],
                        radii[visibility_filter],
                    )
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(
                            opt.densify_grad_threshold, 0.005, scene.cameras_extent,
                            size_threshold,
                        )

                        mp = MAX_POINTS_BUDGET
                        
                        if mp > 0 and hasattr(gaussians, "enforce_max_points"):
                            gaussians.enforce_max_points(mp)
                        torch.cuda.empty_cache()

                    if iteration % opt.opacity_reset_interval == 0 or (
                            dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()


            if iteration < opt.iterations:
                gaussians.zero_frozen_gradients()
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration),
                           scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        if hasattr(gaussians, 'update_alpha'):
            gaussians.update_alpha()
        if hasattr(gaussians, 'prepare_scaling_rot'):
            gaussians.prepare_scaling_rot()


    scene.save(opt.iterations)


def prune_never_contributing_seeds(gaussians, cameras, pipe, background):
    """Exact final visibility sweep."""
    if gaussians.seed_mask is None or not bool(gaussians.seed_mask.any()):
        return 0

    n = gaussians.get_xyz.shape[0]
    contributing = torch.zeros(n, dtype=torch.bool, device="cuda")
    gaussians.optimizer.zero_grad(set_to_none=True)

    for cam in cameras:
        render_pkg = render(cam, gaussians, pipe, background)
        render_pkg["render"].sum().backward()
        if gaussians._opacity.grad is not None:
            contributing |= gaussians._opacity.grad.detach().abs().squeeze(-1) > 0
        gaussians.optimizer.zero_grad(set_to_none=True)

    prune_mask = (~contributing) & gaussians.seed_mask
    n_prune = int(prune_mask.sum().item())
    if n_prune > 0:
        gaussians.prune_points(prune_mask)
    print(f"[visibility sweep] {int(contributing.sum())}/{n} Gaussians contribute "
          f"to >=1 training view; pruning {n_prune} invisible seeds")
    return n_prune


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")

    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--save_xyz", action='store_true')

    parser.add_argument("--seed_opacity_init", type=float, default=0.05)
    parser.add_argument("--seed_mode", type=str, default="sky_shell",
        choices=["upper_ground_mixture", "sky_shell"])

    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--seed_num_points", type=int, default=10_000)

    parser.add_argument("--save_renders_every", type=int, default=0,
                        help="Save a GT/render/diff image every N iterations "
                             "to <model_path>/renders (0 = off).")

    lp = ModelParams(parser)

    op = OptimizationParamsSky(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])

    print("torch cuda: ", torch.cuda.is_available())
    print("Optimizing " + args.model_path)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    dataset = lp.extract(args)

    training(
        dataset, op.extract(args), pp.extract(args),
        args.test_iterations, args.save_iterations, args.checkpoint_iterations,
        args.start_checkpoint, args.debug_from, args.save_xyz,
        seed_num_points=args.seed_num_points,
        seed_opacity_init=args.seed_opacity_init,
        seed_mode=args.seed_mode,
        save_renders_every=args.save_renders_every,
    )

    print("\nTraining complete.")
