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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import (
        preprocess3dgs_and_all2all,
        render
    )
import torchvision
from utils.general_utils import safe_state, set_args, init_distributed, set_log_file, get_args, set_cur_iter
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from gaussian_renderer.workload_division import get_division_strategy_history, get_local_running_time_by_modes
from arguments import (
    AuxiliaryParams,
    ModelParams, 
    PipelineParams, 
    OptimizationParams, 
    DistributionParams, 
    BenchmarkParams, 
    DebugParams, 
    print_all_args, 
    init_args
)
import utils.general_utils as utils

def render_zoom_out(model_path, name, iteration, views, gaussians, pipeline, background, generate_num):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    set_cur_iter(iteration)

    # idx = 3
    # view = views[idx]
    # for (dx, dy, dz) in [(0, 0, 0), (0, -1, 0), (0, -2, 0), (0, -3, 0), (0, -4, 0), (-1, 0, 0), (-2, 0, 0), (-3, 0, 0), (-4, 0, 0)]:
    # for (dx, dy, dz) in [(0, -3, 0), (-1, -1, 0), (-2, -2, 0), (-1, -2, 0), (-3, -3, 0), (-4, -4, 0)]:
    # for (dx, dy, dz) in [(-2.5, -6, 0), (-2.5, -6, 1), (-2.5, -6, 2)]:
    # for (dx, dy, dz) in [(-2.5, -6.5, 1), (-2.5, -7, 1)]:
    # for (dx, dy, dz) in [(-2.5, -7, 3), (-2.5, -8, 3), (-2.5, -9, 3)]:
    # for (dx, dy, dz) in [(-2.5, -10, 3), (-2.5, -11, 3), (-2.5, -12, 3)]:
    # for (dx, dy, dz) in [(-2.5, -15, 3), (-2.5, -18, 3), (-2.5, -20, 3)]:
    # for (dx, dy, dz) in [(-2, -7, 1), (-2, -6.5, 1)]:
    # for (dx, dy, dz) in [(-1.5, -7, 1), (-1.5, -6.5, 1), (-1, -7, 1), (-1, -6.5, 1), (-1.25, -7, 1), (-1.25, -6.5, 1)]:
    # for (dx, dy, dz) in [(-1.5, -7, 2), (-1.5, -7, 2), (-1, -7, 2), (-1, -6.5, 2), (-1.25, -7, 2), (-1.25, -6.5, 2)]:

    (dx, dy, dz) = (0, -6, 0)
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        file_name = '{0:05d}'.format(idx) + f"dx={dx}_dy={dy}_dz={dz}"
        if os.path.exists(os.path.join(render_path, file_name + ".png")):
            continue

        view.update(dx=dx, dy=dy, dz=dz)
        cameraId2StrategyHistory = {}
        strategy_history = get_division_strategy_history(cameraId2StrategyHistory, view, "evaluation")
        strategy = strategy_history.start_strategy()
        screenspace_pkg = preprocess3dgs_and_all2all([view], gaussians, pipeline, background,
                                                        [strategy],
                                                        mode="test")
        rendered_image, _ = render(screenspace_pkg, strategy)
        gt_image = torch.clamp(view.original_image_backup[0:3, :, :].cuda() / 255.0, 0, 1.0)
        torchvision.utils.save_image(rendered_image, os.path.join(render_path, file_name + ".png"))

def render_one(model_path, name, iteration, views, gaussians, pipeline, background, generate_num):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    set_cur_iter(iteration)
    cameraId2StrategyHistory = {}
    idx = 220
    view = views[idx]
    # for (dx, dy, dz) in [(0, 0, 0), (0, -1, 0), (0, -2, 0), (0, -3, 0), (0, -4, 0), (-1, 0, 0), (-2, 0, 0), (-3, 0, 0), (-4, 0, 0)]:
    # for (dx, dy, dz) in [(0, -3, 0), (-1, -1, 0), (-2, -2, 0), (-1, -2, 0), (-3, -3, 0), (-4, -4, 0)]:
    # for (dx, dy, dz) in [(-2.5, -6, 0), (-2.5, -6, 1), (-2.5, -6, 2)]:
    # for (dx, dy, dz) in [(-2.5, -6.5, 1), (-2.5, -7, 1)]:
    # for (dx, dy, dz) in [(-2.5, -7, 3), (-2.5, -8, 3), (-2.5, -9, 3)]:
    # for (dx, dy, dz) in [(-2.5, -10, 3), (-2.5, -11, 3), (-2.5, -12, 3)]:
    # for (dx, dy, dz) in [(-2.5, -15, 3), (-2.5, -18, 3), (-2.5, -20, 3)]:
    # for (dx, dy, dz) in [(-2, -7, 1), (-2, -6.5, 1)]:
    # for (dx, dy, dz) in [(-1.5, -7, 1), (-1.5, -6.5, 1), (-1, -7, 1), (-1, -6.5, 1), (-1.25, -7, 1), (-1.25, -6.5, 1)]:
    # for (dx, dy, dz) in [(-1.5, -7, 2), (-1.5, -7, 2), (-1, -7, 2), (-1, -6.5, 2), (-1.25, -7, 2), (-1.25, -6.5, 2)]:

    # for (dx, dy, dz) in [(-1.5, -5.5, 1), (-1.5, -4.5, 1), (-1.5, -3.5, 1), (-1.5, -2.5, 1), (-1.5, -1.5, 1), (-1.5, -0.5, 1), (-1.5, 0, 1)]:
    # for (dx, dy, dz) in [(-1, -5.5, 2), (-1, -4.5, 1), (-1, -3.5, 1), (-1, -2.5, 1), (-1, -1.5, 1), (-1, -0.5, 1), (-1, 0, 1)]:
    # for (dx, dy, dz) in [(-0.5, -5.5, 1), (-0.5, -4.5, 1), (-0.5, -3.5, 1), (-0.5, -2.5, 1), (-0.5, -1.5, 1), (-0.5, -0.5, 1), (-0.5, -1, 1), (-0.5, 0, 1)]+[(-1, -5.5, 1), (-1, -4.5, 1), (-1, -3.5, 1), (-1, -2.5, 1), (-1, -1.5, 1), (-1, -0.5, 1), (-1, -1, 1), (-1, 0, 1)]:
    # for (dx, dy, dz) in [(-0.5, -5.5, 1.5), (-0.5, -4.5, 1.5), (-0.5, -3.5, 1.5), (-0.5, -2.5, 1.5), (-0.5, -1.5, 1.5), (-0.5, -0.5, 1.5), (-0.5, -1, 1.5), (-0.5, 0, 1.5)]:
    # for (dx, dy, dz) in [(-1, -0, 0), (0, -1, 0), (0, 0, -1), (-2, -0, 0), (0, -2, 0), (0, 0, -2), (-4, -0, 0), (0, -4, 0), (0, 0, -4)]:

    for (dx, dy, dz) in [(-1, 0, 16), (-2, 0, 16), (-4, 0, 16), (-8, 0, 16), (-16, 0, 16), (0, -1, 16), (0, -2, 16), (0, -4, 16), (0, -8, 16), (0, -16, 16),
                         (0, -12, 16), (-4, -12, 16), (-4, -12, 12), (-4, -12, 10), (-4, -10, 12), (-4, -10, 12), (-4, -8, 12), (-4, -8, 6), (-4, -6, 6), (-4, -4, 4),
                         (-4, -3, 3), (-4, -2, 2), (-4, -1, 1), (-3, -3, 3), (-5, -3, 3), (-2, -3, 3), (-6, -3, 3), (-2, -2, 2), (-2, -1, 1),
                         (-2, -2, 1), (-2, -1, 0), (-2, -2, 0), (-2, -1, -1), (-2, -2, -1), (-2.5, -0.5, 0), (-2.5, -1, 0), (-2.5, -1, -1), (-2.5, 0.5, 0),
                         (-2.5, 0.5, -0.5), (-2.5, 0.5, -1), (-2.5, 0.5, -1.5)]:
        file_name = '{0:05d}'.format(idx) + f"dx={dx}_dy={dy}_dz={dz}"
        if os.path.exists(os.path.join(render_path, file_name + ".png")):
            continue

        view.update(dx=dx, dy=dy, dz=dz)
        strategy_history = get_division_strategy_history(cameraId2StrategyHistory, view, "evaluation")
        strategy = strategy_history.start_strategy()
        screenspace_pkg = preprocess3dgs_and_all2all([view], gaussians, pipeline, background,
                                                        [strategy],
                                                        mode="test")
        rendered_image, _ = render(screenspace_pkg, strategy)
        torchvision.utils.save_image(rendered_image, os.path.join(render_path, file_name + ".png"))


def visualization_generation(dataset : ModelParams, iteration : int, pipeline : PipelineParams):
    with torch.no_grad():
        args = utils.get_args()
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(args, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_one(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    ap = AuxiliaryParams(parser)
    lp = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    dist_p = DistributionParams(parser)
    bench_p = BenchmarkParams(parser)
    debug_p = DebugParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--generate_num", default=-1, type=int)
    parser.add_argument("--sample_freq", default=-1, type=int)
    parser.add_argument("--distributed_load", action="store_true")
    parser.add_argument("--render_one", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    init_distributed(args)
    # I need to put the flags here because the render() function need it.
    # However, disable them during render.py because they are only needed during training.

    log_file = open(args.model_path+f"/vis_generate_ws={utils.DEFAULT_GROUP.size()}_rk_{utils.DEFAULT_GROUP.rank()}.log", 'w')
    set_log_file(log_file)

    ## Prepare arguments.
    # Check arguments
    init_args(args)
    if args.skip_train:
        args.num_train_cameras = 0
    if args.skip_test:
        args.num_test_cameras = 0
    # Set up global args
    set_args(args)
    print_all_args(args, log_file)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    visualization_generation(lp.extract(args), args.iteration, pp.extract(args))