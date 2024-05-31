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
from scene import Scene, SceneDataset
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import (
        # preprocess3dgs_and_all2all,
        # render
        distributed_preprocess3dgs_and_all2all_final,
        render_final
    )
import torchvision
from utils.general_utils import safe_state, set_args, init_distributed, set_log_file, set_cur_iter
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from gaussian_renderer.loss_distribution import load_camera_from_cpu_to_all_gpu_for_eval
from gaussian_renderer.workload_division import start_strategy_final, DivisionStrategyHistoryFinal
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

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    dataset = SceneDataset(views)

    set_cur_iter(iteration)
    generated_cnt = 0
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if args.sample_freq != -1 and idx % args.sample_freq != 0:
            continue
        if generated_cnt == args.generate_num:
            break
        generated_cnt += 1

        strategy_history = DivisionStrategyHistoryFinal(dataset, utils.DEFAULT_GROUP.size(), utils.DEFAULT_GROUP.rank())
        batched_strategies, gpuid2tasks = start_strategy_final([view], strategy_history)
        load_camera_from_cpu_to_all_gpu_for_eval([view], batched_strategies, gpuid2tasks)

        batched_screenspace_pkg = distributed_preprocess3dgs_and_all2all_final([view], gaussians, pipeline, background,
                                                                                batched_strategies=batched_strategies,
                                                                                mode="test")
        rendered_image, _ = render_final(batched_screenspace_pkg, batched_strategies)
        # strategy_history = get_division_strategy_history(cameraId2StrategyHistory, view, "evaluation")
        # strategy = strategy_history.start_strategy()
        # screenspace_pkg = preprocess3dgs_and_all2all([view], gaussians, pipeline, background,
        #                                              [strategy],
        #                                              mode="test")
        # rendered_image, _ = render(screenspace_pkg, strategy)

        gt_image = torch.clamp(view.original_image_backup[0:3, :, :].cuda() / 255.0, 0, 1.0)
        torchvision.utils.save_image(rendered_image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt_image, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        args = utils.get_args()
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(args, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

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
    parser.add_argument("--distributed_load", action="store_true")# TODO: delete this.
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    init_distributed(args)# This script only supports single-gpu rendering. 
    # I need to put the flags here because the render() function need it.
    # However, disable them during render.py because they are only needed during training.

    log_file = open(args.model_path+f"/render_ws={utils.DEFAULT_GROUP.size()}_rk_{utils.DEFAULT_GROUP.rank()}.log", 'w')
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

    render_sets(lp.extract(args), args.iteration, pp.extract(args), args.skip_train, args.skip_test)