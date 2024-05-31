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
from scene import Scene, GaussianModel, SceneDataset
import os
import sys
import json
from os import makedirs
from gaussian_renderer import (
    distributed_preprocess3dgs_and_all2all_final,
    render_final
)
import torchvision
from utils.general_utils import safe_state, set_args, init_distributed, set_log_file, get_args, set_cur_iter, get_log_file
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from gaussian_renderer.loss_distribution import load_camera_from_cpu_to_all_gpu, load_camera_from_cpu_to_all_gpu_for_eval, batched_loss_computation
from gaussian_renderer.workload_division import start_strategy_final, finish_strategy_final, DivisionStrategyHistoryFinal
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
from tqdm import tqdm

def render_one_path(model_path, views, gaussians, pipeline, background, path):
    render_path = os.path.join(model_path, "renders")
    makedirs(render_path, exist_ok=True)

    dataset = SceneDataset(views)
    args = get_args()
    view = views[args.reference_idx]
    set_cur_iter(1)

    config = {}
    for idx, (dx, dy, dz) in tqdm(enumerate(path)):
        # print(f"Rendering {idx} / {len(path)}")
        file_name = '{0:05d}'.format(args.reference_idx) + f"_{idx}"
        config[idx] = (args.reference_idx, dx, dy, dz)
        if os.path.exists(os.path.join(render_path, file_name + ".png")):
            continue

        view.update(dx=dx, dy=dy, dz=dz)
        strategy_history = DivisionStrategyHistoryFinal(dataset, utils.DEFAULT_GROUP.size(), utils.DEFAULT_GROUP.rank())
        batched_strategies, gpuid2tasks = start_strategy_final([view], strategy_history)
        load_camera_from_cpu_to_all_gpu_for_eval([view], batched_strategies, gpuid2tasks)

        batched_screenspace_pkg = distributed_preprocess3dgs_and_all2all_final([view], gaussians, pipeline, background,
                                                                                batched_strategies=batched_strategies,
                                                                                mode="test")
        rendered_image, _ = render_final(batched_screenspace_pkg, batched_strategies)
        torchvision.utils.save_image(rendered_image, os.path.join(render_path, file_name + ".png"))
        json.dump(config, open(os.path.join(render_path, "config.json"), "w"))


def visualization_generation(dataset : ModelParams, pipeline : PipelineParams):
    with torch.no_grad():
        args = utils.get_args()
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(args, gaussians, load_iteration=None, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


        rubble_start = (-1, -7, 1)
        rubble_end = (-1, -0.5, 0.5)
        rubble_path_length = 400
        rubble_path = []
        for i in range(rubble_path_length):# This generated path diverges a little bit
            t = float(i) / float(rubble_path_length)
            dx = rubble_start[0] * (1 - t) + rubble_end[0] * t
            dy = rubble_start[1] * (1 - t) + rubble_end[1] * t
            dz = rubble_start[2] * (1 - t) + rubble_end[2] * t
            rubble_path.append((dx, dy, dz))
        path = rubble_path + [(0,0,0)]

        mat_start = (-4, -12, 16)
        mat_end = (-2.5, -0.5, 0)
        mat_path_length = 400
        mat_path = []
        for i in range(mat_path_length):
            t = float(i) / float(mat_path_length)
            dx = mat_start[0] * (1 - t) + mat_end[0] * t
            dy = mat_start[1] * (1 - t) + mat_end[1] * t
            dz = mat_start[2] * (1 - t) + mat_end[2] * t
            mat_path.append((dx, dy, dz))
        # path = mat_path + [(0,0,0)]

        render_one_path(dataset.model_path, scene.getTestCameras(), gaussians, pipeline, background,
                        path)


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
    parser.add_argument("--load_ply_path", default="", type=str)
    parser.add_argument("--distributed_load", action="store_true")
    parser.add_argument("--reference_idx", default=-1, type=int)
    args = get_combined_args(parser) #TODO: load cfg_args path fix, this is a weird bug to be None.
    if args.load_ply_path == "":
        print("Please provide a ply path.")
        exit()
    print("Rendering " + args.model_path)
    init_distributed(args)
    args.num_train_cameras = 0
    args.num_test_cameras = args.reference_idx + 1

    ## Prepare arguments.
    # Check arguments
    init_args(args)

    # os.makedirs(args.model_path, exist_ok=True)
    log_file = open(args.model_path+f"/vis_generate_ws={utils.DEFAULT_GROUP.size()}_rk_{utils.DEFAULT_GROUP.rank()}.log", 'w')
    set_log_file(log_file)

    # Set up global args
    set_args(args)
    print_all_args(args, log_file)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    visualization_generation(lp.extract(args), pp.extract(args))