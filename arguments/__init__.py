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

from argparse import ArgumentParser, Namespace
import sys
import os
from gaussian_renderer.distribution_config import init_image_distribution_config
import utils.general_utils as utils
import diff_gaussian_rasterization

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                elif t == list:
                    type_to_use = int
                    if len(value) > 0:
                        type_to_use = type(value[0])
                    group.add_argument("--" + key, default=value, nargs="+", type=type_to_use)
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class AuxiliaryParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.debug_from = -1
        self.detect_anomaly = False
        self.test_iterations = [7_000, 30_000]
        self.save_iterations = [7_000, 30_000]
        self.quiet = False
        self.checkpoint_iterations = []
        self.start_checkpoint = ""
        self.auto_start_checkpoint = False
        self.log_folder = "/tmp/gaussian_splatting"
        self.log_interval = 250
        self.llffhold = 8
        self.backend = "diff"
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        return g

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = "/tmp/gaussian_splatting"
        self._images = "images"
        self._white_background = False
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.lr_scale_loss = 1.0
        self.lr_scale_pos_and_scale = 1.0
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.densify_memory_limit_percentage = 0.9
        self.disable_auto_densification = False
        self.opacity_reset_until_iter = -1
        self.random_background = False
        self.min_opacity = 0.005
        self.lr_scale_mode = "sqrt" # can be "linear", "sqrt", or "accumu"
        super().__init__(parser, "Optimization Parameters")

class DistributionParams(ParamGroup):
    def __init__(self, parser):
        # Distribution for pixel-wise workloads.
        self.image_distribution = True
        self.image_distribution_mode = "final"
        self.heuristic_decay = 0.0
        self.no_heuristics_update = False
        self.border_divpos_coeff = 1.0
        self.adjust_strategy_warmp_iterations = -1
        self.save_strategy_history = False

        # Distribution for 3DGS-wise workloads.
        self.gaussians_distribution = True
        self.redistribute_gaussians_mode = "random_redistribute" # "no_redistribute"
        self.redistribute_gaussians_frequency = 10 # redistribution frequency for 3DGS storage location.
        self.redistribute_gaussians_threshold = 1.1 # threshold to apply redistribution for 3DGS storage location
        self.sync_grad_mode = "dense" # "dense", "sparse", "fused_dense", "fused_sparse" gradient synchronization. Only use when gaussians_distribution is False.
        self.grad_normalization_mode = "none" # "divide_by_visible_count", "square_multiply_by_visible_count", "multiply_by_visible_count", "none" gradient normalization mode. 

        # Dataset and Model save
        self.bsz = 1 # batch size.
        self.distributed_dataset_storage = True # if True, we store dataset only on rank 0 and broadcast to other ranks.
        self.distributed_save = True
        self.local_sampling = True
        self.preload_dataset_to_gpu = False # By default, we do not preload dataset to GPU.
        self.preload_dataset_to_gpu_threshold = 10 # unit is GB, by default 10GB memory limit for dataset.
        self.multiprocesses_image_loading = False
        self.num_train_cameras = -1
        self.num_test_cameras = -1

        super().__init__(parser, "Distribution Parameters")

class BenchmarkParams(ParamGroup):
    def __init__(self, parser):
        self.enable_timer = False # Log running time from python side.
        self.end2end_time = True # Log end2end training time.
        self.zhx_time = False # Log running time from gpu side.
        self.check_gpu_memory = False # check gpu memory usage.
        self.check_cpu_memory = False # check cpu memory usage.
        self.log_memory_summary = False

        super().__init__(parser, "Benchmark Parameters")

class DebugParams(ParamGroup):
    def __init__(self, parser):
        self.zhx_debug = False # log debug information that zhx needs.
        self.stop_update_param = False # stop updating parameters. No optimizer.step() will be called.
        self.time_image_loading = False # Log image loading time.

        self.nsys_profile = False # profile with nsys.
        self.drop_initial_3dgs_p = 0.0 # profile with nsys.
        self.drop_duplicate_gaussians_coeff = 1.0

        super().__init__(parser, "Debug Parameters")

def get_combined_args(parser : ArgumentParser, auto_find_cfg_args_path=False):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        if auto_find_cfg_args_path:
            if hasattr(args_cmdline, "load_ply_path"):
                path = args_cmdline.load_ply_path
                while not os.path.exists(os.path.join(path, "cfg_args")) and os.path.exists(path):
                    path = os.path.join(path, "..")
                cfgfilepath = os.path.join(path, "cfg_args")
        else:
            cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)

def print_all_args(args, log_file):
    # print all arguments in a readable format, each argument in a line.
    log_file.write("arguments:\n")
    log_file.write("-"*30+"\n")
    for arg in vars(args):
        log_file.write("{}: {}\n".format(arg, getattr(args, arg)))
    log_file.write("-"*30+"\n\n")
    log_file.write("world_size: " + str(utils.WORLD_SIZE)+" rank: " + str(utils.GLOBAL_RANK) + "; bsz: " + str(args.bsz)+"\n")

    # Make sure block size match between python and cuda code.
    cuda_block_x, cuda_block_y, one_dim_block_size = diff_gaussian_rasterization._C.get_block_XY()
    utils.set_block_size(cuda_block_x, cuda_block_y, one_dim_block_size)
    log_file.write("cuda_block_x: {}; cuda_block_y: {}; one_dim_block_size: {};\n".format(cuda_block_x, cuda_block_y, one_dim_block_size))

def find_latest_checkpoint(log_folder):
    checkpoint_folder = os.path.join(log_folder, "checkpoints")
    if os.path.exists(checkpoint_folder):
        all_sub_folders = os.listdir(checkpoint_folder)
        if len(all_sub_folders) > 0:
            all_sub_folders.sort(key=lambda x: int(x), reverse=True)
            return os.path.join(checkpoint_folder, all_sub_folders[0])
    return ""

def init_args(args):

    if args.opacity_reset_until_iter == -1:
        args.opacity_reset_until_iter = args.densify_until_iter + args.bsz

    # Logging are saved with where model is saved.
    args.log_folder = args.model_path

    if args.auto_start_checkpoint:
        args.start_checkpoint = find_latest_checkpoint(args.log_folder)
    
    if utils.DEFAULT_GROUP.size() == 1:
        args.gaussians_distribution = False
        args.image_distribution = False
        args.image_distribution_mode = ""
        args.distributed_dataset_storage = False
        args.distributed_save = False

    if args.preload_dataset_to_gpu:
        args.distributed_dataset_storage = False

    if not args.gaussians_distribution:
        args.distributed_save = False

    # sort test_iterations
    args.test_iterations.sort()
    args.save_iterations.sort()
    if len(args.save_iterations) > 0 and args.iterations not in args.save_iterations:
        args.save_iterations.append(args.iterations)
    args.checkpoint_iterations.sort()

    # Set up global args
    utils.set_args(args)
    # TODO: handle the warning: https://github.com/pytorch/pytorch/blob/bae409388cfc20cce656bf7b671e45aaf81dd1c8/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L1849-L1852
