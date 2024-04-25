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
        self.ip = "127.0.0.1"
        self.port = 6009
        self.debug_from = -1
        self.detect_anomaly = False
        self.test_iterations = [7_000, 30_000]
        self.save_iterations = []
        self.quiet = False
        self.checkpoint_iterations = []
        self.start_checkpoint = ""
        self.log_folder = "experiments/default_folder"
        self.log_interval = 50
        self.debug_why = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        return g

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = "/scratch/hz3496/gs_tmp"
        self._images = "images"
        self._resolution = 1 # set it to 1 to disable resizing. In current project, we do not resize images because we want to support larger resolution image. 
        self.train_resolution_scale = 1.0
        self.test_resolution_scale = 1.0
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
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
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.random_background = False
        self.min_opacity = 0.005
        self.lr_scale_mode = "linear" # can be "linear", "sqrt", or "accumu"
        super().__init__(parser, "Optimization Parameters")

class DistributionParams(ParamGroup):
    def __init__(self, parser):
        # Distribution for pixel-wise render computation.
        self.image_distribution = True
        self.image_distribution_mode = ""
        # self.render_distribution_adjust_mode = "1" # render distribution strategy adjustment mode for pixel-wise render: choices are "1", "2", ... ; TODO: rename. 
        self.heuristic_decay = 0.0 # decay factor for heuristic used in pixel-wise render distribution adjustment. 
        self.stop_adjust_if_workloads_well_balanced = True # if current strategy is well balanced, we do not have to update strategy. 
        self.lazy_load_image = True # lazily move image to gpu. Dataset is always large, saving all images on gpu always leads to OOM. 
        self.dist_global_strategy = "" # if self.render_distribution_adjust_mode == "3", we set the flag `global distribution strategy` for pixel-wise render computation. 
        self.adjust_strategy_warmp_iterations = 20 # do not use the running statistics in the first self.adjust_strategy_warmp_iterations iterations.
        # self.render_distribution_unbalance_threshold = 0.06 # threshold to adjust distribution ratio for pixel-wise render computation: min*self.render_distribution_unbalance_threshold < max --> redistribute.
        self.image_distribution_unbalance_threshold = 0.06 # threshold to adjust distribution ratio for pixel-wise render computation: min*self.render_distribution_unbalance_threshold < max --> redistribute.

        # Distribution for 3DGS-wise workloads.
        # self.memory_distribution_mode = "0"
        # "0" no shard 3dgs storage. memory_distribution must be false <==> memory_distribution_mode is 0.
        # "1" is shard 3dgs storage across MP group and gradient sync. 
        # "2" is shard 3dgs storage across the global group and use all2all to replace the gradient sync. 
        self.gaussians_distribution = False
        self.redistribute_gaussians_mode = "no_redistribute" # enable redistribution for 3DGS storage location. "no_redistribute" is no redistribution. 
        self.redistribute_gaussians_frequency = 10 # redistribution frequency for 3DGS storage location.
        self.redistribute_gaussians_threshold = 1.1 # threshold to apply redistribution for 3DGS storage location: min*self.redistribute_gaussian_threshold < max --> redistribute.

        # Distribution for pixel-wise loss computation.
        # self.loss_distribution_mode = "general" # "no_distribution", "general", "fast_less_comm", "fast_less_comm_noallreduceloss", "allreduce", "fast" and "functional_allreduce". 
        self.get_global_exact_loss = False # if True, we recompute loss without redundant border pixel computation to get exact number. This is for debugging.

        # Data Parallel
        self.bsz = 1 # batch size. currently, our implementation is just gradient accumulation. 
        self.dp_size = 1 # data parallel degree.
        self.grad_normalization_mode = "none" # "divide_by_visible_count", "square_multiply_by_visible_count", "multiply_by_visible_count", "none" gradient normalization mode. 
        self.mp_size = -1 # model parallel degree.
        self.sync_grad_mode = "dense" # "dense", "sparse", "fused_dense", "fused_sparse" gradient synchronization. 

        self.distributed_dataset_storage = False # if True, we store dataset only on rank 0 and broadcast to other ranks.
        self.async_load_gt_image = False
        self.multiprocesses_image_loading = False
        self.num_train_cameras = -1
        self.distributed_save = False

        super().__init__(parser, "Distribution Parameters")

class BenchmarkParams(ParamGroup):
    def __init__(self, parser):
        self.zhx_time = False # log some steps' running time with cuda events timer.
        self.zhx_python_time = False # log some steps' running time with python timer.
        self.end2end_time = False # log end2end training time.
        self.check_memory_usage = False # check memory usage.
        self.log_iteration_memory_usage = False # log memory usage for every iteration.

        self.check_cpu_memory = False # check cpu memory usage.

        self.benchmark_stats = False # Benchmark mode: it will enable some flags to log detailed statistics to research purposes.
        self.performance_stats = False # Performance mode: to know its generation quality, it will evaluate/save models at some iterations and use them for render.py and metrics.py .

        self.analyze_3dgs_change = False # log some 3dgs parameters change to analyze 3dgs change.

        super().__init__(parser, "Benchmark Parameters")

class DebugParams(ParamGroup):
    def __init__(self, parser):
        self.zhx_debug = False # log debug information that zhx needs.
        self.fixed_training_image = -1 # if not -1, use this image as the training image.
        self.disable_auto_densification = False # disable auto densification.
        self.stop_update_param = False # stop updating parameters. No optimizer.step() will be called.
        self.force_python_timer_iterations = [600, 700, 800] # forcibly print timers at these iterations.

        self.save_i2jsend = False # Deprecated. It was used to save i2jsend_size to file for debugging. Now, we save size of communication from gpui to gpuj in strategy_history_ws=4_rk=0.json .
        self.time_image_loading = False # Log image loading time.
        self.save_send_to_gpui_cnt = False # Save send_to_gpui_cnt to file for debugging. save in send_to_gpui_cnt_ws=4_rk=0.json .

        self.nsys_profile = False # profile with nsys.
        self.drop_initial_3dgs_p = 0.0 # profile with nsys.

        super().__init__(parser, "Debug Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
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

    log_file.write("world_size: " + str(utils.WORLD_SIZE)+" rank: " + str(utils.GLOBAL_RANK) + "; dp_size: " + str(args.dp_size) + " dp_rank: " + str(utils.DP_GROUP.rank()) + "; mp_size: " + str(args.mp_size) + " mp_rank: " + str(utils.MP_GROUP.rank())+"\n")

    # Make sure block size match between python and cuda code.
    cuda_block_x, cuda_block_y, one_dim_block_size = diff_gaussian_rasterization._C.get_block_XY()
    utils.set_block_size(cuda_block_x, cuda_block_y, one_dim_block_size)
    log_file.write("cuda_block_x: {}; cuda_block_y: {}; one_dim_block_size: {};\n".format(cuda_block_x, cuda_block_y, one_dim_block_size))


def init_args(args):

    # Check arguments
    assert not (args.benchmark_stats and args.performance_stats), "benchmark_stats and performance_stats can not be enabled at the same time."

    if len(args.save_iterations) > 0 and args.iterations not in args.save_iterations:
        args.save_iterations.append(args.iterations)

    if args.benchmark_stats:
        args.zhx_time = True
        args.zhx_python_time = True
        args.log_iteration_memory_usage = True
        args.check_memory_usage = True
        args.end2end_time = True

        args.save_iterations = []
        assert args.fixed_training_image == -1, "benchmark mode does not support fixed_training_image."
        assert not args.disable_auto_densification, "benchmark mode needs auto densification."
        assert not args.save_i2jsend, "benchmark mode does not support save_i2jsend."
        assert not args.stop_update_param, "benchmark mode does not support stop_update_param."

    if args.performance_stats:
        args.eval = True
        args.zhx_time = False
        args.zhx_python_time = False
        args.end2end_time = True
        args.log_iteration_memory_usage = False
        args.check_memory_usage = False
        args.test_iterations = [500]+ [i for i in range(2000, args.iterations+1, 1000)]

        args.lazy_load_image = True

        assert args.fixed_training_image == -1, "performance_stats mode does not support fixed_training_image."
        assert not args.disable_auto_densification, "performance_stats mode needs auto densification."
        assert not args.save_i2jsend, "performance_stats mode does not support save_i2jsend."
        assert not args.stop_update_param, "performance_stats mode does not support stop_update_param."

    if args.fixed_training_image != -1:
        args.test_iterations = [] # disable testing during training.
        args.disable_auto_densification = True
    
    if args.log_iteration_memory_usage:
        args.check_memory_usage = True

    if utils.DEFAULT_GROUP.size() == 1:
        args.gaussians_distribution = False
        args.image_distribution = False
        args.image_distribution_mode = "0"
        args.distributed_dataset_storage = False

    if utils.MP_GROUP.size() == 1:
        args.image_distribution_mode = "0"

    if not args.gaussians_distribution:
        args.distributed_save = False

    assert args.bsz % args.dp_size == 0, "dp worker should compute equal number of samples, for now."

    # sort test_iterations
    args.test_iterations.sort()

    init_image_distribution_config(args)