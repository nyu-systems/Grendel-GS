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
import utils.general_utils as utils

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
                    group.add_argument("--" + key, default=value, nargs="+", type=type(value[0]))
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = "/scratch/hz3496/gs_tmp"
        self._images = "images"
        self._resolution = 1 # set it to 1 to disable resizing. In current project, we do not resize images because we want to support larger resolution image. 
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
        super().__init__(parser, "Optimization Parameters")

class DistributionParams(ParamGroup):
    def __init__(self, parser):
        # Distribution for pixel-wise render computation.
        self.render_distribution = True # default to be True if world_size > 1.
        self.render_distribution_mode = "1" # render distribution strategy adjustment mode for pixel-wise render: choices are "1", "2", ... ; TODO: rename. 
        self.heuristic_decay = 0.0 # decay factor for heuristic used in pixel-wise render distribution adjustment. 
        self.stop_adjust_if_workloads_well_balanced = True # if current strategy is well balanced, we do not have to update strategy. 
        self.lazy_load_image = True # lazily move image to gpu. Dataset is always large, saving all images on gpu always leads to OOM. 
        self.dist_global_strategy = "" # if self.adjust_mode == "3", we set the flag `global distribution strategy` for pixel-wise render computation. 
        self.adjust_strategy_warmp_iterations = 20 # do not use the running statistics in the first self.adjust_strategy_warmp_iterations iterations.
        self.render_distribution_unbalance_threshold = 0.06 # threshold to adjust distribution ratio for pixel-wise render computation: min*self.render_distribution_unbalance_threshold < max --> redistribute.

        # Distribution for 3DGS-wise workloads.
        self.memory_distribution = True # enable distribution for 3DGS storage memory and preprocess forward and backward compute. 
        self.redistribute_gaussians_mode = "no_redistribute" # enable redistribution for 3DGS storage location. "no_redistribute" is no redistribution. 
        self.redistribute_gaussians_frequency = 10 # redistribution frequency for 3DGS storage location.
        self.redistribute_gaussians_threshold = 1.1 # threshold to apply redistribution for 3DGS storage location: min*self.redistribute_gaussian_threshold < max --> redistribute.

        # Distribution for pixel-wise loss computation.
        self.loss_distribution = True # enable distribution for pixel-wise loss computation.
        self.loss_distribution_mode = "general" # "general", "fast_less_comm", "fast_less_comm_noallreduceloss", "allreduce", "fast" and "functional_allreduce". 
        self.get_global_exact_loss = False # if True, we recompute loss without redundant border pixel computation to get exact number. This is for debugging.

        # Data Parallel
        self.bsz = 1 # batch size. currently, our implementation is just gradient accumulation. 
        self.dp_size = 1 # data parallel degree.
        self.mp_size = -1 # model parallel degree.
        # data parallel mode.
        # "1" is shard 3dgs storage across MP group and gradient sync. 
        # "2" is shard 3dgs storage across the global group and use all2all to replace the gradient sync. 
        self.dp_mode = "1" 


        # Deprecated Arguments
        self.dist_division_mode = "tile_num" # Deprecated
        self.adjust_div_stra = False # Deprecated. Distribution strategy adjustment during training for pixel-wise render computation.
        self.enable_redistribute = True # Deprecated. Enable redistribution for 3dgs storage location.
        self.redistribution_mode = "0" # Deprecated. Redistribution mode for 3dgs storage location.
        self.redistribute_frequency = 10 # Deprecated. Redistribution frequency for 3dgs storage location.
        self.stop_adjust2_well_balanced = False # Deprecated. Stop adjustment if workloads are well balanced.
        self.img_dist_compile_mode = "general" # Deprecated. Distribution mode for pixel-wise loss computation.
        self.image_distribution = True # Deprecated. Distribution for pixel-wise loss computation.
        self.adjust_mode = "1" # Deprecated. 
        self.avoid_pixel_all2all = False # Deprecated. avoid pixel-wise all2all communication by replicated border pixel rendering during forward. 
        self.avoid_pixel_all2all_log_correctloss = False # Deprecated. log correct loss for pixel-wise all2all communication.


        super().__init__(parser, "Distribution Parameters")

class BenchmarkParams(ParamGroup):
    def __init__(self, parser):
        self.zhx_time = False # log some steps' running time with cuda events timer.
        self.zhx_python_time = False # log some steps' running time with python timer.
        self.end2end_time = False # log end2end training time.
        self.check_memory_usage = False # check memory usage.
        self.log_iteration_memory_usage = False # log memory usage for every iteration.

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
        self.disable_checkpoint_and_save = False # Disable checkpoint and save.
        self.save_send_to_gpui_cnt = False # Save send_to_gpui_cnt to file for debugging. save in send_to_gpui_cnt_ws=4_rk=0.json .

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

def check_args(args):

    # Check arguments
    assert not (args.benchmark_stats and args.performance_stats), "benchmark_stats and performance_stats can not be enabled at the same time."

    if args.benchmark_stats:
        args.zhx_time = True
        args.zhx_python_time = True
        args.log_iteration_memory_usage = True
        args.check_memory_usage = True
        args.end2end_time = True
        args.disable_checkpoint_and_save = True
        args.checkpoint_iterations = []
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
        args.save_iterations = [2000, 7000, 15000, 30000]
        args.test_iterations = [500]+ [i for i in range(2000, args.iterations+1, 1000)]
        args.checkpoint_iterations = []

        # use the fastest mode.
        args.adjust_mode = "2"
        args.lazy_load_image = True
        args.memory_distribution = True
        args.loss_distribution = True

        assert args.fixed_training_image == -1, "performance_stats mode does not support fixed_training_image."
        assert not args.disable_auto_densification, "performance_stats mode needs auto densification."
        assert not args.save_i2jsend, "performance_stats mode does not support save_i2jsend."
        assert not args.stop_update_param, "performance_stats mode does not support stop_update_param."

    if utils.WORLD_SIZE == 1:
        args.render_distribution = False
        args.memory_distribution = False
        args.loss_distribution = False

    assert not (args.memory_distribution and len(args.checkpoint_iterations)>0 ), "memory_distribution does not support checkpoint yet!"
    assert not (args.save_i2jsend and not args.memory_distribution), "save_i2jsend needs memory_distribution!"
    assert not (args.loss_distribution and not args.memory_distribution), "loss_distribution needs memory_distribution!"

    if args.render_distribution_mode == "3":
        assert not args.dist_global_strategy == "", "dist_global_strategy must be set if adjust_mode is 3."

    if args.render_distribution_mode == "5":
        args.loss_distribution_mode = "avoid_pixel_all2all"
        utils.print_rank_0("NOTE! set loss_distribution_mode to `avoid_pixel_all2all` because render_distribution_mode is 5.")

    if args.fixed_training_image != -1:
        args.test_iterations = [] # disable testing during training.
        args.disable_auto_densification = True

    if args.redistribute_gaussians_mode != "no_redistribute":
       utils.print_rank_0(args.memory_distribution)
       assert args.memory_distribution, "enable_redistribute needs memory_distribution!"
       args.disable_checkpoint_and_save = True # checkpoint and save are not implemented in mode of enable_redistribute.

    if args.disable_checkpoint_and_save:
        utils.print_rank_0("Attention! disable_checkpoint_and_save is enabled. disable checkpoint and save.")
        args.checkpoint_iterations = []
        args.save_iterations = []
    
    if args.log_iteration_memory_usage:
        args.check_memory_usage = True
 