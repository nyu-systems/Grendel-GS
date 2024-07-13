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
from torch import nn
import sys
from datetime import datetime
import numpy as np
import random
import os
import torch.distributed as dist
import time
from argparse import Namespace
import psutil

ARGS = None
LOG_FILE = None
CUR_ITER = None
GLOBAL_RANK = None # rank in all nodes
LOCAL_RANK = 0 # local rank in the node
WORLD_SIZE = 1
DP_GROUP = None
MP_GROUP = None
DEFAULT_GROUP = None
IN_NODE_GROUP = None
TIMERS = None
DENSIFY_ITER = 0

def set_args(args):
    global ARGS
    ARGS = args

def get_args():
    global ARGS
    return ARGS

def set_log_file(log_file):
    global LOG_FILE
    LOG_FILE = log_file

def get_log_file():
    global LOG_FILE
    return LOG_FILE

def set_cur_iter(cur_iter):
    global CUR_ITER
    CUR_ITER = cur_iter

def get_cur_iter():
    global CUR_ITER
    return CUR_ITER

def set_timers(timers):
    global TIMERS
    TIMERS = timers

def get_timers():
    global TIMERS
    return TIMERS

BLOCK_X, BLOCK_Y = 16, 16
ONE_DIM_BLOCK_SIZE = 256
IMG_H, IMG_W = None, None
TILE_Y, TILE_X = None, None

def set_block_size(x, y, z):
    global BLOCK_X, BLOCK_Y, ONE_DIM_BLOCK_SIZE
    BLOCK_X, BLOCK_Y, ONE_DIM_BLOCK_SIZE = x, y, z

def set_img_size(h, w):
    global IMG_H, IMG_W, TILE_Y, TILE_X
    IMG_H, IMG_W = h, w
    TILE_Y = (IMG_H + BLOCK_Y - 1) // BLOCK_Y
    TILE_X = (IMG_W + BLOCK_X - 1) // BLOCK_X

def get_img_size():
    global IMG_H, IMG_W
    return IMG_H, IMG_W

def get_img_width():
    global IMG_W
    return IMG_W

def get_img_height():
    global IMG_H
    return IMG_H

def get_num_pixels():
    global IMG_H, IMG_W
    return IMG_H * IMG_W

def get_denfify_iter():
    global DENSIFY_ITER
    return DENSIFY_ITER

def inc_densify_iter():
    global DENSIFY_ITER
    DENSIFY_ITER += 1

def print_rank_0(str):
    global GLOBAL_RANK
    if GLOBAL_RANK == 0:
        print(str)


def check_enable_python_timer():
    args = get_args()
    iteration = get_cur_iter()
    return args.enable_timer and check_update_at_this_iter(iteration, args.bsz, args.log_interval, 1)

def globally_sync_for_timer():
    global DEFAULT_GROUP
    if check_enable_python_timer() and DEFAULT_GROUP.size() > 1:
        torch.distributed.barrier(group=DEFAULT_GROUP)

def check_update_at_this_iter(iteration, bsz, update_interval, update_residual):
    residual_l = iteration % update_interval
    residual_r = residual_l + bsz
    # residual_l <= update_residual < residual_r
    if residual_l <= update_residual and update_residual < residual_r:
        return True
    # residual_l <= update_residual+update_interval < residual_r
    if residual_l <= update_residual+update_interval and update_residual+update_interval < residual_r:
        return True
    return False


class SingleGPUGroup:
    def __init__(self):
        pass

    def rank(self):
        return 0

    def size(self):
        return 1

def check_comm_group():
    tensor = torch.ones(1, device="cuda")
    if WORLD_SIZE > 1:
        torch.distributed.all_reduce(tensor, group=DEFAULT_GROUP)
        print(f"DEFAULT_GROUP.rank() {DEFAULT_GROUP.rank()} tensor: {tensor.item()}\n", flush=True)
    tensor = torch.ones(1, device="cuda")
    if DP_GROUP.size() > 1:
        torch.distributed.all_reduce(tensor, group=DP_GROUP)
        print(f"DP_GROUP.rank() {DP_GROUP.rank()} tensor: {tensor.item()}\n", flush=True)
    tensor = torch.ones(1, device="cuda")
    if MP_GROUP.size() > 1:
        torch.distributed.all_reduce(tensor, group=MP_GROUP)
        print(f"MP_GROUP.rank() {MP_GROUP.rank()} tensor: {tensor.item()}\n", flush=True)

def init_distributed(args):
    global GLOBAL_RANK, LOCAL_RANK, WORLD_SIZE, DEFAULT_GROUP, IN_NODE_GROUP
    GLOBAL_RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    if WORLD_SIZE > 1:
        torch.distributed.init_process_group("nccl", rank=GLOBAL_RANK, world_size=WORLD_SIZE)
        assert torch.cuda.is_available(), "Distributed mode requires CUDA"
        assert torch.distributed.is_initialized(), "Distributed mode requires init_distributed() to be called first"

        DEFAULT_GROUP = dist.group.WORLD

        num_gpu_per_node = one_node_device_count()
        n_of_nodes = WORLD_SIZE // num_gpu_per_node
        all_in_node_group = []
        for rank in range(n_of_nodes):
            in_node_group_ranks = list(range(rank*num_gpu_per_node, (rank+1)*num_gpu_per_node))
            all_in_node_group.append(dist.new_group(in_node_group_ranks))
        node_rank = GLOBAL_RANK // num_gpu_per_node
        IN_NODE_GROUP = all_in_node_group[node_rank]
        print("Initializing -> "+" world_size: " + str(WORLD_SIZE)+" rank: " + str(DEFAULT_GROUP.rank()) + "     in_node_size: " + str(IN_NODE_GROUP.size()) + " in_node_rank: " + str(IN_NODE_GROUP.rank()))

    else:
        DEFAULT_GROUP = SingleGPUGroup()
        IN_NODE_GROUP = SingleGPUGroup()

def one_node_device_count():
    global WORLD_SIZE
    return min(torch.cuda.device_count(), WORLD_SIZE)

def get_first_rank_on_cur_node():
    global GLOBAL_RANK
    NODE_ID = GLOBAL_RANK // one_node_device_count()
    first_rank_in_node = NODE_ID * one_node_device_count()
    return first_rank_in_node

def our_allgather_among_cpu_processes_float_list(data, group):
    ## official implementation: torch.distributed.all_gather_object()
    # all_data = [None for _ in range(group.size())]
    # torch.distributed.all_gather_object(all_data, data, group=group)

    ## my hand-written allgather.
    # data should a list of floats
    assert isinstance(data, list) and isinstance(data[0], float), "data should be a list of float"
    data_gpu = torch.tensor(data, dtype=torch.float32, device="cuda")
    all_data_gpu = torch.empty( (group.size(), len(data_gpu)), dtype=torch.float32, device="cuda")
    if group.size() > 1:
        torch.distributed.all_gather_into_tensor(all_data_gpu, data_gpu, group=group)
    else:
        all_data_gpu = data_gpu.unsqueeze(0)

    all_data = all_data_gpu.cpu().tolist()
    return all_data

def get_local_chunk_l_r(array_length, world_size, rank):
    chunk_size = (array_length + world_size - 1) // world_size
    l = rank * chunk_size
    r = min(l + chunk_size, array_length)
    return l, r

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def check_initial_gpu_memory_usage(prefix):
    if get_cur_iter() not in [0, 1]:
        return
    args = get_args()
    log_file = get_log_file()
    if hasattr(args, "check_gpu_memory") and args.check_gpu_memory and log_file is not None:
        log_file.write("check_gpu_memory["+prefix+"]: Memory usage: {} GB. Max Memory usage: {} GB.\n".format(
            torch.cuda.memory_allocated() / 1024 / 1024 / 1024,
            torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024)
        )

def check_memory_usage(log_file, args, iteration, gaussians, before_densification_stop):
    global DEFAULT_GROUP

    memory_usage = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
    max_memory_usage = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
    max_reserved_memory = torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024
    now_reserved_memory = torch.cuda.memory_reserved() / 1024 / 1024 / 1024
    log_str = ""
    log_str += "iteration[{},{}) {}Now num of 3dgs: {}. Now Memory usage: {} GB. Max Memory usage: {} GB. Max Reserved Memory: {} GB. Now Reserved Memory: {} GB. \n".format(
        iteration, iteration+args.bsz,
        "densify_and_prune. " if not before_densification_stop else "",
        gaussians.get_xyz.shape[0], 
        memory_usage, max_memory_usage, max_reserved_memory, now_reserved_memory)
    if args.log_memory_summary:
        log_str += "Memory Summary: {} GB \n".format(torch.cuda.memory_summary())

    if args.check_gpu_memory:
        log_file.write(log_str)

    if before_densification_stop:
        memory_usage_list = our_allgather_among_cpu_processes_float_list([max_reserved_memory], DEFAULT_GROUP)
        # print("total memory: ", torch.cuda.get_device_properties(0).total_memory)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
        if max([a[0] for a in memory_usage_list]) > args.densify_memory_limit_percentage * total_memory:# If memory usage is reaching the upper bound of GPU memory, stop densification to avoid OOM by fragmentation and etc.
            print("Reserved Memory usage is reaching the upper bound of GPU memory. stop densification.\n")
            log_file.write("Reserved Memory usage is reaching the upper bound of GPU memory. stop densification.\n")
            args.disable_auto_densification = True

def PILtoTorch(pil_image, resolution, args, log_file, decompressed_image=None):
    if decompressed_image is not None:
        return decompressed_image
    pil_image.load()
    resized_image_PIL = pil_image.resize(resolution)
    if args.time_image_loading:
        start_time = time.time()
    resized_image = torch.from_numpy(np.array(resized_image_PIL))
    if args.time_image_loading:
        log_file.write(f"pil->numpy->torch in {time.time() - start_time} seconds\n")
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    global LOCAL_RANK
    torch.cuda.set_device(torch.device("cuda", LOCAL_RANK))

def prepare_output_and_logger(args):
    global GLOBAL_RANK

    # Set up output folder
    if GLOBAL_RANK != 0:
        return
    print_rank_0("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:# TODO: I want to delete cfg_args file.
        cfg_log_f.write(str(Namespace(**vars(args))))

def log_cpu_memory_usage(position_str):
    args = get_args()
    if not args.check_cpu_memory:
        return
    LOG_FILE.write("[Check CPU Memory]"+position_str+" ->  Memory Usage: {} GB. Available Memory: {} GB. Total memory: {} GB\n".format(
        psutil.virtual_memory().used / 1024 / 1024 / 1024,
        psutil.virtual_memory().available / 1024 / 1024 / 1024,
        psutil.virtual_memory().total / 1024 / 1024 / 1024))

def merge_multiple_checkpoints(checkpoint_files):
    global LOCAL_RANK

    all_model_params = []
    start_from_this_iteration = 0
    for checkpoint_file in checkpoint_files:
        (model_params, start_from_this_iteration) = torch.load(checkpoint_file, map_location=f"cuda:{LOCAL_RANK}")
        all_model_params.append(model_params)
    
    active_sh_degree = all_model_params[0][0]

    xyz = torch.cat([model_params[1] for model_params in all_model_params], dim=0)
    features_dc = torch.cat([model_params[2] for model_params in all_model_params], dim=0)
    features_rest = torch.cat([model_params[3] for model_params in all_model_params], dim=0)
    scaling = torch.cat([model_params[4] for model_params in all_model_params], dim=0)
    rotation = torch.cat([model_params[5] for model_params in all_model_params], dim=0)
    opacity = torch.cat([model_params[6] for model_params in all_model_params], dim=0)
    max_radii2D = torch.cat([model_params[7] for model_params in all_model_params], dim=0)
    xyz_gradient_accum = torch.cat([model_params[8] for model_params in all_model_params], dim=0)
    denom = torch.cat([model_params[9] for model_params in all_model_params], dim=0)
    opt_dict = None
    spatial_lr_scale = all_model_params[0][-1]

    merged_model_params = (active_sh_degree,
                           nn.Parameter(xyz.requires_grad_(True)),
                           nn.Parameter(features_dc.requires_grad_(True)),
                           nn.Parameter(features_rest.requires_grad_(True)),
                           nn.Parameter(scaling.requires_grad_(True)),
                           nn.Parameter(rotation.requires_grad_(True)),
                           nn.Parameter(opacity.requires_grad_(True)),
                           max_radii2D, 
                           xyz_gradient_accum, 
                           denom, 
                           opt_dict, 
                           spatial_lr_scale)

    return merged_model_params, start_from_this_iteration

def get_part_of_checkpoints(checkpoint_file, num_parts, part_id):
    global LOCAL_RANK

    (model_params, start_from_this_iteration) = torch.load(checkpoint_file, map_location=f"cuda:{LOCAL_RANK}")

    num_gaussians = model_params[1].shape[0]
    num_gaussians_per_part = num_gaussians // num_parts + 1
    start_idx = part_id * num_gaussians_per_part
    end_idx = min((part_id+1) * num_gaussians_per_part, num_gaussians)

    active_sh_degree = model_params[0]
    xyz = model_params[1][start_idx:end_idx]
    features_dc = model_params[2][start_idx:end_idx]
    features_rest = model_params[3][start_idx:end_idx]
    scaling = model_params[4][start_idx:end_idx]
    rotation = model_params[5][start_idx:end_idx]
    opacity = model_params[6][start_idx:end_idx]
    max_radii2D = model_params[7][start_idx:end_idx]
    xyz_gradient_accum = model_params[8][start_idx:end_idx]
    denom = model_params[9][start_idx:end_idx]
    opt_dict = None
    spatial_lr_scale = model_params[11]

    new_model_params = (active_sh_degree,
            nn.Parameter(xyz.requires_grad_(True)),
            nn.Parameter(features_dc.requires_grad_(True)),
            nn.Parameter(features_rest.requires_grad_(True)),
            nn.Parameter(scaling.requires_grad_(True)),
            nn.Parameter(rotation.requires_grad_(True)),
            nn.Parameter(opacity.requires_grad_(True)),
            max_radii2D, 
            xyz_gradient_accum, 
            denom, 
            opt_dict, 
            spatial_lr_scale)
    return new_model_params, start_from_this_iteration


def drop_duplicate_gaussians(model_params, drop_duplicate_gaussians_coeff):
    if drop_duplicate_gaussians_coeff == 1.0:
        return model_params

    active_sh_degree = model_params[0]
    xyz = model_params[1]
    features_dc = model_params[2]
    features_rest = model_params[3]
    scaling = model_params[4]
    rotation = model_params[5]
    opacity = model_params[6]
    max_radii2D = model_params[7]
    xyz_gradient_accum = model_params[8]
    denom = model_params[9]
    opt_dict = None
    spatial_lr_scale = model_params[11]

    all_indices = torch.arange(int(xyz.shape[0]*drop_duplicate_gaussians_coeff), device=xyz.device)
    keep_indices = all_indices % xyz.shape[0]

    return (
        active_sh_degree,
        nn.Parameter(xyz[keep_indices].requires_grad_(True)),
        nn.Parameter(features_dc[keep_indices].requires_grad_(True)),
        nn.Parameter(features_rest[keep_indices].requires_grad_(True)),
        nn.Parameter(scaling[keep_indices].requires_grad_(True)),
        nn.Parameter(rotation[keep_indices].requires_grad_(True)),
        nn.Parameter(opacity[keep_indices].requires_grad_(True)),
        max_radii2D[keep_indices],
        xyz_gradient_accum[keep_indices],
        denom[keep_indices],
        opt_dict,
        spatial_lr_scale
    )


def load_checkpoint(args):
    # TODO: merge these loading functions into a single one.

    global DEFAULT_GROUP

    number_files = len(os.listdir(args.start_checkpoint))
    if args.start_checkpoint[-1] != "/":
        args.start_checkpoint += "/"
    if number_files == DEFAULT_GROUP.size():
        # file_name = args.start_checkpoint+"chkpnt" + str(DEFAULT_GROUP.rank()) + ".pth"
        file_name = args.start_checkpoint+"chkpnt_ws="+str(number_files)+"_rk="+str(DEFAULT_GROUP.rank())+".pth"
        (model_params, start_from_this_iteration) = torch.load(file_name)

    elif number_files > DEFAULT_GROUP.size():
        assert number_files % DEFAULT_GROUP.size() == 0, "The number of files in the checkpoint folder must be a multiple of the number of processes."
        local_processed_file_names = []
        for i in range(DEFAULT_GROUP.rank(), number_files, DEFAULT_GROUP.size()):
            local_processed_file_names.append(args.start_checkpoint+"chkpnt_ws="+str(number_files)+"_rk="+str(i)+".pth")
        (model_params, start_from_this_iteration) = merge_multiple_checkpoints(local_processed_file_names)
        file_name = local_processed_file_names
    elif number_files < DEFAULT_GROUP.size():
        assert DEFAULT_GROUP.size() % number_files == 0, "The number of files in the checkpoint folder must be a divisor of the number of processes."
        # file_name = args.start_checkpoint+"chkpnt" + str(DEFAULT_GROUP.rank() % number_files) + ".pth"
        file_name = args.start_checkpoint+"chkpnt_ws="+str(number_files)+"_rk="+str(DEFAULT_GROUP.rank() % number_files)+".pth"
        (model_params, start_from_this_iteration) = get_part_of_checkpoints(file_name, DEFAULT_GROUP.size()//number_files, DEFAULT_GROUP.rank()//number_files)

    if args.drop_duplicate_gaussians_coeff != 1.0:
        model_params = drop_duplicate_gaussians(model_params, args.drop_duplicate_gaussians_coeff)
    
    return model_params, start_from_this_iteration