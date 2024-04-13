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
import sys
from datetime import datetime
import numpy as np
import random
import os
import torch.distributed as dist
# from torch.distributed.device_mesh import init_device_mesh
import time
from argparse import Namespace

ARGS = None
LOG_FILE = None
CUR_ITER = None
GLOBAL_RANK = None # rank in all nodes
LOCAL_RANK = 0 # local rank in the node
WORLD_SIZE = 1
DP_GROUP = None
MP_GROUP = None
DEFAULT_GROUP = None
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

def set_block_size(x, y, z):
    global BLOCK_X, BLOCK_Y, ONE_DIM_BLOCK_SIZE
    BLOCK_X, BLOCK_Y, ONE_DIM_BLOCK_SIZE = x, y, z

def set_img_size(h, w):
    global IMG_H, IMG_W
    IMG_H, IMG_W = h, w

def get_img_size():
    global IMG_H, IMG_W
    return IMG_H, IMG_W

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
    return args.zhx_python_time and ( check_update_at_this_iter(iteration, args.bsz, args.log_interval, 1) or iteration in args.force_python_timer_iterations)

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
    global GLOBAL_RANK, LOCAL_RANK, WORLD_SIZE
    GLOBAL_RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    if WORLD_SIZE > 1:
        torch.distributed.init_process_group("nccl", rank=GLOBAL_RANK, world_size=WORLD_SIZE)
        assert torch.cuda.is_available(), "Distributed mode requires CUDA"
        assert torch.distributed.is_initialized(), "Distributed mode requires init_distributed() to be called first"

        assert WORLD_SIZE % args.dp_size == 0, "World size should be divisible by dp_size"
        args.mp_size = WORLD_SIZE // args.dp_size

        global DP_GROUP, MP_GROUP, DEFAULT_GROUP

        # mesh_2d = init_device_mesh("cuda", (args.dp_size, args.mp_size), mesh_dim_names=("dp", "mp"))
        # # Users can access the underlying process group thru `get_group` API.
        # DP_GROUP = mesh_2d.get_group(mesh_dim="dp")
        # MP_GROUP = mesh_2d.get_group(mesh_dim="mp")
        # DEFAULT_GROUP = dist.group.WORLD

        dp_rank = GLOBAL_RANK // args.mp_size
        mp_rank = GLOBAL_RANK % args.mp_size

        all_DP_GROUP = []
        for rank in range(args.mp_size):
            dp_group_ranks = list(range(rank, WORLD_SIZE, args.mp_size))
            all_DP_GROUP.append(dist.new_group(dp_group_ranks))
        DP_GROUP = all_DP_GROUP[mp_rank]

        all_MP_GROUP = []
        for rank in range(args.dp_size):
            mp_group_ranks = list(range(rank*args.mp_size, (rank+1)*args.mp_size))
            all_MP_GROUP.append(dist.new_group(mp_group_ranks))
        MP_GROUP = all_MP_GROUP[dp_rank]

        DEFAULT_GROUP = dist.group.WORLD

        print("Initializing -> "+" world_size: " + str(WORLD_SIZE)+" rank: " + str(DEFAULT_GROUP.rank()) + "     dp_size: " + str(args.dp_size) + " dp_rank: " + str(DP_GROUP.rank()) + "     mp_size: " + str(args.mp_size) + " mp_rank: " + str(MP_GROUP.rank()))
    else:
        DP_GROUP = SingleGPUGroup()
        MP_GROUP = SingleGPUGroup()
        DEFAULT_GROUP = SingleGPUGroup()

def our_allgather_among_cpu_processes_float_list(data, group):
    ## official implementation: torch.distributed.all_gather_object()
    # all_data = [None for _ in range(group.size())]
    # torch.distributed.all_gather_object(all_data, data, group=group)

    ## my hand-written allgather.
    # data should a list of floats
    assert isinstance(data, list) and isinstance(data[0], float), "data should be a list of float"
    data_gpu = torch.tensor(data, dtype=torch.float32, device="cuda")
    all_data_gpu = torch.empty( (group.size(), len(data_gpu)), dtype=torch.float32, device="cuda")
    torch.distributed.all_gather_into_tensor(all_data_gpu, data_gpu, group=group)

    all_data = all_data_gpu.cpu().tolist()
    return all_data

def get_local_chunk_l_r(array_length, world_size, rank):
    chunk_size = (array_length + world_size - 1) // world_size
    l = rank * chunk_size
    r = min(l + chunk_size, array_length)
    return l, r

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def check_memory_usage_logging(prefix):
    if get_cur_iter() not in [0, 1]:
        return
    args = get_args()
    log_file = get_log_file()
    if hasattr(args, "check_memory_usage") and args.check_memory_usage and log_file is not None:
        log_file.write("check_memory_usage["+prefix+"]: Memory usage: {} GB. Max Memory usage: {} GB.\n".format(
            torch.cuda.memory_allocated() / 1024 / 1024 / 1024,
            torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024)
        )

def PILtoTorch(pil_image, resolution, args, log_file):
    assert pil_image.size == resolution, f"Should not resize. image size {pil_image.size} and {resolution} mismatch should not happen in this current project!"
    # resized_image_PIL = pil_image.resize(resolution)
    resized_image_PIL = pil_image
    if args.time_image_loading:
        start_time = time.time()
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
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
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    global GLOBAL_RANK

    # Set up output folder
    if GLOBAL_RANK != 0:
        return None
    print_rank_0("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer. Disable for now. 
    tb_writer = None
    return tb_writer