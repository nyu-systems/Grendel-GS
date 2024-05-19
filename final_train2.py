import os
import torch
import json
from utils.loss_utils import l1_loss
from gaussian_renderer import (
        distributed_preprocess3dgs_and_all2all_final,
        render_final
    )
from torch.cuda import nvtx
from scene import Scene, GaussianModel, SceneDataset
from gaussian_renderer.workload_division import division_pos_heuristic
from utils.general_utils import prepare_output_and_logger, globally_sync_for_timer
import utils.general_utils as utils
from utils.timer import Timer, End2endTimer
from tqdm import tqdm
from utils.image_utils import psnr
import torch.distributed as dist
from densification import densification
import diff_gaussian_rasterization
import time
from utils.loss_utils import pixelwise_l1_with_mask, pixelwise_ssim_with_mask
from torch import nn

class DivisionStrategyFinal:

    def __init__(self, camera, world_size, gpu_ids, division_pos, gpu_for_this_camera_tilelr):
        assert world_size > 0, "The world_size must be greater than 0."
        assert len(gpu_ids) == world_size, "The number of gpu_ids must be equal to the world_size."
        assert len(division_pos) == world_size+1, "The number of division_pos must be equal to the world_size+1."
        assert division_pos[0] == 0, "The first element of division_pos must be 0."
        assert division_pos[-1] == utils.TILE_Y, "The last element of division_pos must be equal to the total number of tiles."
        for i in range(1, len(division_pos)):
            assert division_pos[i] > division_pos[i-1], "The division_pos must be in ascending order."

        for idx in range(len(gpu_for_this_camera_tilelr)):
            assert gpu_for_this_camera_tilelr[idx][0] == division_pos[idx] and gpu_for_this_camera_tilelr[idx][1] == division_pos[idx+1], "The division_pos must be consistent with gpu_for_this_camera_tilelr."

        self.camera = camera
        self.world_size = world_size
        self.gpu_ids = gpu_ids
        if utils.GLOBAL_RANK in gpu_ids:
            self.rank = gpu_ids.index(utils.GLOBAL_RANK)
        else:
            self.rank = -1

        self.division_pos = division_pos
    
    def get_local2j_ids(self, means2D, radii, raster_settings, cuda_args):
        dist_global_strategy_tensor = torch.tensor(self.division_pos, dtype=torch.int, device=means2D.device) * utils.TILE_X

        args = (
            raster_settings.image_height,
            raster_settings.image_width,
            self.rank,
            self.world_size,
            means2D,
            radii,
            dist_global_strategy_tensor,
            cuda_args
        )

        local2j_ids_bool = diff_gaussian_rasterization._C.get_local2j_ids_bool(*args)

        local2j_ids = []
        for rk in range(self.world_size):
            local2j_ids.append(local2j_ids_bool[:, rk].nonzero())

        return local2j_ids, local2j_ids_bool

    def get_compute_locally(self):
        if utils.GLOBAL_RANK not in self.gpu_ids:
            return None
        rank = self.gpu_ids.index(utils.GLOBAL_RANK)

        tile_ids_l, tile_ids_r = self.division_pos[rank]*utils.TILE_X, self.division_pos[rank+1]*utils.TILE_X
        compute_locally = torch.zeros(utils.TILE_Y*utils.TILE_X, dtype=torch.bool, device="cuda")
        compute_locally[tile_ids_l:tile_ids_r] = True
        compute_locally = compute_locally.view(utils.TILE_Y, utils.TILE_X)
        return compute_locally
    
    def get_extended_compute_locally(self):
        return None


class DivisionStrategyHistoryFinal:
    def __init__(self, dataset, world_size, rank):
        self.world_size = world_size
        self.rank = rank
        self.accum_heuristic = {}
        for camera in dataset.cameras:
            self.accum_heuristic[camera.uid] = torch.ones((utils.TILE_Y, ), dtype=torch.float32, device="cuda", requires_grad=False)

        self.history = []

    def store_stats(self, batched_cameras, gpu_camera_running_time, batched_strategies):
        batched_camera_info = []
        all_camera_running_time = [0 for _ in range(len(batched_cameras))]
        all_gpu_running_time = [0 for _ in range(self.world_size)]
        for camera_id, camera in enumerate(batched_cameras):
            each_gpu_running_time = []
            for gpu_i in batched_strategies[camera_id].gpu_ids:
                all_camera_running_time[camera_id] += gpu_camera_running_time[gpu_i][camera_id]
                all_gpu_running_time[gpu_i] += gpu_camera_running_time[gpu_i][camera_id]
                each_gpu_running_time.append(gpu_camera_running_time[gpu_i][camera_id])

            batched_camera_info.append({
                "camera_id": camera.uid,
                "gpu_ids": batched_strategies[camera_id].gpu_ids,
                "division_pos": batched_strategies[camera_id].division_pos,
                "each_gpu_running_time": each_gpu_running_time,
            })
        self.history.append({
            "iteration": utils.get_cur_iter(),
            "all_gpu_running_time": all_gpu_running_time,
            "all_camera_running_time": all_camera_running_time,
            "batched_camera_info": batched_camera_info,
        })

    def to_json(self):
        return self.history

def start_strategy_final(batched_cameras, strategy_history):
    args = utils.get_args()

    n_tiles_per_image = utils.TILE_Y
    total_tiles = n_tiles_per_image * len(batched_cameras)

    batched_accum_heuristic = [strategy_history.accum_heuristic[camera.uid] for camera in batched_cameras] # batch_size * [tile_y* tile_x]
    catted_accum_heuristic = torch.cat(batched_accum_heuristic, dim=0) # [batch_size * tile_y * tile_x]

    division_pos = division_pos_heuristic(catted_accum_heuristic, total_tiles, utils.DEFAULT_GROUP.size(), right=True)
    # slightly adjust the division_pos to avoid redundant computation.
    for i in range(1, len(division_pos)-1):
        if (division_pos[i] % n_tiles_per_image + args.border_divpos_coeff >= n_tiles_per_image):
            division_pos[i] = division_pos[i] // n_tiles_per_image * n_tiles_per_image + n_tiles_per_image
        elif division_pos[i] % n_tiles_per_image - args.border_divpos_coeff <= 0:
            division_pos[i] = division_pos[i] // n_tiles_per_image * n_tiles_per_image
    for i in range(0, len(division_pos)-1):
        assert division_pos[i] + args.border_divpos_coeff < division_pos[i+1], "The division_pos must be large enough to boundary case error."

    batched_strategies = []
    gpuid2tasks = [[] for _ in range(utils.DEFAULT_GROUP.size())] # map from gpuid to a list of tasks (camera_id, tile_l, tile_r) it should do.
    for idx, camera in enumerate(batched_cameras):
        offset = idx * n_tiles_per_image
        
        gpu_for_this_camera = []
        gpu_for_this_camera_tilelr = []
        for gpu_id in range(utils.DEFAULT_GROUP.size()):
            gpu_tile_l, gpu_tile_r = division_pos[gpu_id], division_pos[gpu_id+1]
            if gpu_tile_r <= offset or offset+n_tiles_per_image <= gpu_tile_l:
                continue
            gpu_for_this_camera.append(gpu_id)
            local_tile_l, local_tile_r = max(gpu_tile_l, offset)-offset, min(gpu_tile_r, offset+n_tiles_per_image)-offset
            gpu_for_this_camera_tilelr.append((local_tile_l, local_tile_r))
            gpuid2tasks[gpu_id].append((idx, local_tile_l, local_tile_r))

        ws_for_this_camera = len(gpu_for_this_camera)
        division_pos_for_this_viewpoint = [0] + [tilelr[1] for tilelr in gpu_for_this_camera_tilelr]
        strategy = DivisionStrategyFinal(camera, ws_for_this_camera, gpu_for_this_camera, division_pos_for_this_viewpoint, gpu_for_this_camera_tilelr)
        batched_strategies.append(strategy)
    return batched_strategies, gpuid2tasks

def finish_strategy_final(batched_cameras, strategy_history, batched_strategies, batched_statistic_collector):
    batched_running_time = []
    for idx, strategy in enumerate(batched_strategies):
        if utils.GLOBAL_RANK not in strategy.gpu_ids:
            batched_running_time.append(-1.0)
            continue

        batched_running_time.append(batched_statistic_collector[idx]["forward_render_time"]+
                                    batched_statistic_collector[idx]["backward_render_time"]+
                                    batched_statistic_collector[idx]["forward_loss_time"]*2)
    
    gpu_camera_running_time = utils.our_allgather_among_cpu_processes_float_list(batched_running_time, utils.DEFAULT_GROUP)
    strategy_history.store_stats(batched_cameras, gpu_camera_running_time, batched_strategies)

    args = utils.get_args()

    if utils.get_cur_iter() <= args.adjust_strategy_warmp_iterations or utils.DEFAULT_GROUP.size() == 1:
        # do not update heuristic in these warmup iterations since these timing are not accurate.
        return

    if args.no_heuristics_update:
        return

    for camera_id, (camera, strategy) in enumerate(zip(batched_cameras, batched_strategies)):
        new_heuristic = torch.zeros((utils.TILE_Y, ), dtype=torch.float32, device="cuda")
        for local_id, gpu_id in enumerate(strategy.gpu_ids):
            tile_ids_l, tile_ids_r = strategy.division_pos[local_id], strategy.division_pos[local_id+1]
            new_heuristic[tile_ids_l:tile_ids_r] = gpu_camera_running_time[gpu_id][camera_id] / (tile_ids_r-tile_ids_l)
        if args.heuristic_decay == 0:
            strategy_history.accum_heuristic[camera.uid] = new_heuristic
        else:
            strategy_history.accum_heuristic[camera.uid] = strategy_history.accum_heuristic[camera.uid] * args.heuristic_decay + new_heuristic * (1-args.heuristic_decay)

def get_coverage_y_min_max(tile_ids_l, tile_ids_r):
    return tile_ids_l*utils.BLOCK_Y, min(tile_ids_r*utils.BLOCK_Y, utils.IMG_H)

def get_coverage_y_min(tile_ids_l):
    return tile_ids_l*utils.BLOCK_Y

def get_coverage_y_max(tile_ids_r):
    return min(tile_ids_r*utils.BLOCK_Y, utils.IMG_H)

def load_camera_from_cpu_to_all_gpu_for_eval(batched_cameras, batched_strategies, gpuid2tasks):
    timers = utils.get_timers()
    args = utils.get_args()

    if args.distributed_dataset_storage:
        if utils.IN_NODE_GROUP.rank() == 0:
            for camera in batched_cameras:
                camera.original_image = camera.original_image_cpu.cuda()
                scatter_list = [camera.original_image for _ in range(utils.IN_NODE_GROUP.size())]
                torch.distributed.scatter(camera.original_image, scatter_list=scatter_list, src=utils.get_first_rank_on_cur_node(), group=utils.IN_NODE_GROUP)
        else:
            for camera in batched_cameras:
                camera.original_image = torch.zeros((3, utils.IMG_H, utils.IMG_W), dtype=torch.uint8, device="cuda")
                torch.distributed.scatter(camera.original_image, scatter_list=None, src=utils.get_first_rank_on_cur_node(), group=utils.IN_NODE_GROUP)
    else:
        for camera in batched_cameras:
            camera.original_image = camera.original_image_cpu.cuda()


def load_camera_from_cpu_to_all_gpu(batched_cameras, batched_strategies, gpuid2tasks):
    timers = utils.get_timers()
    args = utils.get_args()

    # Asynchronously load ground-truth image to GPU
    timers.start("load_gt_image_to_gpu")

    def load_camera_from_cpu_to_gpu(first_task, last_task):
        coverage_min_max_y = {}
        coverage_min_y_first_task = get_coverage_y_min(first_task[1])
        coverage_max_y_last_task = get_coverage_y_max(last_task[2])
        for camera_id_in_batch in range(first_task[0], last_task[0]+1):
            coverage_min_y = 0
            if camera_id_in_batch == first_task[0]:
                coverage_min_y = coverage_min_y_first_task
            coverage_max_y = utils.IMG_H
            if camera_id_in_batch == last_task[0]:
                coverage_max_y = coverage_max_y_last_task

            batched_cameras[camera_id_in_batch].original_image = batched_cameras[camera_id_in_batch].original_image_cpu[:, coverage_min_y:coverage_max_y, :].cuda()
            coverage_min_max_y[camera_id_in_batch] = (coverage_min_y, coverage_max_y)
        return coverage_min_max_y

    if args.distributed_dataset_storage:
        if utils.IN_NODE_GROUP.rank() == 0:
            in_node_first_rank = utils.GLOBAL_RANK
            in_node_last_rank = in_node_first_rank + utils.IN_NODE_GROUP.size() - 1
            first_task = gpuid2tasks[in_node_first_rank][0]
            last_task = gpuid2tasks[in_node_last_rank][-1]
            coverage_min_max_y_gpu0 = load_camera_from_cpu_to_gpu(first_task, last_task)
    else:
        first_task = gpuid2tasks[utils.GLOBAL_RANK][0]
        last_task = gpuid2tasks[utils.GLOBAL_RANK][-1]
        _ = load_camera_from_cpu_to_gpu(first_task, last_task)

    # if utils.DEFAULT_GROUP.size() > 1:
    #     torch.distributed.barrier(group=utils.DEFAULT_GROUP)
    timers.stop("load_gt_image_to_gpu")

    # Asynchronously send the original image from gpu0 to all GPUs in the same node.
    timers.start("scatter_gt_image")
    if args.distributed_dataset_storage:
        comm_ops = []
        if utils.IN_NODE_GROUP.rank() == 0:
            in_node_first_rank = utils.get_first_rank_on_cur_node()
            in_node_last_rank = in_node_first_rank + utils.IN_NODE_GROUP.size() - 1
            for rank in range(in_node_first_rank, in_node_last_rank+1):
                if rank == utils.GLOBAL_RANK:
                    continue
                for task in gpuid2tasks[rank]:
                    camera_id = task[0]
                    coverage_min_y = get_coverage_y_min(task[1])
                    coverage_max_y = get_coverage_y_max(task[2])

                    coverage_min_y_gpu0, coverage_max_y_gpu0 = coverage_min_max_y_gpu0[camera_id]
                    if coverage_min_y == coverage_min_y_gpu0 and coverage_max_y == coverage_max_y_gpu0:
                        # less memory copy
                        op = torch.distributed.P2POp(dist.isend, batched_cameras[camera_id].original_image.contiguous(), rank)
                    else:
                        send_tensor = batched_cameras[camera_id].original_image[:,
                                                                                coverage_min_y-coverage_min_y_gpu0:coverage_max_y-coverage_min_y_gpu0,
                                                                                :].contiguous()
                        op = torch.distributed.P2POp(dist.isend, send_tensor, rank)
                    comm_ops.append(op)

            reqs = torch.distributed.batch_isend_irecv(comm_ops)
            for req in reqs:
                req.wait()

            for task in gpuid2tasks[utils.GLOBAL_RANK]:
                camera_id = task[0]
                coverage_min_y_gpu0, coverage_max_y_gpu0 = coverage_min_max_y_gpu0[camera_id]
                coverage_min_y = get_coverage_y_min(task[1])
                coverage_max_y = get_coverage_y_max(task[2])
                batched_cameras[camera_id].original_image = batched_cameras[camera_id].original_image[:,
                                                                            coverage_min_y-coverage_min_y_gpu0:coverage_max_y-coverage_min_y_gpu0,
                                                                            :].contiguous()
        else:
            in_node_first_rank = utils.get_first_rank_on_cur_node()
            recv_buffer_list = []
            for task in gpuid2tasks[utils.GLOBAL_RANK]:
                coverage_min_y = get_coverage_y_min(task[1])
                coverage_max_y = get_coverage_y_max(task[2])
                recv_buffer = torch.zeros((3, coverage_max_y-coverage_min_y, utils.IMG_W), dtype=torch.uint8, device="cuda")
                recv_buffer_list.append(recv_buffer)
                op = torch.distributed.P2POp(dist.irecv, recv_buffer, in_node_first_rank)
                comm_ops.append(op)

            reqs = torch.distributed.batch_isend_irecv(comm_ops)
            for req in reqs:
                req.wait()

            for idx, task in enumerate(gpuid2tasks[utils.GLOBAL_RANK]):
                batched_cameras[task[0]].original_image = recv_buffer_list[idx]

    timers.stop("scatter_gt_image")

def final_system_loss_computation(image, viewpoint_cam, compute_locally, strategy, statistic_collector):
    timers = utils.get_timers()
    args = utils.get_args()

    timers.start("prepare_image_rect_and_mask")
    assert utils.GLOBAL_RANK in strategy.gpu_ids, "The current gpu must be used to render this camera."
    rank = strategy.gpu_ids.index(utils.GLOBAL_RANK)
    tile_ids_l, tile_ids_r = strategy.division_pos[rank], strategy.division_pos[rank+1]
    coverage_min_y, coverage_max_y = get_coverage_y_min_max(tile_ids_l, tile_ids_r)

    local_image_rect = image[:, coverage_min_y:coverage_max_y, :].contiguous()
    local_image_rect_pixels_compute_locally = torch.ones((coverage_max_y-coverage_min_y, utils.IMG_W), dtype=torch.bool, device="cuda")
    timers.stop("prepare_image_rect_and_mask")

    # Move partial image_gt which is needed to GPU.
    timers.start("prepare_gt_image")
    local_image_rect_gt = torch.clamp(viewpoint_cam.original_image / 255.0, 0.0, 1.0)
    timers.stop("prepare_gt_image")

    # Loss computation
    timers.start("local_loss_computation")
    torch.cuda.synchronize()# TODO: improve the time measurement here.
    start_time = time.time()
    pixelwise_Ll1 = pixelwise_l1_with_mask(local_image_rect,
                                           local_image_rect_gt,
                                           local_image_rect_pixels_compute_locally)
    Ll1 = pixelwise_Ll1.sum()/(utils.get_num_pixels()*3)
    # utils.check_memory_usage_logging("after l1_loss")
    pixelwise_ssim_loss = pixelwise_ssim_with_mask(local_image_rect,
                                                   local_image_rect_gt,
                                                   local_image_rect_pixels_compute_locally)
    ssim_loss = pixelwise_ssim_loss.sum()/(utils.get_num_pixels()*3)

    torch.cuda.synchronize()
    statistic_collector["forward_loss_time"] = (time.time() - start_time)*1000
    # utils.check_memory_usage_logging("after ssim_loss")
    timers.stop("local_loss_computation") # measure time before allreduce, so that we can get the real local time. 

    return Ll1, ssim_loss, None

def batched_loss_computation(batched_image, batched_cameras, batched_compute_locally, batched_strategies, batched_statistic_collector):
    args = utils.get_args()
    timers = utils.get_timers()

    # Loss computation
    timers.start("loss_computation")
    losses_for_saving = []
    loss_sum = 0
    all_test_losses = []
    for idx, (image, camera, compute_locally, strategy, statistic_collector) in enumerate(zip(batched_image, batched_cameras, batched_compute_locally, batched_strategies, batched_statistic_collector)):
        if image is None:
            losses_for_saving.append(None)
            all_test_losses.append(0)
            continue
        if len(image.shape) == 3:
            Ll1, ssim_loss, test_loss = final_system_loss_computation(image,
                                                        camera,
                                                        compute_locally,
                                                        strategy,
                                                        statistic_collector)
            loss = (1.0 - args.lambda_dssim) * Ll1 + args.lambda_dssim * (1.0 - ssim_loss)
            all_test_losses.append(test_loss)
        elif len(image.shape) == 0:
            loss = image*0
            all_test_losses.append(0)
            print("[WARN]: The image is a scalar tensor.")
        else:
            raise ValueError("The shape of image is not correct.")
        losses_for_saving.append(loss)
        loss_sum += loss

    # loss_sum is a scalar tensor
    assert loss_sum.dim() == 0, "The loss_sum must be a scalar."
    timers.stop("loss_computation")
    return loss_sum * args.lr_scale_loss, losses_for_saving, all_test_losses

def merge_multiple_checkpoints(checkpoint_files):
    all_model_params = []
    start_from_this_iteration = 0
    for checkpoint_file in checkpoint_files:
        (model_params, start_from_this_iteration) = torch.load(checkpoint_file, map_location=f"cuda:{utils.LOCAL_RANK}")
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

        # self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        # self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        # self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        # self._scaling = nn.Parameter(scales.requires_grad_(True))
        # self._rotation = nn.Parameter(rots.requires_grad_(True))
        # self._opacity = nn.Parameter(opacities.requires_grad_(True))

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
    (model_params, start_from_this_iteration) = torch.load(checkpoint_file, map_location=f"cuda:{utils.LOCAL_RANK}")

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

def training(dataset_args, opt_args, pipe_args, args, log_file):
    args.no_avoid_pixel_all2all = True
    print("set args.no_avoid_pixel_all2all to True")

    # dataset_args, opt_args, pipe_args, args contain arguments containing all kinds of settings and configurations. 
    # In which, the first three are sub-domains, and the fourth one contains all of them. 

    # init auxiliary tools
    timers = Timer(args)
    utils.set_timers(timers)
    utils.set_log_file(log_file)
    prepare_output_and_logger(dataset_args)
    utils.log_cpu_memory_usage("at the beginning of training")

    start_from_this_iteration = 1

    # init parameterized scene
    gaussians = GaussianModel(dataset_args.sh_degree)
    with torch.no_grad():
        scene = Scene(args, gaussians)
        gaussians.training_setup(opt_args)

        if args.start_checkpoint != "":
            number_files = len(os.listdir(args.start_checkpoint))
            if args.start_checkpoint[-1] != "/":
                args.start_checkpoint += "/"
            if number_files == utils.DEFAULT_GROUP.size():
                file_name = args.start_checkpoint+"chkpnt" + str(utils.DEFAULT_GROUP.rank()) + ".pth"
                (model_params, start_from_this_iteration) = torch.load(file_name)

            elif number_files > utils.DEFAULT_GROUP.size():
                assert number_files % utils.DEFAULT_GROUP.size() == 0, "The number of files in the checkpoint folder must be a multiple of the number of processes."
                local_processed_file_names = []
                for i in range(utils.DEFAULT_GROUP.rank(), number_files, utils.DEFAULT_GROUP.size()):
                    local_processed_file_names.append(args.start_checkpoint+"chkpnt" + str(i) + ".pth")
                (model_params, start_from_this_iteration) = merge_multiple_checkpoints(local_processed_file_names)
                file_name = local_processed_file_names
            elif number_files < utils.DEFAULT_GROUP.size():
                assert utils.DEFAULT_GROUP.size() % number_files == 0, "The number of files in the checkpoint folder must be a divisor of the number of processes."
                file_name = args.start_checkpoint+"chkpnt" + str(utils.DEFAULT_GROUP.rank() % number_files) + ".pth"
                (model_params, start_from_this_iteration) = get_part_of_checkpoints(file_name, utils.DEFAULT_GROUP.size()//number_files, utils.DEFAULT_GROUP.rank()//number_files)

            if args.drop_duplicate_gaussians_coeff != 1.0:
                model_params = drop_duplicate_gaussians(model_params, args.drop_duplicate_gaussians_coeff)

            gaussians.restore(model_params, opt_args)
            start_from_this_iteration += args.bsz
            utils.print_rank_0("Restored from checkpoint: {}".format(file_name))
            log_file.write("Restored from checkpoint: {}\n".format(file_name))

        scene.log_scene_info_to_file(log_file, "Scene Info Before Training")
    # exit()
    utils.check_memory_usage_logging("after init and before training loop")

    # init dataset
    train_dataset = SceneDataset(scene.getTrainCameras(dataset_args.train_resolution_scale))
    if args.adjust_strategy_warmp_iterations == -1:
        args.adjust_strategy_warmp_iterations = len(train_dataset.cameras)
        # use one epoch to warm up. do not use the first epoch's running time for adjustment of strategy.

    utils.set_img_size(train_dataset.cameras[0].image_height, train_dataset.cameras[0].image_width)
    # init workload division strategy
    strategy_history = DivisionStrategyHistoryFinal(train_dataset, utils.DEFAULT_GROUP.size(), utils.DEFAULT_GROUP.rank())
    # init background
    bg_color = [1, 1, 1] if dataset_args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Training Loop
    end2end_timers = End2endTimer(args)
    end2end_timers.start()
    progress_bar = tqdm(range(1, opt_args.iterations + 1), desc="Training progress", disable=(utils.LOCAL_RANK != 0))
    progress_bar.update(start_from_this_iteration - 1)
    num_trained_batches = 0
    for iteration in range(start_from_this_iteration, opt_args.iterations + 1, args.bsz):
        torch.cuda.synchronize()
        # DEBUG
        # if utils.DEFAULT_GROUP.rank() == 0:
        #     print("\niteration: ", iteration, flush=True)

        # Step Initialization
        progress_bar.update(args.bsz)
        utils.set_cur_iter(iteration)
        gaussians.update_learning_rate(iteration)
        num_trained_batches += 1
        timers.clear()
        if args.nsys_profile:
            nvtx.range_push(f"iteration[{iteration},{iteration+args.bsz})")
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if utils.check_update_at_this_iter(iteration, args.bsz, 1000, 0):
            gaussians.oneupSHdegree()

        # Prepare data: Pick random Cameras for training
        batched_cameras = train_dataset.get_batched_cameras(args.bsz, load_now=False)
        # DEBUG
        # if utils.LOCAL_RANK == 0:
        #     for camera in batched_cameras:
        #         print(camera.image_name)

        with torch.no_grad():
            # Prepare Workload division strategy
            timers.start("prepare_strategies")
            batched_strategies, gpuid2tasks = start_strategy_final(batched_cameras, strategy_history)
            # print("rank ", utils.GLOBAL_RANK, "gpuid2tasks", gpuid2tasks)
            timers.stop("prepare_strategies")

            # Load ground-truth images to GPU
            timers.start("load_cameras")
            load_camera_from_cpu_to_all_gpu(batched_cameras, batched_strategies, gpuid2tasks)
            timers.stop("load_cameras")


        batched_screenspace_pkg = distributed_preprocess3dgs_and_all2all_final(batched_cameras, gaussians, pipe_args, background, 
                                                                               batched_strategies=batched_strategies,
                                                                               mode="train")
        batched_image, batched_compute_locally = render_final(batched_screenspace_pkg, batched_strategies)
        # print("rank ", utils.GLOBAL_RANK, "batched_image", batched_image)
        batch_statistic_collector = [cuda_args["stats_collector"] for cuda_args in batched_screenspace_pkg["batched_cuda_args"]]

        loss_sum, losses_for_saving, all_test_losses = batched_loss_computation(
            batched_image,
            batched_cameras,
            batched_compute_locally,
            batched_strategies,
            batch_statistic_collector
        )

        timers.start("backward")
        # if loss is a tensor
        loss_sum.backward()
        timers.stop("backward")
        utils.check_memory_usage_logging("after backward")

        with torch.no_grad():
            # Adjust workload division strategy. 
            globally_sync_for_timer()
            timers.start("finish_strategy_final")
            finish_strategy_final(batched_cameras, strategy_history, batched_strategies, batch_statistic_collector)
            timers.stop("finish_strategy_final")

            # Update Epoch Statistics: allgather loss into a tensor across DP GROUP
            timers.start("allgather_loss_and_log")
            losses_gpu = torch.zeros((args.bsz, ), device="cuda")
            for i, loss in enumerate(losses_for_saving):
                if loss is not None:
                    losses_gpu[i] = loss
            # allreduce losses across default group
            if utils.DEFAULT_GROUP.size() > 1:
                dist.all_reduce(losses_gpu, op=dist.ReduceOp.SUM, group=utils.DEFAULT_GROUP)
            losses_cpu = losses_gpu.cpu().numpy()
            train_dataset.update_losses(losses_cpu)

            # Logging
            losses_cpu = [round(loss, 6) for loss in losses_cpu]
            log_string = "iteration[{},{}) loss: {} image: {}\n".format(iteration, iteration+args.bsz,
                                                                        losses_cpu,
                                                                        [viewpoint_cam.image_name for viewpoint_cam in batched_cameras])
            if args.get_global_exact_loss:
                all_test_losses_tensor = torch.tensor(all_test_losses, device="cuda")
                if utils.DEFAULT_GROUP.size() > 1:
                    dist.all_reduce(all_test_losses_tensor, op=dist.ReduceOp.SUM, group=utils.DEFAULT_GROUP)
                log_string += "iteration[{},{}) exact losses: {}\n".format(iteration, iteration+args.bsz, all_test_losses_tensor.cpu().numpy())
            log_file.write(log_string)
            timers.stop("allgather_loss_and_log")

            # Log and save
            end2end_timers.stop()
            training_report(iteration, l1_loss, args.test_iterations, scene, pipe_args, background, dataset_args.test_resolution_scale)
            end2end_timers.start()

            # Densification
            densification(iteration, scene, gaussians, batched_screenspace_pkg)

            # Save Gaussians
            # if for some save_iteration in save_iterations, iteration <= save_iteration < iteration+args.bsz, then save the gaussians.
            if any([iteration <= save_iteration < iteration+args.bsz for save_iteration in args.save_iterations]):
                end2end_timers.stop()
                end2end_timers.print_time(log_file, iteration+args.bsz)
                utils.print_rank_0("\n[ITER {}] Saving Gaussians".format(iteration))
                log_file.write("[ITER {}] Saving Gaussians\n".format(iteration))
                scene.save(iteration)

                with open(args.log_folder+"/strategy_history_ws="+str(utils.WORLD_SIZE)+"_rk="+str(utils.GLOBAL_RANK)+".json", 'w') as f:
                    json.dump(strategy_history.to_json(), f)
                end2end_timers.start()

            if any([iteration <= checkpoint_iteration < iteration+args.bsz for checkpoint_iteration in args.checkpoint_iterations]):
                end2end_timers.stop()
                utils.print_rank_0("\n[ITER {}] Saving Checkpoint".format(iteration))
                log_file.write("\n[ITER {}] Saving Checkpoint".format(iteration))
                # end2end_timers.print_time(log_file, iteration+args.bsz)
                save_folder = scene.model_path + "/checkpoints/" + str(iteration) + "/"
                if utils.DEFAULT_GROUP.rank() == 0:
                    os.makedirs(save_folder, exist_ok=True)
                    if utils.DEFAULT_GROUP.size() > 1:
                        torch.distributed.barrier(group=utils.DEFAULT_GROUP)
                elif utils.DEFAULT_GROUP.size() > 1:
                    torch.distributed.barrier(group=utils.DEFAULT_GROUP)
                torch.save((gaussians.capture(), iteration), save_folder + "/chkpnt" + str(utils.DEFAULT_GROUP.rank()) + ".pth")
                end2end_timers.start()

            # Optimizer step
            if iteration < opt_args.iterations:
                timers.start("optimizer_step")

                if args.lr_scale_mode != "accumu": # we scale the learning rate rather than accumulate the gradients.
                    for param in gaussians.all_parameters():
                        if param.grad is not None:
                            param.grad /= args.bsz

                if not args.stop_update_param:
                    gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                timers.stop("optimizer_step")
                utils.check_memory_usage_logging("after optimizer step")

        # Finish a iteration and clean up
        torch.cuda.synchronize()
        # Release memory of locally rendered original_image
        for viewpoint_cam in batched_cameras:
            viewpoint_cam.original_image = None
        if args.nsys_profile:
            nvtx.range_pop()
        if utils.check_enable_python_timer():
            timers.printTimers(iteration, mode="sum")
        log_file.flush()

    # Finish training
    end2end_timers.print_time(log_file, opt_args.iterations)
    log_file.write("Max Memory usage: {} GB.\n".format(torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024))

def training_report(iteration, l1_loss, testing_iterations, scene : Scene, pipe_args, background, test_resolution_scale=1.0):
    args = utils.get_args()
    log_file = utils.get_log_file()
    # Report test and samples of training set
    while len(testing_iterations) > 0 and iteration > testing_iterations[0]:
        testing_iterations.pop(0)
    if len(testing_iterations) > 0 and utils.check_update_at_this_iter(iteration, utils.get_args().bsz, testing_iterations[0], 0):
        testing_iterations.pop(0)
        utils.print_rank_0("\n[ITER {}] Start Testing".format(iteration))
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras(test_resolution_scale)}, 
                            {'name': 'train', 'cameras' : [scene.getTrainCameras(test_resolution_scale)[idx*args.llffhold % len(scene.getTrainCameras())]
                                                           for idx in range(len(scene.getTrainCameras()) // args.llffhold)]})
                    # HACK: if we do not set --eval, then scene.getTestCameras is None; and there will be some errors. 
        # init workload division strategy
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = torch.scalar_tensor(0.0, device="cuda")
                psnr_test = torch.scalar_tensor(0.0, device="cuda")

                num_cameras = len(config['cameras'])
                eval_dataset = SceneDataset(config['cameras'])
                strategy_history = DivisionStrategyHistoryFinal(eval_dataset, utils.DEFAULT_GROUP.size(), utils.DEFAULT_GROUP.rank())
                for idx in range(1, num_cameras+1, args.bsz):
                    if args.empty_cache_more and idx % 10 == 0:
                        torch.cuda.empty_cache()
                    num_camera_to_load = min(args.bsz, num_cameras-idx+1)
                    batched_cameras = eval_dataset.get_batched_cameras(num_camera_to_load, load_now=False)
                    batched_strategies, gpuid2tasks = start_strategy_final(batched_cameras, strategy_history)
                    load_camera_from_cpu_to_all_gpu_for_eval(batched_cameras, batched_strategies, gpuid2tasks)

                    batched_screenspace_pkg = distributed_preprocess3dgs_and_all2all_final(batched_cameras, scene.gaussians, pipe_args, background, 
                                                                                           batched_strategies=batched_strategies, 
                                                                                           mode="test")
                    batched_image, batched_compute_locally = render_final(batched_screenspace_pkg, batched_strategies)
                    for camera_id, (image, gt_camera) in enumerate(zip(batched_image, batched_cameras)):
                        if image is None or len(image.shape) == 0:
                            image = torch.zeros(gt_camera.original_image.shape, device="cuda", dtype=torch.float32)
    
                        if utils.DEFAULT_GROUP.size() > 1:
                            torch.distributed.all_reduce(image, op=dist.ReduceOp.SUM, group=utils.DEFAULT_GROUP)

                        image = torch.clamp(image, 0.0, 1.0)
                        gt_image = torch.clamp(gt_camera.original_image / 255.0, 0.0, 1.0)

                        if idx + camera_id < num_cameras + 1:
                            l1_test += l1_loss(image, gt_image).mean().double()
                            psnr_test += psnr(image, gt_image).mean().double()
                        gt_camera.original_image = None
                psnr_test /= num_cameras
                l1_test /= num_cameras
                utils.print_rank_0("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                log_file.write("[ITER {}] Evaluating {}: L1 {} PSNR {}\n".format(iteration, config['name'], l1_test, psnr_test))

        torch.cuda.empty_cache()