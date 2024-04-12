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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
import utils.general_utils as utils
import torch.distributed.nn.functional as dist_func


def get_cuda_args(strategy, mode="train"):# "test"
    args = utils.get_args()
    iteration = utils.get_cur_iter()

    if mode == "train":
        for x in range(args.bsz):# This is to make sure we will get the 
            if (iteration+x) % args.log_interval == 1:
                iteration += x
                break
        avoid_pixel_all2all = args.image_distribution_config.avoid_pixels_all2all
    elif mode == "test":
        iteration = -1
        avoid_pixel_all2all = False
    else:
        raise ValueError("mode should be train or test.")

    cuda_args = {
            "mode": mode,
            "world_size": str(utils.WORLD_SIZE),
            "global_rank": str(utils.GLOBAL_RANK),
            "local_rank": str(utils.LOCAL_RANK),
            "mp_world_size": str(utils.MP_GROUP.size()),
            "mp_rank": str(utils.MP_GROUP.rank()),
            "log_folder": args.log_folder,
            "log_interval": str(args.log_interval),
            "iteration": str(iteration),
            "zhx_debug": str(args.zhx_debug),
            "zhx_time": str(args.zhx_time),
            "dist_global_strategy": strategy.get_global_strategy_str(),
            "avoid_pixel_all2all": avoid_pixel_all2all,
            "stats_collector": {},
        }
    return cuda_args






def replicated_preprocess3dgs(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0,
                              strategy=None,
                              mode="train"):
    """
    preprocess 3dgs.

    all 3DGS are stored replicatedly on all GPUs.
    """
    ########## [START] Prepare CUDA Rasterization Settings ##########
    timers = utils.get_timers()
    if timers is not None:
        timers.start("forward_prepare_args_and_settings")
    # only locally render one camera in a batched cameras
    cuda_args = get_cuda_args(strategy, mode)

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    if timers is not None:
        timers.stop("forward_prepare_args_and_settings")
    ########## [END] Prepare CUDA Rasterization Settings ##########



    ########## [START] Prepare Gaussians for rendering ##########
    if timers is not None:
        timers.start("forward_prepare_gaussians")
    means3D = pc.get_xyz
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    shs = pc.get_features
    if timers is not None:
        timers.stop("forward_prepare_gaussians")

    utils.check_memory_usage_logging("after forward_prepare_gaussians")
    ########## [END] Prepare Gaussians for rendering ##########



    ########## [START] CUDA Rasterization Call ##########
    # Rasterize visible Gaussians to image, obtain their screen-space intermedia parameters. 
    if timers is not None:
        timers.start("forward_preprocess_gaussians")
    #[3DGS-wise preprocess]
    means2D, rgb, conic_opacity, radii, depths = rasterizer.preprocess_gaussians(
        means3D=means3D,
        scales=scales,
        rotations=rotations,
        shs=shs,
        opacities=opacity,
        cuda_args=cuda_args
    )
    if mode == "train":
        means2D.retain_grad()
    if timers is not None:
        timers.stop("forward_preprocess_gaussians")
    utils.check_memory_usage_logging("after forward_preprocess_gaussians")

    screenspace_pkg = {
                "rasterizer": rasterizer,
                "cuda_args": cuda_args,
                "locally_preprocessed_mean2D": means2D,
                "locally_preprocessed_radii": radii,
                "means2D_for_render": means2D,
                "rgb_for_render": rgb,
                "conic_opacity_for_render": conic_opacity,
                "radii_for_render": radii,
                "depths_for_render": depths
    }
    return screenspace_pkg





def all_to_all_communication(batched_rasterizers, batched_screenspace_params, batched_cuda_args, batched_strategies):
    # TODO: fix this. 
    batched_local2j_ids = []
    batched_local2j_ids_bool = []
    for i in range(utils.DP_GROUP.size()):
        means2D, rgb, conic_opacity, radii, depths = batched_screenspace_params[i]
        local2j_ids, local2j_ids_bool = batched_strategies[i].get_local2j_ids(means2D, radii, batched_rasterizers[i].raster_settings, batched_cuda_args[i])
        batched_local2j_ids.append(local2j_ids)
        batched_local2j_ids_bool.append(local2j_ids_bool)

    catted_batched_local2j_ids_bool = torch.cat(batched_local2j_ids_bool, dim=1)

    i2j_send_size = torch.zeros((utils.DEFAULT_GROUP.size(), utils.DEFAULT_GROUP.size()), dtype=torch.int, device="cuda")
    local2j_send_size = []
    for i in range(utils.DP_GROUP.size()):
        for j in range(utils.MP_GROUP.size()):
            local2j_send_size.append(len(batched_local2j_ids[i][j]))
    local2j_send_size = torch.tensor(local2j_send_size, dtype=torch.int, device="cuda")
    torch.distributed.all_gather_into_tensor(i2j_send_size, local2j_send_size, group=utils.DEFAULT_GROUP)
    i2j_send_size = i2j_send_size.cpu().numpy().tolist()

    def one_all_to_all(batched_tensors, use_function_version=False):
        tensor_to_rki = []
        tensor_from_rki = []
        for d_i in range(utils.DP_GROUP.size()):
            for d_j in range(utils.MP_GROUP.size()):
                i = d_i * utils.MP_GROUP.size() + d_j
                tensor_to_rki.append(batched_tensors[d_i][batched_local2j_ids[d_i][d_j]].contiguous()) # NCCL communication requires contiguous memory.
                tensor_from_rki.append(torch.zeros((i2j_send_size[i][utils.DEFAULT_GROUP.rank()], ) + batched_tensors[0].shape[1:], 
                                                   dtype=batched_tensors[0].dtype, device="cuda"))

        if use_function_version:# FIXME: there is error if I use torch.distributed.nn.functional to replace dist_func here. So weird. 
            dist_func.all_to_all(
                output_tensor_list=tensor_from_rki,
                input_tensor_list=tensor_to_rki,
                group=utils.DEFAULT_GROUP
            )# The function version could naturally enable communication during backward. 
        else:
            torch.distributed.all_to_all(
                output_tensor_list=tensor_from_rki,
                input_tensor_list=tensor_to_rki,
                group=utils.DEFAULT_GROUP
            )
        return torch.cat(tensor_from_rki, dim=0).contiguous()# TODO: I have too many contiguous(), will it cause large overhead?

    # Merge means2D, rgb, conic_opacity into one functional all-to-all communication call.
    batched_catted_screenspace_states = []
    batched_catted_screenspace_auxiliary_states = []
    for i in range(utils.DP_GROUP.size()):
        means2D, rgb, conic_opacity, radii, depths = batched_screenspace_params[i]
        if i == 0:
            mean2d_dim1 = means2D.shape[1]
            rgb_dim1 = rgb.shape[1]
            conic_opacity_dim1 = conic_opacity.shape[1]
        batched_catted_screenspace_states.append(torch.cat([means2D, rgb, conic_opacity], dim=1).contiguous())
        batched_catted_screenspace_auxiliary_states.append(torch.cat([radii.float().unsqueeze(1), depths.unsqueeze(1)], dim=1).contiguous())

    params_redistributed = one_all_to_all(batched_catted_screenspace_states, use_function_version=True)
    means2D_redistributed, rgb_redistributed, conic_opacity_redistributed = torch.split(
        params_redistributed,
        [mean2d_dim1, rgb_dim1, conic_opacity_dim1],
        dim=1
    )
    radii_depth_redistributed = one_all_to_all(batched_catted_screenspace_auxiliary_states, use_function_version=False)
    radii_redistributed, depths_redistributed = torch.split(
        radii_depth_redistributed,
        [1, 1],
        dim=1
    )
    radii_redistributed = radii_redistributed.squeeze(1).int()
    depths_redistributed = depths_redistributed.squeeze(1)

    return means2D_redistributed, rgb_redistributed, conic_opacity_redistributed, radii_redistributed, depths_redistributed, i2j_send_size, catted_batched_local2j_ids_bool


def distributed_preprocess3dgs_and_all2all(batched_viewpoint_cameras, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0,
                                           batched_strategies=None,
                                           mode="train"):
    """
    Render the scene. 

    distribute gaussians parameters across all GPUs.
    """
    assert utils.DEFAULT_GROUP.size() > 1, "This function is only for distributed training. "

    timers = utils.get_timers()

    ########## [START] Prepare Gaussians for rendering ##########
    if timers is not None:
        timers.start("forward_prepare_gaussians")
    means3D = pc.get_xyz
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    shs = pc.get_features
    if timers is not None:
        timers.stop("forward_prepare_gaussians")
    utils.check_memory_usage_logging("after forward_prepare_gaussians")
    ########## [END] Prepare Gaussians for rendering ##########

    if timers is not None:
        timers.start("forward_preprocess_gaussians")
    batched_rasterizers = []
    batched_cuda_args = []
    batched_screenspace_params = []
    batched_means2D = []
    batched_radii = []
    for i, (viewpoint_camera, strategy) in enumerate(zip(batched_viewpoint_cameras, batched_strategies)):
        ########## [START] Prepare CUDA Rasterization Settings ##########
        cuda_args = get_cuda_args(strategy, mode)
        batched_cuda_args.append(cuda_args)

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        ########## [END] Prepare CUDA Rasterization Settings ##########

        #[3DGS-wise preprocess]
        means2D, rgb, conic_opacity, radii, depths = rasterizer.preprocess_gaussians(
            means3D=means3D,
            scales=scales,
            rotations=rotations,
            shs=shs,
            opacities=opacity,
            cuda_args=cuda_args
        )
        if mode == "train":
            means2D.retain_grad()
        batched_means2D.append(means2D)
        screenspace_params = [means2D, rgb, conic_opacity, radii, depths]
        batched_rasterizers.append(rasterizer)
        batched_screenspace_params.append(screenspace_params)
        batched_radii.append(radii)
    utils.check_memory_usage_logging("after forward_preprocess_gaussians")
    if timers is not None:
        timers.stop("forward_preprocess_gaussians")


    if timers is not None:
        timers.start("forward_all_to_all_communication")
    means2D_redistributed, rgb_redistributed, conic_opacity_redistributed, radii_redistributed, depths_redistributed, i2j_send_size, local2j_ids_bool = \
        all_to_all_communication(batched_rasterizers, batched_screenspace_params, batched_cuda_args, batched_strategies)
    utils.check_memory_usage_logging("after forward_all_to_all_communication")
    if timers is not None:
        timers.stop("forward_all_to_all_communication")
    
    screenspace_pkg = {
                "batched_locally_preprocessed_mean2D": batched_means2D,
                "batched_locally_preprocessed_radii": batched_radii,
                "rasterizer": batched_rasterizers[utils.DP_GROUP.rank()],
                "cuda_args": batched_cuda_args[utils.DP_GROUP.rank()],
                "means2D_for_render": means2D_redistributed,
                "rgb_for_render": rgb_redistributed,
                "conic_opacity_for_render": conic_opacity_redistributed,
                "radii_for_render": radii_redistributed,
                "depths_for_render": depths_redistributed,
                "i2j_send_size": i2j_send_size,
                "local2j_ids_bool": local2j_ids_bool
    }
    return screenspace_pkg


def preprocess3dgs_and_all2all(batched_cameras, gaussians, pipe_args, background, batched_strategies, mode):
    args = utils.get_args()

    local_render_viewpoint_cam = batched_cameras[utils.DP_GROUP.rank()]
    local_render_strategy = batched_strategies[utils.DP_GROUP.rank()]

    if args.gaussians_distribution:
        screenspace_pkg = distributed_preprocess3dgs_and_all2all(batched_cameras, gaussians, pipe_args, background,
                                                                 batched_strategies=batched_strategies,
                                                                 mode=mode)
        if mode == "test":
            return screenspace_pkg

        screenspace_pkg["batched_locally_preprocessed_visibility_filter"] = [radii > 0 for radii in screenspace_pkg["batched_locally_preprocessed_radii"]]
    else:
        screenspace_pkg = replicated_preprocess3dgs(local_render_viewpoint_cam, gaussians, pipe_args, background,
                                                    strategy=local_render_strategy,
                                                    mode=mode)
        if mode == "test":
            return screenspace_pkg

        screenspace_pkg["batched_locally_preprocessed_radii"] = [screenspace_pkg["locally_preprocessed_radii"]]
        screenspace_pkg["batched_locally_preprocessed_visibility_filter"] = [screenspace_pkg["locally_preprocessed_radii"]>0]
        screenspace_pkg["batched_locally_preprocessed_mean2D"] = [screenspace_pkg["locally_preprocessed_mean2D"]]

    return screenspace_pkg


def render(screenspace_pkg, strategy=None):
    """
    Render the scene. 
    """
    timers = utils.get_timers()

    # get compute_locally to know local workload in the end2end distributed training.
    if timers is not None:
        timers.start("forward_compute_locally")    
    compute_locally = strategy.get_compute_locally()
    extended_compute_locally = strategy.get_extended_compute_locally()
    if timers is not None:
        timers.stop("forward_compute_locally")
    utils.check_memory_usage_logging("after forward_compute_locally")

    # render
    if timers is not None:
        timers.start("forward_render_gaussians")
    rendered_image, n_render, n_consider, n_contrib = screenspace_pkg["rasterizer"].render_gaussians(
        means2D=screenspace_pkg["means2D_for_render"],
        conic_opacity=screenspace_pkg["conic_opacity_for_render"],
        rgb=screenspace_pkg["rgb_for_render"],
        depths=screenspace_pkg["depths_for_render"],
        radii=screenspace_pkg["radii_for_render"],
        compute_locally=compute_locally,
        extended_compute_locally=extended_compute_locally,
        cuda_args=screenspace_pkg["cuda_args"]
    )
    if timers is not None:
        timers.stop("forward_render_gaussians")
    utils.check_memory_usage_logging("after forward_render_gaussians")

    ########## [END] CUDA Rasterization Call ##########
    return rendered_image, compute_locally