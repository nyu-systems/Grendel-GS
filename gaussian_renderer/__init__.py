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
from utils.sh_utils import eval_sh
import utils.general_utils as utils
import torch.distributed.nn.functional as dist_func

def all_to_all_communication(rasterizer, means2D, rgb, conic_opacity, radii, depths, cuda_args):# TODO: support DP_all2all here. Now, this is just for MP_all2all.
    local2j_ids, local2j_ids_bool = rasterizer.get_local2j_ids(means2D, radii, cuda_args)
    # (world_size,) matrix: local2j_ids[j] is the local 3dgs ids that should be sent to gpu j.
    # (# of 3dgs, world_size) ubt: local2j_ids_bool[i,j] is True if 3dg i should be sent to gpu j.

    # NOTE: i2j_send_size is an  world_size*world_size integer tensor, containing the number of 3dgs that should be sent from gpu i to gpu j.
    i2j_send_size = torch.zeros((utils.WORLD_SIZE, utils.WORLD_SIZE), dtype=torch.int, device="cuda")
    local2j_send_size = torch.tensor([len(local2j_ids[i]) for i in range(utils.WORLD_SIZE)], dtype=torch.int, device="cuda")
    torch.distributed.all_gather_into_tensor(i2j_send_size, local2j_send_size, group=utils.MP_GROUP)
    i2j_send_size = i2j_send_size.cpu().numpy().tolist()

    def one_all_to_all(tensor, use_function_version=False):
        tensor_to_rki = []
        tensor_from_rki = []
        for i in range(utils.WORLD_SIZE):
            tensor_to_rki.append(tensor[local2j_ids[i]].contiguous())# NCCL communication requires contiguous memory.
            tensor_from_rki.append(torch.zeros((i2j_send_size[i][utils.LOCAL_RANK], ) + tensor.shape[1:], dtype=tensor.dtype, device="cuda"))

        if use_function_version:# FIXME: there is error if I use torch.distributed.nn.functional to replace dist_func here. So weird. 
            dist_func.all_to_all(
                output_tensor_list=tensor_from_rki,
                input_tensor_list=tensor_to_rki,
                group=utils.MP_GROUP
            )# The function version could naturally enable communication during backward. 
        else:
            torch.distributed.all_to_all(
                output_tensor_list=tensor_from_rki,
                input_tensor_list=tensor_to_rki,
                group=utils.MP_GROUP
            )
        return torch.cat(tensor_from_rki, dim=0).contiguous()# TODO: I have too many contiguous(), will it cause large overhead?

    # Merge means2D, rgb, conic_opacity into one functional all-to-all communication call.
    params = [means2D, rgb, conic_opacity]
    params_cancatenated = torch.cat(params, dim=1).contiguous()
    params_redistributed = one_all_to_all(params_cancatenated, use_function_version=True)
    means2D_redistributed, rgb_redistributed, conic_opacity_redistributed = torch.split(
        params_redistributed,
        [means2D.shape[1], rgb.shape[1], conic_opacity.shape[1]],
        dim=1
    )

    # Merge radii and depths into one all-to-all communication call. 
    radii = radii.float() # XXX: I am not sure whether it will affect accuracy. 
    radii_depth_float = torch.cat([radii.float().unsqueeze(1), depths.unsqueeze(1)], dim=1).contiguous()
    radii_depth_redistributed = one_all_to_all(radii_depth_float, use_function_version=False)
    radii_redistributed, depths_redistributed = torch.split(
        radii_depth_redistributed,
        [1, 1],
        dim=1
    )
    radii_redistributed = radii_redistributed.squeeze(1).int()
    depths_redistributed = depths_redistributed.squeeze(1)

    # radii_redistributed = one_all_to_all(radii, use_function_version=False)
    # depths_redistributed = one_all_to_all(depths, use_function_version=False)
    return means2D_redistributed, rgb_redistributed, conic_opacity_redistributed, radii_redistributed, depths_redistributed, i2j_send_size, local2j_ids_bool

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None,
           cuda_args=None,
           timers=None,
           strategy=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    

    ########## [START] Prepare CUDA Rasterization Settings ##########
    if timers is not None:
        timers.start("forward_prepare_args_and_settings")
    args = utils.get_args()

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

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    if timers is not None:
        timers.stop("forward_prepare_gaussians")

    utils.check_memory_usage_logging("after forward_prepare_gaussians")
    ########## [END] Prepare Gaussians for rendering ##########



    ########## [START] CUDA Rasterization Call ##########
    # Rasterize visible Gaussians to image, obtain their screen-space intermedia parameters. 
    assert colors_precomp is None, "sep_rendering mode does not support precomputed colors."
    assert cov3D_precomp is None, "sep_rendering mode does not support precomputed 3d covariance."

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
    if timers is not None:
        timers.stop("forward_preprocess_gaussians")
    utils.check_memory_usage_logging("after forward_preprocess_gaussians")

    if cuda_args["mode"] == "train":
        means2D.retain_grad()
        # NOTE: means2D is (P, 2) tensor. This is different from means2D in not sep_rendering mode i.e. (P, 3). TODO: double check. 

    #[3DGS all2all]: all to all communication for means2D, rgb, conic_opacity, radii, depths
    if args.memory_distribution:
        if timers is not None:
            timers.start("forward_all_to_all_communication")
        means2D_redistributed, rgb_redistributed, conic_opacity_redistributed, radii_redistributed, depths_redistributed, i2j_send_size, local2j_ids_bool = \
            all_to_all_communication(rasterizer, means2D, rgb, conic_opacity, radii, depths, cuda_args)
        if timers is not None:
            timers.stop("forward_all_to_all_communication")
        utils.check_memory_usage_logging("after forward_all_to_all_communication")

    # get compute_locally to know local workload in the end2end distributed training.
    if timers is not None:
        timers.start("forward_compute_locally")
    
    compute_locally = strategy.get_compute_locally()

    if timers is not None:
        timers.stop("forward_compute_locally")
    utils.check_memory_usage_logging("after forward_compute_locally")

    # render
    if timers is not None:
        timers.start("forward_render_gaussians")
    #[Pixel-wise render]
    rendered_image, n_render, n_consider, n_contrib = rasterizer.render_gaussians(
        means2D=means2D if not args.memory_distribution else means2D_redistributed,
        conic_opacity=conic_opacity if not args.memory_distribution else conic_opacity_redistributed,
        rgb=rgb if not args.memory_distribution else rgb_redistributed,
        depths=depths if not args.memory_distribution else depths_redistributed,
        radii=radii if not args.memory_distribution else radii_redistributed,
        compute_locally=compute_locally,
        cuda_args=cuda_args
    )
    if timers is not None:
        timers.stop("forward_render_gaussians")
    utils.check_memory_usage_logging("after forward_render_gaussians")

    ########## [END] CUDA Rasterization Call ##########

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return_data = {"render": rendered_image,
                    "viewspace_points": means2D,
                    "visibility_filter" : radii > 0,
                    "radii": radii,
                    "n_render": n_render,
                    "n_consider": n_consider,
                    "n_contrib": n_contrib}
    if args.memory_distribution:
        return_data["i2j_send_size"] = i2j_send_size
        return_data["compute_locally"] = compute_locally
        return_data["local2j_ids_bool"] = local2j_ids_bool
    return return_data
