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


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, adjust_div_stra_timer=None, cuda_args=None, timers=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    
    args = utils.get_args()

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

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

    means3D = pc.get_xyz
    means2D = screenspace_points
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

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    if adjust_div_stra_timer is not None:
        adjust_div_stra_timer.start("forward")

    if not args.sep_rendering:
        rendered_image, radii, n_render, n_consider, n_contrib = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp,
            cuda_args = cuda_args)
    else:
        if timers is not None:
            timers.start("forward_preprocess_gaussians")
        means2D, rgb, conic_opacity, radii, depths, tiles_touched = rasterizer.preprocess_gaussians(
            means3D=means3D,
            scales=scales,
            rotations=rotations,
            shs=shs,
            opacities=opacity,
            cuda_args=cuda_args
        )
        if timers is not None:
            timers.stop("forward_preprocess_gaussians")

        if cuda_args["mode"] == "train":
            means2D.retain_grad()
        # means2D is (P, 2) tensor. This is different from screenspace_points (P, 3). TODO: double check. 

        # all to all communication for means2D, rgb, conic_opacity, radii, depths, tiles_touched;
        if args.memory_distribution:#TODO: check whether these computation will affect gradient flow?
        
            if timers is not None:
                timers.start("forward_all_to_all_communication")

            local2j_ids = rasterizer.get_local2j_ids(means2D, radii, cuda_args)# (world_size,) matrix: local2j_ids[j] is the local 3dgs ids that should be sent to gpu j.

            # i2j_send_size is an  world_size*world_size integer tensor, containing the number of 3dgs that should be sent from gpu i to gpu j.
            i2j_send_size = torch.zeros((utils.WORLD_SIZE, utils.WORLD_SIZE), dtype=torch.int, device="cuda")
            i2j_send_size[utils.LOCAL_RANK, :] = torch.tensor([len(local2j_ids[i]) for i in range(utils.WORLD_SIZE)], dtype=torch.int, device="cuda")
            # sync i2j_send_size
            torch.distributed.all_reduce(i2j_send_size, op=torch.distributed.ReduceOp.SUM)
            i2j_send_size = i2j_send_size.cpu().numpy().tolist()

            def all_to_all(tensor):
                tensor_to_rki = []
                tensor_from_rki = []
                for i in range(utils.WORLD_SIZE):
                    tensor_to_rki.append(tensor[local2j_ids[i]].contiguous())# NCCL communication requires contiguous memory.
                    tensor_from_rki.append(torch.zeros((i2j_send_size[i][utils.LOCAL_RANK], ) + tensor.shape[1:], dtype=tensor.dtype, device="cuda"))
    
                # if utils.LOCAL_RANK==0:
                #     print(utils.LOCAL_RANK, utils.WORLD_SIZE, len(tensor_to_rki), len(tensor_from_rki), tensor_to_rki[0].shape, tensor_from_rki[0].shape)

                dist_func.all_to_all(
                    output_tensor_list=tensor_from_rki,
                    input_tensor_list=tensor_to_rki
                )# The function version could naturally enable communication during backward. 
                return torch.cat(tensor_from_rki, dim=0).contiguous()# a question: I have too many contiguous(), will it cause large overhead?

            means2D_redistributed = all_to_all(means2D)
            rgb_redistributed = all_to_all(rgb)
            conic_opacity_redistributed = all_to_all(conic_opacity)
            radii_redistributed = all_to_all(radii)
            depths_redistributed = all_to_all(depths)
            tiles_touched_redistributed = all_to_all(tiles_touched)
            # TODO: tiles_touched does not need to be communicate.

            if timers is not None:
                timers.stop("forward_all_to_all_communication")

            if timers is not None:
                timers.start("forward_render_gaussians")
            rendered_image, n_render, n_consider, n_contrib = rasterizer.render_gaussians(
                means2D=means2D_redistributed,
                conic_opacity=conic_opacity_redistributed,
                rgb=rgb_redistributed,
                depths=depths_redistributed,
                radii=radii_redistributed,
                tiles_touched=tiles_touched_redistributed,
                cuda_args=cuda_args
            )
            if timers is not None:
                timers.stop("forward_render_gaussians")
        else:
            # TODO: tiles_touched will be in-place modified in rasterizer.render_gaussians.
            # Actually, tiles_touched is re-generated from scratch in rasterizer.render_gaussians.
            # will this conflict pytorch graph/autograd functionality?
            # Maybe tile_touched could be viewed as assitance tensor which is not used in gradient computation. 
            if timers is not None:
                timers.start("forward_render_gaussians")
            rendered_image, n_render, n_consider, n_contrib = rasterizer.render_gaussians(
                means2D=means2D,
                conic_opacity=conic_opacity,
                rgb=rgb,
                depths=depths,
                radii=radii,
                tiles_touched=tiles_touched,
                cuda_args=cuda_args
            )
            if timers is not None:
                timers.stop("forward_render_gaussians")

    if adjust_div_stra_timer is not None:
        adjust_div_stra_timer.stop("forward")

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points if not args.sep_rendering else means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "n_render": n_render,
            "n_consider": n_consider,
            "n_contrib": n_contrib}
