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
from utils.general_utils import memory_logging

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, offload=False, log_file=None, my_timer=None, enable_send2gpu_sparsity=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
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

    memory_logging(log_file, "before preparation for rasterizer")
    my_timer.start("prepare for rasterizer(parameters.cuda())")

    means3D = None
    means2D = None
    opacity = None
    scales = None
    rotations = None
    shs = None
    cov3D_precomp = None
    colors_precomp = None
    parameters = []
    send2gpu_filter = None
    if not offload:
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        means2D = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0

        means3D = pc.get_xyz
        opacity = pc.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if pipe.compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier)# TODO: support swap mode for it.
        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            if pipe.convert_SHs_python:# TODO: support swap mode for it.
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = pc.get_features
        else:
            colors_precomp = override_color
    else:
        #TODO: only take out the needed indices of the 3dgs and put them to the gpu. make sure the tensor on gpu is differentiable.
        if enable_send2gpu_sparsity:
            my_timer.start("move data to gpu for rasterizer.getTouchedIndices")
            means3D_all = pc._xyz.detach().cuda()
            my_timer.stop("move data to gpu for rasterizer.getTouchedIndices")

            my_timer.start("rasterizer.getTouchedIndices")
            send2gpu_filter = rasterizer.getTouchedIndices(means3D_all)
            my_timer.stop("rasterizer.getTouchedIndices")

            num_touched = int(send2gpu_filter.sum().item())

            means2D = torch.zeros((num_touched, ) + means3D_all.shape[1:], dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
            means3D = means3D_all[send2gpu_filter].detach().requires_grad_(True)
            my_timer.start("move send2gpu_filter to cpu for rasterizer.getTouchedIndices")
            send2gpu_filter = send2gpu_filter.cpu()
            my_timer.stop("move send2gpu_filter to cpu for rasterizer.getTouchedIndices")
        else:
            means2D = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
            means3D = pc._xyz.detach().cuda().requires_grad_(True)
            send2gpu_filter = torch.full((means3D.shape[0],), True, dtype=torch.bool, device="cpu")

        parameters.append(means3D)

        # TODO: detach will create a new tensor which is bit redundant. can we avoid it?
        # TODO: do i need to make it contiguous?
        opacity = pc._opacity[send2gpu_filter].detach().cuda().requires_grad_(True)
        parameters.append(opacity)
        opacity = pc.opacity_activation(opacity)

        scales = pc._scaling[send2gpu_filter].detach().cuda().requires_grad_(True)
        parameters.append(scales)
        scales = pc.scaling_activation(scales)

        rotations = pc._rotation[send2gpu_filter].detach().cuda().requires_grad_(True)
        parameters.append(rotations)
        rotations = pc.rotation_activation(rotations)
        
        features_dc = pc._features_dc[send2gpu_filter].detach().cuda().requires_grad_(True)
        features_rest = pc._features_rest[send2gpu_filter].detach().cuda().requires_grad_(True)
        parameters.append(features_dc)
        parameters.append(features_rest)
        shs = torch.cat([features_dc, features_rest], dim=1)

    try:
        means2D.retain_grad()
    except:
        pass

    my_timer.stop("prepare for rasterizer(parameters.cuda())")

    memory_logging(log_file, "after preparation for rasterizer/before cuda_rasterizer")

    my_timer.start("cuda_rasterizer")
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    my_timer.stop("cuda_rasterizer")
    memory_logging(log_file, "after cuda_rasterizer")

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    #TODO: return the used indices and tensors with their gradients on gpu;
    return {"render": rendered_image,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "parameters": parameters,
            "send2gpu_filter": send2gpu_filter}
