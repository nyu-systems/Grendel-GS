
from collections import namedtuple
import utils.general_utils as utils

ImageDistributionConfig = namedtuple('ImageDistributionConfig', [
    'loss_distribution_mode',
    'workloads_division_mode',
    'avoid_pixels_all2all',
    'local_running_time_mode'
])

def init_image_distribution_config(args):

    if args.image_distribution_mode == "0":
        args.global_image_distribution_config = ImageDistributionConfig(
            loss_distribution_mode="replicated_loss_computation",
            workloads_division_mode="DivisionStrategyUniform",
            avoid_pixels_all2all=False,
            local_running_time_mode=["backward_render_time"]
        )

    elif args.image_distribution_mode == "1":
        args.global_image_distribution_config = ImageDistributionConfig(
            loss_distribution_mode="general_distributed_loss_computation",
            workloads_division_mode="DivisionStrategyUniform",
            avoid_pixels_all2all=False,
            local_running_time_mode=["backward_render_time"]
        )

    elif args.image_distribution_mode == "2":
        args.global_image_distribution_config = ImageDistributionConfig(
            loss_distribution_mode="general_distributed_loss_computation",
            workloads_division_mode="DivisionStrategyDynamicAdjustment",
            avoid_pixels_all2all=False,
            local_running_time_mode=["backward_render_time"]
        )

    elif args.image_distribution_mode == "3":
        args.global_image_distribution_config = ImageDistributionConfig(
            loss_distribution_mode="avoid_pixel_all2all_loss_computation",
            workloads_division_mode="DivisionStrategyDynamicAdjustment",
            avoid_pixels_all2all=True,
            local_running_time_mode=["backward_render_time", "forward_render_time", "forward_loss_time", "forward_loss_time"]
        )

    elif args.image_distribution_mode == "4":
        args.global_image_distribution_config = ImageDistributionConfig(
            loss_distribution_mode="avoid_pixel_all2all_loss_computation_adjust_mode6",
            workloads_division_mode="DivisionStrategyDynamicAdjustment",
            avoid_pixels_all2all=True,
            local_running_time_mode=["backward_render_time", "forward_render_time", "forward_loss_time", "forward_loss_time"]
        )

    else:
        raise ValueError(f"Unknown image_distribution_mode: {args.image_distribution_mode}")
