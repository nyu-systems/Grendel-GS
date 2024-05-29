import torch
import utils.general_utils as utils

def densification(iteration, scene, gaussians, batched_screenspace_pkg):
    args = utils.get_args()
    timers = utils.get_timers()
    log_file = utils.get_log_file()

    # Densification
    if not args.disable_auto_densification and iteration <= args.densify_until_iter:
        # TODO: more check on this: originally the code is < args.densify_until_iter, but for bsz=1 it does not update at densify_until_iter iteration but other bsz>1 updates at densify_until_iter - (bsz - 1) iteration, thus there is different number of densifications for different bsz, which is not fair. 
        # the same issue for opacity reset, which has more severe implications.

        # Keep track of max radii in image-space for pruning
        timers.start("densification")

        timers.start("densification_update_stats")
        for radii, visibility_filter, screenspace_mean2D in zip(batched_screenspace_pkg["batched_locally_preprocessed_radii"],
                                                                batched_screenspace_pkg["batched_locally_preprocessed_visibility_filter"],
                                                                batched_screenspace_pkg["batched_locally_preprocessed_mean2D"]):
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            gaussians.add_densification_stats(screenspace_mean2D, visibility_filter)
        timers.stop("densification_update_stats")

        if iteration > args.densify_from_iter and utils.check_update_at_this_iter(iteration, args.bsz, args.densification_interval, 0):
            assert args.stop_update_param == False, "stop_update_param must be false for densification; because it is a flag for debugging."
            # utils.print_rank_0("iteration: {}, bsz: {}, update_interval: {}, update_residual: {}".format(iteration, args.bsz, args.densification_interval, 0))

            timers.start("densify_and_prune")
            size_threshold = 20 if iteration > args.opacity_reset_interval else None
            gaussians.densify_and_prune(args.densify_grad_threshold, args.min_opacity, scene.cameras_extent, size_threshold)
            timers.stop("densify_and_prune")

            # redistribute after densify_and_prune, because we have new gaussians to distribute evenly.
            if utils.get_denfify_iter() % args.redistribute_gaussians_frequency == 0:
                num_3dgs_before_redistribute = gaussians.get_xyz.shape[0]
                timers.start("redistribute_gaussians")
                gaussians.redistribute_gaussians()
                timers.stop("redistribute_gaussians")
                num_3dgs_after_redistribute = gaussians.get_xyz.shape[0]

                log_file.write("iteration[{},{}) redistribute. Now num of 3dgs before redistribute: {}. Now num of 3dgs after redistribute: {}. \n".format(
                    iteration, iteration+args.bsz, num_3dgs_before_redistribute, num_3dgs_after_redistribute))

            utils.check_memory_usage(log_file, args, iteration, gaussians, before_densification_stop=True)

            utils.inc_densify_iter()
        
        if utils.check_update_at_this_iter(iteration, args.bsz, args.opacity_reset_interval, 0) and iteration+args.bsz <= args.opacity_reset_until_iter:
            # TODO: do opacity reset if dataset_args.white_background and iteration == opt_args.densify_from_iter
            timers.start("reset_opacity")
            gaussians.reset_opacity()
            timers.stop("reset_opacity")

        timers.stop("densification")
    else:
        if iteration > args.densify_from_iter and utils.check_update_at_this_iter(iteration, args.bsz, args.densification_interval, 0):
            utils.check_memory_usage(log_file, args, iteration, gaussians, before_densification_stop=False)



