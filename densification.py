import torch
from torch.cuda import nvtx
import utils.general_utils as utils

def densification(iteration, scene, gaussians, batched_screenspace_pkg):
    args = utils.get_args()
    timers = utils.get_timers()
    log_file = utils.get_log_file()

    # Update Statistics for redistribution
    # if args.gaussians_distribution:
    #     for local2j_ids_bool in batched_screenspace_pkg["batched_local2j_ids_bool"]:
    #         gaussians.send_to_gpui_cnt += local2j_ids_bool

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
            if args.redistribute_gaussians_mode != "no_redistribute" and ( utils.get_denfify_iter() % args.redistribute_gaussians_frequency == 0 ):
                num_3dgs_before_redistribute = gaussians.get_xyz.shape[0]
                timers.start("redistribute_gaussians")
                gaussians.redistribute_gaussians()
                timers.stop("redistribute_gaussians")
                num_3dgs_after_redistribute = gaussians.get_xyz.shape[0]

                log_file.write("iteration[{},{}) redistribute. Now num of 3dgs before redistribute: {}. Now num of 3dgs after redistribute: {}. \n".format(
                    iteration, iteration+args.bsz, num_3dgs_before_redistribute, num_3dgs_after_redistribute))

            # torch.cuda.empty_cache()
            memory_usage = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
            max_memory_usage = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
            max_reserved_memory = torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024
            now_reserved_memory = torch.cuda.memory_reserved() / 1024 / 1024 / 1024
            log_file.write("iteration[{},{}) densify_and_prune. Now num of 3dgs: {}. Now Memory usage: {} GB. Max Memory usage: {} GB. Max Reserved Memory: {} GB. Now Reserved Memory: {} GB. \n".format(
                iteration, iteration+args.bsz, gaussians.get_xyz.shape[0], memory_usage, max_memory_usage, max_reserved_memory, now_reserved_memory))
            if args.log_memory_summary:
                log_file.write("Memory Summary: {} GB \n".format(torch.cuda.memory_summary()))

            # all_gather the memory usage and log it.
            memory_usage_list = utils.our_allgather_among_cpu_processes_float_list([memory_usage], utils.DEFAULT_GROUP)
            if max([a[0] for a in memory_usage_list]) > args.densify_memory_limit:# In expe `rubble_2k_mp_9`, memory_usage>18GB leads to OOM.
                print("Memory usage is over 18GB per GPU. stop densification.\n")
                log_file.write("Memory usage is over 20GB per GPU. stop densification.\n")
                args.disable_auto_densification = True

            utils.inc_densify_iter()
        
        if utils.check_update_at_this_iter(iteration, args.bsz, args.opacity_reset_interval, 0) and iteration+args.bsz <= args.opacity_reset_until_iter:
            # TODO: do opacity reset if dataset_args.white_background and iteration == opt_args.densify_from_iter
            timers.start("reset_opacity")
            gaussians.reset_opacity()
            timers.stop("reset_opacity")

        timers.stop("densification")
    else:

        if args.clear_floaters and iteration > args.densify_until_iter:
            # clear floaters
            if utils.check_update_at_this_iter(iteration, args.bsz, args.prune_based_on_opacity_interval, 0):
                gaussians.prune_based_on_opacity(args.min_opacity)
                # if iteration == 240001 or iteration == 210001 or iteration == 220001 or iteration == 230001:
                #     gaussians.reset_opacity()

        if iteration > args.densify_from_iter and utils.check_update_at_this_iter(iteration, args.bsz, args.densification_interval, 0):
            # measue the memory usage.
            # torch.cuda.empty_cache()
            memory_usage = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
            max_memory_usage = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
            max_reserved_memory = torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024
            now_reserved_memory = torch.cuda.memory_reserved() / 1024 / 1024 / 1024
            log_file.write("iteration[{},{}) Now num of 3dgs: {}. Now Memory usage: {} GB. Max Memory usage: {} GB. Max Reserved Memory: {} GB. Now Reserved Memory: {} GB. \n".format(
                iteration, iteration+args.bsz, gaussians.get_xyz.shape[0], memory_usage, max_memory_usage, max_reserved_memory, now_reserved_memory))



