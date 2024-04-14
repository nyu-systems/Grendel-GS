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

import os
import torch
import json
from utils.loss_utils import l1_loss
from gaussian_renderer import (
        preprocess3dgs_and_all2all,
        render
    )
from torch.cuda import nvtx
from gaussian_renderer.loss_distribution import loss_computation
import sys
from scene import Scene, GaussianModel, SceneDataset
from scene.workload_division import create_division_strategy_history, get_local_running_time_by_modes
from utils.general_utils import safe_state, init_distributed, prepare_output_and_logger
import utils.general_utils as utils
from utils.timer import Timer
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from arguments import (
    AuxiliaryParams,
    ModelParams, 
    PipelineParams, 
    OptimizationParams, 
    DistributionParams, 
    BenchmarkParams, 
    DebugParams, 
    print_all_args, 
    check_args
)
import time
import torch.distributed as dist

def globally_sync_for_timer():
    if utils.check_enable_python_timer() and utils.MP_GROUP.size() > 1:
        torch.distributed.barrier(group=utils.MP_GROUP)

def densification(iteration, scene, gaussians, batched_screenspace_pkg):
    args = utils.get_args()
    timers = utils.get_timers()
    log_file = utils.get_log_file()

    # Update Statistics for redistribution
    if args.memory_distribution_mode != "0":
        for local2j_ids_bool in batched_screenspace_pkg["batched_local2j_ids_bool"]:
            gaussians.send_to_gpui_cnt += local2j_ids_bool

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
            gaussians.densify_and_prune(args.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
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

            memory_usage = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
            max_memory_usage = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
            log_file.write("iteration[{},{}) densify_and_prune. Now num of 3dgs: {}. Now Memory usage: {} GB. Max Memory usage: {} GB. \n".format(
                iteration, iteration+args.bsz, gaussians.get_xyz.shape[0], memory_usage, max_memory_usage))

            utils.inc_densify_iter()
        
        if utils.check_update_at_this_iter(iteration, args.bsz, args.opacity_reset_interval, 0):
            # TODO: do opacity reset if dataset_args.white_background and iteration == opt_args.densify_from_iter
            timers.start("reset_opacity")
            gaussians.reset_opacity()
            timers.stop("reset_opacity")

        timers.stop("densification")



def training(dataset_args, opt_args, pipe_args, args, log_file):
    # dataset_args, opt_args, pipe_args, args contain arguments containing all kinds of settings and configurations. 
    # In which, the first three are sub-domains, and the fourth one contains all of them.

    # init auxiliary tools
    timers = Timer(args)
    utils.set_timers(timers)
    utils.set_log_file(log_file)
    prepare_output_and_logger(dataset_args)

    # init parameterized scene
    gaussians = GaussianModel(dataset_args.sh_degree)
    with torch.no_grad():
        scene = Scene(dataset_args, gaussians)
        scene.log_scene_info_to_file(log_file, "Scene Info Before Training")
        gaussians.training_setup(opt_args)
    utils.check_memory_usage_logging("after init and before training loop")

    # init dataset
    train_dataset = SceneDataset(scene.getTrainCameras(dataset_args.train_resolution_scale))
    # init workload division strategy
    cameraId2StrategyHistory = {}
    # init background
    bg_color = [1, 1, 1] if dataset_args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Training Loop
    train_start_time = time.time()
    progress_bar = tqdm(range(1, opt_args.iterations + 1), desc="Training progress", disable=(utils.LOCAL_RANK != 0))
    num_trained_batches = 0
    for iteration in range(1, opt_args.iterations + 1, args.bsz):

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
        batched_cameras = train_dataset.get_batched_cameras(args.bsz)
        batched_screenspace_pkg = {"batched_locally_preprocessed_radii":[],
                                   "batched_locally_preprocessed_visibility_filter":[],
                                   "batched_locally_preprocessed_mean2D":[],
                                   "batched_local2j_ids_bool":[],
                                   "statistic_collectors":[],
                                   "losses": []}
        batched_parameter_gradients_pkg = {}

        # Prepare Workload division strategy
        timers.start("prepare_strategies")
        batched_strategies = []
        batched_strategy_histories = []
        for viewpoint_cam in batched_cameras:
            if viewpoint_cam.uid not in cameraId2StrategyHistory:
                cameraId2StrategyHistory[viewpoint_cam.uid] = create_division_strategy_history(viewpoint_cam, 
                                                                                               args.render_distribution_adjust_mode)
            strategy_history = cameraId2StrategyHistory[viewpoint_cam.uid]
            strategy = strategy_history.start_strategy()
            batched_strategies.append(strategy)
            batched_strategy_histories.append(strategy_history)
        timers.stop("prepare_strategies")

        assert args.bsz % args.dp_size == 0, "dp_size must be a divisor of bsz."
        micro_bsz_size = args.dp_size
        num_samples_per_dp_rank = args.bsz // args.dp_size
        for micro_step in range(args.bsz // args.dp_size):
            # Micro Step Initialization
            micro_batched_cameras = batched_cameras[micro_step::num_samples_per_dp_rank]
            micro_batched_strategies = batched_strategies[micro_step::num_samples_per_dp_rank]
            local_render_viewpoint_cam = micro_batched_cameras[utils.DP_GROUP.rank()]
            utils.set_img_size(local_render_viewpoint_cam.image_height, local_render_viewpoint_cam.image_width)
            local_render_strategy = micro_batched_strategies[utils.DP_GROUP.rank()]

            # 3DGS preprocess and all2all communication
            globally_sync_for_timer()
            screenspace_pkg = preprocess3dgs_and_all2all(micro_batched_cameras, gaussians, pipe_args, background,
                                                         micro_batched_strategies,
                                                         mode="train")
            statistic_collector = screenspace_pkg["cuda_args"]["stats_collector"]

            # Pixel-wise Render
            globally_sync_for_timer() # NOTE: this is to make sure: we are measuring time for local work. where to add this barrier depends on: whether there will be global communication(i.e. allreduce) in the following code.
            image, compute_locally = render(screenspace_pkg, local_render_strategy)

            # Pixel-wise Loss Computation
            globally_sync_for_timer()
            Ll1, ssim_loss = loss_computation(image,
                                              local_render_viewpoint_cam,
                                              compute_locally,
                                              local_render_strategy,
                                              statistic_collector)
            loss = (1.0 - opt_args.lambda_dssim) * Ll1 + opt_args.lambda_dssim * (1.0 - ssim_loss)
            utils.check_memory_usage_logging("after loss")

            # Backward
            globally_sync_for_timer()
            timers.start("backward")
            loss.backward()
            timers.stop("backward")
            utils.check_memory_usage_logging("after backward")

            batched_screenspace_pkg["batched_locally_preprocessed_radii"].extend(screenspace_pkg["batched_locally_preprocessed_radii"])
            batched_screenspace_pkg["batched_locally_preprocessed_visibility_filter"].extend(screenspace_pkg["batched_locally_preprocessed_visibility_filter"])
            batched_screenspace_pkg["batched_locally_preprocessed_mean2D"].extend(screenspace_pkg["batched_locally_preprocessed_mean2D"])
            batched_screenspace_pkg["statistic_collectors"].append(statistic_collector)
            batched_screenspace_pkg["losses"].append(loss)
            
            # TODO: store this gradient of accumulation step and then implement all gather at rank 0 to get the gradients of all accumulation steps. to compute intra batch statistics like cosine similarity and noise signal ratio.
            if args.memory_distribution_mode != "0":
                batched_screenspace_pkg["batched_local2j_ids_bool"].append(screenspace_pkg["local2j_ids_bool"])

        with torch.no_grad():
            # Sync gradients across replicas, if some 3dgs are stored replicatedly.
            globally_sync_for_timer()
            timers.start("sync_gradients_for_replicated_3dgs_storage")
            gaussians.sync_gradients_for_replicated_3dgs_storage(batched_screenspace_pkg)
            if args.memory_distribution_mode == "0" and utils.MP_GROUP.size() > 1:
                for local_render_screenspace_mean2D in batched_screenspace_pkg["batched_locally_preprocessed_mean2D"]:
                    torch.distributed.all_reduce(local_render_screenspace_mean2D.grad.data, op=dist.ReduceOp.SUM, group=utils.MP_GROUP)
            timers.stop("sync_gradients_for_replicated_3dgs_storage")

            # Adjust workload division strategy. 
            globally_sync_for_timer()
            timers.start("strategy.update_stats")
            if iteration > args.adjust_strategy_warmp_iterations and utils.DEFAULT_GROUP.size() > 1:
                ### orginal implementation. When the new implementation is stable, I will delete the old implementation. 
                # all_statistic_collector = [None for _ in range(utils.DP_GROUP.size())]
                # timers.start("strategy.update_stats.all_statistic_collector")
                # if utils.DP_GROUP.size() > 1:
                #     torch.distributed.all_gather_object(all_statistic_collector, batched_screenspace_pkg["statistic_collectors"], group=utils.DP_GROUP)
                #     all_statistic_collector = sum(all_statistic_collector, [])
                # else:
                #     all_statistic_collector = batched_screenspace_pkg["statistic_collectors"]
                # timers.stop("strategy.update_stats.all_statistic_collector")
                # timers.start("strategy.update_stats.update_stats")
                # for idx in range(args.bsz):
                #     batched_strategies[idx].update_stats(all_statistic_collector[idx])
                #     batched_strategy_histories[idx].finish_strategy()
                # timers.stop("strategy.update_stats.update_stats")
                pass

                ### new implementation
                assert args.render_distribution_adjust_mode in ["2", "5", "6"], "Only support mode 2, 5 and 6 for now."
                timers.start("strategy.update_stats.all_gather_running_time")
                batched_local_running_time = []
                for statistic_collector in batched_screenspace_pkg["statistic_collectors"]:
                    batched_local_running_time.append(get_local_running_time_by_modes(statistic_collector))

                all_running_time = utils.our_allgather_among_cpu_processes_float_list(batched_local_running_time, utils.DEFAULT_GROUP)

                timers.stop("strategy.update_stats.all_gather_running_time")

                for dp_rk in range(utils.DP_GROUP.size()):
                    for accum_idx_in_one_dp_rk in range(num_samples_per_dp_rank):
                        idx_in_one_batch = dp_rk * num_samples_per_dp_rank + accum_idx_in_one_dp_rk
                        all_running_time_cross_mp = []
                        for mp_rk in range(utils.MP_GROUP.size()):
                            all_running_time_cross_mp.append(all_running_time[ dp_rk * utils.MP_GROUP.size() + mp_rk ][accum_idx_in_one_dp_rk])
                        batched_strategies[idx_in_one_batch].update_stats(all_running_time_cross_mp)
                        batched_strategy_histories[idx_in_one_batch].finish_strategy()

            timers.stop("strategy.update_stats")


            # Update Epoch Statistics: allgather loss into a tensor across DP GROUP
            timers.start("allgather_loss_and_log")
            if utils.DP_GROUP.size() > 1:
                local_losses = torch.stack(batched_screenspace_pkg["losses"])
                losses = torch.empty( (args.dp_size, num_samples_per_dp_rank), dtype=torch.float32, device="cuda")
                torch.distributed.all_gather_into_tensor(losses, local_losses, group=utils.DP_GROUP)
                losses_cpu = losses.flatten().cpu().tolist()
            else:
                losses_cpu = torch.stack(batched_screenspace_pkg["losses"]).cpu().tolist()
            train_dataset.update_losses(losses_cpu)

            # Logging
            losses_cpu = [round(loss, 6) for loss in losses_cpu]
            log_string = "iteration[{},{}) loss: {} image: {}\n".format(iteration, iteration+args.bsz,
                                                                        losses_cpu,
                                                                        [viewpoint_cam.image_name for viewpoint_cam in batched_cameras])
            log_file.write(log_string)
            timers.stop("allgather_loss_and_log")

            if utils.check_update_at_this_iter(iteration, args.bsz, args.log_interval, 0):
                if args.debug_why and iteration > args.densify_until_iter:
                    breakpoint()
                stats, exp_avg_dict, exp_avg_sq_dict = gaussians.log_gaussian_stats()
                log_file.write("iteration[{},{}) gaussian stats: {}\n".format(iteration, iteration+args.bsz, stats))
                log_file.write("iteration[{},{}) exp_avg_dict: {}\n".format(iteration, iteration+args.bsz, exp_avg_dict))
                log_file.write("iteration[{},{}) exp_avg_sq_dict: {}\n".format(iteration, iteration+args.bsz, exp_avg_sq_dict))

            # Log and save
            training_report(iteration, l1_loss, args.test_iterations, scene, pipe_args, background, dataset_args.test_resolution_scale)

            # Densification
            densification(iteration, scene, gaussians, batched_screenspace_pkg)

            # Save Gaussians
            # if for some save_iteration in save_iterations, iteration <= save_iteration < iteration+args.bsz, then save the gaussians.
            if any([iteration <= save_iteration < iteration+args.bsz for save_iteration in args.save_iterations]):
                utils.print_rank_0("\n[ITER {}] Saving Gaussians".format(iteration))
                log_file.write("[ITER {}] Saving Gaussians\n".format(iteration))
                scene.save(iteration)

            # Optimizer step
            if iteration < opt_args.iterations:
                timers.start("optimizer_step")

                # if "visible_count" in args.grad_normalization_mode:
                #     # sum_visibility_filter_int = batched_screenspace_pkg["sum_batched_locally_preprocessed_visibility_filter_int"]
                #     # # compute histogram of visible count
                #     # max_count = torch.max(sum_visibility_filter_int)
                #     # visibility_hist = torch.histc(sum_visibility_filter_int, bins=max_count+1, min=0, max=max_count+1)
                #     # # normalize to ratio
                #     # visibility_hist = visibility_hist / torch.sum(visibility_hist)
                #     # log_file.write("iteration[{},{}) visibility_hist: {}\n".format(iteration, iteration+args.bsz, visibility_hist.cpu().tolist()))
                    
                #     if args.grad_normalization_mode == "divide_by_visible_count":
                #         gradient_multiplier = 1 / batched_screenspace_pkg["sum_batched_locally_preprocessed_visibility_filter_int"]
                #         gradient_multiplier[gradient_multiplier.isnan()] = 0.0
                #     elif args.grad_normalization_mode == "multiply_by_visible_count":
                #         gradient_multiplier = batched_screenspace_pkg["sum_batched_locally_preprocessed_visibility_filter_int"]
                #     elif args.grad_normalization_mode == "square_multiply_by_visible_count":
                #         gradient_multiplier = batched_screenspace_pkg["sum_batched_locally_preprocessed_visibility_filter_int"] ** 2
                    
                #     try:
                #         for param in gaussians.all_parameters():
                #             if param.grad is None:
                #                 continue
                #             if len(param.shape) == 2:
                #                 param.grad *= gradient_multiplier.unsqueeze(1)
                #             elif len(param.shape) == 3:
                #                 param.grad *= gradient_multiplier.unsqueeze(1).unsqueeze(2)
                #             else:
                #                 raise NotImplementedError("Not implemented for this shape of parameter.")
                #     except Exception as e:
                #         if utils.LOCAL_RANK == 0:
                #             print("Error in grad normalization: ", e)
                #             breakpoint()
                #         else:
                #             time.sleep(1000)

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
        if args.nsys_profile:
            nvtx.range_pop()
        if utils.check_enable_python_timer():
            timers.printTimers(iteration, mode="sum")
        log_file.flush()

    # Finish training
    if args.end2end_time:
        torch.cuda.synchronize()
        log_file.write("end2end total_time: {:.6f} ms, iterations: {}, throughput {:.2f} it/s\n".format(time.time() - train_start_time, opt_args.iterations, opt_args.iterations/(time.time() - train_start_time)))
    
    log_file.write("Max Memory usage: {} GB.\n".format(torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024))

    # Save some running statistics to file.
    if not args.performance_stats:
        data_json = {}
        for camera_id, strategy_history in cameraId2StrategyHistory.items():
            data_json[camera_id] = strategy_history.to_json()
        
        with open(args.log_folder+"/strategy_history_ws="+str(utils.WORLD_SIZE)+"_rk="+str(utils.GLOBAL_RANK)+".json", 'w') as f:
            json.dump(data_json, f)

def training_report(iteration, l1_loss, testing_iterations, scene : Scene, pipe_args, background, test_resolution_scale=1.0):
    log_file = utils.get_log_file()
    # Report test and samples of training set
    if len(testing_iterations) > 0 and utils.check_update_at_this_iter(iteration, utils.get_args().bsz, testing_iterations[0], 0):
        testing_iterations.pop(0)
        utils.print_rank_0("\n[ITER {}] Start Testing".format(iteration))
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras(test_resolution_scale)}, 
                            {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = torch.scalar_tensor(0.0, device="cuda")
                psnr_test = torch.scalar_tensor(0.0, device="cuda")

                num_cameras = len(config['cameras'])
                eval_dataset = SceneDataset(config['cameras'])
                for idx in range(1, num_cameras+1, args.dp_size):
                    batched_cameras = eval_dataset.get_batched_cameras(args.dp_size)
                    local_render_camera = batched_cameras[utils.DP_GROUP.rank()]
                    batched_strategies = []
                    for viewpoint in batched_cameras:
                        hack_history = create_division_strategy_history(viewpoint, "evaluation")
                        batched_strategies.append(hack_history.start_strategy())
                    local_render_strategy = batched_strategies[utils.DP_GROUP.rank()]
                    screenspace_pkg = preprocess3dgs_and_all2all(batched_cameras, scene.gaussians, pipe_args, background,
                                                                 batched_strategies,
                                                                 mode="test")
                    image, _ = render(screenspace_pkg, local_render_strategy)

                    if utils.MP_GROUP.size() > 1:
                        torch.distributed.all_reduce(image, op=dist.ReduceOp.SUM, group=utils.MP_GROUP)
                    image = torch.clamp(image, 0.0, 1.0)
                    gt_image = torch.clamp(local_render_camera.original_image.to("cuda"), 0.0, 1.0)

                    if idx + utils.DP_GROUP.rank() < num_cameras + 1:
                        l1_test += l1_loss(image, gt_image).mean().double()
                        psnr_test += psnr(image, gt_image).mean().double()
                if utils.DP_GROUP.size() > 1:
                    torch.distributed.all_reduce(l1_test, op=dist.ReduceOp.SUM, group=utils.DP_GROUP)
                    torch.distributed.all_reduce(psnr_test, op=dist.ReduceOp.SUM, group=utils.DP_GROUP)
                psnr_test /= num_cameras
                l1_test /= num_cameras
                utils.print_rank_0("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                log_file.write("[ITER {}] Evaluating {}: L1 {} PSNR {}\n".format(iteration, config['name'], l1_test, psnr_test))

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    ap = AuxiliaryParams(parser)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    dist_p = DistributionParams(parser)
    bench_p = BenchmarkParams(parser)
    debug_p = DebugParams(parser)
    args = parser.parse_args(sys.argv[1:])

    # Set up distributed training
    init_distributed(args)

    ## Prepare arguments.
    # Check arguments
    check_args(args)
    # Set up global args
    utils.set_args(args)


    # create log folder
    if utils.GLOBAL_RANK == 0:
        os.makedirs(args.log_folder, exist_ok = True)
        os.makedirs(args.model_path, exist_ok = True)
    if utils.WORLD_SIZE > 1:
        torch.distributed.barrier(group=utils.DEFAULT_GROUP)# make sure log_folder is created before other ranks start writing log.

    if utils.LOCAL_RANK == 0:
        with open(args.log_folder+"/args.json", 'w') as f:
            json.dump(vars(args), f)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # Initialize log file and print all args
    log_file = open(args.log_folder+"/python_ws="+str(utils.WORLD_SIZE)+"_rk="+str(utils.GLOBAL_RANK)+".log", 'w')
    print_all_args(args, log_file)

    training(lp.extract(args), op.extract(args), pp.extract(args), args, log_file)

    # All done
    utils.print_rank_0("\nTraining complete.")