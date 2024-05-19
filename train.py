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
from gaussian_renderer.workload_division import get_division_strategy_history, get_local_running_time_by_modes
from utils.general_utils import safe_state, init_distributed, prepare_output_and_logger, globally_sync_for_timer, init_distributed_final
import utils.general_utils as utils
from utils.timer import Timer, End2endTimer
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
    init_args
)
import time
import torch.distributed as dist
from densification import densification
import final_train
import final_train2


def training(dataset_args, opt_args, pipe_args, args, log_file):
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
            assert number_files == utils.DEFAULT_GROUP.size(), "The number of files in the checkpoint folder must be equal to the number of processes."
            if args.start_checkpoint[-1] != "/":
                args.start_checkpoint += "/"
            file_name = args.start_checkpoint+"chkpnt" + str(utils.DEFAULT_GROUP.rank()) + ".pth"
            (model_params, start_from_this_iteration) = torch.load(file_name)
            gaussians.restore(model_params, opt_args)
            start_from_this_iteration += args.dp_size
            utils.print_rank_0("Restored from checkpoint: {}".format(file_name))
            log_file.write("Restored from checkpoint: {}\n".format(file_name))

        scene.log_scene_info_to_file(log_file, "Scene Info Before Training")

    utils.check_memory_usage_logging("after init and before training loop")

    # init dataset
    train_dataset = SceneDataset(scene.getTrainCameras(dataset_args.train_resolution_scale))
    if args.adjust_strategy_warmp_iterations == -1:
        args.adjust_strategy_warmp_iterations = len(train_dataset.cameras)
        # use one epoch to warm up. do not use the first epoch's running time for adjustment of strategy.
    # init workload division strategy
    cameraId2StrategyHistory = {}
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
        batched_cameras = train_dataset.get_batched_cameras(args.bsz)
        batched_screenspace_pkg = {"batched_locally_preprocessed_radii":[],
                                   "batched_locally_preprocessed_visibility_filter":[],
                                   "batched_locally_preprocessed_mean2D":[],
                                   "batched_local2j_ids_bool":[],
                                   "statistic_collectors":[],
                                   "losses": []}
        batched_parameter_gradients_pkg = {}
        # DEBUG
        # if utils.LOCAL_RANK == 0:
        #     for camera in batched_cameras:
        #         print(camera.image_name)

        # Prepare Workload division strategy
        timers.start("prepare_strategies")
        batched_strategy_histories = [get_division_strategy_history(cameraId2StrategyHistory,
                                                                    viewpoint_cam,
                                                                    args.image_distribution_config.workloads_division_mode)
                                                    for viewpoint_cam in batched_cameras]
        batched_strategies = [strategy_history.start_strategy() for strategy_history in batched_strategy_histories]
        timers.stop("prepare_strategies")

        assert args.bsz % args.dp_size == 0, "dp_size must be a divisor of bsz."
        micro_bsz_size = args.dp_size
        num_samples_per_dp_worker = args.bsz // args.dp_size
        for micro_step in range(num_samples_per_dp_worker):
            # Micro Step Initialization
            micro_batched_cameras = batched_cameras[micro_step::num_samples_per_dp_worker]
            micro_batched_strategies = batched_strategies[micro_step::num_samples_per_dp_worker]
            local_render_viewpoint_cam = micro_batched_cameras[utils.DP_GROUP.rank()]
            utils.set_img_size(local_render_viewpoint_cam.image_height, local_render_viewpoint_cam.image_width)
            local_render_strategy = micro_batched_strategies[utils.DP_GROUP.rank()]
            if args.sync_more:
                print("local_render_strategy",utils.LOCAL_RANK, local_render_viewpoint_cam.image_name, local_render_strategy.get_global_strategy_str())
            # 3DGS preprocess and all2all communication
            globally_sync_for_timer()
            screenspace_pkg = preprocess3dgs_and_all2all(micro_batched_cameras, gaussians, pipe_args, background,
                                                         micro_batched_strategies,
                                                         mode="train")
            statistic_collector = screenspace_pkg["cuda_args"]["stats_collector"]
            if args.sync_more:
                torch.cuda.synchronize()
            # Pixel-wise Render
            globally_sync_for_timer() # NOTE: this is to make sure: we are measuring time for local work. where to add this barrier depends on: whether there will be global communication(i.e. allreduce) in the following code.
            image, compute_locally = render(screenspace_pkg, local_render_strategy)

            if args.sync_more:
                torch.cuda.synchronize()
                print("afterrenderforward",utils.LOCAL_RANK, local_render_viewpoint_cam.image_name, statistic_collector)
                log_file.write("afterrenderforward: {} {} {}\n".format(utils.LOCAL_RANK, local_render_viewpoint_cam.image_name, statistic_collector))

            # Pixel-wise Loss Computation
            globally_sync_for_timer()# adding this also creates some unstability in the time measurement.
            Ll1, ssim_loss = loss_computation(image,
                                              local_render_viewpoint_cam,
                                              compute_locally,
                                              local_render_strategy,
                                              statistic_collector,
                                              args.image_distribution_config.loss_distribution_mode)
            loss = (1.0 - opt_args.lambda_dssim) * Ll1 + opt_args.lambda_dssim * (1.0 - ssim_loss)
            loss = loss * args.lr_scale_loss
            utils.check_memory_usage_logging("after loss")

            if args.sync_more:
                torch.cuda.synchronize()

            # Backward
            globally_sync_for_timer()
            timers.start("backward")
            # if loss is a tensor
            loss.backward()
            timers.stop("backward")
            utils.check_memory_usage_logging("after backward")

            if args.sync_more:
                torch.cuda.synchronize()
                print("afterbackward", utils.LOCAL_RANK, local_render_viewpoint_cam.image_name, statistic_collector)

            batched_screenspace_pkg["batched_locally_preprocessed_radii"].extend(screenspace_pkg["batched_locally_preprocessed_radii"])
            batched_screenspace_pkg["batched_locally_preprocessed_visibility_filter"].extend(screenspace_pkg["batched_locally_preprocessed_visibility_filter"])
            batched_screenspace_pkg["batched_locally_preprocessed_mean2D"].extend(screenspace_pkg["batched_locally_preprocessed_mean2D"])
            batched_screenspace_pkg["statistic_collectors"].append(statistic_collector)
            batched_screenspace_pkg["losses"].append(loss)
            
            # TODO: store this gradient of accumulation step and then implement all gather at rank 0 to get the gradients of all accumulation steps. to compute intra batch statistics like cosine similarity and noise signal ratio.
            if args.gaussians_distribution:
                batched_screenspace_pkg["batched_local2j_ids_bool"].append(screenspace_pkg["local2j_ids_bool"])

            # Release memory of locally rendered original_image
            torch.cuda.synchronize()
            local_render_viewpoint_cam.gt_image_comm_op = None
            local_render_viewpoint_cam.original_image = None


        with torch.no_grad():
            # Sync gradients across replicas, if some 3dgs are stored replicatedly.
            globally_sync_for_timer()
            timers.start("sync_gradients_for_replicated_3dgs_storage")
            gaussians.sync_gradients_for_replicated_3dgs_storage(batched_screenspace_pkg)
            if not args.gaussians_distribution and utils.MP_GROUP.size() > 1:
                for local_render_screenspace_mean2D in batched_screenspace_pkg["batched_locally_preprocessed_mean2D"]:
                    torch.distributed.all_reduce(local_render_screenspace_mean2D.grad.data, op=dist.ReduceOp.SUM, group=utils.MP_GROUP)
            timers.stop("sync_gradients_for_replicated_3dgs_storage")

            # Adjust workload division strategy. 
            globally_sync_for_timer()
            timers.start("strategy.update_stats")
            if iteration > args.adjust_strategy_warmp_iterations and utils.DEFAULT_GROUP.size() > 1:
                ### new implementation
                timers.start("strategy.update_stats.all_gather_running_time")
                batched_local_running_time = []
                for statistic_collector in batched_screenspace_pkg["statistic_collectors"]:
                    batched_local_running_time.append(get_local_running_time_by_modes(statistic_collector))
                all_running_time = utils.our_allgather_among_cpu_processes_float_list(batched_local_running_time, utils.DEFAULT_GROUP)
                timers.stop("strategy.update_stats.all_gather_running_time")

                for dp_rk in range(utils.DP_GROUP.size()):
                    for accum_idx_in_one_dp_rk in range(num_samples_per_dp_worker):
                        idx_in_one_batch = dp_rk * num_samples_per_dp_worker + accum_idx_in_one_dp_rk
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
                losses = torch.empty( (args.dp_size, num_samples_per_dp_worker), dtype=torch.float32, device="cuda")
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

            if utils.check_update_at_this_iter(iteration, args.bsz, args.log_interval, 0) and args.debug_why:
                # if args.debug_why and iteration > args.densify_until_iter:
                #     breakpoint()
                stats, exp_avg_dict, exp_avg_sq_dict = gaussians.log_gaussian_stats()
                log_file.write("iteration[{},{}) gaussian stats: {}\n".format(iteration, iteration+args.bsz, stats))
                log_file.write("iteration[{},{}) exp_avg_dict: {}\n".format(iteration, iteration+args.bsz, exp_avg_dict))
                log_file.write("iteration[{},{}) exp_avg_sq_dict: {}\n".format(iteration, iteration+args.bsz, exp_avg_sq_dict))

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
                data_json = {}
                for camera_id, strategy_history in cameraId2StrategyHistory.items():
                    data_json[camera_id] = strategy_history.to_json()
                
                with open(args.log_folder+"/strategy_history_ws="+str(utils.WORLD_SIZE)+"_rk="+str(utils.GLOBAL_RANK)+".json", 'w') as f:
                    json.dump(data_json, f)
                end2end_timers.start()

            if any([iteration <= checkpoint_iteration < iteration+args.bsz for checkpoint_iteration in args.checkpoint_iterations]):
                end2end_timers.stop()
                utils.print_rank_0("\n[ITER {}] Saving Checkpoint".format(iteration))
                log_file.write("\n[ITER {}] Saving Checkpoint".format(iteration))
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
        if args.nsys_profile:
            nvtx.range_pop()
        if utils.check_enable_python_timer():
            timers.printTimers(iteration, mode="sum")
        log_file.flush()

    # Finish training
    end2end_timers.print_time(log_file, opt_args.iterations)
    log_file.write("Max Memory usage: {} GB.\n".format(torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024))

def training_report(iteration, l1_loss, testing_iterations, scene : Scene, pipe_args, background, test_resolution_scale=1.0):
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
                cameraId2StrategyHistory = {}
                for idx in range(1, num_cameras+1, args.dp_size):
                    if args.empty_cache_more and idx % 10 == 0:
                        torch.cuda.empty_cache()
                    batched_cameras = eval_dataset.get_batched_cameras(args.dp_size)
                    local_render_camera = batched_cameras[utils.DP_GROUP.rank()]
                    batched_strategies = []
                    for viewpoint in batched_cameras:
                        hack_history = get_division_strategy_history(cameraId2StrategyHistory, viewpoint, "evaluation")
                        batched_strategies.append(hack_history.start_strategy())
                    local_render_strategy = batched_strategies[utils.DP_GROUP.rank()]
                    screenspace_pkg = preprocess3dgs_and_all2all(batched_cameras, scene.gaussians, pipe_args, background,
                                                                 batched_strategies,
                                                                 mode="test")
                    image, _ = render(screenspace_pkg, local_render_strategy)

                    if len(image.shape) == 0:
                        print(f"warning[{utils.DEFAULT_GROUP.rank()}]: image is not rendered, but set to zero tensor.")
                        image = torch.zeros(gt_image.shape, device="cuda", dtype=torch.float32)

                    if utils.MP_GROUP.size() > 1:
                        torch.distributed.all_reduce(image, op=dist.ReduceOp.SUM, group=utils.MP_GROUP)
                    image = torch.clamp(image, 0.0, 1.0)
                    gt_image = torch.clamp(local_render_camera.original_image / 255.0, 0.0, 1.0)

                    if idx + utils.DP_GROUP.rank() < num_cameras + 1:
                        l1_test += l1_loss(image, gt_image).mean().double()
                        psnr_test += psnr(image, gt_image).mean().double()
                    local_render_camera.original_image = None # release memory
                if utils.DP_GROUP.size() > 1:
                    torch.distributed.all_reduce(l1_test, op=dist.ReduceOp.SUM, group=utils.DP_GROUP)
                    torch.distributed.all_reduce(psnr_test, op=dist.ReduceOp.SUM, group=utils.DP_GROUP)
                psnr_test /= num_cameras
                l1_test /= num_cameras
                utils.print_rank_0("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                log_file.write("[ITER {}] Evaluating {}: L1 {} PSNR {}\n".format(iteration, config['name'], l1_test, psnr_test))

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # DEBUG purpose
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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
    if args.use_final_system or args.use_final_system2:
        init_distributed_final(args)
    else:
        init_distributed(args)

    ## Prepare arguments.
    # Check arguments
    init_args(args)
    # Set up global args
    utils.set_args(args)


    # create log folder
    if utils.GLOBAL_RANK == 0:
        os.makedirs(args.log_folder, exist_ok = True)
        os.makedirs(args.model_path, exist_ok = True)
    if utils.WORLD_SIZE > 1:
        torch.distributed.barrier(group=utils.DEFAULT_GROUP)# make sure log_folder is created before other ranks start writing log.

    if utils.GLOBAL_RANK == 0:
        with open(args.log_folder+"/args.json", 'w') as f:
            json.dump(vars(args), f)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # Initialize log file and print all args
    log_file = open(args.log_folder+"/python_ws="+str(utils.WORLD_SIZE)+"_rk="+str(utils.GLOBAL_RANK)+".log", 'a' if args.auto_start_checkpoint else 'w')
    print_all_args(args, log_file)

    if args.use_final_system:
        final_train.training(lp.extract(args), op.extract(args), pp.extract(args), args, log_file)
    elif args.use_final_system2:
        final_train2.training(lp.extract(args), op.extract(args), pp.extract(args), args, log_file)
    else:
        training(lp.extract(args), op.extract(args), pp.extract(args), args, log_file)

    # All done
    torch.distributed.barrier(group=utils.DEFAULT_GROUP)
    utils.print_rank_0("\nTraining complete.")