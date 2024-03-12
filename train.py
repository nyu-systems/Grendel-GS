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
import numpy as np
import json
from random import randint
from utils.loss_utils import l1_loss
from gaussian_renderer import render, network_gui
from gaussian_renderer.loss_distribution import distributed_loss_computation, replicated_loss_computation
import sys
from scene import Scene, GaussianModel
from scene.workload_division import get_evenly_global_strategy_str, create_division_strategy_history
from utils.general_utils import safe_state, init_distributed
import utils.general_utils as utils
from utils.timer import Timer
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import (
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
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import torch.distributed as dist
import diff_gaussian_rasterization

def training(dataset, opt, pipe, args, log_file):
    (testing_iterations,
     saving_iterations, 
     checkpoint_iterations, 
     checkpoint, 
     debug_from, 
     fixed_training_image, 
     disable_auto_densification) = (
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.fixed_training_image,
        args.disable_auto_densification
    )

    timers = Timer(args)
    utils.set_timers(timers)
    utils.set_log_file(log_file)
    utils.set_cur_iter(0)

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)

    with torch.no_grad():
        scene = Scene(dataset, gaussians)
        if args.duplicate_gs_cnt > 0:
            gaussians.duplicate_gaussians(args.duplicate_gs_cnt)
        gaussians.training_setup(opt)

    if checkpoint:
        assert not args.memory_distribution, "memory_distribution does not support checkpoint yet!"
        (model_params, first_iter) = torch.load(checkpoint, map_location=torch.device("cuda", utils.LOCAL_RANK))
        gaussians.restore(model_params, opt)
        print("rank {} Restored from checkpoint: {}, at iteration {}".format(utils.LOCAL_RANK, checkpoint, first_iter))
        log_file.write("rank {} Restored from checkpoint: {}, at iteration {}\n".format(utils.LOCAL_RANK, checkpoint, first_iter))

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # Print shape of gaussians parameters. 
    log_file.write("xyz shape: {}\n".format(gaussians._xyz.shape))
    log_file.write("f_dc shape: {}\n".format(gaussians._features_dc.shape))
    log_file.write("f_rest shape: {}\n".format(gaussians._features_rest.shape))
    log_file.write("opacity shape: {}\n".format(gaussians._opacity.shape))
    log_file.write("scaling shape: {}\n".format(gaussians._scaling.shape))
    log_file.write("rotation shape: {}\n".format(gaussians._rotation.shape))


    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # init epoch stats
    epoch_loss = 0
    epoch_id = 0
    epoch_progress_cnt = 0
    epoch_camera_size = len(scene.getTrainCameras())

    # init workload division strategy stuff
    cameraId2StrategyHistory = {}

    utils.check_memory_usage_logging("after init and before training loop")    

    train_start_time = time.time()

    for iteration in range(first_iter, opt.iterations + 1):        

        # Step Initialization
        utils.set_cur_iter(iteration)
        timers.clear()
        timers.start("pre_forward")
        iter_start.record()
        gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        # DEBUG: understand time for one cuda synchronize call.
        # timers.start("test_cuda_synchronize_time")
        # timers.stop("test_cuda_synchronize_time")


        # Prepara data: Pick a random Camera
        if not viewpoint_stack:
            log_file.write("reset viewpoint stack\n")
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = None
        if fixed_training_image == -1:
            camera_id = randint(0, len(viewpoint_stack)-1)
            viewpoint_cam = viewpoint_stack.pop(camera_id)
        else:
            viewpoint_cam = viewpoint_stack[fixed_training_image]
        utils.set_img_size(viewpoint_cam.image_height, viewpoint_cam.image_width)


        # Prepare Workload division strategy

        if viewpoint_cam.uid not in cameraId2StrategyHistory:
            cameraId2StrategyHistory[viewpoint_cam.uid] = create_division_strategy_history(viewpoint_cam, 
                                                                                           args.render_distribution_mode)
        strategy_history = cameraId2StrategyHistory[viewpoint_cam.uid]
        strategy = strategy_history.start_strategy()

        # Prepare arguments for rendering cuda code. The values should all be string.
        cuda_args = {
            "mode": "train",
            "world_size": str(utils.WORLD_SIZE),
            "local_rank": str(utils.LOCAL_RANK),
            "log_folder": args.log_folder,
            "log_interval": str(args.log_interval),
            "iteration": str(iteration),
            "zhx_debug": str(args.zhx_debug),
            "zhx_time": str(args.zhx_time),
            "dist_global_strategy": strategy.get_gloabl_strategy_str(),
            "avoid_pixel_all2all": strategy.is_avoid_pixel_all2all(),
            "stats_collector": {},
        }


        memory_iteration_begin = torch.cuda.memory_allocated() / 1024 / 1024 / 1024

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        timers.stop("pre_forward")

        # NOTE: this is to make sure: we are measuring time for local work.
        # where to add this barrier depends on: whether there will be global communication(i.e. allreduce) in the following code.
        if utils.check_enable_python_timer() and utils.WORLD_SIZE > 1:
            torch.distributed.barrier()

        # Forward
        timers.start("forward")
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg,
                            cuda_args=cuda_args,
                            timers=timers,
                            strategy=strategy)
        timers.stop("forward")
        (image, 
         viewspace_point_tensor, 
         visibility_filter, 
         radii, 
         n_render,
         n_consider,
         n_contrib) = (
            render_pkg["render"],
            render_pkg["viewspace_points"], 
            render_pkg["visibility_filter"],
            render_pkg["radii"], 
            render_pkg["n_render"], 
            render_pkg["n_consider"], 
            render_pkg["n_contrib"]
        )
        if args.memory_distribution:
            i2j_send_size = render_pkg["i2j_send_size"]
            compute_locally = render_pkg["compute_locally"]
            local2j_ids_bool = render_pkg["local2j_ids_bool"]
        else:
            i2j_send_size = None
            compute_locally = None
            local2j_ids_bool = None

        memory_after_forward = torch.cuda.memory_allocated() / 1024 / 1024 / 1024

        if utils.check_enable_python_timer() and utils.WORLD_SIZE > 1:
            torch.distributed.barrier()


        # Loss Computation
        if args.loss_distribution:
            # Distributed Loss Computation
            Ll1, ssim_loss = distributed_loss_computation(image, viewpoint_cam, compute_locally, strategy, cuda_args)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_loss)
        else:
            # Replicated Loss Computation
            Ll1, ssim_loss = replicated_loss_computation(image, viewpoint_cam)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_loss)
        utils.check_memory_usage_logging("after loss")
        memory_after_loss = torch.cuda.memory_allocated() / 1024 / 1024 / 1024

        # Logging
        log_string = "iteration {} image: {} loss: {}\n".format(iteration, viewpoint_cam.image_name, loss.item())
        if args.log_iteration_memory_usage:
            log_string += "memory_iteration_begin: {:.4f} GB. memory_after_forward: {:.4f} GB. memory_after_loss: {:.4f} GB.\n".format(
                memory_iteration_begin, memory_after_forward, memory_after_loss
            )

        epoch_progress_cnt = epoch_progress_cnt + 1
        epoch_loss = epoch_loss + loss.item()
        if epoch_progress_cnt == epoch_camera_size:
            assert args.fixed_training_image or viewpoint_stack == None or len(viewpoint_stack) == 0, \
                "viewpoint_stack should be empty at the end of epoch."
            log_file.write("epoch {} loss: {}\n".format(epoch_id, epoch_loss/epoch_progress_cnt))
            epoch_id = epoch_id + 1
            epoch_progress_cnt = 0
            epoch_loss = 0


        if utils.check_enable_python_timer() and utils.WORLD_SIZE > 1:
            torch.distributed.barrier()


        # Backward
        timers.start("backward")
        loss.backward()
        timers.stop("backward")
        utils.check_memory_usage_logging("after backward")

        if args.analyze_3dgs_change:
            # save some statistics in log_file for analyzing 3dgs change.
            with torch.no_grad():
                local_n_3dgs = visibility_filter.shape[0]
                local_visible_3dgs = visibility_filter.sum().item()
                local_sum_3dgs_radii = radii.sum().item() # only visibility_filter position are non-zero.
                local_sum_3dgs_grad_norm = torch.norm(viewspace_point_tensor.grad[:, :2], dim=-1).sum().item() # only visibility_filter is non-zero.
                log_string += "local_n_3dgs: {}; local_visible_3dgs: {}; local_sum_3dgs_radii: {}; local_sum_3dgs_grad_norm: {};\n".format(
                    local_n_3dgs, local_visible_3dgs, local_sum_3dgs_radii, local_sum_3dgs_grad_norm
                )

        if utils.check_enable_python_timer() and utils.WORLD_SIZE > 1:
            torch.distributed.barrier()


        # Adjust workload division strategy. 
        timers.start("strategy.update_stats")
        if iteration > args.adjust_strategy_warmp_iterations:
            strategy.update_stats(cuda_args["stats_collector"],
                                    n_render,
                                    n_consider,
                                    n_contrib,
                                    i2j_send_size)
            strategy_history.finish_strategy()
        timers.stop("strategy.update_stats")


        if utils.check_enable_python_timer() and utils.WORLD_SIZE > 1:
            torch.distributed.barrier()


        # Sync Gradients. NOTE: do not sync grad in args.memory_distribution mode
        if not args.memory_distribution:
            timers.start("sync_gradients")
            sparse_ids_mask = gaussians.sync_gradients(viewspace_point_tensor, utils.WORLD_SIZE)
            non_zero_indices_cnt = sparse_ids_mask.sum().item()
            total_indices_cnt = sparse_ids_mask.shape[0]
            log_file.write("iteration {} non_zero_indices_cnt: {} total_indices_cnt: {} ratio: {}\n".format(iteration, non_zero_indices_cnt, total_indices_cnt, non_zero_indices_cnt/total_indices_cnt))
            timers.stop("sync_gradients")


        if utils.check_enable_python_timer() and utils.WORLD_SIZE > 1:
            torch.distributed.barrier()


        iter_end.record()


        log_file.write(log_string)

        with torch.no_grad():
            # Update Statistics for redistribution
            if args.memory_distribution:
                gaussians.send_to_gpui_cnt += local2j_ids_bool

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if utils.LOCAL_RANK == 0:
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), log_file)
            if iteration in saving_iterations: # Do not check rk here. Because internal implementation maybe distributed save.
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                log_file.write("[ITER {}] Saving Gaussians\n".format(iteration))
                scene.save(iteration)

            # Densification
            if not disable_auto_densification and iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                timers.start("densification")

                timers.start("densification_update_stats")
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                timers.stop("densification_update_stats")

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    assert args.stop_update_param == False, "stop_update_param must be false for densification; because it is a flag for debugging."

                    timers.start("densify_and_prune")
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    timers.stop("densify_and_prune")

                    # redistribute after densify_and_prune, because we have new gaussians to distribute evenly.
                    if args.redistribute_gaussians_mode != "no_redistribute" and ( utils.get_denfify_iter() % args.redistribute_gaussians_frequency == 0 ):
                        num_3dgs_before_redistribute = gaussians.get_xyz.shape[0]
                        timers.start("redistribute_gaussians")
                        gaussians.redistribute_gaussians()
                        timers.stop("redistribute_gaussians")
                        num_3dgs_after_redistribute = gaussians.get_xyz.shape[0]

                        log_file.write("iteration {} redistribute. Now num of 3dgs before redistribute: {}. Now num of 3dgs after redistribute: {}. \n".format(
                            iteration, num_3dgs_before_redistribute, num_3dgs_after_redistribute))

                    memory_usage = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
                    max_memory_usage = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
                    log_file.write("iteration {} densify_and_prune. Now num of 3dgs: {}. Now Memory usage: {} GB. Max Memory usage: {} GB. \n".format(
                        iteration, gaussians.get_xyz.shape[0], memory_usage, max_memory_usage))

                    utils.inc_densify_iter()
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    timers.start("reset_opacity")
                    gaussians.reset_opacity()
                    timers.stop("reset_opacity")

                timers.stop("densification")

            # Optimizer step
            if iteration < opt.iterations and iteration % args.bsz == 0:
                timers.start("optimizer_step")
                if not args.stop_update_param:
                    gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                timers.stop("optimizer_step")
                utils.check_memory_usage_logging("after optimizer step")

            if utils.LOCAL_RANK == 0 and (iteration in checkpoint_iterations): #TODO: have not handled args.memory_distribution yet.
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                log_file.write("[ITER {}] Saving Checkpoint\n".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
    
        # Finish a iteration and clean up
        if utils.check_enable_python_timer():
            timers.printTimers(iteration)

        log_file.flush()

    # Finish training
    if args.end2end_time:
        torch.cuda.synchronize()
        log_file.write("end2end total_time: {:.6f} ms, iterations: {}, throughput {:.2f} it/s\n".format(time.time() - train_start_time, opt.iterations, opt.iterations/(time.time() - train_start_time)))
    
    log_file.write("Max Memory usage: {} GB.\n".format(torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024))

    if not args.performance_stats:
        # Save some statistics to file for future usage. 
        data_json = {}
        for camera_id, strategy_history in cameraId2StrategyHistory.items():
            data_json[camera_id] = strategy_history.to_json()
        
        with open(args.log_folder+"/strategy_history_ws="+str(utils.WORLD_SIZE)+"_rk="+str(utils.LOCAL_RANK)+".json", 'w') as f:
            json.dump(data_json, f)


        # DEBUG
        if args.memory_distribution and args.save_send_to_gpui_cnt:
            # save gaussians.send_to_gpui_cnt to file.
            with open(args.log_folder+"/send_to_gpui_cnt_ws="+str(utils.WORLD_SIZE)+"_rk="+str(utils.LOCAL_RANK)+".json", 'w') as f:
                send_to_gpui_cnt_cpu = gaussians.send_to_gpui_cnt.cpu().numpy().tolist()
                data2save = []
                for i in range(len(send_to_gpui_cnt_cpu)):
                    data2save.append( ",".join([str(x) for x in send_to_gpui_cnt_cpu[i]]) )
                json.dump(data2save, f, indent=4)


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    if utils.LOCAL_RANK != 0:
        return None
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, log_file):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        # prepare arguments for render cuda code.
        cuda_args = {
            "mode": "test",
            "world_size": str(utils.WORLD_SIZE),
            "local_rank": str(utils.LOCAL_RANK),
            "log_folder": args.log_folder,
            "log_interval": str(args.log_interval),
            "iteration": str(iteration),# this is training iteration, not testing iteration.
            "zhx_debug": str(args.zhx_debug),
            "zhx_time": str(args.zhx_time),
            "avoid_pixel_all2all": False, # during testing, we use image allreduce, thus different GPUs should compute different parts of the image.
            "stats_collector": {}
        }
        renderKwargs = {"scaling_modifier": 1.0, "override_color": None, "cuda_args": cuda_args}

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    # TODO: refactor code here. 
                    cuda_args["dist_global_strategy"] = get_evenly_global_strategy_str(viewpoint)# HACK: Use naive distribution strategy during testing.
                    hack_history = create_division_strategy_history(viewpoint, "evaluation")
                    renderKwargs["strategy"] = hack_history.start_strategy()# HACK: Use naive distribution strategy during testing.
                    image = renderFunc(viewpoint, scene.gaussians, *renderArgs, **renderKwargs)["render"]
                    if utils.WORLD_SIZE > 1:
                        torch.distributed.all_reduce(image, op=dist.ReduceOp.SUM)
                    image = torch.clamp(image, 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] rank {} Evaluating {}: L1 {} PSNR {}".format(iteration, utils.LOCAL_RANK, config['name'], l1_test, psnr_test))
                log_file.write("[ITER {}] Evaluating {}: L1 {} PSNR {}\n".format(iteration, config['name'], l1_test, psnr_test))

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    dist_p = DistributionParams(parser)
    bench_p = BenchmarkParams(parser)
    debug_p = DebugParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None) # NOTE: restart from a checkpoint will give a different loss because of training image orders are different.
    parser.add_argument("--log_folder", type=str, default = "logs")
    parser.add_argument("--log_interval", type=int, default=50)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # Set up distributed training
    init_distributed()
    print("Local rank: " + str(utils.LOCAL_RANK) + " World size: " + str(utils.WORLD_SIZE))

    ## Prepare arguments.
    # Check arguments
    check_args(args)
    # Set up global args
    utils.set_args(args)


    # create log folder
    if utils.LOCAL_RANK == 0:
        os.makedirs(args.log_folder, exist_ok = True)
        os.makedirs(args.model_path, exist_ok = True)
    if utils.WORLD_SIZE > 1:
        torch.distributed.barrier()# make sure log_folder is created before other ranks start writing log.

    # Initialize system state (RNG)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # Initialize log file and print all args
    log_file = open(args.log_folder+"/python_ws="+str(utils.WORLD_SIZE)+"_rk="+str(utils.LOCAL_RANK)+".log", 'w')
    print_all_args(args, log_file)

    # Make sure block size match between python and cuda code. TODO: modify block size from python code without slow down training.
    cuda_block_x, cuda_block_y, one_dim_block_size = diff_gaussian_rasterization._C.get_block_XY()
    utils.set_block_size(cuda_block_x, cuda_block_y, one_dim_block_size)
    log_file.write("cuda_block_x: {}; cuda_block_y: {}; one_dim_block_size: {};\n".format(cuda_block_x, cuda_block_y, one_dim_block_size))

    training(lp.extract(args), op.extract(args), pp.extract(args), args, log_file)

    # All done
    print("\nTraining complete.")