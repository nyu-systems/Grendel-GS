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
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
# try:
#     from torch.utils.tensorboard import SummaryWriter
#     TENSORBOARD_FOUND = True
# except ImportError:
#     TENSORBOARD_FOUND = False

import wandb

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    assert pipe.batch_grad_stats, "batch_grad_stats must be True for analyse"

    for bsz in [2**i for i in range(7)]:
        trajectory_pkg = {}
        batch_gradient_pkg = {}
        grad_prev = {}
        batch_loss = []
        weights_pkg = {}
        exp_avg_pkg = {}
        exp_avg_sq_pkg = {}
        
        print(f"Training with batch size {bsz}")
        opt.bsz = bsz
        first_iter = 0
        tb_writer = prepare_output_and_logger(dataset)
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians)
        gaussians.training_setup(opt)
        if checkpoint:
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        iter_start = torch.cuda.Event(enable_timing = True)
        iter_end = torch.cuda.Event(enable_timing = True)

        batched_cameras_uid = []
        ema_loss_for_log = 0.0
        progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
        for iteration in range(first_iter, opt.iterations):        
            if network_gui.conn == None:
                network_gui.try_connect()
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                    if custom_cam != None:
                        net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    network_gui.send(net_image_bytes, dataset.source_path)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    network_gui.conn = None

            iter_start.record()

            gaussians.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration > first_iter and iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            # Pick a random Camera
            viewpoint_cam = scene.get_one_camera(batched_cameras_uid)
            batched_cameras_uid.append(viewpoint_cam.uid)

            # Render
            if iteration == debug_from:
                pipe.debug = True

            bg = torch.rand((3), device="cuda") if opt.random_background else background

            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()

            iter_end.record()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == opt.iterations - 1:
                    progress_bar.close()

                # Log and save
                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)

                # Densification
                if not pipe.disable_densification and iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration >= opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration >= opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    
                    if iteration > first_iter and iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()
                    
                # record gradient for each sample in the mini batch
                for param_group in gaussians.optimizer.param_groups:
                    name = param_group["name"]
                    if name == "f_rest":
                        continue
                    grad = param_group["params"][0].grad.clone().detach()
                    if name not in grad_prev:
                        grad_prev[name] = grad
                        batch_gradient_pkg[name] = []
                    else:
                        grad_prev[name], grad = grad, grad - grad_prev[name]
                    batch_gradient_pkg[name].append(grad)
                # if bsz != 1:
                #     breakpoint()
                batch_loss.append(loss.item())

                # Optimizer step
                if len(batched_cameras_uid) == opt.bsz:
                    batched_cameras_uid = []
                    # divide grad by bsz
                    for param in gaussians.all_parameters():
                        if param.grad is not None:
                            param.grad.div_(opt.bsz)
                        
                    # record weight before optimizer step
                    for param_group in gaussians.optimizer.param_groups:
                        name = param_group["name"]
                        if name == "f_rest":
                            continue
                        weights_pkg[name] = param_group["params"][0].clone().detach()

                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                    
                    # record optimizer states after optimizer step
                    for param_group in gaussians.optimizer.param_groups:
                        name = param_group["name"]
                        if name == "f_rest":
                            continue
                        stored_state = gaussians.optimizer.state.get(param_group["params"][0], None)
                        if stored_state is not None:
                            exp_avg_pkg[name] = stored_state["exp_avg"].clone().detach()
                            exp_avg_sq_pkg[name] = stored_state["exp_avg_sq"].clone().detach()
                    
                    # save stats of this step to trajectory_pkg
                    save_dict = {}
                    for name in batch_gradient_pkg:
                        save_dict[name] = (
                            torch.stack(batch_gradient_pkg[name]).cpu(),
                            weights_pkg[name].cpu(),
                            exp_avg_pkg[name].cpu(),
                            exp_avg_sq_pkg[name].cpu(),
                            torch.tensor(batch_loss).cpu()
                        )
                    trajectory_pkg[iteration+1-opt.bsz] = save_dict

                    # clear batch_gradient_pkg
                    batch_gradient_pkg = {}
                    grad_prev = {}
                    batch_loss = []

                if (iteration in checkpoint_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
        if bsz == 1:
            base_trajectory_pkg = trajectory_pkg
        else:
            def suffix_dict_keys(suffix, d):
                return {k + suffix: v for k, v in d.items()}
            # compare the following metrics with base_trajectory_pkg. take bsz = 4 as example.
            # 1. gradient: the batched_gradient [st, st+1, st+2, st+3] is saved at iteration st, compare with the batched_gradient of bsz = 1 at iteration st, st+1, st+2, st+3. For every iteration, compare the gradient norm ratio and the cosine similarity.
            # 2. weight difference. the weight at iteration st is saved at iteration st, compare weight[st+4] - weight[st] that of bsz = 1 at iteration st. For every 4 iteration, compare the weight difference.
            # 3. optimizer states. the optimizer states at iteration st is saved at iteration st, compare optimizer states[st+4] that of bsz = 1 at iteration st. For every 4 iteration, compare the optimizer states.
            # 4. loss. the loss at iteration [st, st+1, st+2, st+3] is saved at iteration st, compare with the loss of bsz = 1 at iteration st, st+1, st+2, st+3. For every iteration, compare the loss.
            wandb.init(project="3dgs-analyse", config=vars(args), group="analyse")
            wandb.run.config.update({"bsz": bsz, "start_iter": first_iter}, allow_val_change=True)

            initial_logdict = {
                "weight_diff_ratio": 1.0,
                "weight_diff_cosine": 1.0,
                "exp_avg_ratio": 1.0,
                "exp_avg_cosine": 1.0,
                "exp_avg_sq_ratio": 1.0,
                "exp_avg_sq_cosine": 1.0,
                "gradient_norm_ratio": 1.0,
                "gradient_cosine": 1.0,
            }
            for name in save_dict:
                wandb.log(suffix_dict_keys("/" + name, initial_logdict), step=list(trajectory_pkg.keys())[0])
                
            for st in trajectory_pkg.keys():
                if st + opt.bsz not in trajectory_pkg or st + opt.bsz not in base_trajectory_pkg:
                    break
                # gradient
                for idx, iteration in enumerate(range(st, st + opt.bsz)):
                    gradient_log_dicts = {}
                    for name in save_dict:
                        try:
                            grad = trajectory_pkg[st][name][0][idx].cuda()
                        except IndexError:
                            breakpoint()
                        grad_base = base_trajectory_pkg[st+idx][name][0][0].cuda()
                        grad_norm_ratio = (torch.norm(grad) / torch.norm(grad_base)).item()
                        grad_cosine = torch.nn.functional.cosine_similarity(grad.view(1, -1), grad_base.view(1, -1)).item()

                        log_dict = {
                            "gradient_norm_ratio": grad_norm_ratio,
                            "gradient_cosine": grad_cosine,
                        }
                        gradient_log_dicts.update(suffix_dict_keys("/" + name, log_dict))
                    wandb.log(gradient_log_dicts, step=iteration+1)

                # weight difference and optimizer states
                log_dicts = {}
                for name in save_dict:
                    weight_diff = trajectory_pkg[st+opt.bsz][name][1].cuda() - trajectory_pkg[st][name][1].cuda()
                    weight_diff_base = base_trajectory_pkg[st+opt.bsz][name][1].cuda() - base_trajectory_pkg[st][name][1].cuda()
                    weight_diff_ratio = (torch.norm(weight_diff) / torch.norm(weight_diff_base)).item()
                    weight_diff_cosine = torch.nn.functional.cosine_similarity(weight_diff, weight_diff_base).mean().item()

                    exp_avg = trajectory_pkg[st][name][2].cuda()
                    exp_avg_base = base_trajectory_pkg[st][name][2].cuda()
                    exp_avg_ratio = (torch.norm(exp_avg) / torch.norm(exp_avg_base)).item()
                    exp_avg_cosine = torch.nn.functional.cosine_similarity(exp_avg, exp_avg_base).mean().item()
                    
                    exp_avg_sq = trajectory_pkg[st][name][3].cuda()
                    exp_avg_sq_base = base_trajectory_pkg[st][name][3].cuda()
                    exp_avg_sq_ratio = (torch.norm(exp_avg_sq) / torch.norm(exp_avg_sq_base)).item()
                    exp_avg_sq_cosine = torch.nn.functional.cosine_similarity(exp_avg_sq, exp_avg_sq_base).mean().item()

                    log_dict = {
                        "weight_diff_ratio": weight_diff_ratio,
                        "weight_diff_cosine": weight_diff_cosine,
                        "exp_avg_ratio": exp_avg_ratio,
                        "exp_avg_cosine": exp_avg_cosine,
                        "exp_avg_sq_ratio": exp_avg_sq_ratio,
                        "exp_avg_sq_cosine": exp_avg_sq_cosine,
                    }
                    log_dicts.update(suffix_dict_keys("/" + name, log_dict))
                wandb.log(log_dicts, step=st + opt.bsz)
            wandb.finish()


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # # Create Tensorboard writer
    # tb_writer = None
    # if TENSORBOARD_FOUND:
    #     tb_writer = SummaryWriter(args.model_path)
    # else:
    #     print("Tensorboard not available: not logging progress")
    # return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
    # wandb.log({"train/l1_loss": Ll1.item(), "train/total_loss": loss.item(), "iter_time": elapsed}, step=iteration)
    # wandb.log({"scene/total_points": scene.gaussians.get_xyz.shape[0]}, step=iteration)
    # wandb.log({"scene/opacity_histogram": scene.gaussians.get_opacity}, step=iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                # wandb.log({'test/' + config['name'] + ' - l1_loss': l1_test, 'test/' + config['name'] + ' - psnr': psnr_test}, step=iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
