import os
import torch
import json
from utils.loss_utils import l1_loss
from gaussian_renderer import (
    distributed_preprocess3dgs_and_all2all_final,
    render_final,
    gsplat_distributed_preprocess3dgs_and_all2all_final,
    gsplat_render_final,
)
from torch.cuda import nvtx
from scene import Scene, GaussianModel, SceneDataset
from gaussian_renderer.workload_division import (
    start_strategy_final,
    finish_strategy_final,
    DivisionStrategyHistoryFinal,
)
from gaussian_renderer.loss_distribution import (
    load_camera_from_cpu_to_all_gpu,
    load_camera_from_cpu_to_all_gpu_for_eval,
    batched_loss_computation,
)
from utils.general_utils import prepare_output_and_logger, globally_sync_for_timer
import utils.general_utils as utils
from utils.timer import Timer, End2endTimer
from tqdm import tqdm
from utils.image_utils import psnr
import torch.distributed as dist
from densification import densification, gsplat_densification


def training(dataset_args, opt_args, pipe_args, args, log_file):

    # Init auxiliary tools

    timers = Timer(args)
    utils.set_timers(timers)
    prepare_output_and_logger(dataset_args)
    utils.log_cpu_memory_usage("at the beginning of training")
    start_from_this_iteration = 1

    # Init parameterized scene
    gaussians = GaussianModel(dataset_args.sh_degree)

    with torch.no_grad():
        scene = Scene(args, gaussians)
        gaussians.training_setup(opt_args)

        if args.start_checkpoint != "":
            model_params, start_from_this_iteration = utils.load_checkpoint(args)
            gaussians.restore(model_params, opt_args)
            utils.print_rank_0(
                "Restored from checkpoint: {}".format(args.start_checkpoint)
            )
            log_file.write(
                "Restored from checkpoint: {}\n".format(args.start_checkpoint)
            )

        scene.log_scene_info_to_file(log_file, "Scene Info Before Training")
    utils.check_initial_gpu_memory_usage("after init and before training loop")

    # Init dataset
    train_dataset = SceneDataset(scene.getTrainCameras())
    if args.adjust_strategy_warmp_iterations == -1:
        args.adjust_strategy_warmp_iterations = len(train_dataset.cameras)
        # use one epoch to warm up. do not use the first epoch's running time for adjustment of strategy.

    # Init distribution strategy history
    strategy_history = DivisionStrategyHistoryFinal(
        train_dataset, utils.DEFAULT_GROUP.size(), utils.DEFAULT_GROUP.rank()
    )

    # Init background
    background = None
    if args.backend == "gsplat":
        bg_color = [1, 1, 1] if dataset_args.white_background else None
    else:
        bg_color = [1, 1, 1] if dataset_args.white_background else [0, 0, 0]

    if bg_color is not None:
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Training Loop
    end2end_timers = End2endTimer(args)
    end2end_timers.start()
    progress_bar = tqdm(
        range(1, opt_args.iterations + 1),
        desc="Training progress",
        disable=(utils.LOCAL_RANK != 0),
    )
    progress_bar.update(start_from_this_iteration - 1)
    num_trained_batches = 0

    ema_loss_for_log = 0
    for iteration in range(
        start_from_this_iteration, opt_args.iterations + 1, args.bsz
    ):
        # Step Initialization
        if iteration // args.bsz % 30 == 0:
            progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
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
        if args.local_sampling:
            assert (
                args.bsz % utils.WORLD_SIZE == 0
            ), "Batch size should be divisible by the number of GPUs."
            batched_cameras_idx = train_dataset.get_batched_cameras_idx(
                args.bsz // utils.WORLD_SIZE
            )
            batched_all_cameras_idx = torch.zeros(
                (utils.WORLD_SIZE, len(batched_cameras_idx)), device="cuda", dtype=int
            )
            batched_cameras_idx = torch.tensor(
                batched_cameras_idx, device="cuda", dtype=int
            )
            torch.distributed.all_gather_into_tensor(
                batched_all_cameras_idx, batched_cameras_idx, group=utils.DEFAULT_GROUP
            )
            batched_all_cameras_idx = batched_all_cameras_idx.cpu().numpy().squeeze()
            batched_cameras = train_dataset.get_batched_cameras_from_idx(
                batched_all_cameras_idx
            )
        else:
            batched_cameras = train_dataset.get_batched_cameras(args.bsz)

        with torch.no_grad():
            # Prepare Workload division strategy
            timers.start("prepare_strategies")
            batched_strategies, gpuid2tasks = start_strategy_final(
                batched_cameras, strategy_history
            )
            timers.stop("prepare_strategies")

            # Load ground-truth images to GPU
            timers.start("load_cameras")
            load_camera_from_cpu_to_all_gpu(
                batched_cameras, batched_strategies, gpuid2tasks
            )
            timers.stop("load_cameras")

        if args.backend == "gsplat":
            batched_screenspace_pkg = (
                gsplat_distributed_preprocess3dgs_and_all2all_final(
                    batched_cameras,
                    gaussians,
                    pipe_args,
                    background,
                    batched_strategies=batched_strategies,
                    mode="train",
                )
            )
            batched_image, batched_compute_locally = gsplat_render_final(
                batched_screenspace_pkg, batched_strategies
            )
            batch_statistic_collector = [
                cuda_args["stats_collector"]
                for cuda_args in batched_screenspace_pkg["batched_cuda_args"]
            ]
        else:
            batched_screenspace_pkg = distributed_preprocess3dgs_and_all2all_final(
                batched_cameras,
                gaussians,
                pipe_args,
                background,
                batched_strategies=batched_strategies,
                mode="train",
            )
            batched_image, batched_compute_locally = render_final(
                batched_screenspace_pkg, batched_strategies
            )
            batch_statistic_collector = [
                cuda_args["stats_collector"]
                for cuda_args in batched_screenspace_pkg["batched_cuda_args"]
            ]

        loss_sum, batched_losses = batched_loss_computation(
            batched_image,
            batched_cameras,
            batched_compute_locally,
            batched_strategies,
            batch_statistic_collector,
        )

        timers.start("backward")
        loss_sum.backward()
        timers.stop("backward")
        utils.check_initial_gpu_memory_usage("after backward")

        with torch.no_grad():
            # Adjust workload division strategy.
            globally_sync_for_timer()
            timers.start("finish_strategy_final")
            finish_strategy_final(
                batched_cameras,
                strategy_history,
                batched_strategies,
                batch_statistic_collector,
            )
            timers.stop("finish_strategy_final")

            # Sync losses in the batch
            timers.start("sync_loss_and_log")
            batched_losses = torch.tensor(batched_losses, device="cuda")
            if utils.DEFAULT_GROUP.size() > 1:
                dist.all_reduce(
                    batched_losses, op=dist.ReduceOp.SUM, group=utils.DEFAULT_GROUP
                )
            batched_loss = (1.0 - args.lambda_dssim) * batched_losses[
                :, 0
            ] + args.lambda_dssim * (1.0 - batched_losses[:, 1])
            batched_loss_cpu = batched_loss.cpu().numpy()
            ema_loss_for_log = (
                batched_loss_cpu.mean()
                if ema_loss_for_log is None
                else 0.6 * ema_loss_for_log + 0.4 * batched_loss_cpu.mean()
            )
            # Update Epoch Statistics
            train_dataset.update_losses(batched_loss_cpu)
            # Logging
            batched_loss_cpu = [round(loss, 6) for loss in batched_loss_cpu]
            log_string = "iteration[{},{}) loss: {} image: {}\n".format(
                iteration,
                iteration + args.bsz,
                batched_loss_cpu,
                [viewpoint_cam.image_name for viewpoint_cam in batched_cameras],
            )
            log_file.write(log_string)
            timers.stop("sync_loss_and_log")

            # Evaluation
            end2end_timers.stop()
            training_report(
                iteration,
                l1_loss,
                args.test_iterations,
                scene,
                pipe_args,
                background,
                args.backend,
            )
            end2end_timers.start()

            # Densification
            if args.backend == "gsplat":
                gsplat_densification(
                    iteration, scene, gaussians, batched_screenspace_pkg
                )
            else:
                densification(iteration, scene, gaussians, batched_screenspace_pkg)

            # Save Gaussians
            if any(
                [
                    iteration <= save_iteration < iteration + args.bsz
                    for save_iteration in args.save_iterations
                ]
            ):
                end2end_timers.stop()
                end2end_timers.print_time(log_file, iteration + args.bsz)
                utils.print_rank_0("\n[ITER {}] Saving Gaussians".format(iteration))
                log_file.write("[ITER {}] Saving Gaussians\n".format(iteration))
                scene.save(iteration)

                if args.save_strategy_history:
                    with open(
                        args.log_folder
                        + "/strategy_history_ws="
                        + str(utils.WORLD_SIZE)
                        + "_rk="
                        + str(utils.GLOBAL_RANK)
                        + ".json",
                        "w",
                    ) as f:
                        json.dump(strategy_history.to_json(), f)
                end2end_timers.start()

            # Save Checkpoints
            if any(
                [
                    iteration <= checkpoint_iteration < iteration + args.bsz
                    for checkpoint_iteration in args.checkpoint_iterations
                ]
            ):
                end2end_timers.stop()
                utils.print_rank_0("\n[ITER {}] Saving Checkpoint".format(iteration))
                log_file.write("[ITER {}] Saving Checkpoint\n".format(iteration))
                save_folder = scene.model_path + "/checkpoints/" + str(iteration) + "/"
                if utils.DEFAULT_GROUP.rank() == 0:
                    os.makedirs(save_folder, exist_ok=True)
                    if utils.DEFAULT_GROUP.size() > 1:
                        torch.distributed.barrier(group=utils.DEFAULT_GROUP)
                elif utils.DEFAULT_GROUP.size() > 1:
                    torch.distributed.barrier(group=utils.DEFAULT_GROUP)
                torch.save(
                    (gaussians.capture(), iteration + args.bsz),
                    save_folder
                    + "/chkpnt_ws="
                    + str(utils.WORLD_SIZE)
                    + "_rk="
                    + str(utils.GLOBAL_RANK)
                    + ".pth",
                )
                end2end_timers.start()

            # Optimizer step
            if iteration < opt_args.iterations:
                timers.start("optimizer_step")

                if (
                    args.lr_scale_mode != "accumu"
                ):  # we scale the learning rate rather than accumulate the gradients.
                    for param in gaussians.all_parameters():
                        if param.grad is not None:
                            param.grad /= args.bsz

                if not args.stop_update_param:
                    gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                timers.stop("optimizer_step")
                utils.check_initial_gpu_memory_usage("after optimizer step")

        # Finish a iteration and clean up
        torch.cuda.synchronize()
        for (
            viewpoint_cam
        ) in batched_cameras:  # Release memory of locally rendered original_image
            viewpoint_cam.original_image = None
        if args.nsys_profile:
            nvtx.range_pop()
        if utils.check_enable_python_timer():
            timers.printTimers(iteration, mode="sum")
        log_file.flush()

    # Finish training
    if opt_args.iterations not in args.save_iterations:
        end2end_timers.print_time(log_file, opt_args.iterations)
    log_file.write(
        "Max Memory usage: {} GB.\n".format(
            torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
        )
    )
    progress_bar.close()


def training_report(
    iteration, l1_loss, testing_iterations, scene: Scene, pipe_args, background, backend
):
    args = utils.get_args()
    log_file = utils.get_log_file()
    # Report test and samples of training set
    while len(testing_iterations) > 0 and iteration > testing_iterations[0]:
        testing_iterations.pop(0)
    if len(testing_iterations) > 0 and utils.check_update_at_this_iter(
        iteration, utils.get_args().bsz, testing_iterations[0], 0
    ):
        testing_iterations.pop(0)
        utils.print_rank_0("\n[ITER {}] Start Testing".format(iteration))

        if args.local_sampling:
            validation_configs = ({"name": "test", "cameras": scene.getTestCameras()},)
        else:
            validation_configs = (
                {"name": "test", "cameras": scene.getTestCameras()},
                {
                    "name": "train",
                    "cameras": [
                        scene.getTrainCameras()[
                            idx * args.llffhold % len(scene.getTrainCameras())
                        ]
                        for idx in range(len(scene.getTrainCameras()) // args.llffhold)
                    ],
                },
            )
        # init workload division strategy
        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = torch.scalar_tensor(0.0, device="cuda")
                psnr_test = torch.scalar_tensor(0.0, device="cuda")

                num_cameras = len(config["cameras"])
                # TODO: if not divisible by world size
                num_cameras = num_cameras // args.bsz * args.bsz
                eval_dataset = SceneDataset(config["cameras"])
                strategy_history = DivisionStrategyHistoryFinal(
                    eval_dataset, utils.DEFAULT_GROUP.size(), utils.DEFAULT_GROUP.rank()
                )
                for idx in range(1, num_cameras + 1, args.bsz):
                    num_camera_to_load = min(args.bsz, num_cameras - idx + 1)
                    if args.local_sampling:
                        # TODO: if not divisible by world size
                        batched_cameras_idx = eval_dataset.get_batched_cameras_idx(
                            args.bsz // utils.WORLD_SIZE
                        )
                        batched_all_cameras_idx = torch.zeros(
                            (utils.WORLD_SIZE, len(batched_cameras_idx)),
                            device="cuda",
                            dtype=int,
                        )
                        batched_cameras_idx = torch.tensor(
                            batched_cameras_idx, device="cuda", dtype=int
                        )
                        torch.distributed.all_gather_into_tensor(
                            batched_all_cameras_idx,
                            batched_cameras_idx,
                            group=utils.DEFAULT_GROUP,
                        )
                        batched_all_cameras_idx = (
                            batched_all_cameras_idx.cpu().numpy().squeeze()
                        )
                        batched_cameras = eval_dataset.get_batched_cameras_from_idx(
                            batched_all_cameras_idx
                        )
                    else:
                        batched_cameras = eval_dataset.get_batched_cameras(
                            num_camera_to_load
                        )
                    batched_strategies, gpuid2tasks = start_strategy_final(
                        batched_cameras, strategy_history
                    )
                    load_camera_from_cpu_to_all_gpu_for_eval(
                        batched_cameras, batched_strategies, gpuid2tasks
                    )
                    if backend == 'gsplat':
                        batched_screenspace_pkg = gsplat_distributed_preprocess3dgs_and_all2all_final(batched_cameras, scene.gaussians, pipe_args, background,
                                                                                           batched_strategies=batched_strategies,
                                                                                           mode="test")
                        batched_image, _ = gsplat_render_final(batched_screenspace_pkg, batched_strategies)
                    else:    
                        batched_screenspace_pkg = distributed_preprocess3dgs_and_all2all_final(batched_cameras, scene.gaussians, pipe_args, background,
                                                                                            batched_strategies=batched_strategies,
                                                                                            mode="test")
                        batched_image, _ = render_final(batched_screenspace_pkg, batched_strategies)
                    for camera_id, (image, gt_camera) in enumerate(
                        zip(batched_image, batched_cameras)
                    ):
                        if (
                            image is None or len(image.shape) == 0
                        ):  # The image is not rendered locally.
                            image = torch.zeros(
                                gt_camera.original_image.shape,
                                device="cuda",
                                dtype=torch.float32,
                            )

                        if utils.DEFAULT_GROUP.size() > 1:
                            torch.distributed.all_reduce(
                                image, op=dist.ReduceOp.SUM, group=utils.DEFAULT_GROUP
                            )

                        image = torch.clamp(image, 0.0, 1.0)
                        gt_image = torch.clamp(
                            gt_camera.original_image / 255.0, 0.0, 1.0
                        )

                        if idx + camera_id < num_cameras + 1:
                            l1_test += l1_loss(image, gt_image).mean().double()
                            psnr_test += psnr(image, gt_image).mean().double()
                        gt_camera.original_image = None
                psnr_test /= num_cameras
                l1_test /= num_cameras
                utils.print_rank_0(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                        iteration, config["name"], l1_test, psnr_test
                    )
                )
                log_file.write(
                    "[ITER {}] Evaluating {}: L1 {} PSNR {}\n".format(
                        iteration, config["name"], l1_test, psnr_test
                    )
                )

        torch.cuda.empty_cache()
