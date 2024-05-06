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
import random
import json
from random import randint
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import utils.general_utils as utils
import torch

class Scene:

    gaussians : GaussianModel

    def __init__(self, args, gaussians : GaussianModel, load_iteration=None, shuffle=True):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        utils.log_cpu_memory_usage("before loading images meta data")

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.llffhold)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train_my.json")):
        #     print("Found transforms_train.json file, assuming Blender data set!")
        #     scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        # else:
            scene_info = sceneLoadTypeCallbacks["City"](args.source_path,
                                                        args.random_background,
                                                        args.white_background,
                                                        args.eval,
                                                        ds=1,
                                                        llffhold=args.llffhold)
        else:
            raise ValueError("No valid dataset found in the source path")

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        utils.log_cpu_memory_usage("before decoding images")

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        log_file = utils.get_log_file()
        for resolution_scale in [args.train_resolution_scale]:
            utils.print_rank_0("Decoding Training Cameras")
            if args.num_train_cameras > 0:
                train_cameras = scene_info.train_cameras[:args.num_train_cameras]
            else:
                train_cameras = scene_info.train_cameras
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(train_cameras, resolution_scale, args)
            # output the number of cameras in the training set and image size to the log file
            log_file.write("Train Resolution Scale: {}\n".format(resolution_scale))
            log_file.write("Number of local training cameras: {}\n".format(len(self.train_cameras[resolution_scale])))
            log_file.write("Image size: {}x{}\n".format(self.train_cameras[resolution_scale][0].image_height, self.train_cameras[resolution_scale][0].image_width))

        if args.eval:
            for resolution_scale in [args.test_resolution_scale]:
                utils.print_rank_0("Decoding Test Cameras")
                if args.num_test_cameras > 0:
                    test_cameras = scene_info.test_cameras[:args.num_test_cameras]
                else:
                    test_cameras = scene_info.test_cameras
                self.test_cameras[resolution_scale] = cameraList_from_camInfos(test_cameras, resolution_scale, args)
                # output the number of cameras in the training set and image size to the log file
                log_file.write("Test Resolution Scale: {}\n".format(resolution_scale))
                log_file.write("Number of local test cameras: {}\n".format(len(self.test_cameras[resolution_scale])))
                log_file.write("Image size: {}x{}\n".format(self.test_cameras[resolution_scale][0].image_height, self.test_cameras[resolution_scale][0].image_width))

        utils.check_memory_usage_logging("after Loading all images")
        utils.log_cpu_memory_usage("after decoding images")

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

        utils.check_memory_usage_logging("after initializing point cloud")
        utils.log_cpu_memory_usage("after loading initial 3dgs points")

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        if scale not in self.train_cameras:
            return []
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        if scale not in self.test_cameras:
            return []
        return self.test_cameras[scale]

    def log_scene_info_to_file(self, log_file, prefix_str=""):

        # Print shape of gaussians parameters. 
        log_file.write("xyz shape: {}\n".format(self.gaussians._xyz.shape))
        log_file.write("f_dc shape: {}\n".format(self.gaussians._features_dc.shape))
        log_file.write("f_rest shape: {}\n".format(self.gaussians._features_rest.shape))
        log_file.write("opacity shape: {}\n".format(self.gaussians._opacity.shape))
        log_file.write("scaling shape: {}\n".format(self.gaussians._scaling.shape))
        log_file.write("rotation shape: {}\n".format(self.gaussians._rotation.shape))

class SceneDataset:
    def __init__(self, cameras):
        self.cameras = cameras
        self.camera_size = len(self.cameras)

        self.cur_epoch_cameras = []
        self.cur_iteration = 0

        self.iteration_loss = []
        self.epoch_loss = []
        
        self.log_file = utils.get_log_file()
        self.args = utils.get_args()

    @property
    def cur_epoch(self):
        return len(self.epoch_loss)
    
    @property
    def cur_iteration_in_epoch(self):
        return len(self.iteration_loss)

    def get_one_camera(self, batched_cameras_uid):
        if len(self.cur_epoch_cameras) == 0:
            self.cur_epoch_cameras = self.cameras.copy()
        self.cur_iteration += 1

        # TODO: fixed_training_image not implemented. 
        while True:
            camera_id = randint(0, len(self.cur_epoch_cameras)-1)
            if self.cur_epoch_cameras[camera_id].uid not in batched_cameras_uid:
                break
        viewpoint_cam = self.cur_epoch_cameras.pop(camera_id)
        return viewpoint_cam

    def get_batched_cameras(self, batch_size):
        assert batch_size <= self.camera_size, "Batch size is larger than the number of cameras in the scene."
        batched_cameras = []
        batched_cameras_uid = []
        for i in range(batch_size):
            batched_cameras.append(self.get_one_camera(batched_cameras_uid))
            batched_cameras_uid.append(batched_cameras[-1].uid)

        args = self.args
        timers = utils.get_timers()

        # if utils.DEFAULT_GROUP.size() > 1:
        #     torch.distributed.barrier(group=utils.DEFAULT_GROUP)

        # Asynchronously load ground-truth image to GPU
        timers.start("load_gt_image_to_gpu")
        if args.distributed_dataset_storage:
            assert args.bsz == args.dp_size, "I have not implemented this for bsz != dp_size"
            if utils.IN_NODE_GROUP.rank() == 0:
                dp_rank_2_image = {}
                for gpu_id in range(0, utils.IN_NODE_GROUP.size()):
                    its_global_rank = utils.GLOBAL_RANK + gpu_id
                    its_dp_rank = its_global_rank // args.mp_size

                    if its_dp_rank not in dp_rank_2_image:
                        dp_rank_2_image[its_dp_rank] = batched_cameras[its_dp_rank].original_image_cpu.cuda(non_blocking=args.async_load_gt_image)
        else:
            batched_cameras[utils.DP_GROUP.rank()].original_image = batched_cameras[utils.DP_GROUP.rank()].original_image_cpu.cuda(non_blocking=args.async_load_gt_image)
        # if utils.DEFAULT_GROUP.size() > 1:
        #     torch.distributed.barrier(group=utils.DEFAULT_GROUP)
        timers.stop("load_gt_image_to_gpu")

        # Asynchronously send the original image from gpu0 to all GPUs in the same node.
        timers.start("scatter_gt_image")
        if args.distributed_dataset_storage:
            if utils.IN_NODE_GROUP.rank() == 0:
                all_original_image_to_send = []
                for gpu_id in range(0, utils.IN_NODE_GROUP.size()):
                    its_global_rank = utils.GLOBAL_RANK + gpu_id
                    its_dp_rank = its_global_rank // args.mp_size
                    dp_rank_2_image[its_dp_rank] = dp_rank_2_image[its_dp_rank].contiguous() # the data loading is non-blocking, so this contiguous() may block until the data is loaded.
                    all_original_image_to_send.append(dp_rank_2_image[its_dp_rank])
                recv_buffer = all_original_image_to_send[0]
                batched_cameras[utils.DP_GROUP.rank()].gt_image_comm_op = torch.distributed.scatter(recv_buffer,
                                            scatter_list=all_original_image_to_send,
                                            src=utils.get_first_rank_on_cur_node(),# NOTE: this rank is on global process group (regardless of group argument)
                                            group=utils.IN_NODE_GROUP,
                                            async_op=args.async_load_gt_image)
                batched_cameras[utils.DP_GROUP.rank()].original_image = recv_buffer
            else:
                recv_buffer = torch.zeros((3, batched_cameras[utils.DP_GROUP.rank()].image_height, batched_cameras[utils.DP_GROUP.rank()].image_width), dtype=torch.uint8, device="cuda")
                batched_cameras[utils.DP_GROUP.rank()].gt_image_comm_op = torch.distributed.scatter(recv_buffer,
                                            scatter_list=None,
                                            src=utils.get_first_rank_on_cur_node(),
                                            group=utils.IN_NODE_GROUP,
                                            async_op=args.async_load_gt_image)
                batched_cameras[utils.DP_GROUP.rank()].original_image = recv_buffer
        else:
            batched_cameras[utils.DP_GROUP.rank()].gt_image_comm_op = None
        if "gt_image_comm_op" in batched_cameras[utils.DP_GROUP.rank()].__dict__ and batched_cameras[utils.DP_GROUP.rank()].gt_image_comm_op is not None:
            batched_cameras[utils.DP_GROUP.rank()].gt_image_comm_op.wait()
            batched_cameras[utils.DP_GROUP.rank()].gt_image_comm_op = None
        # if utils.DEFAULT_GROUP.size() > 1:
        #     torch.distributed.barrier(group=utils.DEFAULT_GROUP)
        timers.stop("scatter_gt_image")

        return batched_cameras

    def update_losses(self, losses):
        for loss in losses:
            self.iteration_loss.append(loss)
            if len(self.iteration_loss) % self.camera_size == 0:
                self.epoch_loss.append(
                    sum(self.iteration_loss[-self.camera_size:]) / self.camera_size
                )
                self.log_file.write("epoch {} loss: {}\n".format(len(self.epoch_loss), self.epoch_loss[-1]))
                self.iteration_loss = []
