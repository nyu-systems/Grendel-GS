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
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
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

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

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

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

        self.cur_epoch_cameras = []
        self.random_sequence = None
        if args.fixed_random_sequence:
            self.random_sequence = self.generate_random_squence()

    def generate_random_squence(self, max_bsz=16, len=128, scale=1.0):
        # Generate a random sequence of cameras, each max_bsz cameras are in a batch.
        # only need to generate the index of the cameras as and int array.
        # The actual cameras are fetched in get_batched_cameras.
        assert len % max_bsz == 0, "len must be divisible by max_bsz"
        random_sequence = []
        if os.path.exists(f"random_sequence_max_bsz={max_bsz}_len={len}.txt"):
            with open(f"random_sequence_max_bsz={max_bsz}_len={len}.txt", "r") as f:
                batched_cameras_uid = [int(line.strip()) for line in f]
            for uid in batched_cameras_uid:
                for cam in self.train_cameras[scale]:
                    if cam.uid == uid:
                        random_sequence.append(cam)
                        break
            return random_sequence

        for _ in range(len // max_bsz):
            batched_cameras_uid = []
            for _ in range(max_bsz):
                random_sequence.append(self.get_one_camera(batched_cameras_uid))
                batched_cameras_uid.append(random_sequence[-1].uid)
        # save to txt
        with open(f"random_sequence_max_bsz={max_bsz}_len={len}.txt", "w") as f:
            for cam in random_sequence:
                f.write(str(cam.uid) + "\n")
        return random_sequence

    def get_one_camera(self, batched_cameras_uid, scale=1.0):
        if self.random_sequence:
            return self.random_sequence.pop(0)
        
        if len(self.cur_epoch_cameras) == 0:
            self.cur_epoch_cameras = self.train_cameras[scale].copy()

        while True:
            camera_id = random.randint(0, len(self.cur_epoch_cameras)-1)
            if self.cur_epoch_cameras[camera_id].uid not in batched_cameras_uid:
                break
        viewpoint_cam = self.cur_epoch_cameras.pop(camera_id)
        return viewpoint_cam

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]