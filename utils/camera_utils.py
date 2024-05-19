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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch, get_args, get_log_file
import utils.general_utils as utils
from tqdm import tqdm
from utils.graphics_utils import fov2focal
import time
import multiprocessing
from multiprocessing import shared_memory
import torch

WARNED = False

def loadCam(args, id, cam_info, resolution_scale, decompressed_image=None, return_image=False):
    if id == 0:
        orig_w, orig_h = cam_info.image.size
        utils.set_img_size(orig_h, orig_w)
    orig_h, orig_w = utils.get_img_size()

    args = get_args()
    log_file = get_log_file()
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    if (args.distributed_dataset_storage and utils.LOCAL_RANK == 0) or (not args.distributed_dataset_storage):
        if args.time_image_loading:
            start_time = time.time()
        resized_image_rgb = PILtoTorch(cam_info.image, resolution, args, log_file, decompressed_image=decompressed_image)
        if args.time_image_loading:
            log_file.write(f"PILtoTorch image in {time.time() - start_time} seconds\n")

        # assert resized_image_rgb.shape[0] == 3, "Image should have exactly 3 channels!"
        gt_image = resized_image_rgb[:3, ...].contiguous()
        loaded_mask = None

        # TODO: change this back.
        # if resized_image_rgb.shape[1] == 4:
        #     loaded_mask = resized_image_rgb[3:4, ...]

        # Free the memory: because the PIL image has been converted to torch tensor, we don't need it anymore. And it takes up lots of cpu memory. 
        cam_info.image.close()
    else:
        gt_image = None
        loaded_mask = None
    

    if return_image:
        return gt_image

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

def load_decompressed_image(params):
    args, id, cam_info, resolution_scale = params
    return loadCam(args, id, cam_info, resolution_scale, decompressed_image=None, return_image=True)

# Modify this code to support shared_memory.SharedMemory to make inter-process communication faster
def decompressed_images_from_camInfos_multiprocess(cam_infos, resolution_scale, args):
    args = get_args()
    decompressed_images = []
    total_cameras = len(cam_infos)

    # Create a pool of processes
    with multiprocessing.Pool(processes=16) as pool:
        # Prepare data for processing
        tasks = [(args, id, cam_info, resolution_scale) for id, cam_info in enumerate(cam_infos)]
        
        # Map load_camera_data to the tasks
        # results = pool.map(load_decompressed_image, tasks)
        results = list(tqdm(pool.imap(load_decompressed_image, tasks), total=total_cameras))

        # Build the camera list considering 'fixed_training_image'
        for id, result in enumerate(results):
            decompressed_images.append(result)

    return decompressed_images

def decompress_and_scale_image(cam_info):
    pil_image = cam_info.image
    resolution = cam_info.image.size # (w, h)
    # print("cam_info.image.size: ", cam_info.image.size)
    pil_image.load()
    resized_image_PIL = pil_image.resize(resolution) 
    resized_image = np.array(resized_image_PIL) # (h, w, 3)
    # print("resized_image.shape: ", resized_image.shape)
    if len(resized_image.shape) == 3:
        return resized_image.transpose(2, 0, 1)
    else:
        return resized_image[..., np.newaxis].transpose(2, 0, 1)

def load_decompressed_image_shared(params):
    print("For shared memory loading, I currently disable any computation to test the speed.")
    return 
    shared_mem_name, args, id, cam_info, resolution_scale = params
    # Retrieve the shared memory block
    existing_shm = shared_memory.SharedMemory(name=shared_mem_name)
    
    # Assume each image will be stored as a flat array in shared memory
    # Example: using numpy for manipulation; adjust size and dtype as needed
    resolution_width, resolution_height = cam_info.image.size
    image_shape = (3, resolution_height, resolution_width)  # Set appropriate values
    dtype = np.uint8  # Adjust as per your image data type
    
    # Calculate the offset for this particular image
    offset = id * np.prod(image_shape)
    np_image_array = np.ndarray(image_shape, dtype=dtype, buffer=existing_shm.buf, offset=offset)
    
    # Decompress image into the numpy array directly
    decompressed_image = decompress_and_scale_image(cam_info)  # Implement this
    np_image_array[:] = decompressed_image
    
    # Clean up
    existing_shm.close()

def decompressed_images_from_camInfos_multiprocess_sharedmem(cam_infos, resolution_scale, args):
    args = get_args()
    decompressed_images = []
    total_cameras = len(cam_infos)

    # Assume each image shape and dtype
    resolution_width, resolution_height = cam_infos[0].image.size
    image_shape = (3, resolution_height, resolution_width)  # Define these as per your data
    dtype = np.uint8
    image_size = np.prod(image_shape) * np.dtype(dtype).itemsize
    
    # Create shared memory
    total_size = image_size * total_cameras
    shm = shared_memory.SharedMemory(create=True, size=total_size)

    # Create a pool of processes
    with multiprocessing.Pool(16) as pool:
        # Prepare data for processing
        tasks = [(shm.name, args, id, cam_info, resolution_scale) for id, cam_info in enumerate(cam_infos)]
        
        # print("Start Parallel loading...")
        # Map load_camera_data to the tasks
        list(tqdm(pool.imap(load_decompressed_image_shared, tasks), total=total_cameras))

    # Read images from shared memory
    decompressed_images = []
    for id in range(total_cameras):
        offset = id * np.prod(image_shape)
        np_image_array = np.ndarray(image_shape, dtype=dtype, buffer=shm.buf, offset=offset)
        decompressed_images.append(torch.from_numpy(np_image_array))  # Make a copy if necessary

    # Clean up shared memory
    shm.close()
    shm.unlink()

    return decompressed_images

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    args = get_args()

    if args.multiprocesses_image_loading:
        # print("start multiprocesses_image_loading")
        decompressed_images = decompressed_images_from_camInfos_multiprocess(cam_infos, resolution_scale, args)
        # decompressed_images = decompressed_images_from_camInfos_multiprocess_sharedmem(cam_infos, resolution_scale, args)
    else:
        decompressed_images = [None for _ in cam_infos]

    # print("finish multiprocesses_image_loading")

    camera_list = []
    for id, c in tqdm(enumerate(cam_infos), total=len(cam_infos), disable=(utils.LOCAL_RANK != 0)):
        if not hasattr(args, "fixed_training_image") or args.fixed_training_image == -1 or id == args.fixed_training_image:
            camera_list.append(loadCam(args, id, c, resolution_scale, decompressed_image=decompressed_images[id], return_image=False))
        else:
            camera_list.append(None)

    if utils.DEFAULT_GROUP.size() > 1:
        torch.distributed.barrier(group=utils.DEFAULT_GROUP)

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
