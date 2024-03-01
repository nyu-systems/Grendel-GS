import torch
import utils.general_utils as utils
import torch.distributed as dist
import diff_gaussian_rasterization
from utils.loss_utils import pixelwise_l1_with_mask, pixelwise_ssim_with_mask

def get_touched_tile_rect(touched_locally):
    nonzero_pos = touched_locally.nonzero()
    min_tile_y = nonzero_pos[:, 0].min().item()
    max_tile_y = nonzero_pos[:, 0].max().item()+1
    min_tile_x = nonzero_pos[:, 1].min().item()
    max_tile_x = nonzero_pos[:, 1].max().item()+1
    return min_tile_y, max_tile_y, min_tile_x, max_tile_x


def get_touched_pixels_rect(touched_locally=None, tile_rect=None):
    assert touched_locally is not None or tile_rect is not None, "Either touched_locally or tile_rect should be provided"
    if tile_rect is None:
        min_tile_y, max_tile_y, min_tile_x, max_tile_x = get_touched_tile_rect(touched_locally)
    else:
        min_tile_y, max_tile_y, min_tile_x, max_tile_x = tile_rect

    min_pixel_y = min_tile_y * utils.BLOCK_Y
    max_pixel_y = min(max_tile_y * utils.BLOCK_Y, utils.IMG_H)
    min_pixel_x = min_tile_x * utils.BLOCK_X
    max_pixel_x = min(max_tile_x * utils.BLOCK_X, utils.IMG_W)
    return min_pixel_y, max_pixel_y, min_pixel_x, max_pixel_x


def get_all_pos_send_to_j(compute_locally, touched_locally):
    # TODO: implement this function using DistributionStrategy Statistics. Could reduce the launching overhead of image distribution.


    timers = utils.get_timers()

    timers.start("[all_pos_send_to_j]all_gather_locally_compute")
    # allgather locally_compute. For 4K, the size of one locally_compute is 4000*4000 = 16MB. It is not a short time. 
    # because bool tensor use 1 byte for 1 element: https://discuss.pytorch.org/t/why-does-each-torch-bool-value-take-up-an-entire-byte/183822
    # TODO: This could be optimized by using bitset. Divide communication volume by 8. 
    all_locally_compute = torch.empty((utils.WORLD_SIZE,)+ compute_locally.shape, dtype=torch.bool, device="cuda")
    torch.distributed.all_gather_into_tensor(all_locally_compute, compute_locally)
    # Suppose we are sending from i to j. 
    pos_mask_to_recv_from_i = [None for _ in range(utils.WORLD_SIZE)]
    pos_recv_from_i = [None for _ in range(utils.WORLD_SIZE)]
    recv_from_i_size = [None for _ in range(utils.WORLD_SIZE)]
    for i in range(utils.WORLD_SIZE):
        if i != utils.LOCAL_RANK:
            pos_mask_to_recv_from_i[i] = torch.logical_and(all_locally_compute[i], touched_locally)
            pos_recv_from_i[i] = torch.nonzero(pos_mask_to_recv_from_i[i], as_tuple=False).contiguous() # (num_pos, 2); contiguous() is needed here.
            recv_from_i_size[i] = pos_recv_from_i[i].shape[0]
        else:
            pos_recv_from_i[i] = torch.zeros((0, 2), dtype=torch.long, device="cuda")
            recv_from_i_size[i] = pos_recv_from_i[i].shape[0]
    timers.stop("[all_pos_send_to_j]all_gather_locally_compute")
    
    timers.start("[all_pos_send_to_j]all_gather_send_to_j_size")# NOTE: This is slow because all_gather_object involves cpu2gpu+gpu2gpu+gpu2cpu communication here. 
    all_pos_recv_from_i = torch.cat(pos_recv_from_i, dim=0)
    j_recv_from_i_size = [None for _ in range(utils.WORLD_SIZE)] # jth row and ith column(j_recv_from_i_size[j][i]) is the size of sending from i to j.
    # each element should be a list of size of pos_recv_from_i[i] for all i. i.e. [None for _ in range(utils.WORLD_SIZE)]
    torch.distributed.all_gather_object(j_recv_from_i_size, recv_from_i_size)
    send_to_j_size = [j_recv_from_i_size[j][utils.LOCAL_RANK] for j in range(utils.WORLD_SIZE)]
    timers.stop("[all_pos_send_to_j]all_gather_send_to_j_size")

    timers.start("[all_pos_send_to_j]all_to_all_pos_send_to_j")
    # Use send_to_j_size[i] as the shape of the tensor
    pos_send_to_j = [None for _ in range(utils.WORLD_SIZE)]
    for j in range(utils.WORLD_SIZE):
        pos_send_to_j[j] = torch.empty((send_to_j_size[j], 2), dtype=torch.long, device="cuda")
    torch.distributed.all_to_all(pos_send_to_j, pos_recv_from_i)
    all_pos_send_to_j = torch.cat(pos_send_to_j, dim=0).contiguous()
    timers.stop("[all_pos_send_to_j]all_to_all_pos_send_to_j")

    return send_to_j_size, recv_from_i_size, all_pos_send_to_j, all_pos_recv_from_i


def get_remote_tiles(send_to_j_size, recv_from_i_size, all_tiles_send_to_j):
    # split all_tiles_send_to_j into tiles_send_to_j, according to the size of pos_send_to_j[j]
    tiles_send_to_j = [None for _ in range(utils.WORLD_SIZE)]
    tiles_recv_from_i = [None for _ in range(utils.WORLD_SIZE)]
    start = 0
    for j in range(utils.WORLD_SIZE):
        end = start + send_to_j_size[j]
        tiles_send_to_j[j] = all_tiles_send_to_j[start:end].contiguous()
        start = end

        i = j
        tiles_recv_from_i[i] = torch.empty((recv_from_i_size[i], 3, utils.BLOCK_Y, utils.BLOCK_X), dtype=torch.float32, device="cuda")
        # XXX: Double check the empty behavior. Because the boundary condition is not clear.

    # all_to_all the pixels
    torch.distributed.nn.functional.all_to_all(tiles_recv_from_i, tiles_send_to_j) # The gradient successfully goes back. 
    
    all_tiles_recv_from_i = torch.cat(tiles_recv_from_i, dim=0).contiguous()
    return all_tiles_recv_from_i



def distributed_loss_computation(image, viewpoint_cam, compute_locally):

    timers = utils.get_timers()


    timers.start("[loss]prepare_for_distributed_loss_computation")


    # Get locally touched tiles and image rect.
    timers.start("[loss]get_touched_locally_and_local_image_rect")
    touched_locally = diff_gaussian_rasterization._C.get_touched_locally(# touched_locally[i,j] is true if pixel (i,j) is touched during local loss computation.
        compute_locally,
        utils.IMG_H,
        utils.IMG_W,
        1,# HACK: extension distance is currently only 1 here because window size is 11 which is less than BLOCK_X and BLOCK_Y. And we only have 1 layer of convolution.
    )
    min_tile_y, max_tile_y, min_tile_x, max_tile_x = get_touched_tile_rect(touched_locally)
    min_pixel_y, max_pixel_y, min_pixel_x, max_pixel_x = get_touched_pixels_rect(tile_rect=(min_tile_y, max_tile_y, min_tile_x, max_tile_x))
    touched_pixels_rect = [min_pixel_y, max_pixel_y, min_pixel_x, max_pixel_x]
    touched_tiles_rect = [min_tile_y, max_tile_y, min_tile_x, max_tile_x]
    # Get image_rect touched locally. shape: (3, max_pixel_y-min_pixel_y, max_pixel_x-min_pixel_x)
    local_image_rect = image[:, min_pixel_y:max_pixel_y, min_pixel_x:max_pixel_x].contiguous()
    timers.stop("[loss]get_touched_locally_and_local_image_rect")


    # Get positions of tiles to send/recv remotely. 
    timers.start("[loss]get_all_pos_send_to_j")
    send_to_j_size, recv_from_i_size, all_pos_send_to_j, all_pos_recv_from_i = get_all_pos_send_to_j(compute_locally, touched_locally)
    timers.stop("[loss]get_all_pos_send_to_j")


    # Load local tiles to send remotely.
    timers.start("[loss]load_image_tiles_by_pos")
    all_tiles_send_to_j = diff_gaussian_rasterization.load_image_tiles_by_pos(
        local_image_rect, # local image rect
        all_pos_send_to_j, # in global coordinates. 
        utils.IMG_H, utils.IMG_W,
        touched_pixels_rect,
        touched_tiles_rect
    )
    timers.stop("[loss]load_image_tiles_by_pos")


    # Receive remote tiles to use locally.
    timers.start("[loss]get_remote_tiles")
    all_tiles_recv_from_i = get_remote_tiles(send_to_j_size, recv_from_i_size, all_tiles_send_to_j)
    timers.stop("[loss]get_remote_tiles")


    # Assemble the image from all the remote tiles, excluding the local ones. shape: (3, max_pixel_y-min_pixel_y, max_pixel_x-min_pixel_x)
    timers.start("[loss]merge_local_tiles_and_remote_tiles")
    local_image_rect_from_remote_tiles = diff_gaussian_rasterization.merge_image_tiles_by_pos(
        all_pos_recv_from_i, # in global coordinates. 
        all_tiles_recv_from_i,
        utils.IMG_H, utils.IMG_W,
        touched_pixels_rect,
        touched_tiles_rect
    )# in local coordinates. 
    local_image_rect_with_remote_tiles = local_image_rect + local_image_rect_from_remote_tiles # in local coordinates. 
    timers.stop("[loss]merge_local_tiles_and_remote_tiles")


    # Get pixels to compute locally. shape: (max_pixel_y-min_pixel_y, max_pixel_x-min_pixel_x)
    # timers.start("[loss]get_pixels_compute_locally_and_in_rect")# very small time
    local_image_rect_pixels_compute_locally = diff_gaussian_rasterization._C.get_pixels_compute_locally_and_in_rect(
        compute_locally,
        utils.IMG_H, utils.IMG_W,
        min_pixel_y, max_pixel_y, min_pixel_x, max_pixel_x
    )
    # timers.stop("[loss]get_pixels_compute_locally_and_in_rect")


    timers.stop("[loss]prepare_for_distributed_loss_computation")



    utils.check_memory_usage_logging("after preparation for image loss distribution")


    # Move image_gt to GPU. its shape: (3, max_pixel_y-min_pixel_y, max_pixel_x-min_pixel_x)
    timers.start("gt_image_load_to_gpu")
    local_image_rect_gt = viewpoint_cam.original_image[:, min_pixel_y:max_pixel_y, min_pixel_x:max_pixel_x].cuda().contiguous()
    timers.stop("gt_image_load_to_gpu")


    # Loss computation
    timers.start("local_loss_computation")
    pixelwise_Ll1 = pixelwise_l1_with_mask(local_image_rect_with_remote_tiles,
                                           local_image_rect_gt,
                                           local_image_rect_pixels_compute_locally)
    pixelwise_Ll1_sum = pixelwise_Ll1.sum()
    utils.check_memory_usage_logging("after l1_loss")
    pixelwise_ssim_loss = pixelwise_ssim_with_mask(local_image_rect_with_remote_tiles,
                                                   local_image_rect_gt,
                                                   local_image_rect_pixels_compute_locally)
    pixelwise_ssim_loss_sum = pixelwise_ssim_loss.sum()
    utils.check_memory_usage_logging("after ssim_loss")
    two_losses = torch.stack([pixelwise_Ll1_sum, pixelwise_ssim_loss_sum]) / (utils.get_num_pixels()*3)
    timers.stop("local_loss_computation") # measure time before allreduce, so that we can get the real local time. 
    torch.distributed.all_reduce(two_losses, op=dist.ReduceOp.SUM)
    # NOTE: We do not have to use allreduce here. It does not affect gradients' correctness. If we want to measure the speed, disable it.


    Ll1 = two_losses[0]
    ssim_loss = two_losses[1]
    return Ll1, ssim_loss



def replicated_loss_computation(image, viewpoint_cam):

    timers = utils.get_timers()


    # Image allreduce
    timers.start("image_allreduce")
    if utils.WORLD_SIZE > 1:
        torch.distributed.all_reduce(image, op=dist.ReduceOp.SUM)
        # make sure non-local pixels are 0 instead of background, otherwise all_reduce sum will give 2*background.
    timers.stop("image_allreduce")


    # Move gt_image to gpu: if args.lazy_load_image is true, then the transfer will actually happen.
    timers.start("gt_image_load_to_gpu")
    gt_image = viewpoint_cam.original_image.cuda()
    timers.stop("gt_image_load_to_gpu")
    utils.check_memory_usage_logging("after gt_image_load_to_gpu")


    # Loss computation
    timers.start("loss")
    pixelwise_Ll1 = pixelwise_l1_with_mask(image, gt_image, torch.ones((utils.IMG_H, utils.IMG_W), dtype=torch.bool, device="cuda"))
    Ll1 = pixelwise_Ll1.mean()
    utils.check_memory_usage_logging("after l1_loss")
    pixelwise_ssim_loss = pixelwise_ssim_with_mask(image, gt_image, torch.ones((utils.IMG_H, utils.IMG_W), dtype=torch.bool, device="cuda"))
    ssim_loss = pixelwise_ssim_loss.mean()
    utils.check_memory_usage_logging("after ssim_loss")
    timers.stop("loss")

    return Ll1, ssim_loss
