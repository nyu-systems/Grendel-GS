import torch
import numpy as np
import utils.general_utils as utils


def save_image_for_debug(image, file_name, keep_digits=8):
    # save image for visualization

    file = open(file_name, 'w')

    image_cpu = image.detach().cpu().numpy()

    for c in range(3):
        for i in range(image.shape[1]):
            for j in range(image.shape[2]):
                float_value = image_cpu[c, i, j]
                file.write(f"{float_value:.{keep_digits}f} ")
            file.write("\n")
        file.write("\n")


def save_image_tiles_for_debug(image_tiles, file_name, keep_digits=3):
    # save image for visualization

    file = open(file_name, 'w')

    image_tiles_cpu = image_tiles.detach().cpu().numpy()

    for tile_idx in range(image_tiles.shape[0]):
        file.write(f"tile_idx " + str(tile_idx) + "\n")
        for c in range(3):
            file.write(f"channel " + str(c) + "\n")
            for i in range(utils.BLOCK_X):
                for j in range(utils.BLOCK_Y):
                    float_value = image_tiles_cpu[tile_idx, c, i, j]
                    file.write(f"{float_value:.{keep_digits}f} ")
                file.write("\n")

def save_all_pos_for_debug(all_pos, file_name):

    file = open(file_name, 'w')

    all_pos_cpu = all_pos.detach().cpu().numpy()

    for i in range(all_pos.shape[0]):
        for j in range(all_pos.shape[1]):
            int_value = all_pos_cpu[i, j]
            file.write(f"{int_value} ")
        file.write("\n")

def save_compute_locally_for_debug(compute_locally, file_name):
    file = open(file_name, 'w')

    compute_locally_cpu = compute_locally.detach().cpu().numpy()

    for i in range(compute_locally.shape[0]):
        for j in range(compute_locally.shape[1]):
            int_value = int(compute_locally_cpu[i, j])
            file.write(f"{int_value}")
        file.write("\n")

def save_pixels_compute_locally_for_debug(pixels_compute_locally, file_name):
    file = open(file_name, 'w')

    pixels_compute_locally_cpu = pixels_compute_locally.detach().cpu().numpy()

    for i in range(pixels_compute_locally.shape[0]):
        for j in range(pixels_compute_locally.shape[1]):
            int_value = int(pixels_compute_locally_cpu[i, j])
            file.write(f"{int_value}")
        file.write("\n")

def save_pixel_loss_for_debug(pixel_loss, file_name, keep_digits=3):
    file = open(file_name, 'w')

    pixel_loss_cpu = pixel_loss.detach().cpu().numpy()

    for i in range(pixel_loss.shape[0]):
        for j in range(pixel_loss.shape[1]):
            float_value = pixel_loss_cpu[i, j]
            file.write(f"{float_value:.{keep_digits}f} ")
        file.write("\n")
