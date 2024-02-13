import os
import numpy as np

rk = 0
exp_id = 16
file1 = f"experiments/image_dist_{exp_id}_debug/image_for_debug_grad_ws=4_rk={rk}_iter=1.txt"
file2 = f"experiments/image_dist_{exp_id}_debug_gt/image_for_debug_grad_ws=4_rk={rk}_iter=1.txt"
folder = f"experiments/image_dist_{exp_id}_debug/"

# file1 = f"experiments/image_dist_16_debug/image_for_debug_grad_ws=4_rk={rk}_iter=1.txt"
# file2 = f"experiments/image_dist_16_debug_gt/image_for_debug_grad_ws=4_rk={rk}_iter=1.txt"


def read_file(file):
    a = []
    file_lines = open(file, "r").readlines()
    for line in file_lines:
        values = [float(x) for x in line.split()]
        a.append(values)
    return a


def compare(file1, file2):
    diff_file = open(folder+"diff.txt", "w")
    a1 = read_file(file1)
    a2 = read_file(file2)

    # print the differences into another file
    for i in range(len(a1)):
        for j in range(len(a1[i])):
            if abs(a1[i][j] - a2[i][j]) > 0 and abs(a1[i][j]) > 0:
                # i, j, a1[i][j], a2[i][j]
                diff_file.write(str(i) + " " + str(j) + " " + str(a1[i][j]) + " " + str(a2[i][j]) + "\n")

    diff_file.close()

compare(file1, file2)

def cal_loss():
    loss = 0.0
    non_zero = 0
    # for rk in range(4):
    #     file1 = f"experiments/image_dist_7_debug/pixelwise_ssim_loss_ws=4_rk={rk}_iter=1.txt"
    #     a = read_file(file1)
    #     for i in range(len(a)):
    #         for j in range(len(a[i])):
    #             loss += a[i][j]
    #             if a[i][j] > 0:
    #                 non_zero += 1
    file1 = f"experiments/image_dist_8_debug_gt/pixelwise_ssim_loss_ws=4_rk=0_iter=1.txt"
    a = read_file(file1)
    for i in range(len(a)):
        for j in range(len(a[i])):
            loss += a[i][j]
            if a[i][j] != 0:
                non_zero += 1

    print(loss / (980*545*3))
    print(np.array(a).mean()/3)
    print(non_zero / (980*545))

# cal_loss()

