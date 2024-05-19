import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib.ticker import MultipleLocator
import re
import ast

def get_suffix_in_folder(folder):

    if not os.path.exists(folder):
        return None
    
    if not folder.endswith("/"):
        folder += "/"
    
    suffix_list_candidates = []
    for ws in [1,2,4,8,16,32]:
        for rk in range(ws):
            suffix_list_candidates.append(f"ws={ws}_rk={rk}")
    
    suffix_list = []
    for suffix in suffix_list_candidates:
        if os.path.exists(folder + "python_" + suffix + ".log"):
            suffix_list.append(suffix)

    return suffix_list


def get_n3dgs_list_per_rank_from_log(folder):
    suffixes = get_suffix_in_folder(folder)
    stats = {}
    iterations = []
    start_iteration = 0
    for rk, suffix in enumerate(suffixes):
        file = f"python_{suffix}.log"
        file_path = os.path.join(folder, file)
        with open(file_path, 'r') as f:
            lines = f.readlines()
        stats[f"n_3dgs_rk={rk}"] = []
        for line in lines:
            if line.startswith("start_checkpoint:"):
                if "checkpoints/" in line:
                    start_iteration = int(line.split("checkpoints/")[1].split("/")[0])
                else:
                    start_iteration = 0

            # xyz shape: torch.Size([182686, 3])
            if line.startswith("xyz shape:"):
                # example
                # xyz shape: torch.Size([182686, 3])
                n_3dgs = int(line.split("[")[1].split(",")[0])
                stats[f"n_3dgs_rk={rk}"].append(n_3dgs)
                if rk == 0:
                    iterations.append(start_iteration)

            if "Now num of 3dgs:" in line:
                # example
                # iteration[600,601) densify_and_prune. Now num of 3dgs: 183910. Now Memory usage: 0.23658323287963867 GB. Max Memory usage: 0.399813175201416 GB. 
                iteration = int(line.split("iteration[")[1].split(",")[0])
                n_3dgs = int(line.split("Now num of 3dgs: ")[1].split(".")[0])
                if rk == 0:
                    iterations.append(iteration)
                stats[f"n_3dgs_rk={rk}"].append(n_3dgs)
    return stats, iterations

def get_n3dgs_list_from_log(folder):
    rk2n3dgs, iterations = get_n3dgs_list_per_rank_from_log(folder)
    n3dgs_at_iterations = []
    for i in range(len(iterations)):
        n = 0
        for key in rk2n3dgs:
            n += rk2n3dgs[key][i]
        n3dgs_at_iterations.append(n)
    return n3dgs_at_iterations, iterations

def get_final_n3dgs_from_log(folder):
    n3dgs_at_iterations, iterations = get_n3dgs_list_from_log(folder)
    return n3dgs_at_iterations[-1]



def get_results_test(folder):
    result_test_file_path = os.path.join(folder, "results_test.json")
    with open(result_test_file_path, "r") as f:
        results_test = json.load(f)
        #         {
        # "ours_199985": {
        # "SSIM": 0.8189770579338074,
        # "PSNR": 27.135982513427734,
        # "LPIPS": 0.3035728335380554
        # }
        # }
        key = list(results_test.keys())[0]
        results_test = results_test[key]
    return results_test

def draw_n3dgs_metrics(folders, save_folder):

    all_results = []
    all_n3dgs = []
    all_points = []
    for folder in folders:
        result = get_results_test(folder)
        n3dgs = get_final_n3dgs_from_log(folder)
        # all_results.append(get_results_test(folder))
        # all_n3dgs.append(get_final_n3dgs_from_log(folder))
        expe_name = folder.split("/")[-1]
        if expe_name == "rub_16g_7_c2":
            expe_name = "rub_16g_7"
        point = (n3dgs, result, "Experiment: " + expe_name)
        all_points.append(point)

    all_points = sorted(all_points, key=lambda x: x[0])

    # Save these in csv file
    # columes: Expe_name, n3dgs, PSNR, SSIM, LPIPS
    df = pd.DataFrame(columns=["Expe_name", "n3dgs", "PSNR", "SSIM", "LPIPS"])
    for point in all_points:
        # df = df._append({"Expe_name": point[2], "n3dgs": point[0], "PSNR": point[1]["PSNR"], "SSIM": point[1]["SSIM"], "LPIPS": point[1]["LPIPS"]}, ignore_index=True)
        # keep 3 decimal places
        df = df._append({"Expe_name": point[2], 
                        "n3dgs": point[0], 
                        "PSNR": round(point[1]["PSNR"], 3), 
                        "SSIM": round(point[1]["SSIM"], 3), 
                        "LPIPS": round(point[1]["LPIPS"], 3)}, 
                    ignore_index=True)
    df.to_csv(os.path.join(save_folder, "n3dgs_metrics.csv"), index=False)

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(30, 10))
    fig.subplots_adjust(hspace=0.5)

    ax[0].set_title("PSNR vs. n_of_3dgs")
    ax[0].set_xlabel("log n_of_3dgs")
    ax[0].set_xscale('log')
    ax[0].set_ylabel("PSNR")
    for point in all_points:
        ax[0].scatter(point[0], point[1]["PSNR"], label=point[2])
    ax[0].legend()

    ax[1].set_title("SSIM vs. n_of_3dgs")
    ax[1].set_xlabel("log n_of_3dgs")
    ax[1].set_xscale('log')
    ax[1].set_ylabel("SSIM")
    for point in all_points:
        ax[1].scatter(point[0], point[1]["SSIM"], label=point[2])
    ax[1].legend()

    ax[2].set_title("LPIPS vs. n_of_3dgs")
    ax[2].set_xlabel("log n_of_3dgs")
    ax[2].set_xscale('log')
    ax[2].set_ylabel("LPIPS")
    for point in all_points:
        ax[2].scatter(point[0], point[1]["LPIPS"], label=point[2])
    ax[2].legend()

    plt.savefig(os.path.join(save_folder, "n3dgs_metrics.png"))

def draw_speed(scene_name, folder, save_folder):

    ngpu_bsz_2_throughput = {}
    for n_gpu in [1, 2, 4, 8, 16, 32]:
        for bsz in [1, 2, 4, 8, 16, 32, 64]:
            ngpu_bsz_2_throughput[str((n_gpu, bsz))] = []
            expe_name = f"{scene_name}_speed_{n_gpu}g_{bsz}b"
            expe_folder = os.path.join(folder, expe_name)
            log_file = os.path.join(expe_folder, f"python_ws={n_gpu}_rk=0.log")
            if not os.path.exists(log_file):
                continue

            with open(log_file, "r") as f:
                lines = f.readlines()
            for line in lines:
                #epoch 1 time: 580.436 s, n_samples: 1657, throughput 2.85 it/s
                if "n_samples: " in line and "throughput" in line:
                    throughput = float(line.split("throughput ")[1].split(" it/s")[0])
                    ngpu_bsz_2_throughput[str((n_gpu, bsz))].append(throughput)
    json.dump(ngpu_bsz_2_throughput, open(os.path.join(save_folder, f"{scene_name}_speed.json"), "w"), indent=4)

    ngpu_bsz_2_1stepoch_throughput = {}
    ngpu_bsz_2_2rdepoch_throughput = {}
    ngpu_bsz_2_final_throughput = {}
    ngpu_bsz_2_final_ave_throughput = {}
    for n_gpu in [1, 2, 4, 8, 16, 32]:
        for bsz in [1, 2, 4, 8, 16, 32, 64]:
            if len(ngpu_bsz_2_throughput[str((n_gpu, bsz))]) == 0:
                continue
            if len(ngpu_bsz_2_throughput[str((n_gpu, bsz))]) >= 1:
                ngpu_bsz_2_1stepoch_throughput[str((n_gpu, bsz))] = ngpu_bsz_2_throughput[str((n_gpu, bsz))][0]
            if len(ngpu_bsz_2_throughput[str((n_gpu, bsz))]) >= 2:
                ngpu_bsz_2_2rdepoch_throughput[str((n_gpu, bsz))] = ngpu_bsz_2_throughput[str((n_gpu, bsz))][1]
            if len(ngpu_bsz_2_throughput[str((n_gpu, bsz))]) >= 3:
                # ngpu_bsz_2_final_throughput[str((n_gpu, bsz))] = max(ngpu_bsz_2_throughput[str((n_gpu, bsz))][2:])
                ngpu_bsz_2_final_throughput[str((n_gpu, bsz))] = ngpu_bsz_2_throughput[str((n_gpu, bsz))][2]
            if len(ngpu_bsz_2_throughput[str((n_gpu, bsz))]) >= 5:
                ngpu_bsz_2_final_ave_throughput[str((n_gpu, bsz))] = round(sum(ngpu_bsz_2_throughput[str((n_gpu, bsz))][3:6]) / 3, 3)
    json.dump(ngpu_bsz_2_1stepoch_throughput, open(os.path.join(save_folder, f"{scene_name}_speed_1stepoch.json"), "w"), indent=4)
    json.dump(ngpu_bsz_2_2rdepoch_throughput, open(os.path.join(save_folder, f"{scene_name}_speed_2rdepoch.json"), "w"), indent=4)
    json.dump(ngpu_bsz_2_final_throughput, open(os.path.join(save_folder, f"{scene_name}_speed_final.json"), "w"), indent=4)
    json.dump(ngpu_bsz_2_final_ave_throughput, open(os.path.join(save_folder, f"{scene_name}_speed_final_ave.json"), "w"), indent=4)

    # Draw a dataframe for the ngpu_bsz_2_2rdepoch_throughput, Rows are ngpu, Columns are bsz
    df = pd.DataFrame(columns=["#_GPU", "bsz=1", "bsz=2", "bsz=4", "bsz=8", "bsz=16", "bsz=32", "bsz=64"])
    for n_gpu in [1, 2, 4, 8, 16, 32]:
        row = [str(n_gpu)+"_gpu"]
        for bsz in [1, 2, 4, 8, 16, 32, 64]:
            if str((n_gpu, bsz)) in ngpu_bsz_2_2rdepoch_throughput:
                row.append(str(ngpu_bsz_2_2rdepoch_throughput[str((n_gpu, bsz))])+"it/s")
            else:
                row.append(None)
        df.loc[n_gpu] = row
    df.to_csv(os.path.join(save_folder, f"{scene_name}_speed_without_loadbalancing.csv"))

    # Draw a dataframe for the ngpu_bsz_2_final_ave_throughput, Rows are ngpu, Columns are bsz
    df = pd.DataFrame(columns=["#_GPU", "bsz=1", "bsz=2", "bsz=4", "bsz=8", "bsz=16", "bsz=32", "bsz=64"])
    for n_gpu in [1, 2, 4, 8, 16, 32]:
        row = [str(n_gpu)+"_gpu"]
        for bsz in [1, 2, 4, 8, 16, 32, 64]:
            if str((n_gpu, bsz)) in ngpu_bsz_2_final_ave_throughput:
                row.append(str(ngpu_bsz_2_final_ave_throughput[str((n_gpu, bsz))])+"it/s")
            else:
                row.append(None)
        df.loc[n_gpu] = row
    df.to_csv(os.path.join(save_folder, f"{scene_name}_speed_with_loadbalancing.csv"))

def draw_memory(scene_name, folder, save_folder):

    ngpu_bsz_2_n3dgs = {}

    for bsz in [1, 4]:
        for n_gpu in [1, 2, 4, 8, 16]:
            expe_name = f"{scene_name}_mem_{n_gpu}g_{bsz}b"
            expe_folder = os.path.join(folder, expe_name)
            if not os.path.exists(expe_folder):
                continue

            ngpu_bsz_2_n3dgs[str((n_gpu, bsz))] = []
            n_3dgs_at_end = 0
            for rk in range(n_gpu):
                log_file = os.path.join(expe_folder, f"python_ws={n_gpu}_rk={rk}.log")
                lines = open(log_file, "r").readlines()
                # read from the end
                for line in reversed(lines):
                    if "Now num of 3dgs:" in line:
                        n_3dgs = int(line.split("Now num of 3dgs: ")[1].split(".")[0])
                        n_3dgs_at_end += n_3dgs
                        break
            ngpu_bsz_2_n3dgs[str((n_gpu, bsz))].append(n_3dgs_at_end)

    json.dump(ngpu_bsz_2_n3dgs, open(os.path.join(save_folder, f"{scene_name}_memory.json"), "w"), indent=4)

    # Draw a graph for this; the name is # of 3dgs supported by different # of GPUs
    df = pd.DataFrame(columns=["#_GPU", "bsz=1", "bsz=4"])
    for n_gpu in [1, 2, 4, 8, 16]:
        row = [str(n_gpu)+"_gpu"]
        for bsz in [1, 4]:
            if str((n_gpu, bsz)) in ngpu_bsz_2_n3dgs:
                row.append(str(round(ngpu_bsz_2_n3dgs[str((n_gpu, bsz))][0]/1000000, 2))+" millions gaussian")
            else:
                row.append(None)
        df.loc[n_gpu] = row
    df.to_csv(os.path.join(save_folder, f"{scene_name}_memory.csv"))

    pass


def plot_rubble():
    plot_rubble_folder = "/pscratch/sd/j/jy-nyu/final_expes/plot_rubble/"
    if not os.path.exists(plot_rubble_folder):
        os.makedirs(plot_rubble_folder)

    rubble_16g_expes = [
        # "/pscratch/sd/j/jy-nyu/final_expes/rub_16g_1",
        # "/pscratch/sd/j/jy-nyu/final_expes/rub_16g_1_b16",
        # "/pscratch/sd/j/jy-nyu/final_expes/rub_16g_2",
        "/pscratch/sd/j/jy-nyu/final_expes/rub_16g_3",
        # "/pscratch/sd/j/jy-nyu/final_expes/rub_16g_4",
        "/pscratch/sd/j/jy-nyu/final_expes/rub_16g_5",
        "/pscratch/sd/j/jy-nyu/final_expes/rub_16g_6",
        "/pscratch/sd/j/jy-nyu/final_expes/rub_16g_7_c2",
        "/pscratch/sd/j/jy-nyu/final_expes/rub_16g_8",
        "/pscratch/sd/j/jy-nyu/final_expes/rub_16g_9",
        # "/pscratch/sd/j/jy-nyu/final_expes/rub_16g_a",
        "/pscratch/sd/j/jy-nyu/final_expes/rub_16g_b",
        "/pscratch/sd/j/jy-nyu/final_expes/rub_16g_c",
        # "/pscratch/sd/j/jy-nyu/final_expes/rub_16g_d",
        "/pscratch/sd/j/jy-nyu/final_expes/rub_16g_e",
        "/pscratch/sd/j/jy-nyu/final_expes/rub_16g_f",
    ]
    # draw_n3dgs_metrics(rubble_16g_expes, plot_rubble_folder)

    draw_speed("rub",
                "/pscratch/sd/j/jy-nyu/final_expes/rub_16g_7_speed_saved/",
               plot_rubble_folder)


    # draw_memory("rub",
    #             "/pscratch/sd/j/jy-nyu/final_expes/rub_memory",
    #             plot_rubble_folder)


def plot_matrixcity_blockall():
    plot_mat_folder = "/pscratch/sd/j/jy-nyu/final_expes/plot_mat/"
    if not os.path.exists(plot_mat_folder):
        os.makedirs(plot_mat_folder)

    matrixcity_blockall_expes = [
        "/pscratch/sd/j/jy-nyu/final_expes/mball2_16g_1",
        "/pscratch/sd/j/jy-nyu/final_expes/mball2_16g_2",
        "/pscratch/sd/j/jy-nyu/final_expes/mball2_16g_3",
        "/pscratch/sd/j/jy-nyu/final_expes/mball2_16g_4",
        "/pscratch/sd/j/jy-nyu/final_expes/mball2_16g_5",
        "/pscratch/sd/j/jy-nyu/final_expes/mball2_16g_6_re",
        "/pscratch/sd/j/jy-nyu/final_expes/mball2_16g_7_re",
        "/pscratch/sd/j/jy-nyu/final_expes/mball2_16g_8",
    ]

    draw_n3dgs_metrics(matrixcity_blockall_expes, plot_mat_folder)


def get_end_running_time(expe_folder):
    a_suffix = get_suffix_in_folder(expe_folder)[0]
    a_log_file = os.path.join(expe_folder, f"python_{a_suffix}.log")
    lines = open(a_log_file, "r").readlines()
    for line in reversed(lines):
        # end2end total_time: 1373.681 s, iterations: 50000, throughput 36.40 it/s
        if "end2end total_time:" in line:
            running_time = int(float(line.split("end2end total_time: ")[1].split(" s")[0]))
            return running_time
    return None

def get_running_time_at_iterations(expe_folder, iterations):
    a_suffix = get_suffix_in_folder(expe_folder)[0]
    a_log_file = os.path.join(expe_folder, f"python_{a_suffix}.log")
    lines = open(a_log_file, "r").readlines()
    results = []
    bsz = 1
    for line in lines:
        # bsz: 1
        if "bsz: " in line:
            bsz = int(line.split("bsz: ")[1])
        # end2end total_time: 443.026 s, iterations: 7001, throughput 15.80 it/s
        if "end2end total_time:" not in line:
            continue
        iteration = int(line.split("iterations: ")[1].split(",")[0])
        running_time = int(float(line.split("end2end total_time: ")[1].split(" s")[0]))

        for r in range(iteration-bsz, iteration):
            if r in iterations:
                results.append(running_time)
                break
    return results

def get_test_psnr_at_iterations(expe_folder, iterations):
    a_suffix = get_suffix_in_folder(expe_folder)[0]
    a_log_file = os.path.join(expe_folder, f"python_{a_suffix}.log")
    lines = open(a_log_file, "r").readlines()
    results = []
    bsz = 1
    for line in lines:
        # bsz: 1
        if "bsz: " in line:
            bsz = int(line.split("bsz: ")[1])
        # [ITER 50000] Evaluating test: L1 0.01809605024755001 PSNR 29.30947494506836
        if "Evaluating test:" not in line:
            continue
        iteration = int(line.split("[ITER ")[1].split("]")[0])
        L1 = float(line.split("L1 ")[1].split(" PSNR")[0])
        PSNR = float(line.split("PSNR ")[1])
        for r in range(iteration, iteration+bsz):
            if r in iterations:
                results.append(round(PSNR, 3))
    return results

def get_test_psnr_list_from_logfile(expe_folder):
    a_suffix = get_suffix_in_folder(expe_folder)[0]
    a_log_file = os.path.join(expe_folder, f"python_{a_suffix}.log")
    lines = open(a_log_file, "r").readlines()
    results = []
    for line in lines:
        # [ITER 50000] Evaluating test: L1 0.01809605024755001 PSNR 29.30947494506836
        if "Evaluating test:" not in line:
            continue
        iteration = int(line.split("[ITER ")[1].split("]")[0])
        L1 = float(line.split("L1 ")[1].split(" PSNR")[0])
        PSNR = float(line.split("PSNR ")[1])
        results.append({
            "iteration": iteration,
            "L1": round(L1, 3),
            "PSNR": round(PSNR, 3)
        })
    return results




def plot_mip360():
    plot_mip_folder = "/pscratch/sd/j/jy-nyu/final_expes/plot_mip360/"
    if not os.path.exists(plot_mip_folder):
        os.makedirs(plot_mip_folder)

    scene_names = [
        "garden",
        "bicycle",
        "counter",
        "kitchen",
        "room",
        "stump",
    ]
    expe_name = {
        "garden":{
            1:"/pscratch/sd/j/jy-nyu/final_expes/ga1080_4g_1_b1",
            4:"/pscratch/sd/j/jy-nyu/final_expes/ga1080_4g_1_b4"
        },
        "bicycle":{
            1:"/pscratch/sd/j/jy-nyu/final_expes/bi1080_4g_1_b1",
            4:"/pscratch/sd/j/jy-nyu/final_expes/bi1080_4g_1_b4"
        },
        "counter":{
            1:"/pscratch/sd/j/jy-nyu/final_expes/360v21080_4g_counter_bsz1",
            4:"/pscratch/sd/j/jy-nyu/final_expes/360v21080_4g_counter_bsz4"
        },
        "kitchen":{
            1:"/pscratch/sd/j/jy-nyu/final_expes/360v21080_4g_kitchen_bsz1",
            4:"/pscratch/sd/j/jy-nyu/final_expes/360v21080_4g_kitchen_bsz4"
        },
        "room":{
            1:"/pscratch/sd/j/jy-nyu/final_expes/360v21080_4g_room_bsz1",
            4:"/pscratch/sd/j/jy-nyu/final_expes/360v21080_4g_room_bsz4"
        },
        "stump":{
            1:"/pscratch/sd/j/jy-nyu/final_expes/360v21080_4g_stump_bsz1",
            4:"/pscratch/sd/j/jy-nyu/final_expes/360v21080_4g_stump_bsz4"
        },
    }
    
    results = {}
    for scene_name in scene_names:
        for bsz in [1,4]:
            expe = expe_name[scene_name][bsz]
            if not os.path.exists(expe):
                continue
            running_time_50k = get_end_running_time(expe)
            psnr_50k = get_test_psnr_list_from_logfile(expe)[-1]["PSNR"]
            results[f"{scene_name}_bsz{bsz}"] = {
                "running_time_50k": running_time_50k,
                "psnr_50k": psnr_50k
            }
    
    # make a dataframe table to visualize
    df = pd.DataFrame(columns=["Scene", "bsz=1", "bsz=4"])
    for scene_name in scene_names:
        row = [scene_name]
        for bsz in [1, 4]:
            result_str = ""
            if f"{scene_name}_bsz{bsz}" in results:
                result_str = f"Time: {results[f'{scene_name}_bsz{bsz}']['running_time_50k']}s\nPSNR: {results[f'{scene_name}_bsz{bsz}']['psnr_50k']}"
            row.append(result_str)
        df.loc[len(df)] = row
    df.to_csv(os.path.join(plot_mip_folder, "mip360.csv"))


def plot_tan_db():
    plot_tan_folder = "/pscratch/sd/j/jy-nyu/final_expes/plot_tandb/"
    if not os.path.exists(plot_tan_folder):
        os.makedirs(plot_tan_folder)
    
    scene_names = [
        "train",
        "truck",
        "playground",
        "drjohnson"
    ]

    expe_name = {
        "train":{
            4:"/pscratch/sd/j/jy-nyu/final_expes/tra_4g_1",
        },
        "truck":{
            4:"/pscratch/sd/j/jy-nyu/final_expes/tru_4g_1",
        },
        "playground":{
            4:"/pscratch/sd/j/jy-nyu/final_expes/pla_4g_1",
        },
        "drjohnson":{
            4:"/pscratch/sd/j/jy-nyu/final_expes/drj_4g_1",
        },
    }

    results = {}
    for scene_name in scene_names:
        for bsz in [4]:
            expe = expe_name[scene_name][bsz]
            if not os.path.exists(expe):
                continue
            running_time_30k = get_end_running_time(expe)
            psnr_30k = get_test_psnr_list_from_logfile(expe)[-1]["PSNR"]
            results[f"{scene_name}_bsz{bsz}"] = {
                "running_time_30k": running_time_30k,
                "psnr_30k": psnr_30k
            }
    
    # make a dataframe table to visualize
    df = pd.DataFrame(columns=["Scene", "bsz=4"])
    for scene_name in scene_names:
        row = [scene_name]
        for bsz in [4]:
            result_str = ""
            if f"{scene_name}_bsz{bsz}" in results:
                result_str = f"Time: {results[f'{scene_name}_bsz{bsz}']['running_time_30k']}s\nPSNR: {results[f'{scene_name}_bsz{bsz}']['psnr_30k']}"
            row.append(result_str)
        df.loc[len(df)] = row
    df.to_csv(os.path.join(plot_tan_folder, "tandb.csv"))



def extract_from_mip360_all9scene(folder):
    if os.path.exists(os.path.join(folder, "mip360_all9scene.json")):
        print("mip360_all9scene.json already exists for ", folder)
        return
    # counter kitchen room stump bicycle garden bonsai flowers treehill
    scene_names = [
        "counter",
        "kitchen",
        "room",
        "stump",
        "bicycle",
        "garden",
        "bonsai",
        "flowers",
        "treehill"
    ]
    check_iterations = [7000, 15000, 30000, 50000]
    results = {}
    for scene in scene_names:
        scene_folder = os.path.join(folder, "e_"+scene)
        if not os.path.exists(scene_folder):
            continue
        running_time_all = get_running_time_at_iterations(scene_folder, check_iterations)
        psnr_all = get_test_psnr_at_iterations(scene_folder, check_iterations)
        results[scene] = {}
        for iteration, running_time, psnr in zip(check_iterations, running_time_all, psnr_all):
            results[scene][iteration] = {
                "running_time": running_time,
                "psnr": psnr
            }

    json.dump(results, open(os.path.join(folder, "mip360_all9scene.json"), "w"), indent=4)
    print("Generated mip360_all9scene.json for ", folder)



def compare_mip360():
    extract_from_mip360_all9scene("/pscratch/sd/j/jy-nyu/final_expes/360v21080_1g_1b/")
    extract_from_mip360_all9scene("/pscratch/sd/j/jy-nyu/final_expes/360v21080_4g_1b/")
    extract_from_mip360_all9scene("/pscratch/sd/j/jy-nyu/final_expes/360v21080_4g_1b_nobalan/")
    extract_from_mip360_all9scene("/pscratch/sd/j/jy-nyu/final_expes/360v21080_4g_1b_nobalangsdist/")
    extract_from_mip360_all9scene("/pscratch/sd/j/jy-nyu/final_expes/360v21080_4g_1b_nobalanimgdist/")
    extract_from_mip360_all9scene("/pscratch/sd/j/jy-nyu/final_expes/360v21080_4g_4b/")
    extract_from_mip360_all9scene("/pscratch/sd/j/jy-nyu/final_expes/360v21080_4g_4b_nobalan/")
    extract_from_mip360_all9scene("/pscratch/sd/j/jy-nyu/final_expes/360v21080_4g_4b_nobalangsdist/")
    extract_from_mip360_all9scene("/pscratch/sd/j/jy-nyu/final_expes/360v21080_4g_4b_nobalanimgdist/")

    analyze_360v21080_folder = "/pscratch/sd/j/jy-nyu/final_expes/360v21080_analyze/"
    if not os.path.exists(analyze_360v21080_folder):
        os.makedirs(analyze_360v21080_folder)
    
    # compare_different_gpu_and_different_bsz
    compare_iterations = ["7000", "30000", "50000"]
    compare_metrics = ["running_time", "psnr"]

    for iteration in compare_iterations:
        for metric in compare_metrics:
            df = pd.DataFrame(columns=["Scene", "1gpu1bsz", "4gpu1bsz", "4gpu4bsz"])
            for scene in ["counter", "kitchen", "room", "stump", "bicycle", "garden", "bonsai", "flowers", "treehill"]:
                row = [scene]
                for expe in ["360v21080_1g_1b", "360v21080_4g_1b", "360v21080_4g_4b"]:
                    expe_folder = os.path.join("/pscratch/sd/j/jy-nyu/final_expes/", expe)
                    results = json.load(open(os.path.join(expe_folder, "mip360_all9scene.json"), "r"))
                    row.append(results[scene][iteration][metric])
                df.loc[len(df)] = row
            df.to_csv(os.path.join(analyze_360v21080_folder, f"mip360_comparegpubsz_{metric}_{iteration}.csv"))

    # check load balance for 4gpu1bsz
    for iteration in compare_iterations:
        for metric in compare_metrics:
            df = pd.DataFrame(columns=["Scene", "4gpu1bsz", "4gpu1bsz_no_loadbalance", "4gpu1bsz_only_imagedistribution_loadbalance", "4gpu1bsz_only_gaussiandistribution_loadbalance"])
            for scene in ["counter", "kitchen", "room", "stump", "bicycle", "garden", "bonsai", "flowers", "treehill"]:
                row = [scene]
                for expe in ["360v21080_4g_1b", "360v21080_4g_1b_nobalan", "360v21080_4g_1b_nobalangsdist", "360v21080_4g_1b_nobalanimgdist"]:
                    expe_folder = os.path.join("/pscratch/sd/j/jy-nyu/final_expes/", expe)
                    results = json.load(open(os.path.join(expe_folder, "mip360_all9scene.json"), "r"))
                    row.append(results[scene][iteration][metric])
                df.loc[len(df)] = row
            df.to_csv(os.path.join(analyze_360v21080_folder, f"mip360_compare_loadbalance_{metric}_{iteration}_4gpu1bsz.csv"))

    # check load balance for 4gpu4bsz
    for iteration in compare_iterations:
        for metric in compare_metrics:
            df = pd.DataFrame(columns=["Scene", "4gpu4bsz", "4gpu4bsz_no_loadbalance", "4gpu4bsz_only_imagedistribution_loadbalance", "4gpu4bsz_only_gaussiandistribution_loadbalance"])
            for scene in ["counter", "kitchen", "room", "stump", "bicycle", "garden", "bonsai", "flowers", "treehill"]:
                row = [scene]
                for expe in ["360v21080_4g_4b", "360v21080_4g_4b_nobalan", "360v21080_4g_4b_nobalangsdist", "360v21080_4g_4b_nobalanimgdist"]:
                    expe_folder = os.path.join("/pscratch/sd/j/jy-nyu/final_expes/", expe)
                    results = json.load(open(os.path.join(expe_folder, "mip360_all9scene.json"), "r"))
                    row.append(results[scene][iteration][metric])
                df.loc[len(df)] = row
            df.to_csv(os.path.join(analyze_360v21080_folder, f"mip360_compare_loadbalance_{metric}_{iteration}_4gpu4bsz.csv"))


def convert_mip360_to_latex():
    psnr_table_source_df = pd.read_csv("/pscratch/sd/j/jy-nyu/final_expes/360v21080_analyze/mip360_comparegpubsz_psnr_30000.csv")
    psnr_table_source_df = psnr_table_source_df.round(2)
    psnr_table_source_df = psnr_table_source_df.astype(str)
    # add name of this table in latex
    psnr_table_source_df.to_latex("/pscratch/sd/j/jy-nyu/final_expes/360v21080_analyze/mip360_comparegpubsz_psnr_30000.tex", index=False)

    running_time_table_source_df = pd.read_csv("/pscratch/sd/j/jy-nyu/final_expes/360v21080_analyze/mip360_comparegpubsz_running_time_30000.csv")
    running_time_table_source_df = running_time_table_source_df.astype(str)
    running_time_table_source_df.to_latex("/pscratch/sd/j/jy-nyu/final_expes/360v21080_analyze/mip360_comparegpubsz_running_time_30000.tex", index=False)

    loadbalance_4g1b_source_df = pd.read_csv("/pscratch/sd/j/jy-nyu/final_expes/360v21080_analyze/mip360_compare_loadbalance_running_time_30000_4gpu1bsz.csv")
    running_time_4g4b_source_df = pd.read_csv("/pscratch/sd/j/jy-nyu/final_expes/360v21080_analyze/mip360_compare_loadbalance_running_time_30000_4gpu4bsz.csv")
    # take all columes from loadbalance_4g1b_source_df, and add the `4gpu4bsz` colume from running_time_4g4b_source_df as the first colume after the `scene` colume.
    running_time_ablation_study_df = pd.concat([running_time_4g4b_source_df[["Scene", "4gpu4bsz"]],
                                                loadbalance_4g1b_source_df[["4gpu1bsz", "4gpu1bsz_only_imagedistribution_loadbalance", "4gpu1bsz_only_gaussiandistribution_loadbalance", "4gpu1bsz_no_loadbalance"]],
                                                ], axis=1)
    running_time_ablation_study_df = running_time_ablation_study_df.astype(str)
    running_time_ablation_study_df.to_latex("/pscratch/sd/j/jy-nyu/final_expes/360v21080_analyze/mip360_compare_loadbalance_running_time_30000_ablation_study.tex", index=False)
    
    pass
    
    

if __name__ == "__main__":
    plot_rubble()
    # plot_mip360()
    # plot_matrixcity_blockall()
    # plot_tan_db()
    # compare_mip360()
    # convert_mip360_to_latex()

    pass

