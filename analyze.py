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
    for n_gpu in [1, 2, 4, 8, 16]:
        for bsz in [1, 2, 4, 8, 16, 32, 48]:
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
    for n_gpu in [1, 2, 4, 8, 16]:
        for bsz in [1, 2, 4, 8, 16, 32, 48]:
            if len(ngpu_bsz_2_throughput[str((n_gpu, bsz))]) == 0:
                continue
            if len(ngpu_bsz_2_throughput[str((n_gpu, bsz))]) >= 1:
                ngpu_bsz_2_1stepoch_throughput[str((n_gpu, bsz))] = ngpu_bsz_2_throughput[str((n_gpu, bsz))][0]
            if len(ngpu_bsz_2_throughput[str((n_gpu, bsz))]) >= 2:
                ngpu_bsz_2_2rdepoch_throughput[str((n_gpu, bsz))] = ngpu_bsz_2_throughput[str((n_gpu, bsz))][1]
            if len(ngpu_bsz_2_throughput[str((n_gpu, bsz))]) >= 3:
                # ngpu_bsz_2_final_throughput[str((n_gpu, bsz))] = max(ngpu_bsz_2_throughput[str((n_gpu, bsz))][2:])
                ngpu_bsz_2_final_throughput[str((n_gpu, bsz))] = ngpu_bsz_2_throughput[str((n_gpu, bsz))][2]
    json.dump(ngpu_bsz_2_1stepoch_throughput, open(os.path.join(save_folder, f"{scene_name}_speed_1stepoch.json"), "w"), indent=4)
    json.dump(ngpu_bsz_2_2rdepoch_throughput, open(os.path.join(save_folder, f"{scene_name}_speed_2rdepoch.json"), "w"), indent=4)
    json.dump(ngpu_bsz_2_final_throughput, open(os.path.join(save_folder, f"{scene_name}_speed_final.json"), "w"), indent=4)

    # Draw a dataframe for the ngpu_bsz_2_2rdepoch_throughput, Rows are ngpu, Columns are bsz
    df = pd.DataFrame(columns=["#_GPU", "bsz=1", "bsz=2", "bsz=4", "bsz=8", "bsz=16", "bsz=32", "bsz=48"])
    for n_gpu in [1, 2, 4, 8, 16]:
        row = [str(n_gpu)+"_gpu"]
        for bsz in [1, 2, 4, 8, 16, 32, 48]:
            if str((n_gpu, bsz)) in ngpu_bsz_2_2rdepoch_throughput:
                row.append(str(ngpu_bsz_2_2rdepoch_throughput[str((n_gpu, bsz))])+"it/s")
            else:
                row.append(None)
        df.loc[n_gpu] = row
    df.to_csv(os.path.join(save_folder, f"{scene_name}_speed_without_loadbalancing.csv"))

    # Draw a dataframe for the ngpu_bsz_2_final_throughput, Rows are ngpu, Columns are bsz
    df = pd.DataFrame(columns=["#_GPU", "bsz=1", "bsz=2", "bsz=4", "bsz=8", "bsz=16", "bsz=32", "bsz=48"])
    for n_gpu in [1, 2, 4, 8, 16]:
        row = [str(n_gpu)+"_gpu"]
        for bsz in [1, 2, 4, 8, 16, 32, 48]:
            if str((n_gpu, bsz)) in ngpu_bsz_2_final_throughput:
                row.append(str(ngpu_bsz_2_final_throughput[str((n_gpu, bsz))])+"it/s")
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

    # rubble_16g_expes = [
    #     # "/pscratch/sd/j/jy-nyu/final_expes/rub_16g_1",
    #     # "/pscratch/sd/j/jy-nyu/final_expes/rub_16g_1_b16",
    #     "/pscratch/sd/j/jy-nyu/final_expes/rub_16g_2",
    #     "/pscratch/sd/j/jy-nyu/final_expes/rub_16g_3",
    #     "/pscratch/sd/j/jy-nyu/final_expes/rub_16g_4",
    #     "/pscratch/sd/j/jy-nyu/final_expes/rub_16g_5",
    #     "/pscratch/sd/j/jy-nyu/final_expes/rub_16g_6",
    #     "/pscratch/sd/j/jy-nyu/final_expes/rub_16g_7_c2",
    # ]
    # draw_n3dgs_metrics(rubble_16g_expes, plot_rubble_folder)

    # draw_speed("rub",
    #             "/pscratch/sd/j/jy-nyu/final_expes/rub_16g_7_speed/",
    #            plot_rubble_folder)


    draw_memory("rub",
                "/pscratch/sd/j/jy-nyu/final_expes/rub_memory",
                plot_rubble_folder)



    

    


if __name__ == "__main__":
    plot_rubble()
    pass

