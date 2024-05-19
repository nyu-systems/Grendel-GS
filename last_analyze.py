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
    if not os.path.exists(os.path.join(folder, "results_test.json")):
        return None
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
        if result is None:
            continue
        n3dgs = get_final_n3dgs_from_log(folder)
        # all_results.append(get_results_test(folder))
        # all_n3dgs.append(get_final_n3dgs_from_log(folder))
        expe_name = folder.split("/")[-1]
        # if expe_name == "rub_16g_7_c2":
        #     expe_name = "rub_16g_7"
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
    # convert to latex
    convert_df_to_latex(df, os.path.join(save_folder, "n3dgs_metrics.tex"), drop_first_column=False)

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

def convert_df_to_latex(df, save_path, drop_first_column=True):
    # delete the first colume of df
    df_latex = df.copy()
    if drop_first_column:
        df_latex = df_latex.drop(df_latex.columns[0], axis=1)
    # go throught each element, replace _ with space
    for i in range(df_latex.shape[0]):
        for j in range(df_latex.shape[1]):
            if df_latex.iat[i, j] is not None and isinstance(df_latex.iat[i, j], str):
                df_latex.iat[i, j] = df_latex.iat[i, j].replace("_", " ")
            # round to 2
            # if df_latex.iat[i, j] is not None and isinstance(df_latex.iat[i, j], float):
            #     df_latex.iat[i, j] = round(df_latex.iat[i, j], 2)
    # go through each column name, replace _ with space
    for i in range(df_latex.shape[1]):
        df_latex.columns.values[i] = df_latex.columns.values[i].replace("_", " ")
    df_latex.to_latex(save_path, index=False)

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
            if len(ngpu_bsz_2_throughput[str((n_gpu, bsz))]) < 5:
                continue
            ngpu_bsz_2_1stepoch_throughput[str((n_gpu, bsz))] = ngpu_bsz_2_throughput[str((n_gpu, bsz))][0]
            ngpu_bsz_2_2rdepoch_throughput[str((n_gpu, bsz))] = ngpu_bsz_2_throughput[str((n_gpu, bsz))][1]
            ngpu_bsz_2_final_throughput[str((n_gpu, bsz))] = ngpu_bsz_2_throughput[str((n_gpu, bsz))][2]
            ngpu_bsz_2_final_ave_throughput[str((n_gpu, bsz))] = round(sum(ngpu_bsz_2_throughput[str((n_gpu, bsz))][3:6]) / 3, 3)
    json.dump(ngpu_bsz_2_1stepoch_throughput, open(os.path.join(save_folder, f"{scene_name}_speed_1stepoch.json"), "w"), indent=4)
    json.dump(ngpu_bsz_2_2rdepoch_throughput, open(os.path.join(save_folder, f"{scene_name}_speed_2rdepoch.json"), "w"), indent=4)
    json.dump(ngpu_bsz_2_final_throughput, open(os.path.join(save_folder, f"{scene_name}_speed_final.json"), "w"), indent=4)
    json.dump(ngpu_bsz_2_final_ave_throughput, open(os.path.join(save_folder, f"{scene_name}_speed_final_ave.json"), "w"), indent=4)

    # Draw a dataframe for the ngpu_bsz_2_2rdepoch_throughput, Rows are ngpu, Columns are bsz
    df = pd.DataFrame(columns=["GPU count", "bsz=1", "bsz=2", "bsz=4", "bsz=8", "bsz=16", "bsz=32", "bsz=64"])
    for n_gpu in [1, 2, 4, 8, 16, 32]:
        row = [str(n_gpu)+"_gpu"]
        for bsz in [1, 2, 4, 8, 16, 32, 64]:
            if str((n_gpu, bsz)) in ngpu_bsz_2_2rdepoch_throughput:
                row.append(str(ngpu_bsz_2_2rdepoch_throughput[str((n_gpu, bsz))])+"it/s")
            else:
                row.append(None)
        df.loc[n_gpu] = row
    df.to_csv(os.path.join(save_folder, f"{scene_name}_speed_without_loadbalancing.csv"))
    convert_df_to_latex(df, os.path.join(save_folder, f"{scene_name}_speed_without_loadbalancing.tex"), drop_first_column=False)

    # Draw a dataframe for the ngpu_bsz_2_final_ave_throughput, Rows are ngpu, Columns are bsz
    df = pd.DataFrame(columns=["GPU count", "bsz=1", "bsz=2", "bsz=4", "bsz=8", "bsz=16", "bsz=32", "bsz=64"])
    for n_gpu in [1, 2, 4, 8, 16, 32]:
        row = [str(n_gpu)+"_gpu"]
        for bsz in [1, 2, 4, 8, 16, 32, 64]:
            if str((n_gpu, bsz)) in ngpu_bsz_2_final_ave_throughput:
                row.append(str(ngpu_bsz_2_final_ave_throughput[str((n_gpu, bsz))])+"it/s")
            else:
                row.append(None)
        df.loc[n_gpu] = row
    df.to_csv(os.path.join(save_folder, f"{scene_name}_speed_with_loadbalancing.csv"))
    convert_df_to_latex(df, os.path.join(save_folder, f"{scene_name}_speed_with_loadbalancing.tex"), drop_first_column=False)

def draw_memory(scene_name, folder, save_folder):

    ngpu_bsz_2_n3dgs = {}

    for bsz in [1, 4, 16]:
        for n_gpu in [1, 2, 4, 8, 16, 32]:
            expe_name = f"{scene_name}_mem_{n_gpu}g_{bsz}b"
            expe_folder = os.path.join(folder, expe_name)
            if not os.path.exists(expe_folder):
                print(f"Expe {expe_name} does not exist")
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
    df = pd.DataFrame(columns=["GPU count", "bsz=1", "bsz=4", "bsz=16"])
    for n_gpu in [1, 2, 4, 8, 16, 32]:
        row = [str(n_gpu)+"_gpu"]
        for bsz in [1, 4, 16]:
            if str((n_gpu, bsz)) in ngpu_bsz_2_n3dgs:
                row.append(str(round(ngpu_bsz_2_n3dgs[str((n_gpu, bsz))][0]/1000000, 2))+" millions gaussian")
            else:
                row.append(None)
        df.loc[n_gpu] = row
    df.to_csv(os.path.join(save_folder, f"{scene_name}_memory.csv"))
    convert_df_to_latex(df, os.path.join(save_folder, f"{scene_name}_memory.tex"), drop_first_column=False)

    pass


def plot_rubble():
    plot_rubble_folder = "/pscratch/sd/j/jy-nyu/last_scripts/plot_rubble/"
    if not os.path.exists(plot_rubble_folder):
        os.makedirs(plot_rubble_folder)
    rubble_16g_folder = "/pscratch/sd/j/jy-nyu/last_scripts/rub/"

    # list all experiments folders in the rubble_expe_folder
    rubble_16g_expes = []
    for expe in os.listdir(rubble_16g_folder):
        expe_folder = os.path.join(rubble_16g_folder, expe)
        if os.path.isdir(expe_folder):
            rubble_16g_expes.append(expe_folder)

    # draw_n3dgs_metrics(rubble_16g_expes, plot_rubble_folder)

    # draw_speed("rub",
    #             "/pscratch/sd/j/jy-nyu/final_expes/rub_16g_7_speed_saved/",
    #            plot_rubble_folder)


    draw_memory("rub",
                "/pscratch/sd/j/jy-nyu/final_expes/rub_memory",
                plot_rubble_folder)

def plot_bicycle():
    plot_bicycle_folder = "/pscratch/sd/j/jy-nyu/last_scripts/plot_bicycle/"
    if not os.path.exists(plot_bicycle_folder):
        os.makedirs(plot_bicycle_folder)
    bicycle_folder = "/pscratch/sd/j/jy-nyu/last_scripts/bicycle/"

    bicycle_expes = []
    for expe in os.listdir(bicycle_folder):
        expe_folder = os.path.join(bicycle_folder, expe)
        if os.path.isdir(expe_folder):
            bicycle_expes.append(expe_folder)
    
    draw_n3dgs_metrics(bicycle_expes, plot_bicycle_folder)

def plot_matrixcity_blockall():
    plot_mat_folder = "/pscratch/sd/j/jy-nyu/last_scripts/plot_mat/"
    if not os.path.exists(plot_mat_folder):
        os.makedirs(plot_mat_folder)
    matrixcity_blockall_folder = "/pscratch/sd/j/jy-nyu/last_scripts/mball2/"

    # list all experiments folders in the rubble_expe_folder
    matrixcity_blockall_expes = []
    for expe in os.listdir(matrixcity_blockall_folder):
        expe_folder = os.path.join(matrixcity_blockall_folder, expe)
        if os.path.isdir(expe_folder):
            matrixcity_blockall_expes.append(expe_folder)

    # draw_n3dgs_metrics(matrixcity_blockall_expes, plot_mat_folder)
    # save the n3dgs at the beginning and at the end, in file
    n3dgs = {}
    for expe_folder in matrixcity_blockall_expes:
        n3dgs_at_iterations, iterations = get_n3dgs_list_from_log(expe_folder)
        # print(f"Expe {expe_folder} has {n3dgs_at_iterations[0]} n3dgs at the beginning")
        # print(f"Expe {expe_folder} has {n3dgs_at_iterations[-1]} n3dgs at the end")
        n3dgs[expe_folder] = {
            "n3dgs_at_beginning": n3dgs_at_iterations[0],
            "n3dgs_at_end": n3dgs_at_iterations[-1]
        }
    json.dump(n3dgs, open(os.path.join(plot_mat_folder, "n3dgs.json"), "w"), indent=4)




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
                "psnr": psnr,
                "throughput": round(iteration/running_time, 2)
            }

    json.dump(results, open(os.path.join(folder, "mip360_all9scene.json"), "w"), indent=4)
    print("Generated mip360_all9scene.json for ", folder)



def plot_mip360():

    expe_sets = [
        "1g_1b",
        "4g_1b",
        "4g_1b_nobalan",
        "4g_1b_nogsbalan",
        "4g_1b_noimgbalan",
        "4g_4b",
        "4g_4b_nobalan",
        "4g_4b_noimgbalan",
    ]
    for expe_set in expe_sets:
        extract_from_mip360_all9scene(f"/pscratch/sd/j/jy-nyu/last_scripts/mip360_1080p/{expe_set}/")

    all_scenes = ["counter", "kitchen", "room", "stump", "bicycle", "garden", "bonsai", "flowers", "treehill"]

    analyze_360v21080_folder = "/pscratch/sd/j/jy-nyu/last_scripts/mip360_1080p_analyze/"
    if not os.path.exists(analyze_360v21080_folder):
        os.makedirs(analyze_360v21080_folder)
    
    # compare_different_gpu_and_different_bsz
    compare_iterations = ["7000", "30000", "50000"]
    unit = {
        "throughput": "its",
        "psnr": "dB",
        "running_time": "second"
    }

    for iteration in compare_iterations:
        for metric in ["throughput", "psnr"]:
            df = pd.DataFrame(columns=["Scene", "1gpu_bsz=1", "4gpu_bsz=1", "4gpu_bsz=4"])
            for scene in all_scenes:
                row = [scene]
                for expe in ["1g_1b", "4g_1b", "4g_4b_noimgbalan"]:
                    expe_folder = os.path.join("/pscratch/sd/j/jy-nyu/last_scripts/mip360_1080p/", expe)
                    results = json.load(open(os.path.join(expe_folder, "mip360_all9scene.json"), "r"))
                    row.append(results[scene][iteration][metric])
                df.loc[len(df)] = row
            df.to_csv(os.path.join(analyze_360v21080_folder, f"mip360_compare_{metric}_{iteration}_{unit[metric]}.csv"))

    # check load balance for 4gpu1bsz
    for iteration in compare_iterations:
        for metric in ["throughput", "psnr"]:
            df = pd.DataFrame(columns=["Scene", "both_gausssian_image_loadbalance", "only_gausssian_loadbalance", "only_image_loadbalance", "no_loadbalance"])
            for scene in all_scenes:
                row = [scene]
                for expe in ["4g_1b", "4g_1b_noimgbalan", "4g_1b_nogsbalan", "4g_1b_nobalan"]:
                    expe_folder = os.path.join("/pscratch/sd/j/jy-nyu/last_scripts/mip360_1080p/", expe)
                    results = json.load(open(os.path.join(expe_folder, "mip360_all9scene.json"), "r"))
                    row.append(results[scene][iteration][metric])
                df.loc[len(df)] = row
            df.to_csv(os.path.join(analyze_360v21080_folder, f"mip360_compare_4gpu_bsz=1_loadbalance_{metric}_{iteration}_{unit[metric]}.csv"))

    # check load balance for 4gpu4bsz
    for iteration in compare_iterations:
        for metric in ["throughput", "psnr"]:
            df = pd.DataFrame(columns=["Scene", "both_gausssian_image_loadbalance", "only_gausssian_loadbalance", "no_loadbalance"])
            for scene in all_scenes:
                row = [scene]
                for expe in ["4g_4b", "4g_4b_noimgbalan", "4g_4b_nobalan"]:
                    expe_folder = os.path.join("/pscratch/sd/j/jy-nyu/last_scripts/mip360_1080p/", expe)
                    results = json.load(open(os.path.join(expe_folder, "mip360_all9scene.json"), "r"))
                    row.append(results[scene][iteration][metric])
                df.loc[len(df)] = row
            df.to_csv(os.path.join(analyze_360v21080_folder, f"mip360_compare_4gpu_bsz=4_loadbalance_{metric}_{iteration}_{unit[metric]}.csv"))

    # ablation study
    for iteration in ["50000"]:
        for metric in ["throughput", "psnr"]:
            df = pd.DataFrame(columns=["Scene", "1gpu_bsz=1", "Loadbalanced_4gpu_bsz=4", "Loadbalanced_4gpu_bsz=1",  "NoLoadbalance_4gpu_bsz=1",  "OnlyGaussianLoadbalance_4gpu_bsz=1", "OnlyImageLoadbalance_4gpu_bsz=1"])
            for scene in all_scenes:
                row = [scene]
                for expe in ["1g_1b", "4g_4b_noimgbalan", "4g_1b", "4g_1b_nobalan", "4g_1b_noimgbalan", "4g_1b_nogsbalan"]:
                    expe_folder = os.path.join("/pscratch/sd/j/jy-nyu/last_scripts/mip360_1080p/", expe)
                    results = json.load(open(os.path.join(expe_folder, "mip360_all9scene.json"), "r"))
                    row.append(results[scene][iteration][metric])
                df.loc[len(df)] = row
            df.to_csv(os.path.join(analyze_360v21080_folder, f"mip360_rawstatistics_{metric}_{iteration}_{unit[metric]}.csv"))
            
            to_latex_df = df[["Scene", "1gpu_bsz=1", "Loadbalanced_4gpu_bsz=4", "Loadbalanced_4gpu_bsz=1",  "NoLoadbalance_4gpu_bsz=1",  "OnlyGaussianLoadbalance_4gpu_bsz=1", "OnlyImageLoadbalance_4gpu_bsz=1"]]
            to_latex_df = to_latex_df.astype(str)
            to_latex_df.to_latex(os.path.join(analyze_360v21080_folder, f"mip360_rawstatistics_{metric}_{iteration}_{unit[metric]}.tex"), index=False)
    
def extract_from_tandb_all4scene(folder):
    if os.path.exists(os.path.join(folder, "tandb_all4scene.json")):
        print("tandb_all4scene.json already exists for ", folder)
        return
    # train truck playroom drjohnson
    scene_names = [
        "train",
        "truck",
        "playroom",
        "drjohnson"
    ]
    check_iterations = [7000, 15000, 30000]
    results = {}
    for scene in scene_names:
        scene_folder = os.path.join(folder, scene)
        if not os.path.exists(scene_folder):
            continue
        running_time_all = get_running_time_at_iterations(scene_folder, check_iterations)
        psnr_all = get_test_psnr_at_iterations(scene_folder, check_iterations)
        results[scene] = {}
        for iteration, running_time, psnr in zip(check_iterations, running_time_all, psnr_all):
            results[scene][iteration] = {
                "running_time": running_time,
                "psnr": psnr,
                "throughput": round(iteration/running_time, 2)
            }

    json.dump(results, open(os.path.join(folder, "tandb_all4scene.json"), "w"), indent=4)
    print("Generated tandb_all4scene.json for ", folder)

def plot_tandb():
    plot_tan_folder = "/pscratch/sd/j/jy-nyu/last_scripts/plot_tandb/"
    if not os.path.exists(plot_tan_folder):
        os.makedirs(plot_tan_folder)

    scene_names = [
        "train",
        "truck",
        "playroom",
        "drjohnson"
    ]

    expe_sets = [
        "1g1b",
        "4g4b",
        "4g8b",
        "4g16b"
    ]

    for expe_set in expe_sets:
        extract_from_tandb_all4scene(f"/pscratch/sd/j/jy-nyu/last_scripts/tandb/{expe_set}/")
    
    analyze_tandb_folder = "/pscratch/sd/j/jy-nyu/last_scripts/tandb_analyze/"
    if not os.path.exists(analyze_tandb_folder):
        os.makedirs(analyze_tandb_folder)
    
    # compare_different_gpu_and_different_bsz
    compare_iterations = ["7000", "15000", "30000"]
    unit = {
        "throughput": "its",
        "psnr": "dB",
        "running_time": "second"
    }

    for iteration in compare_iterations:
        for metric in ["throughput", "psnr"]:
            df = pd.DataFrame(columns=["Scene", "1gpu_bsz=1", "4gpu_bsz=4", "4gpu_bsz=8", "4gpu_bsz=16"])
            for scene in scene_names:
                row = [scene]
                for expe in expe_sets:
                    expe_folder = os.path.join("/pscratch/sd/j/jy-nyu/last_scripts/tandb/", expe)
                    results = json.load(open(os.path.join(expe_folder, "tandb_all4scene.json"), "r"))
                    row.append(results[scene][iteration][metric])
                df.loc[len(df)] = row
            df.to_csv(os.path.join(analyze_tandb_folder, f"tandb_compare_{metric}_{iteration}_{unit[metric]}.csv"))
            convert_df_to_latex(df, os.path.join(analyze_tandb_folder, f"tandb_compare_{metric}_{iteration}_{unit[metric]}.tex"), drop_first_column=False)

def extract_from_some_expes(expe_paths, check_iterations):
    results = {}
    for expe_folder in expe_paths:
        results[expe_folder] = {}
        running_time_all = get_running_time_at_iterations(expe_folder, check_iterations)
        psnr_all = get_test_psnr_at_iterations(expe_folder, check_iterations)
        for iteration, running_time, psnr in zip(check_iterations, running_time_all, psnr_all):
            results[expe_folder][iteration] = {
                "running_time": running_time,
                "psnr": psnr,
                "throughput": round(iteration/running_time, 2)
            }
    return results
    


def plot_tandb_train_scalability():
    folder = "/pscratch/sd/j/jy-nyu/last_scripts/tandb/scalability/"
    analyze_folder = "/pscratch/sd/j/jy-nyu/last_scripts/tandb_analyze/scalability/"
    if not os.path.exists(analyze_folder):
        os.makedirs(analyze_folder)

    all_expes = []
    for n_g in [1, 4, 8, 16]:
        for bsz in [1, 2, 4, 8, 16, 32]:
            # train_16g_16b
            expe_name = f"train_{n_g}g_{bsz}b"
            if os.path.exists(os.path.join(folder, expe_name)):
                all_expes.append(expe_name)


    check_iterations = [7000, 15000, 30000]
    results = extract_from_some_expes([os.path.join(folder, expe) for expe in all_expes], check_iterations)
    json.dump(results, open(os.path.join(folder, "tandb_train_scalability.json"), "w"), indent=4)
    
    compare_iterations = ["7000", "15000", "30000"]
    unit = {
        "throughput": "its",
        "psnr": "dB",
        "running_time": "second"
    }
    columes = []

    for metric in ["throughput", "psnr"]:
        for iteration in compare_iterations:
            columes.append(f"{metric}_{iteration}")
    
    df = pd.DataFrame(columns=["Expe"] + columes)
    for expe in all_expes:
        row = [expe]
        for metric in ["throughput", "psnr"]:
            for iteration in compare_iterations:
                row.append(results[os.path.join(folder, expe)][int(iteration)][metric])
        df.loc[len(df)] = row


    df.to_csv(os.path.join(analyze_folder, "tandb_train_scalability.csv"))
    convert_df_to_latex(df, os.path.join(analyze_folder, "tandb_train_scalability.tex"), drop_first_column=False)





if __name__ == "__main__":
    # plot_rubble()
    plot_bicycle()
    # plot_mip360()
    # plot_matrixcity_blockall()
    # plot_tandb()
    # plot_tandb_train_scalability()

    pass

