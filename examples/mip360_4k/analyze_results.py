import json
import pandas as pd
import os
import sys

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
                results.append(round(PSNR, 2))
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
            "L1": round(L1, 2),
            "PSNR": round(PSNR, 2)
        })
    return results

def get_max_memory_at_iterations(expe_folder, check_iterations):
    a_suffix = get_suffix_in_folder(expe_folder)[0]
    a_log_file = os.path.join(expe_folder, f"python_{a_suffix}.log")
    lines = open(a_log_file, "r").readlines()
    results = []
    for line in lines:
        # iteration[50000,50001) densify_and_prune. Now num of 3dgs: 1322628. Now Memory usage: 1.4657483100891113 GB. Max Memory usage: 4.159056186676025 GB. Max Reserved Memory: 8.357421875 GB. Now Reserved Memory: 2.44921875 GB. 
        if "Max Reserved Memory: " not in line:
            continue
        iteration_start, iteration_end = line.split("iteration[")[1].split(")")[0].split(",")
        iteration_start, iteration_end = int(iteration_start), int(iteration_end)
        max_memory = float(line.split("Max Reserved Memory: ")[1].split(" GB")[0])
        for r in range(iteration_start, iteration_end):
            if r in check_iterations:
                results.append(round(max_memory, 2))
    return results

def extract_from_mip360_all9scene(folder):
    # if os.path.exists(os.path.join(folder, "mip360_all9scene.json")):
    #     print("mip360_all9scene.json already exists for ", folder)
    #     return

    scene_names = [
        "bicycle", 
        "garden", 
    ]
    check_iterations = [7000, 30000]
    results = {}
    for scene in scene_names:
        scene_folder = os.path.join(folder, "e_"+scene)
        if not os.path.exists(scene_folder):
            continue
        running_time_all = get_running_time_at_iterations(scene_folder, check_iterations)
        psnr_all = get_test_psnr_at_iterations(scene_folder, check_iterations)
        memory_all = get_max_memory_at_iterations(scene_folder, check_iterations)
        results[scene] = {}
        for iteration, running_time, psnr, max_memory in zip(check_iterations, running_time_all, psnr_all, memory_all):
            results[scene][iteration] = {
                "running_time": running_time,
                "psnr": psnr,
                "throughput": round(iteration/running_time, 2),
                "max_memory": max_memory
            }

    json.dump(results, open(os.path.join(folder, "mip360_all9scene.json"), "w"), indent=4)
    print("Generated mip360_all9scene.json for ", folder)

def convert2readable(seconds):
    # 3h 30min
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    return f"{hours}h {minutes}min"

def plot_release_mip360_4k(expe_folder_base):

    expe_sets = [
        "1g_1b_4k",
        "4g_1b_4k",
    ]
    for expe_set in expe_sets:
        extract_from_mip360_all9scene(f"{expe_folder_base}/{expe_set}/")

    all_scenes = [
        "bicycle", 
        "garden", 
    ]

    df = pd.DataFrame(columns=["Configuration", "50k Training Time", "Memory Per GPU", "PSNR"])

    for scene in all_scenes:
        for i, expe in enumerate(["1 GPU + Batch Size=1", "4 GPU + Batch Size=1"]):
            row = [scene + " + " + expe]

            expe_folder = os.path.join(expe_folder_base, expe_sets[i])
            results = json.load(open(os.path.join(expe_folder, "mip360_all9scene.json"), "r"))

            row.append(convert2readable(results[scene]["30000"]["running_time"]))
            row.append(results[scene]["30000"]["max_memory"])
            row.append(results[scene]["30000"]["psnr"])
            df.loc[len(df)] = row
    df.to_markdown(os.path.join(expe_folder_base, f"mip360_4k_compare_{scene}.md"), index=False)

if __name__ == "__main__":
    # read args from command line
    expe_folder_base = sys.argv[1]
    print("Expe base folder: ", expe_folder_base)

    plot_release_mip360_4k(expe_folder_base)



