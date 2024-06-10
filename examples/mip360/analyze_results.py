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

def extract_from_mip360_all9scene(folder):
    # if os.path.exists(os.path.join(folder, "mip360_all9scene.json")):
    #     print("mip360_all9scene.json already exists for ", folder)
    #     return

    # counter kitchen room stump bicycle garden bonsai flowers treehill
    scene_names = [
        "counter",
        "kitchen",
        "room",
        "stump",
        "bicycle",
        "garden",
        "bonsai",
    ]
    check_iterations = [7000, 30000]
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

def plot_release_mip360(expe_folder_base):

    expe_sets = [
        "1g_1b",
        "4g_1b",
        "4g_4b",
    ]
    for expe_set in expe_sets:
        extract_from_mip360_all9scene(f"{expe_folder_base}/{expe_set}/")

    all_scenes = [
        "stump",
        "bicycle", 
        "kitchen", 
        "room", 
        "counter", 
        "garden", 
        "bonsai", 
    ]

    compare_iterations = ["7000", "30000"]
    unit_map = {
        "throughput": "its",
        "psnr": "dB",
        "running_time": "min"
    }

    for metric in ["running_time", "psnr"]:
        first_col_name = "30k Train Time(min)" if metric == "running_time" else "30k Test PSNR"
        for iter in compare_iterations:
            df = pd.DataFrame(columns=[first_col_name]+all_scenes)
            for i, expe in enumerate(["1 GPU + Batch Size=1", "4 GPU + Batch Size=1", "4 GPU + Batch Size=4"]):
                row = [expe]
                expe_folder = os.path.join(expe_folder_base, expe_sets[i])
                results = json.load(open(os.path.join(expe_folder, "mip360_all9scene.json"), "r"))
                for scene in all_scenes:
                    if metric == "running_time":
                        row.append(round(results[scene][iter]["running_time"]/60, 2))
                    else:
                        row.append(results[scene][iter][metric])
                df.loc[len(df)] = row
            df.to_markdown(os.path.join(expe_folder, f"mip360_compare_{metric}_{iter}.md"), index=False)

if __name__ == "__main__":
    # read args from command line
    expe_folder_base = sys.argv[1]
    print("Expe base folder: ", expe_folder_base)

    plot_release_mip360(expe_folder_base)


