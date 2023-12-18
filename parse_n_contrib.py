import json
import os
import numpy as np
import pandas as pd
import re


file_path = "experiments/n_contributors_3/n_contrib_ws=1_rk=0.log"

def parse_line_str2json_mode1(line):
    # example: "iteration: 301, iteration: 301, local_rank: 0, world_size: 1, num_local_tiles: 2170, global_num_rendered: 1142835, global_last_n_contrib: 390.050232, global_n_contrib2loss: 113.447990, contrib_ratio: 0.251814"    
    
    items = line.split(", ")
    ans = {"mode": "global"}
    for item in items:
        key, value = item.split(": ")
        ans[key] = value
    # print(ans)
    return ans

def parse_line_str2json_mode2(line):
    # line = "iteration: 351, tile: (0, 0), range: (102, 0), local_num_rendered: 102, local_last_n_contrib: 101.000000, local_real_n_contrib: 40.863281, contrib_ratio: 0.400620"
    
    # get iteration
    iteration = re.findall(r"iteration: (\d+)", line)[0]
    # get tile
    tile = re.findall(r"tile: \((\d+), (\d+)\)", line)[0]
    # get range
    range = re.findall(r"range: \((\d+), (\d+)\)", line)[0]
    # get local_num_rendered
    local_num_rendered = re.findall(r"local_num_rendered: (\d+)", line)[0]
    # get local_last_n_contrib, it is float and has decimal point
    local_last_n_contrib = re.findall(r"local_last_n_contrib: (\d+\.\d+)", line)[0]
    # get local_real_n_contrib
    local_real_n_contrib = re.findall(r"local_real_n_contrib: (\d+\.\d+)", line)[0]
    # get contrib_ratio
    contrib_ratio = re.findall(r"contrib_ratio: (\d+\.\d+)", line)[0]
    ans = {"mode": "local", "iteration": iteration, "tile": tile, "range": range, "local_num_rendered": local_num_rendered, "local_last_n_contrib": local_last_n_contrib, "local_real_n_contrib": local_real_n_contrib, "contrib_ratio": contrib_ratio}
    # print(ans)
    return ans

def prepare():
    with open(file_path, 'r') as f:
        lines = f.readlines()

    all_json = []
    for line in lines:
        # iteration in line twice
        x = line.strip("\n")
        if x.count("iteration") == 2:
            all_json.append(parse_line_str2json_mode1(x))
        else:
            all_json.append(parse_line_str2json_mode2(x))

    
    # save to json
    with open(file_path.replace(".log", ".json"), 'w') as f:
        json.dump(all_json, f, indent=4)

def analysis():
    with open(file_path.replace(".log", ".json"), 'r') as f:
        data = json.load(f)
    
    # get all iterations
    iteration = 1

    all_local_num_rendered = []
    all_local_real_n_contrib = []
    df = pd.DataFrame(columns=["iteration", "mean_local_num_rendered", "mean_local_real_n_contrib", "std_local_num_rendered", "std_local_real_n_contrib", "max_local_num_rendered", "max_local_real_n_contrib", "min_local_num_rendered", "min_local_real_n_contrib", "median_local_num_rendered", "median_local_real_n_contrib", "percentile25_local_num_rendered", "percentile25_local_real_n_contrib", "percentile75_local_num_rendered", "percentile75_local_real_n_contrib"])
    for item in data:
        if item["mode"] == "global":
            # get mean, variance, max, min, median, 25%, 75%
            mean_local_num_rendered = np.mean(all_local_num_rendered)
            mean_local_real_n_contrib = np.mean(all_local_real_n_contrib)
            std_local_num_rendered = np.std(all_local_num_rendered)
            std_local_real_n_contrib = np.std(all_local_real_n_contrib)
            max_local_num_rendered = np.max(all_local_num_rendered)
            max_local_real_n_contrib = np.max(all_local_real_n_contrib)
            min_local_num_rendered = np.min(all_local_num_rendered)
            min_local_real_n_contrib = np.min(all_local_real_n_contrib)
            median_local_num_rendered = np.median(all_local_num_rendered)
            median_local_real_n_contrib = np.median(all_local_real_n_contrib)
            percentile25_local_num_rendered = np.percentile(all_local_num_rendered, 25)
            percentile25_local_real_n_contrib = np.percentile(all_local_real_n_contrib, 25)
            percentile75_local_num_rendered = np.percentile(all_local_num_rendered, 75)
            percentile75_local_real_n_contrib = np.percentile(all_local_real_n_contrib, 75)

            # save in dataframe
            df = df._append({"iteration": int(iteration), "mean_local_num_rendered": mean_local_num_rendered, "mean_local_real_n_contrib": mean_local_real_n_contrib, "std_local_num_rendered": std_local_num_rendered, "std_local_real_n_contrib": std_local_real_n_contrib, "max_local_num_rendered": max_local_num_rendered, "max_local_real_n_contrib": max_local_real_n_contrib, "min_local_num_rendered": min_local_num_rendered, "min_local_real_n_contrib": min_local_real_n_contrib, "median_local_num_rendered": median_local_num_rendered, "median_local_real_n_contrib": median_local_real_n_contrib, "percentile25_local_num_rendered": percentile25_local_num_rendered, "percentile25_local_real_n_contrib": percentile25_local_real_n_contrib, "percentile75_local_num_rendered": percentile75_local_num_rendered, "percentile75_local_real_n_contrib": percentile75_local_real_n_contrib}, ignore_index=True)

            # clear
            iteration += 50
            all_local_num_rendered = []
            all_local_real_n_contrib = []
            continue
        else:
            all_local_num_rendered.append(float(item["local_num_rendered"]))
            all_local_real_n_contrib.append(float(item["local_real_n_contrib"]))

    df.to_csv(file_path.replace(".log", ".csv"), index=False)

analysis()
# prepare()



