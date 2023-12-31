import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def extract_excel(folder, file_paths, iteration):
    # extract frame from all data
    df = None
    for file_path in file_paths:
        if not os.path.exists(file_path):
            continue
        file_path = file_path.removesuffix(".log") + ".json"
        stats = []
        with open(file_path, 'r') as f:
            stats = json.load(f)

        data = None
        for stat in stats:
            if stat["iteration"] == iteration:
                data = stat
                break

        data_for_save = {}
        data_for_save["experiment"] = file_path.split("/")[-2]
        for key in data.keys():
            if key == "iteration":
                continue
            data_for_save[key] = data[key]

        if df is None:
            df = pd.DataFrame(data_for_save, index=[0])
        else:
            df = pd.concat([df, pd.DataFrame([data_for_save])], ignore_index=True)

    df.to_csv(folder + "time_stat_it="+ str(iteration) +".csv", index=False)

def extract_json_from_python_time_log(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    stats = []
    for line in lines:
        if line.startswith("iter"):
            parts = line.split(",")
            iteration = int(parts[0].split(" ")[1])

            if not stats or stats[-1]["iteration"] != iteration:
                stats.append({"iteration": iteration})
            # extract key and time from `TimeFor 'forward': 3.405571 ms`
            key = parts[1].split("'")[1]
            time = float(parts[1].split(":")[1].split(" ")[1])
            stats[-1][key] = time

    # save in file
    with open(file_path.removesuffix(".log") + ".json", 'w') as f:
        json.dump(stats, f, indent=4)
    return stats

def offload_speed0():
    file_names = [
        "experiments/offload_time0/time.log",
        "experiments/offload_no_time0/time.log",
    ]
    for file in file_names:
        extract_json_from_python_time_log(file)
    extract_excel("experiments/offload_time0/", file_names, 50)

def offload_speed1():
    file_names = [
        "experiments/offload_time1/time.log",
        "experiments/offload_no_time1/time.log",
    ]
    for file in file_names:
        extract_json_from_python_time_log(file)
    extract_excel("experiments/offload_time1/", file_names, 50)

if __name__ == "__main__":
    # offload_speed0()
    offload_speed1()




