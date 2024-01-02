import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pandas as pd

def log2json(path):
    file = os.path.join(path, "python.log")
    with open(file) as f:
        lines = f.readlines()

    epoch = 0
    data_cur_epoch = {}
    all_data = []
    for line in lines:
        if "epoch" in line:
            epoch += 1
            data_to_save = {}
            data_to_save["epoch"] = epoch
            for key in data_cur_epoch:
                values = data_cur_epoch[key]
                # get mean, max, min, std, median, 25%, 75% for values, make it a dict
                data_to_save[key] = {"mean": np.mean(values), "max": np.max(values), "min": np.min(values), "std": np.std(values), "median": np.median(values), "25%": np.percentile(values, 25), "75%": np.percentile(values, 75)}
            all_data.append(data_to_save)
            data_cur_epoch = {}

        if " sparsity" in line:# the " " is important. but it is very adhoc. 
            elems = line.strip("\n").split(":")
            key = elems[0].strip()
            value = float(elems[1].strip())
            if key not in data_cur_epoch:
                data_cur_epoch[key] = []
            data_cur_epoch[key].append(value)


    with open(os.path.join(path, "sparsity.json"), "w") as f:
        json.dump(all_data, f, indent=4)

def json2csv(file_path):
    with open(os.path.join(file_path, "sparsity.json")) as f:
        data = json.load(f)

    data_csv = []
    for d in data:
        one_data_csv = {}
        one_data_csv["epoch"] = d["epoch"]
        for key_I_want in ["mean", "std"]:
            for key in d:
                if key != "epoch":
                    one_data_csv[key + ":" + key_I_want] = round(d[key][key_I_want], 4)
        print(one_data_csv)
        data_csv.append(one_data_csv)

    df = pd.DataFrame(data_csv)
    df.to_csv(os.path.join(file_path, "sparsity.csv"), index=False)

def log_sparsity_1():
    log2json("experiments/log_sparsity_1/")

def log_sparsity_2():
    log2json("experiments/log_sparsity_2/")
    json2csv("experiments/log_sparsity_2/")


if __name__ == "__main__":
    # log_sparsity_1()
    log_sparsity_2()    

    pass

