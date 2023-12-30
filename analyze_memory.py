import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pandas as pd

def log2json(path):
    file = os.path.join(path, "python.log")
    with open(file) as f:
        lines = f.readlines()
    data = []
    for line in lines:
        if line.startswith("memory"):
            elems = line.strip("\n").split(",")
            # print(elems)
            # change to json
            # memory,before everything,max,0.0,MB
            data.append({"prefix": elems[1], "mode": elems[2], "value": round(float(elems[3]), 1)})
    print(data)
    with open(os.path.join(path, "memory.json"), "w") as f:
        json.dump(data, f, indent=4)

def json2csv(paths):
    data = []
    for path in paths:
        with open(os.path.join(path, "memory.json")) as f:
            data.append(json.load(f))
    
    n_rows = len(data[0])
    n_cols = len(data)
    print(n_rows, n_cols)
    # 3 columns: prefix+mode, value
    # columnes = [[path.removeprefix("experiments/")+"[max]", path.removeprefix("experiments/")+"[now]"] for path in paths]
    # columnes = sum(columnes, [])
    columnes = [path.removeprefix("experiments/")+"[now]" for path in paths]
    df = pd.DataFrame(columns=["prefix_mode"] + columnes)
    rows = []
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j]["prefix"] not in rows:
                rows.append(data[i][j]["prefix"])

    for name in rows:
        row = [name]
        for i in range(len(data)):
            found = False
            for j in range(len(data[i])):
                if data[i][j]["prefix"] == name and data[i][j]["mode"] == "now":
                    row.append(data[i][j]["value"])
                    found = True
                    break
            if not found:
                row.append(-1)
        df.loc[len(df)] = row

    print(df)
    # delete columns that its name contains max
    # df = df.loc[:,~df.columns.str.contains("max")]
    df.to_csv(os.path.join(paths[0], "statistics.csv"), index=False)

def offload_memory():
    log2json("experiments/offload_memory/")
    log2json("experiments/no_offload_memory/")

    paths = ["experiments/offload_memory/", "experiments/no_offload_memory/"]
    json2csv(paths)

def more_3dgs_no_offload():
    log2json("experiments/more_3dgs_no_offload/")
    log2json("experiments/more_3dgs_offload/")

    paths = ["experiments/more_3dgs_no_offload/", "experiments/more_3dgs_offload/"]
    json2csv(paths)

def offload_image_dataset():
    log2json("experiments/offload_image_dataset/")
    log2json("experiments/more_3dgs_offload/")
    log2json("experiments/more_3dgs_no_offload/")

    paths = ["experiments/offload_image_dataset/", "experiments/more_3dgs_offload/", "experiments/more_3dgs_no_offload/"]
    json2csv(paths)

if __name__ == "__main__":
    offload_memory()
    more_3dgs_no_offload()
    offload_image_dataset()

    pass

