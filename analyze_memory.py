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
    columnes = [[path.removeprefix("experiments/")+"[max]", path.removeprefix("experiments/")+"[now]"] for path in paths]
    columnes = sum(columnes, [])
    df = pd.DataFrame(columns=["prefix_mode"] + columnes)
    for i in range(0, n_rows, 2):
        row = [data[0][i]["prefix"]]
        for j in range(n_cols):
            assert data[j][i]["mode"] == "max" and data[j][i+1]["mode"] == "now", "mode error"
            row.append(data[j][i]["value"])
            row.append(data[j][i+1]["value"])
        df.loc[i//2] = row
    print(df)
    # delete columns that its name contains max
    df = df.loc[:,~df.columns.str.contains("max")]
    df.to_csv(os.path.join(paths[0], "statistics.csv"), index=False)
    

if __name__ == "__main__":
    log2json("experiments/offload_memory/")
    log2json("experiments/no_offload_memory/")

    paths = ["experiments/offload_memory/", "experiments/no_offload_memory/"]
    json2csv(paths)

    pass

