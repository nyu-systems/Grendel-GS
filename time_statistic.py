import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

folder = "experiments/timertest/"
file_names = [
    "gpu_time_ws=1_rk=0.log",
    "gpu_time_ws=2_rk=0.log",
    "gpu_time_ws=2_rk=1.log",
    "time_ws=1_rk=0.log",
    "time_ws=2_rk=0.log",
    "time_ws=2_rk=1.log",
]

num_render_file_names = [
    "num_rendered_ws=1_rk=0.log",
    "num_rendered_ws=2_rk=0.log",
    "num_rendered_ws=2_rk=1.log",
    "num_rendered_ws=1_rk=0.log",
    "num_rendered_ws=2_rk=0.log",
    "num_rendered_ws=2_rk=1.log",
]

def read_file(file_name, num_render_file_name):
    file_path = folder + file_name

    # get rk and ws
    rk = int(file_name[file_name.find("rk=")+3])
    ws = int(file_name[file_name.find("ws=")+3])
    prefix = file_name[:file_name.find("ws=")]
    print("rk: ", rk, "ws: ", ws)


    with open(file_path, 'r') as file:
        file_contents = file.readlines()
    
    # Function to parse each line and extract the statistic and its value
    def parse_line(line):
        # 10 preprocess time: 0.291950 ms
        parts = line.split(":")

        if len(parts) != 2:
            print("Error parsing line: ", line)
            return None

        # Extracting the statistic name and its value
        stat_name = parts[0]
        stat_value = float(parts[1].strip().split("ms")[0].strip())
        return stat_name, stat_value

    # Parsing the file and constructing the JSON object
    stats_json = []
    last_iteration = -1

    for line in file_contents:
        if line.startswith("it="):
            iteration = int(line[3:line.find(",")])
            if iteration != last_iteration:
                last_iteration = iteration
            else:
                continue
            stats_json.append({"iteration": iteration, "prefix": prefix, "rk": rk, "ws": ws})
            continue

        print(line)
        parsed_data = parse_line(line)
        if parsed_data:
            stat_name, stat_value = parsed_data
            stats_json[-1][stat_name] = stat_value

    with open(folder + num_render_file_name, 'r') as file:
        num_render_file_contents = file.readlines()

    idx = 0
    for line in num_render_file_contents:
        # line format:`iteration: 251, num_local_tiles: 398, local_tiles_left_idx: 742, local_tiles_right_idx: 1139, last_local_num_rendered_end: 336199, local_num_rendered_end: 672398, num_rendered: 335791, num_rendered_from_distState: 335791`
        # extract iteration, num_local_tiles, num_rendered
        iteration = int(line[line.find("iteration:")+len("iteration: "):line.find(",")])
        num_local_tiles = int(line[line.find("num_local_tiles:")+len("num_local_tiles: "):line.find(", local_tiles_left_idx:")])
        num_rendered = int(line[line.find("num_rendered:")+len("num_rendered: "):line.find(", num_rendered_from_distState:")])
        assert iteration == stats_json[idx]["iteration"]
        stats_json[idx]["num_local_tiles"] = num_local_tiles
        stats_json[idx]["num_rendered"] = num_rendered
        idx += 1

    # Converting the JSON object to a string for display
    json_data = json.dumps(stats_json, indent=4)
    print(json_data)
    return stats_json

def extract_stats_from_file():
    file2stats = {}
    assert len(file_names) == len(num_render_file_names), "file_names and num_render_file_names should have same length"
    for file_name, num_render_file_name in zip(file_names, num_render_file_names):

        if not os.path.exists(folder + file_name):
            continue        

        file2stats[file_name] = read_file(file_name, num_render_file_name)

        # save in file
        with open(folder + file_name.removesuffix(".log") + ".json", 'w') as f:
            json.dump(file2stats[file_name], f, indent=4)

def draw_graph(file_name, iteration):
    file_path = folder + file_name
    stats = []
    with open(file_path, 'r') as f:
        stats = json.load(f)

    data = None
    for stat in stats:
        if stat["iteration"] == iteration:
            data = stat
            break
    del data["iteration"]
    del data["rk"]
    del data["ws"]
    del data["00 forward time"]
    del data["b00 backward time"]
    del data["21 updateDistributedStatLocally.getGlobalGaussianOnTiles time"]
    del data["22 updateDistributedStatLocally.InclusiveSum time"]
    del data["23 updateDistributedStatLocally.getComputeLocally time"]
    del data["24 updateDistributedStatLocally.updateTileTouched time"]

    # draw a pie chart for the data
    labels = data.keys()
    sizes = data.values()
    # make character smaller
    plt.rc('font', size=6)
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True)
    ax1.axis('equal')
    plt.title(file_name)
    fig_name = file_name.split(".json")[0] + "_it=" + str(iteration) + ".png"
    plt.savefig(folder + fig_name)

def extract_excel(iteration):
    # extract frame from all data
    df = None
    for file_name in file_names:
        if not os.path.exists(folder + file_name):
            continue
        file_path = folder + file_name.removesuffix(".log") + ".json"
        stats = []
        with open(file_path, 'r') as f:
            stats = json.load(f)

        data = None
        for stat in stats:
            if stat["iteration"] == iteration:
                data = stat
                break

        # del data["rk"]
        # del data["ws"]
        data_for_save = {}
        data_for_save["rk"] = data["rk"]
        data_for_save["ws"] = data["ws"]
        for key in data.keys():
            if key == "iteration" or key == "rk" or key == "ws":
                continue
            data_for_save[key] = data[key]

        # print(data)
        if df is None:
            df = pd.DataFrame(data_for_save, index=[0])
        else:
            df = pd.concat([df, pd.DataFrame([data_for_save])], ignore_index=True)
            # df = df.append(data, ignore_index=True)


    df.to_csv(folder + "time_stat_it="+ str(iteration) +".csv", index=False)




if __name__ == "__main__":

    # extract_stats_from_file()

    # draw_graph("time_rk=0_ws=1.json", 161)
    # draw_graph("time_rk=0_ws=2.json", 161)
    # draw_graph("time_rk=1_ws=2.json", 161)
    # draw_graph("time_rk=0_ws=4.json", 161)
    # draw_graph("time_rk=1_ws=4.json", 161)
    # draw_graph("time_rk=2_ws=4.json", 161)
    # draw_graph("time_rk=3_ws=4.json", 161)

    # extract_excel(101)
    # extract_excel(151)
    extract_excel(201)
    extract_excel(251)
    extract_excel(301)
    # extract_excel(3351)
    # extract_excel(3301)
    pass


