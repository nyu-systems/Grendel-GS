import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

folder = "experiments/testTimerRedundantTiles/"
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

def read_file(file_name, num_render_file_name=None):
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

    if num_render_file_name is not None:

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



def extract_stats_from_file_bench_num_tiles():
    global folder
    global file_names
    global num_render_file_names

    base_folder = "experiments/bench_tile_num2/"
    file_names = [
        "gpu_time_ws=1_rk=0.log",
    ]
    num_render_file_names = [
        "num_rendered_ws=1_rk=0.log",
    ]
    x = os.listdir(base_folder)
    # filter out non-folders
    x = [t for t in x if os.path.isdir(base_folder + t)]
    x.sort( key=lambda x: int(x) )
    iterations = [1, 301, 601, 901]
    for t in x:
        folder = base_folder + t + "/"
        print(folder)
        extract_stats_from_file()
        for it in iterations:
            extract_excel(it)

    stats = None
    try:
        # load 
        path = "experiments/n_contributors_3/n_contrib_ws=1_rk=0.json"
        with open(path, 'r') as f:
            stats = json.load(f)
    except:
        pass
    

    # merge all csv together
    for i in iterations:
        df = None
        stat_iteration = None
        if stats is not None:
            stat_iteration = []
            for data in stats:
                if data["mode"] == "local" and data["iteration"] == str(i):
                    stat_iteration.append(data)
            assert len(stat_iteration) == 2170

        # print("len stat_iteration: ", len(stat_iteration))
        # print("stat_iteration", stat_iteration[:10])

        for t in x:
            folder = base_folder + t + "/"
            df_t = pd.read_csv(folder + "time_stat_it="+ str(i) +".csv")
            # add a columne for tile size
            df_t["tile_size"] = t
            df_t["b10 render time/tile_size*100"] = float(df_t.loc[0, "b10 render time"]) / float(df_t.loc[0, "tile_size"])*100
            df_t["b10 render time/num_rendered*100000"] = float(df_t.loc[0, "b10 render time"]) / float(df_t.loc[0, "num_rendered"])*100000
            df_t["70 render time/tile_size*100"] = float(df_t.loc[0, "70 render time"]) / float(df_t.loc[0, "tile_size"])*100
            df_t["70 render time/num_rendered*100000"] = float(df_t.loc[0, "70 render time"]) / float(df_t.loc[0, "num_rendered"])*100000

            if df is None:
                df = df_t
            else:
                df = pd.concat([df, df_t], ignore_index=True)

        # new rendered
        df["new_tile:num_rendered"] = 0.0
        df["new_tile:num_real_contributed"] = 0.0
        df["new_tile:b10 render time"] = 0.0
        df["new_tile:70 render time"] = 0.0

        for k in range(1, len(df)):
            df.loc[k, "new_tile:num_rendered"] = df.loc[k, "num_rendered"] - df.loc[k-1, "num_rendered"]
            df.loc[k, "new_tile:b10 render time"] = df.loc[k, "b10 render time"] - df.loc[k-1, "b10 render time"]
            df.loc[k, "new_tile:70 render time"] = df.loc[k, "70 render time"] - df.loc[k-1, "70 render time"]

            tile_range = (int(df.loc[k-1, "tile_size"]), int(df.loc[k, "tile_size"]))
            num_real_contributed = 0
            ave_contrib_ratio = 0
            if stats is not None:
                for data in stat_iteration:
                    if data["iteration"] == str(i) and data["mode"] == "local":
                        tile_str = data["tile"]
                        tile_xy = (int(tile_str[0]), int(tile_str[1]))
                        tile_id = tile_xy[0]*62 + tile_xy[1]
                        if tile_range[0] <= tile_id and tile_id < tile_range[1]:
                            num_real_contributed += float(data["local_real_n_contrib"])
                            ave_contrib_ratio += float(data["contrib_ratio"])
            
            ave_contrib_ratio = ave_contrib_ratio / float(tile_range[1] - tile_range[0])
            df.loc[k, "new_tile:num_real_contributed"] = num_real_contributed
            df.loc[k, "new_tile:ave_contrib_ratio"] = ave_contrib_ratio
            # print("tile_range: ", tile_range, "num_real_contributed: ", num_real_contributed, "ave_contrib_ratio: ", ave_contrib_ratio)

        df.to_csv(base_folder + "time_stat_it="+ str(i) +".csv", index=False)

def gpu_timer_0():
    global folder
    global file_names
    global num_render_file_names

    folder = "experiments/gpu_timer_0/"
    file_names = [
        "gpu_time_ws=1_rk=0.log",
        "gpu_time_ws=2_rk=0.log",
        "gpu_time_ws=2_rk=1.log",
        "gpu_time_ws=4_rk=0.log",
        "gpu_time_ws=4_rk=1.log",
        "gpu_time_ws=4_rk=2.log",
        "gpu_time_ws=4_rk=3.log",
    ]
    num_render_file_names = [None for i in range(len(file_names))]
    extract_stats_from_file()
    extract_excel(301)

# iter 1001, TimeFor 'forward': 3.405571 ms
# iter 1001, TimeFor 'image_allreduce': 0.006914 ms
# iter 1001, TimeFor 'loss': 2.740145 ms
# iter 1001, TimeFor 'backward': 15.798092 ms
# iter 1001, TimeFor 'sync_gradients': 0.006199 ms
# iter 1001, TimeFor 'optimizer_step': 2.892017 ms
def extract_json_from_python_time_log(file_path):

    file_name = file_path.split("/")[-1]
    ws, rk = file_name.split("_")[2].split("=")[1], file_name.split("_")[3].split("=")[1].split(".")[0]
    ws, rk = int(ws), int(rk)
    # print(file_name, " wk: ", wk, "rk: ", rk)

    with open(file_path, 'r') as f:
        lines = f.readlines()
    stats = []
    for line in lines:
        if line.startswith("iter"):
            parts = line.split(",")
            iteration = int(parts[0].split(" ")[1])

            if not stats or stats[-1]["iteration"] != iteration:
                stats.append({"iteration": iteration, "ws": ws, "rk": rk})
            # extract key and time from `TimeFor 'forward': 3.405571 ms`
            key = parts[1].split("'")[1]
            time = float(parts[1].split(":")[1].split(" ")[1])
            stats[-1][key] = time

    # save in file
    with open(file_path.removesuffix(".log") + ".json", 'w') as f:
        json.dump(stats, f, indent=4)
    return stats

def extract_json_from_python_time_log_many_files():
    for file in file_names:
        extract_json_from_python_time_log(folder + file)

def python_timer_0():
    global folder
    global file_names
    global num_render_file_names

    folder = "experiments/python_timer_0/"
    file_names = [
        "python_time_ws=1_rk=0.log",
        "python_time_ws=2_rk=0.log",
        "python_time_ws=2_rk=1.log",
        "python_time_ws=4_rk=0.log",
        "python_time_ws=4_rk=1.log",
        "python_time_ws=4_rk=2.log",
        "python_time_ws=4_rk=3.log",
    ]
    num_render_file_names = [None for i in range(len(file_names))]

    extract_json_from_python_time_log_many_files()
    extract_excel(301)

def python_timer_1():
    global folder
    global file_names
    global num_render_file_names

    folder = "experiments/python_timer_1/"
    file_names = [
        "python_time_ws=1_rk=0.log",
        "python_time_ws=2_rk=0.log",
        "python_time_ws=2_rk=1.log",
        "python_time_ws=4_rk=0.log",
        "python_time_ws=4_rk=1.log",
        "python_time_ws=4_rk=2.log",
        "python_time_ws=4_rk=3.log",
    ]
    num_render_file_names = [None for i in range(len(file_names))]

    extract_json_from_python_time_log_many_files()
    extract_excel(301)

def python_timer_sync_sparse_grad():
    global folder
    global file_names
    global num_render_file_names

    folder = "experiments/sparse_grad_sync/"
    file_names = [
        "python_time_ws=2_rk=0.log",
        "python_time_ws=2_rk=1.log",
        "python_time_ws=4_rk=0.log",
        "python_time_ws=4_rk=1.log",
        "python_time_ws=4_rk=2.log",
        "python_time_ws=4_rk=3.log",
    ]
    num_render_file_names = [None for i in range(len(file_names))]

    extract_json_from_python_time_log_many_files()
    extract_excel(401)
    extract_excel(801)
    extract_excel(1201)
    extract_excel(1601)
    extract_excel(2001)
    extract_excel(2401)
    extract_excel(2801)

def end2end_timer_0():
    global folder
    global file_names
    global num_render_file_names

    folder = "experiments/end2end_timer_0/"
    file_names = [
        "python_time_ws=1_rk=0.log",
        "python_time_ws=2_rk=0.log",
        "python_time_ws=2_rk=1.log",
        "python_time_ws=4_rk=0.log",
        "python_time_ws=4_rk=1.log",
        "python_time_ws=4_rk=2.log",
        "python_time_ws=4_rk=3.log",
    ]
    num_render_file_names = [None for i in range(len(file_names))]

    extract_json_from_python_time_log_many_files()
    extract_excel(401)
    extract_excel(801)
    extract_excel(1201)
    extract_excel(1601)
    extract_excel(2001)
    extract_excel(2401)
    extract_excel(2801)

if __name__ == "__main__":

    # python_timer_0()
    # python_timer_1()
    python_timer_sync_sparse_grad()
    end2end_timer_0()

    # extract_stats_from_file_bench_num_tiles()

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
    # extract_excel(201)
    # extract_excel(251)
    # extract_excel(301)
    # extract_excel(3351)
    # extract_excel(3301)
    pass


