import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib.ticker import MultipleLocator

# TODO: delete them later
folder = None
file_names = None
num_render_file_names = None

def delete_all_file_paths(file_paths):
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)

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

def extract_excel(iteration, provided_file_names=None):
    # extract frame from all data
    if provided_file_names is not None:
        file_names = provided_file_names

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





############################################################################################################
# New tools for analyzing: extract stats from python time log
############################################################################################################

def extract_data_from_list_by_iteration(data_list, iteration):
    for stat in data_list:
        if stat["iteration"] == iteration:
            return stat
    return None

def get_suffix_in_folder(folder):

    if not os.path.exists(folder):
        return None
    
    if not folder.endswith("/"):
        folder += "/"
    
    suffix_list_candidates = [
        "ws=1_rk=0",
        "ws=2_rk=0",
        "ws=2_rk=1",
        "ws=4_rk=0",
        "ws=4_rk=1",
        "ws=4_rk=2",
        "ws=4_rk=3",
    ]
    suffix_list = []

    for suffix in suffix_list_candidates:
        # python_ws=1_rk=0.log
        if os.path.exists(folder + "python_" + suffix + ".log"):
            suffix_list.append(suffix)

    return suffix_list

def get_end2end_stats(file_path):
    if not os.path.exists(file_path):
        return None

    with open(file_path, 'r') as f:
        lines = f.readlines()

    # end2end total_time: 5473.110746 ms, iterations: 30000, throughput 5.48 it/s
    # Max Memory usage: 8.000114917755127 GB.

    line_for_time = lines[-2]
    line_for_memory = lines[-1]
    
    if not line_for_time.startswith("end2end total_time"):
        return {"expe_name": file_path.strip("experiments")}
    
    print("line_for_time: ", line_for_time)
    print("line_for_memory: ", line_for_memory)
    stats = {}
    stats["expe_name"] = file_path.strip("experiments")
    stats["total_time"] = float(line_for_time.split("total_time: ")[1].split(" ms")[0])
    stats["throughput"] = float(line_for_time.split("throughput ")[1].split(" it/s")[0])
    stats["max_memory_usage"] = float(line_for_memory.split("Max Memory usage: ")[1].split(" GB")[0])
    # round to 3 digits
    stats["total_time"] = round(stats["total_time"], 3)
    stats["throughput"] = round(stats["throughput"], 3)
    stats["max_memory_usage"] = round(stats["max_memory_usage"], 3)
    # stats["iterations"] = int(line_for_time.split("iterations: ")[1].split(",")[0])
    return stats





def extract_time_excel_from_json(folder, file_paths, iteration, mode="python"):# mode = "python" or "gpu"
    # extract frame from all data
    df = None
    for file_path in file_paths:
        if not os.path.exists(file_path):
            continue
        stats = []
        with open(file_path, 'r') as f:
            stats = json.load(f)
        
        # gpu_time_ws=2_rk=0.json, python_time_ws=2_rk=0.log
        ws = int(file_path.split("/")[-1].split("_")[2].split("=")[1])
        rk = int(file_path.split("/")[-1].split("_")[3].split("=")[1].split(".")[0])

        data = extract_data_from_list_by_iteration(stats, iteration)

        # assert data is not None, "Queried iteration statistics should be in the log file."
        if data is None:
            print("Queried iteration statistics is not in the log file.")
            continue

        data_for_save = {}
        data_for_save["rk"] = rk
        data_for_save["ws"] = ws
        for key in data.keys():
            if key == "iteration" or key == "rk" or key == "ws":
                continue
            data_for_save[key] = data[key]

        if df is None:
            df = pd.DataFrame(data_for_save, index=[0])
        else:
            df = pd.concat([df, pd.DataFrame([data_for_save])], ignore_index=True)

    if df is None:
        print("No data to save in csv.")
        return
    print("extract_time_excel_from_json at iteration: ", iteration)
    df.to_csv(folder + f"{mode}_time_it="+ str(iteration) +".csv", index=False)

def merge_csv_which_have_same_columns(file_paths, output_file_path):
    # add another column for file_path
    df = None
    for file_path in file_paths:
        if not os.path.exists(file_path):
            continue
        df_t = pd.read_csv(file_path)
        df_t["file_path"] = file_path
        if df is None:
            df = df_t
        else:
            df = pd.concat([df, df_t], ignore_index=True)
        # add an empty row for better visualization
        empty_row = {col: None for col in df.columns}
        df = df._append(empty_row, ignore_index=True)
    if df is None:
        return
    df.to_csv(output_file_path, index=False)

# iter 1001, TimeFor 'forward': 3.405571 ms
# iter 1001, TimeFor 'image_allreduce': 0.006914 ms
# iter 1001, TimeFor 'loss': 2.740145 ms
# iter 1001, TimeFor 'backward': 15.798092 ms
# iter 1001, TimeFor 'sync_gradients': 0.006199 ms
# iter 1001, TimeFor 'optimizer_step': 2.892017 ms
def extract_json_from_python_time_log(file_path, load_genereated_json=False):

    file_name = file_path.split("/")[-1]
    ws, rk = file_name.split("_")[2].split("=")[1], file_name.split("_")[3].split("=")[1].split(".")[0]
    ws, rk = int(ws), int(rk)
    # print(file_name, " wk: ", wk, "rk: ", rk)

    if load_genereated_json and os.path.exists(file_path.removesuffix(".log") + ".json"):
        print("load from file"+file_path.removesuffix(".log") + ".json")
        with open(file_path.removesuffix(".log") + ".json", 'r') as f:
            return json.load(f)

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
            time = float(parts[1].split("': ")[1].split(" ")[0])
            stats[-1][key] = time

    # save in file
    with open(file_path.removesuffix(".log") + ".json", 'w') as f:
        json.dump(stats, f, indent=4)
    
    print("return data from file"+file_path.removesuffix(".log") + ".json")
    return stats

def extract_3dgs_count_from_python_log(folder):
    files = [
        "python_ws=4_rk=0.log",
        "python_ws=4_rk=1.log",
        "python_ws=4_rk=2.log",
        "python_ws=4_rk=3.log",
    ]
    stats = {}
    iterations = []
    for rk, file in enumerate(files):
        file_path = folder + file
        with open(file_path, 'r') as f:
            lines = f.readlines()
        stats[f"n_3dgs_rk={rk}"] = []
        for line in lines:
            if "densify_and_prune. Now num of 3dgs:" in line:
                # example
                # iteration 1000 densify_and_prune. Now num of 3dgs: 54539. Now Memory usage: 0.45931053161621094 GB. Max Memory usage: 4.580923080444336 GB.
                iteration = int(line.split("iteration ")[1].split(" ")[0])
                n_3dgs = int(line.split("Now num of 3dgs: ")[1].split(".")[0])
                if rk == 0:
                    iterations.append(iteration)
                stats[f"n_3dgs_rk={rk}"].append(n_3dgs)
    return stats, iterations

def extract_comm_count_from_i2jsend_log(folder):
    expe_name = folder.split("/")[-2]
    file_path = folder+"i2jsend_ws=4_rk=0.txt"
    with open(file_path, 'r') as f:
        lines = f.readlines()
    stats = {"total_comm_count": [], "i2jsend": []}
    iterations = []
    for line in lines:
        #example
        #iteration 851:[[511, 6817, 22924, 10372], [534, 5954, 24520, 10415], [1525, 7140, 15090, 11255], [945, 4812, 17584, 9013]]
        if line.startswith("iteration"):
            parts = line.split(":")
            iteration = int(parts[0].split(" ")[1])
            iterations.append(iteration)

            i2jsend_json_data = json.loads(parts[1])
            # stats["i2jsend"].append(i2jsend_json_data)
            stats["total_comm_count"].append(
                sum([sum(i2jsend_json_data[i]) - i2jsend_json_data[i][i] for i in range(len(i2jsend_json_data))])
            )
    return stats, iterations

def extract_json_from_i2jsend_log(file_path):

    file_name = file_path.split("/")[-1]
    ws, rk = file_name.split("_")[1].split("=")[1], file_name.split("_")[2].split("=")[1].split(".")[0]
    ws, rk = int(ws), int(rk)

    with open(file_path, 'r') as f:
        lines = f.readlines()
    stats = []
    for line in lines:
        #example
        #iteration 851:[[511, 6817, 22924, 10372], [534, 5954, 24520, 10415], [1525, 7140, 15090, 11255], [945, 4812, 17584, 9013]]
        if line.startswith("iteration"):
            parts = line.split(":")
            iteration = int(parts[0].split(" ")[1])

            if not stats or stats[-1]["iteration"] != iteration:
                stats.append({"iteration": iteration, "ws": ws})
            i2jsend_json_data = json.loads(parts[1])
            stats[-1]["i2jsend"] = i2jsend_json_data
            stats[-1]["total_comm_count"] = sum([sum(i2jsend_json_data[i]) - i2jsend_json_data[i][i] for i in range(len(i2jsend_json_data))])

    # save in file
    with open(file_path.removesuffix(".txt") + ".json", 'w') as f:
        json.dump(stats, f, indent=4)
    return stats

def extract_csv_from_forward_all_to_all_communication_json(folder, time_data, suffix_list, all2all_stats, process_iterations):
    # save results in csv: i2jsend.csv
    columns = ["iteration", "ws", "rk", "send_volume", "recv_volume", "forward_all_to_all_communication"]
    df = pd.DataFrame(columns=columns)

    ws = int(suffix_list[0].split("_")[0].split("=")[1])

    for iteration in process_iterations:
        python_time_rks = []
        for rk, suffix in enumerate(suffix_list):
            data = time_data[suffix]["python_time"]
            data = extract_data_from_list_by_iteration(data, iteration)
            python_time_rks.append(data["forward_all_to_all_communication"])

        stat = extract_data_from_list_by_iteration(all2all_stats, iteration)

        i2jsend = stat["i2jsend"]
        i2jsend = np.array(i2jsend)
        send_volume = np.sum(i2jsend, axis=1)
        recv_volume = np.sum(i2jsend, axis=0)

        # print experiment name and iteration
        with open(folder + f"i2jsend_ws={ws}.txt", 'a') as f:
            f.write(f"experiment: {folder}\n")
            f.write(f"iteration: {iteration}\n")
            f.write(f"send_volume: {send_volume}\n")
            f.write(f"recv_volume: {recv_volume}\n")
            f.write(f"python_time_rks: {python_time_rks}\n")
            f.write("\n")

        for i, suffix in enumerate(suffix_list):
            this_ws = int(suffix.split("_")[0].split("=")[1])
            rk = int(suffix.split("_")[1].split("=")[1])
            assert rk == i, "rk should be the same as index!"
            assert ws == this_ws, "ws should be the same!"
            df = df._append({
                "iteration": int(iteration),
                "ws": int(ws),
                "rk": int(rk),
                "send_volume": int(send_volume[rk]),
                "recv_volume": int(recv_volume[rk]),
                "forward_all_to_all_communication": python_time_rks[i],
            }, ignore_index=True)
        # append an empty row for better visualization
        df = df._append({
            "iteration": "",
            "ws": "",
            "rk": "",
            "send_volume": "",
            "recv_volume": "",
            "forward_all_to_all_communication": "",
        }, ignore_index=True)
    # save in file
    df.to_csv(folder + f"i2jsend_ws={ws}.csv", index=False)

def extract_memory_json_from_log(folder, file):
    file_path = folder + file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    stats = []
    for line in lines:
        if "densify_and_prune. Now num of 3dgs:" in line:
            # example
            # iteration 1000 densify_and_prune. Now num of 3dgs: 54539. Now Memory usage: 0.45931053161621094 GB. Max Memory usage: 4.580923080444336 GB.
            iteration = int(line.split("iteration ")[1].split(" ")[0])
            n_3dgs = int(line.split("Now num of 3dgs: ")[1].split(".")[0])
            now_memory_usage = float(line.split("Now Memory usage: ")[1].split(" GB")[0])
            max_memory_usage = float(line.split("Max Memory usage: ")[1].split(" GB")[0])
            stats.append({
                "iteration": iteration,
                "n_3dgs": n_3dgs,
                "now_memory_usage": now_memory_usage,
                "max_memory_usage": max_memory_usage,
            })

    # save in file
    memory_log_path = folder+file.removesuffix(".log") + "_mem.json"
    with open(memory_log_path, 'w') as f:
        json.dump(stats, f, indent=4)
    return stats

def extract_all_memory_json_from_log(folder):
    files = [
        "python_ws=1_rk=0.log",
        "python_ws=2_rk=0.log",
        "python_ws=2_rk=1.log",
        "python_ws=4_rk=0.log",
        "python_ws=4_rk=1.log",
        "python_ws=4_rk=2.log",
        "python_ws=4_rk=3.log",
    ]
    stats = []
    for file in files:
        if os.path.exists(folder + file):
            stats.append(extract_memory_json_from_log(folder, file))
    return stats

def extract_json_from_gpu_time_log(file_path, load_genereated_json=False):
    
    file_name = file_path.split("/")[-1]
    ws, rk = file_name.split("_")[2].split("=")[1], file_name.split("_")[3].split("=")[1].split(".")[0]
    ws, rk = int(ws), int(rk)

    if load_genereated_json and os.path.exists(file_path.removesuffix(".log") + ".json"):
        print("load from file"+file_path.removesuffix(".log") + ".json")
        with open(file_path.removesuffix(".log") + ".json", 'r') as f:
            return json.load(f)

    if not os.path.exists(file_path):
        return {}

    with open(file_path, 'r') as file:
        file_contents = file.readlines()
    
    # Function to parse each line and extract the statistic and its value
    def parse_line(line):
        # 10 preprocess time: 0.291950 ms
        parts = line.split(":")

        if len(parts) != 2:
            # print("Error parsing line: ", line)
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
            stats_json.append({"iteration": iteration})
            continue

        # print(line)
        parsed_data = parse_line(line)
        if parsed_data:
            stat_name, stat_value = parsed_data
            stats_json[-1][stat_name] = stat_value
    
    # save in file
    with open(file_path.removesuffix(".log") + ".json", 'w') as f:
        json.dump(stats_json, f, indent=4)
    print("return data from file"+file_path.removesuffix(".log") + ".json")
    return stats_json

def get_number_prefix(s):
    # the number may have multiple digits
    # example: s = "123abc", return 123
    # example: s = "123", return 123

    is_float = False
    for i in range(len(s)):
        if not s[i].isdigit():
            if s[i] == "-" and s[i+1].isdigit():
                continue
            if s[i] == ".":
                is_float = True
                continue
            # print(s[i], i)
            number = float(s[:i]) if is_float else int(s[:i])
            return number, s[i:]

    assert False, "s should have a number prefix"

def get_number_tuple_prefix(s):
    # the number may have multiple digits
    # example: s = "(0, 0)abc", return (0,0)

    left = s.find("(")
    right = s.find(")")
    assert left != -1 and right != -1, "s should have a number tuple prefix"
    tuple_str = s[left+1:right]
    tuple_str = tuple_str.split(",")
    assert len(tuple_str) == 2, "tuple_str should have 2 elements"
    return (int(tuple_str[0].strip()), int(tuple_str[1].strip())), s[right+1:]


def extract_json_from_n_contrib_log(file_path):
    file_name = file_path.split("/")[-1]
    ws, rk = file_name.split("_")[2].split("=")[1], file_name.split("_")[3].split("=")[1].split(".")[0]
    ws, rk = int(ws), int(rk)
    print(file_name, " ws: ", ws, "rk: ", rk)

    # if os.path.exists(file_path.removesuffix(".log") + ".json"):
    #     with open(file_path.removesuffix(".log") + ".json", 'r') as f:
    #         return json.load(f)

    with open(file_path, 'r') as f:
        lines = f.readlines()
    stats = []
    last_iteration = None
    for line in lines:

        # an example
        # iteration: 1, iteration: 1, local_rank: 0, world_size: 1, num_tiles: 62, num_pixels: 534100, num_rendered: 23574, global_ave_n_rendered_per_pix: 380.225800, global_ave_n_considered_per_pix: 11.138024, global_ave_n_contrib2loss_per_pix: 4.609777
        if "world_size" in line:
            tmp_line = line

            # print(tmp_line)
            tmp_line = tmp_line.strip(", ").removeprefix("iteration: ")
            iteration, tmp_line = get_number_prefix(tmp_line)
            # print(tmp_line)
            tmp_line = tmp_line.strip(", ").removeprefix("local_rank: ")
            local_rank, tmp_line = get_number_prefix(tmp_line)
            # print(tmp_line)
            tmp_line = tmp_line.strip(", ").removeprefix("world_size: ")
            world_size, tmp_line = get_number_prefix(tmp_line)
            # print(tmp_line)
            tmp_line = tmp_line.strip(", ").removeprefix("num_tiles: ")
            num_tiles, tmp_line = get_number_prefix(tmp_line)
            # print(tmp_line)
            tmp_line = tmp_line.strip(", ").removeprefix("num_pixels: ")
            num_pixels, tmp_line = get_number_prefix(tmp_line)
            # print(tmp_line)
            tmp_line = tmp_line.strip(", ").removeprefix("num_rendered: ")
            num_rendered, tmp_line = get_number_prefix(tmp_line)
            # print(tmp_line)
            tmp_line = tmp_line.strip(", ").removeprefix("global_ave_n_rendered_per_pix: ")
            global_ave_n_rendered_per_pix, tmp_line = get_number_prefix(tmp_line)
            # print(tmp_line)
            tmp_line = tmp_line.strip(", ").removeprefix("global_ave_n_considered_per_pix: ")
            global_ave_n_considered_per_pix, tmp_line = get_number_prefix(tmp_line)
            # print(tmp_line)
            tmp_line = tmp_line.strip(", ").removeprefix("global_ave_n_contrib2loss_per_pix: ")
            global_ave_n_contrib2loss_per_pix, tmp_line = get_number_prefix(tmp_line)


            assert iteration == last_iteration, "iteration should be the same"
            stats[-1]["stats"] = {
                "iteration": iteration,
                "local_rank": local_rank,
                "world_size": world_size,
                "num_tiles": num_tiles,
                "num_pixels": num_pixels,
                "num_rendered": num_rendered,
                "global_ave_n_rendered_per_pix": global_ave_n_rendered_per_pix,
                "global_ave_n_considered_per_pix": global_ave_n_considered_per_pix,
                "global_ave_n_contrib2loss_per_pix": global_ave_n_contrib2loss_per_pix,
            }

            continue

        # an example            
        # iteration: 1, tile: (0, 12), range: (1639, 1501), num_rendered_this_tile: 138, n_considered_per_pixel: 138.000000, n_contrib2loss_per_pixel: 77.437500, contrib2loss_ratio: 0.302490
        if line.startswith("iteration:"):
            tmp_line = line

            # print(tmp_line)
            tmp_line = tmp_line.strip(", ").removeprefix("iteration: ")
            iteration, tmp_line = get_number_prefix(tmp_line)
            # print(tmp_line)
            tmp_line = tmp_line.strip(", ").removeprefix("tile: ")
            tile, tmp_line = get_number_tuple_prefix(tmp_line)
            # print(tmp_line)
            tmp_line = tmp_line.strip(", ").removeprefix("range: ")
            range, tmp_line = get_number_tuple_prefix(tmp_line)
            # print(tmp_line)
            tmp_line = tmp_line.strip(", ").removeprefix("num_rendered_this_tile: ")
            num_rendered_this_tile, tmp_line = get_number_prefix(tmp_line)
            # print(tmp_line)
            tmp_line = tmp_line.strip(", ").removeprefix("n_considered_per_pixel: ")
            n_considered_per_pixel, tmp_line = get_number_prefix(tmp_line)
            # print(tmp_line)
            tmp_line = tmp_line.strip(", ").removeprefix("n_contrib2loss_per_pixel: ")
            n_contrib2loss_per_pixel, tmp_line = get_number_prefix(tmp_line)
            # print(tmp_line)
            tmp_line = tmp_line.strip(", ").removeprefix("contrib2loss_ratio: ")
            contrib2loss_ratio, tmp_line = get_number_prefix(tmp_line)


            if iteration != last_iteration:
                stats.append({"iteration": iteration, "data": []})
                last_iteration = iteration

            stats[-1]["data"].append(
                {
                    "iteration": iteration,
                    "tile": tile,
                    "range": range,
                    "num_rendered_this_tile": num_rendered_this_tile,
                    "n_considered_per_pixel": n_considered_per_pixel,
                    "n_contrib2loss_per_pixel": n_contrib2loss_per_pixel,
                    # "contrib2loss_ratio": contrib2loss_ratio,
                }
            )


    # save in file
    with open(file_path.removesuffix(".log") + ".json", 'w') as f:
        json.dump(stats, f, indent=4)
    return stats

def extract_json_from_num_rendered_log(file_path):
    file_name = file_path.split("/")[-1]
    ws, rk = file_name.split("_")[2].split("=")[1], file_name.split("_")[3].split("=")[1].split(".")[0]
    ws, rk = int(ws), int(rk)

    # if os.path.exists(file_path.removesuffix(".log") + ".json"):
    #     with open(file_path.removesuffix(".log") + ".json", 'r') as f:
    #         return json.load(f)

    with open(file_path, 'r') as file:
        file_contents = file.readlines()    
    
    stats = []
    for line in file_contents:
        # example
        # iteration: 1, iteration: 1, num_local_tiles: 62, local_tiles_left_idx: 0, local_tiles_right_idx: 61, last_local_num_rendered_end: 0, local_num_rendered_end: 62, num_rendered: 23574, num_rendered_from_distState: 23574

        if line.startswith("iteration:"):
            tmp_line = line

            # print(tmp_line)
            tmp_line = tmp_line.strip(", ").removeprefix("iteration: ")
            iteration, tmp_line = get_number_prefix(tmp_line)
            # print(tmp_line)
            tmp_line = tmp_line.strip(", ").removeprefix("num_local_tiles: ")
            num_local_tiles, tmp_line = get_number_prefix(tmp_line)
            # print(tmp_line)
            tmp_line = tmp_line.strip(", ").removeprefix("local_tiles_left_idx: ")
            local_tiles_left_idx, tmp_line = get_number_prefix(tmp_line)
            # print(tmp_line)
            tmp_line = tmp_line.strip(", ").removeprefix("local_tiles_right_idx: ")
            local_tiles_right_idx, tmp_line = get_number_prefix(tmp_line)
            # print(tmp_line)
            tmp_line = tmp_line.strip(", ").removeprefix("last_local_num_rendered_end: ")
            last_local_num_rendered_end, tmp_line = get_number_prefix(tmp_line)
            # print(tmp_line)
            tmp_line = tmp_line.strip(", ").removeprefix("local_num_rendered_end: ")
            local_num_rendered_end, tmp_line = get_number_prefix(tmp_line)
            # print(tmp_line)
            tmp_line = tmp_line.strip(", ").removeprefix("num_rendered: ")
            num_rendered, tmp_line = get_number_prefix(tmp_line)
            # print(tmp_line)
            tmp_line = tmp_line.strip(", ").removeprefix("num_rendered_from_distState: ")
            num_rendered_from_distState, tmp_line = get_number_prefix(tmp_line)

            stats.append({
                "iteration": iteration,
                "num_local_tiles": num_local_tiles,
                "local_tiles_left_idx": local_tiles_left_idx,
                "local_tiles_right_idx": local_tiles_right_idx,
                "last_local_num_rendered_end": last_local_num_rendered_end,
                "local_num_rendered_end": local_num_rendered_end,
                "num_rendered": num_rendered,
                "num_rendered_from_distState": num_rendered_from_distState,
            })
    
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

def get_all_grad_sync_time(json_file_path):
    with open(json_file_path, 'r') as f:
        stats = json.load(f)
    grad_sync_time = []
    iterations = []
    for stat in stats:
        iterations.append(stat["iteration"])
        grad_sync_time.append(stat["sync_gradients"])
    return iterations, grad_sync_time

def analyze_sparse_grad_speed_up():
    paths = [
        "experiments/end2end_timer_0/python_time_ws=1_rk=0.json",
        "experiments/end2end_timer_0/python_time_ws=2_rk=0.json",
        "experiments/end2end_timer_0/python_time_ws=2_rk=1.json",
        "experiments/end2end_timer_0/python_time_ws=4_rk=0.json",
        "experiments/end2end_timer_0/python_time_ws=4_rk=1.json",
        "experiments/end2end_timer_0/python_time_ws=4_rk=2.json",
        "experiments/end2end_timer_0/python_time_ws=4_rk=3.json",
        "experiments/sparse_grad_sync/python_time_ws=2_rk=0.json",
        "experiments/sparse_grad_sync/python_time_ws=2_rk=1.json",
        "experiments/sparse_grad_sync/python_time_ws=4_rk=0.json",
        "experiments/sparse_grad_sync/python_time_ws=4_rk=1.json",
        "experiments/sparse_grad_sync/python_time_ws=4_rk=2.json",
        "experiments/sparse_grad_sync/python_time_ws=4_rk=3.json",        
    ]
    all_grad_sync_time = []
    iterations = []
    columes = ["iteration"]
    for path in paths:
        iterations, grad_sync_time = get_all_grad_sync_time(path)
        columes.append(path.removeprefix("experiments/").removesuffix(".json"))
        all_grad_sync_time.append(grad_sync_time)
    
    # convert to data frame
    # each row is a iteration
    # each column is a json data
    df = pd.DataFrame(columns=columes)
    df["iteration"] = iterations
    print("columes: ", columes)
    print("iterations: ", iterations)
    print("len iterations: ", len(iterations))
    for i in range(len(all_grad_sync_time)):
        # print length of each grad_sync_time
        print("len grad_sync_time: ", len(all_grad_sync_time[i]))
        df[columes[i+1]] = all_grad_sync_time[i]
        
    df["end2end_timer_0/python_time_ws=2"] = df[["end2end_timer_0/python_time_ws=2_rk=0", "end2end_timer_0/python_time_ws=2_rk=1"]].max(axis=1)
    df["end2end_timer_0/python_time_ws=4"] = df[["end2end_timer_0/python_time_ws=4_rk=0", "end2end_timer_0/python_time_ws=4_rk=1", "end2end_timer_0/python_time_ws=4_rk=2", "end2end_timer_0/python_time_ws=4_rk=3"]].max(axis=1)
    df["sparse_grad_sync/python_time_ws=2"] = df[["sparse_grad_sync/python_time_ws=2_rk=0", "sparse_grad_sync/python_time_ws=2_rk=1"]].max(axis=1)
    df["sparse_grad_sync/python_time_ws=4"] = df[["sparse_grad_sync/python_time_ws=4_rk=0", "sparse_grad_sync/python_time_ws=4_rk=1", "sparse_grad_sync/python_time_ws=4_rk=2", "sparse_grad_sync/python_time_ws=4_rk=3"]].max(axis=1)
    df["speed_up_ws=2"] = df["sparse_grad_sync/python_time_ws=2"] / df["end2end_timer_0/python_time_ws=2"]
    df["speed_up_ws=4"] = df["sparse_grad_sync/python_time_ws=4"] / df["end2end_timer_0/python_time_ws=4"]

    df.to_csv("experiments/sparse_grad_sync/compare_sparse_grad_sync_time.csv", index=False)

    # output the average of colume: speed_up_ws=2 and speed_up_ws=4, discard the first 2 rows
    print("average time spent ratio=2: ", np.mean(df["speed_up_ws=2"][2:]))
    print("average time spent ratio=4: ", np.mean(df["speed_up_ws=4"][2:]))


def prepare_json(folder):
    n_contrib_path = folder + "/n_contrib_ws=1_rk=0.log"
    n_contrib_json = extract_json_from_n_contrib_log(n_contrib_path)
    gpu_time_path = folder + "/gpu_time_ws=1_rk=0.log"
    gpu_time_json = extract_json_from_gpu_time_log(gpu_time_path)
    num_rendered_path = folder + "/num_rendered_ws=1_rk=0.log"
    num_rendered_json = extract_json_from_num_rendered_log(num_rendered_path)
    python_time_path = folder + "/python_time_ws=1_rk=0.log"
    python_time_json = extract_json_from_python_time_log(python_time_path)
    return n_contrib_json, gpu_time_json, num_rendered_json, python_time_json

def bench_train_rows(folder):
    sub_folders = [x for x in os.listdir(folder) if os.path.isdir(folder + x)]
    sub_folders.sort(key=lambda x: int(x.split("_")[1]))

    all_gpu_time_json = []
    all_n_contrib_json = []
    for sub_folder in sub_folders:
        print("sub_folder: ", sub_folder)
        n_contrib_json, gpu_time_json, num_rendered_json, python_time_json = prepare_json(folder + sub_folder)

        # merge all json
        assert len(n_contrib_json) == len(gpu_time_json) == len(num_rendered_json) == len(python_time_json), "length of json should be the same"

        all_gpu_time_json.append(gpu_time_json)
        all_n_contrib_json.append(n_contrib_json)
    
    def get_statistics(stat_id, iteration):
        gpu_time_keys = []
        for key in all_gpu_time_json[0][stat_id].keys():
            if key == "iteration" or key == "rk" or key == "ws":
                continue
            gpu_time_keys.append(key)

        n_contrib_keys = ["num_tiles", "num_pixels", "num_rendered", "global_ave_n_rendered_per_pix", "global_ave_n_considered_per_pix", "global_ave_n_contrib2loss_per_pix"]
        
        columns = ["sub_folder"] + gpu_time_keys + n_contrib_keys
        df = pd.DataFrame(columns=columns)
        for i in range(len(sub_folders)):
            sub_folder = sub_folders[i]
            gpu_time_json = all_gpu_time_json[i]
            n_contrib_json = all_n_contrib_json[i]

            assert gpu_time_json[stat_id]["iteration"] == iteration, "iteration should be the same"
            assert n_contrib_json[stat_id]["stats"]["iteration"] == iteration, "iteration should be the same"

            row = {"sub_folder": sub_folder}
            for key in gpu_time_keys:
                row[key] = gpu_time_json[stat_id][key]
            for key in n_contrib_keys:
                row[key] = n_contrib_json[stat_id]["stats"][key]
            df = df._append(row, ignore_index=True)
        
        df.to_csv(folder + f"statistics_{iteration}.csv", index=False)

    all_iterations = []
    for gpu_time_data in all_gpu_time_json[-1]:
        all_iterations.append(gpu_time_data["iteration"])
    print("all_iterations: ", all_iterations)
    for i, iteration in enumerate(all_iterations):
        if i >=1:
            get_statistics(i, iteration)


def fvalue(x):
    # example: row_21_22_duplicategscnt_4
    # return 21*10000 + 22*100 + 4
    parts = x.split("_")
    return int(parts[1])*10000 + int(parts[2])*100 + int(parts[4])

def bench_sklearn_dataset(folder):
    sub_folders = [x for x in os.listdir(folder) if os.path.isdir(folder + x)]
    sub_folders.sort(key=lambda x: fvalue(x))

    # print("sub_folders: ", sub_folders)
    print("len sub_folders: ", len(sub_folders)) # 2112 = 528*4

    all_gpu_time_json = []
    all_n_contrib_json = []
    for sub_folder in sub_folders:
        print("sub_folder: ", sub_folder)
        n_contrib_json, gpu_time_json, num_rendered_json, python_time_json = prepare_json(folder + sub_folder)

        # merge all json
        assert len(n_contrib_json) == len(gpu_time_json) == len(num_rendered_json) == len(python_time_json), "length of json should be the same"

        all_gpu_time_json.append(gpu_time_json)
        all_n_contrib_json.append(n_contrib_json)
    
    def get_statistics(stat_id, iteration):
        gpu_time_keys = []
        for key in all_gpu_time_json[0][stat_id].keys():
            if key == "iteration" or key == "rk" or key == "ws":
                continue
            gpu_time_keys.append(key)

        n_contrib_keys = ["num_tiles", "num_pixels", "num_rendered", "global_ave_n_rendered_per_pix", "global_ave_n_considered_per_pix", "global_ave_n_contrib2loss_per_pix"]
        
        columns = ["sub_folder"] + gpu_time_keys + n_contrib_keys
        df = pd.DataFrame(columns=columns)
        for i in range(len(sub_folders)):
            sub_folder = sub_folders[i]
            gpu_time_json = all_gpu_time_json[i]
            n_contrib_json = all_n_contrib_json[i]

            assert gpu_time_json[stat_id]["iteration"] == iteration, "iteration should be the same"
            assert n_contrib_json[stat_id]["stats"]["iteration"] == iteration, "iteration should be the same"

            row = {"sub_folder": sub_folder}
            for key in gpu_time_keys:
                row[key] = gpu_time_json[stat_id][key]
            for key in n_contrib_keys:
                row[key] = n_contrib_json[stat_id]["stats"][key]
            df = df._append(row, ignore_index=True)
        
        df.to_csv(folder + f"statistics_{iteration}.csv", index=False)

    all_iterations = []
    for gpu_time_data in all_gpu_time_json[-1]:
        all_iterations.append(gpu_time_data["iteration"])
    # print("all_iterations: ", all_iterations)
    for i, iteration in enumerate(all_iterations):
        print("iteration: ", iteration)
        if i >=1:
            get_statistics(i, iteration)


def div_stra_5_adjust(folder):
    suffix_list = [
        "ws=2_rk=0",
        "ws=2_rk=1",
        "ws=4_rk=0",
        "ws=4_rk=1",
        "ws=4_rk=2",
        "ws=4_rk=3",
    ]
    data = {}
    for suffix in suffix_list:
        file_path = folder + f"gpu_time_{suffix}.log"
        gpu_time_json = extract_json_from_gpu_time_log(file_path)
        file_path = folder + f"python_time_{suffix}.log"
        python_time_json = extract_json_from_python_time_log(file_path)
        data[suffix] = {"gpu_time": gpu_time_json, "python_time": python_time_json}

    file_paths = [
        folder + "gpu_time_ws=2_rk=0.json",
        folder + "gpu_time_ws=2_rk=1.json",
        folder + "gpu_time_ws=4_rk=0.json",
        folder + "gpu_time_ws=4_rk=1.json",
        folder + "gpu_time_ws=4_rk=2.json",
        folder + "gpu_time_ws=4_rk=3.json",
    ]
    extract_time_excel_from_json(folder, file_paths, 4, mode="gpu")
    extract_time_excel_from_json(folder, file_paths, 7, mode="gpu")
    extract_time_excel_from_json(folder, file_paths, 10, mode="gpu")

    file_paths = [
        folder + "python_time_ws=2_rk=0.json",
        folder + "python_time_ws=2_rk=1.json",
        folder + "python_time_ws=4_rk=0.json",
        folder + "python_time_ws=4_rk=1.json",
        folder + "python_time_ws=4_rk=2.json",
        folder + "python_time_ws=4_rk=3.json",
    ]
    extract_time_excel_from_json(folder, file_paths, 4, mode="python")
    extract_time_excel_from_json(folder, file_paths, 7, mode="python")
    extract_time_excel_from_json(folder, file_paths, 10, mode="python")

def merge_csv_for_div_stra_5_adjust():
    folder1 = "experiments/div_stra_5_adjust_none/"
    folder2 = "experiments/div_stra_5_adjust_n_contrib/"

    file_paths = []
    for iteration in [4,7,10]:
        file_paths += [
            folder1 + f"gpu_time_it={iteration}.csv",
            folder2 + f"gpu_time_it={iteration}.csv",
        ]

    merge_csv_which_have_same_columns(file_paths, folder1 + f"merged_gpu_time.csv")

    file_paths = []
    for iteration in [4,7,10]:
        file_paths += [
            folder1 + f"python_time_it={iteration}.csv",
            folder2 + f"python_time_it={iteration}.csv",
        ]

    merge_csv_which_have_same_columns(file_paths, folder1 + f"merged_python_time.csv")


def memory_distribution_4(folder):
    suffix_list = [
        "ws=1_rk=0",
        "ws=2_rk=0",
        "ws=2_rk=1",
        "ws=4_rk=0",
        "ws=4_rk=1",
        "ws=4_rk=2",
        "ws=4_rk=3",
    ]
    data = {}
    for suffix in suffix_list:
        file_path = folder + f"gpu_time_{suffix}.log"
        gpu_time_json = extract_json_from_gpu_time_log(file_path)
        file_path = folder + f"python_time_{suffix}.log"
        python_time_json = extract_json_from_python_time_log(file_path)
        data[suffix] = {"gpu_time": gpu_time_json, "python_time": python_time_json}

    file_paths = [
        folder + "gpu_time_ws=1_rk=0.json",
        folder + "gpu_time_ws=2_rk=0.json",
        folder + "gpu_time_ws=2_rk=1.json",
        folder + "gpu_time_ws=4_rk=0.json",
        folder + "gpu_time_ws=4_rk=1.json",
        folder + "gpu_time_ws=4_rk=2.json",
        folder + "gpu_time_ws=4_rk=3.json",
    ]
    extract_time_excel_from_json(folder, file_paths, 51, mode="gpu")
    extract_time_excel_from_json(folder, file_paths, 101, mode="gpu")
    extract_time_excel_from_json(folder, file_paths, 151, mode="gpu")

    file_paths = [
        folder + "python_time_ws=1_rk=0.json",
        folder + "python_time_ws=2_rk=0.json",
        folder + "python_time_ws=2_rk=1.json",
        folder + "python_time_ws=4_rk=0.json",
        folder + "python_time_ws=4_rk=1.json",
        folder + "python_time_ws=4_rk=2.json",
        folder + "python_time_ws=4_rk=3.json",
    ]
    extract_time_excel_from_json(folder, file_paths, 51, mode="python")
    extract_time_excel_from_json(folder, file_paths, 101, mode="python")
    extract_time_excel_from_json(folder, file_paths, 151, mode="python")

def memory_distribution_4_no(folder):
    suffix_list = [
        "ws=2_rk=0",
        "ws=2_rk=1",
        "ws=4_rk=0",
        "ws=4_rk=1",
        "ws=4_rk=2",
        "ws=4_rk=3",
    ]
    data = {}
    for suffix in suffix_list:
        file_path = folder + f"gpu_time_{suffix}.log"
        gpu_time_json = extract_json_from_gpu_time_log(file_path)
        file_path = folder + f"python_time_{suffix}.log"
        python_time_json = extract_json_from_python_time_log(file_path)
        data[suffix] = {"gpu_time": gpu_time_json, "python_time": python_time_json}

    file_paths = [
        folder + "gpu_time_ws=2_rk=0.json",
        folder + "gpu_time_ws=2_rk=1.json",
        folder + "gpu_time_ws=4_rk=0.json",
        folder + "gpu_time_ws=4_rk=1.json",
        folder + "gpu_time_ws=4_rk=2.json",
        folder + "gpu_time_ws=4_rk=3.json",
    ]
    extract_time_excel_from_json(folder, file_paths, 51, mode="gpu")
    extract_time_excel_from_json(folder, file_paths, 101, mode="gpu")
    extract_time_excel_from_json(folder, file_paths, 151, mode="gpu")

    file_paths = [
        folder + "python_time_ws=2_rk=0.json",
        folder + "python_time_ws=2_rk=1.json",
        folder + "python_time_ws=4_rk=0.json",
        folder + "python_time_ws=4_rk=1.json",
        folder + "python_time_ws=4_rk=2.json",
        folder + "python_time_ws=4_rk=3.json",
    ]
    extract_time_excel_from_json(folder, file_paths, 51, mode="python")
    extract_time_excel_from_json(folder, file_paths, 101, mode="python")
    extract_time_excel_from_json(folder, file_paths, 151, mode="python")

def memory_distribution_4_no_sep_render_ws1(folder):
    suffix_list = [
        "ws=1_rk=0",
    ]
    data = {}
    for suffix in suffix_list:
        file_path = folder + f"gpu_time_{suffix}.log"
        gpu_time_json = extract_json_from_gpu_time_log(file_path)
        file_path = folder + f"python_time_{suffix}.log"
        python_time_json = extract_json_from_python_time_log(file_path)
        data[suffix] = {"gpu_time": gpu_time_json, "python_time": python_time_json}

    file_paths = [
        folder + "gpu_time_ws=1_rk=0.json",
    ]
    extract_time_excel_from_json(folder, file_paths, 51, mode="gpu")
    extract_time_excel_from_json(folder, file_paths, 101, mode="gpu")
    extract_time_excel_from_json(folder, file_paths, 151, mode="gpu")

    file_paths = [
        folder + "python_time_ws=1_rk=0.json",
    ]
    extract_time_excel_from_json(folder, file_paths, 51, mode="python")
    extract_time_excel_from_json(folder, file_paths, 101, mode="python")
    extract_time_excel_from_json(folder, file_paths, 151, mode="python")

def mem_dist_stats_3(folder):
    suffix_list = [
        "ws=4_rk=0",
        "ws=4_rk=1",
        "ws=4_rk=2",
        "ws=4_rk=3",
    ]

    time_data = {}
    for suffix in suffix_list:
        file_path = folder + f"gpu_time_{suffix}.log"
        gpu_time_json = extract_json_from_gpu_time_log(file_path)
        file_path = folder + f"python_time_{suffix}.log"
        python_time_json = extract_json_from_python_time_log(file_path)
        time_data[suffix] = {"gpu_time": gpu_time_json, "python_time": python_time_json}


    file_path = folder + "i2jsend_ws=4_rk=0.txt"
    stats = extract_json_from_i2jsend_log(file_path)

    # save results in csv: i2jsend.csv
    columns = ["iteration", "rk", "send_volume", "recv_volume", "python_time"]
    df = pd.DataFrame(columns=columns)

    process_iterations = [551, 651, 751, 851, 951]
    for iteration in process_iterations:

        python_time_rks = []
        for rk, suffix in enumerate(suffix_list):
            data = time_data[suffix]["python_time"]
            data = extract_data_from_list_by_iteration(data, iteration)
            python_time_rks.append(data["forward_all_to_all_communication"])

        stat = extract_data_from_list_by_iteration(stats, iteration)

        i2jsend = stat["i2jsend"]
        i2jsend = np.array(i2jsend)
        send_volume = np.sum(i2jsend, axis=1)
        recv_volume = np.sum(i2jsend, axis=0)

        # output experiment name and iteration into this file
        # print("experiment: ", folder)
        # print("iteration: ", iteration)
        # print("send_volume: ", send_volume)
        # print("recv_volume: ", recv_volume)
        # print("python_time_rks: ", python_time_rks)
        # print("")

        for rk in range(4):
            df = df._append({
                "iteration": iteration,
                "rk": rk,
                "send_volume": send_volume[rk],
                "recv_volume": recv_volume[rk],
                "python_time": python_time_rks[rk],
            }, ignore_index=True)
        # append an empty row for better visualization
        df = df._append({
            "iteration": "",
            "rk": "",
            "send_volume": "",
            "recv_volume": "",
            "python_time": "",
        }, ignore_index=True)
    # save in file
    df.to_csv(folder + "i2jsend.csv", index=False)


def mem_dist_stats_4(folder):
    suffix_list = [
        "ws=1_rk=0",
        "ws=4_rk=0",
        "ws=4_rk=1",
        "ws=4_rk=2",
        "ws=4_rk=3",
    ]

    time_data = {}
    for suffix in suffix_list:
        file_path = folder + f"gpu_time_{suffix}.log"
        gpu_time_json = extract_json_from_gpu_time_log(file_path)
        file_path = folder + f"python_time_{suffix}.log"
        python_time_json = extract_json_from_python_time_log(file_path)
        time_data[suffix] = {"gpu_time": gpu_time_json, "python_time": python_time_json}

    file_path = folder + "i2jsend_ws=4_rk=0.txt"
    stats = extract_json_from_i2jsend_log(file_path)

    # save results in csv: i2jsend.csv
    columns = ["iteration", "rk", "send_volume", "recv_volume", "forward_all_to_all_communication"]
    df = pd.DataFrame(columns=columns)

    # the following code is only for ws=4
    process_iterations = [551, 1051, 2051, 3051, 4051, 5051, 6051, 6951]
    for iteration in process_iterations:

        python_time_rks = []
        for rk, suffix in enumerate(suffix_list[1:]):
            assert "4" == suffix.split("_")[0].split("=")[1], "ws should be 4!"
            data = time_data[suffix]["python_time"]
            data = extract_data_from_list_by_iteration(data, iteration)
            python_time_rks.append(data["forward_all_to_all_communication"])

        stat = extract_data_from_list_by_iteration(stats, iteration)

        i2jsend = stat["i2jsend"]
        i2jsend = np.array(i2jsend)
        send_volume = np.sum(i2jsend, axis=1)
        recv_volume = np.sum(i2jsend, axis=0)

        # print experiment name and iteration
        print("experiment: ", folder)
        print("iteration: ", iteration)
        print("send_volume: ", send_volume)
        print("recv_volume: ", recv_volume)
        print("python_time_rks: ", python_time_rks)
        print("")

        for rk in range(4):
            df = df._append({
                "iteration": int(iteration),
                "rk": int(rk),
                "send_volume": int(send_volume[rk]),
                "recv_volume": int(recv_volume[rk]),
                "forward_all_to_all_communication": python_time_rks[rk],
            }, ignore_index=True)
        # append an empty row for better visualization
        df = df._append({
            "iteration": "",
            "rk": "",
            "send_volume": "",
            "recv_volume": "",
            "forward_all_to_all_communication": "",
        }, ignore_index=True)
    # save in file
    df.to_csv(folder + "i2jsend.csv", index=False)

    file_paths = [
        folder + "gpu_time_ws=1_rk=0.json",
        folder + "gpu_time_ws=4_rk=0.json",
        folder + "gpu_time_ws=4_rk=1.json",
        folder + "gpu_time_ws=4_rk=2.json",
        folder + "gpu_time_ws=4_rk=3.json",
    ]
    for iteration in process_iterations:
        extract_time_excel_from_json(folder, file_paths, iteration, mode="gpu")
    file_paths = [
        folder + "python_time_ws=1_rk=0.json",
        folder + "python_time_ws=4_rk=0.json",
        folder + "python_time_ws=4_rk=1.json",
        folder + "python_time_ws=4_rk=2.json",
        folder + "python_time_ws=4_rk=3.json",
    ]
    for iteration in process_iterations:
        extract_time_excel_from_json(folder, file_paths, iteration, mode="python")


def mem_dist_stats_4k_garden_2(folder):
    suffix_list = [
        "ws=1_rk=0",
        "ws=2_rk=0",
        "ws=2_rk=1",
        "ws=4_rk=0",
        "ws=4_rk=1",
        "ws=4_rk=2",
        "ws=4_rk=3",
    ]

    time_data = {}
    for suffix in suffix_list:
        file_path = folder + f"gpu_time_{suffix}.log"
        gpu_time_json = extract_json_from_gpu_time_log(file_path)
        file_path = folder + f"python_time_{suffix}.log"
        python_time_json = extract_json_from_python_time_log(file_path)
        time_data[suffix] = {"gpu_time": gpu_time_json, "python_time": python_time_json}

    process_iterations = [551, 751, 951]

    # get i2jsend data csv
    if os.path.exists(folder + f"i2jsend_ws=4.txt"):
        os.remove(folder + f"i2jsend_ws=4.txt")
    if os.path.exists(folder + f"i2jsend_ws=2.txt"):
        os.remove(folder + f"i2jsend_ws=2.txt")
    stats = extract_json_from_i2jsend_log(folder + "i2jsend_ws=4_rk=0.txt")
    extract_csv_from_forward_all_to_all_communication_json(folder, time_data, suffix_list[3:], stats, process_iterations)
    stats = extract_json_from_i2jsend_log(folder + "i2jsend_ws=2_rk=0.txt")
    extract_csv_from_forward_all_to_all_communication_json(folder, time_data, suffix_list[1:3], stats, process_iterations)    

    # get time csv
    file_paths = [
        folder + "gpu_time_ws=1_rk=0.json",
        folder + "gpu_time_ws=2_rk=0.json",
        folder + "gpu_time_ws=2_rk=1.json",
        folder + "gpu_time_ws=4_rk=0.json",
        folder + "gpu_time_ws=4_rk=1.json",
        folder + "gpu_time_ws=4_rk=2.json",
        folder + "gpu_time_ws=4_rk=3.json",
    ]
    for iteration in process_iterations:
        extract_time_excel_from_json(folder, file_paths, iteration, mode="gpu")
    file_paths = [
        folder + "python_time_ws=1_rk=0.json",
        folder + "python_time_ws=2_rk=0.json",
        folder + "python_time_ws=2_rk=1.json",
        folder + "python_time_ws=4_rk=0.json",
        folder + "python_time_ws=4_rk=1.json",
        folder + "python_time_ws=4_rk=2.json",
        folder + "python_time_ws=4_rk=3.json",
    ]
    for iteration in process_iterations:
        extract_time_excel_from_json(folder, file_paths, iteration, mode="python")

    extract_all_memory_json_from_log(folder)

def mem_dist_stats_4k_garden_3(folder):
    suffix_list = [
        "ws=1_rk=0",
        "ws=4_rk=0",
        "ws=4_rk=1",
        "ws=4_rk=2",
        "ws=4_rk=3",
    ]

    time_data = {}
    for suffix in suffix_list:
        file_path = folder + f"gpu_time_{suffix}.log"
        gpu_time_json = extract_json_from_gpu_time_log(file_path)
        file_path = folder + f"python_time_{suffix}.log"
        python_time_json = extract_json_from_python_time_log(file_path)
        time_data[suffix] = {"gpu_time": gpu_time_json, "python_time": python_time_json}

    process_iterations = [551, 1551, 2551]

    # get i2jsend data csv
    if os.path.exists(folder + f"i2jsend_ws=4.txt"):
        os.remove(folder + f"i2jsend_ws=4.txt")
    stats = extract_json_from_i2jsend_log(folder + "i2jsend_ws=4_rk=0.txt")
    extract_csv_from_forward_all_to_all_communication_json(folder, time_data, suffix_list[1:], stats, process_iterations)

    # get time csv
    file_paths = [
        folder + "gpu_time_ws=1_rk=0.json",
        folder + "gpu_time_ws=4_rk=0.json",
        folder + "gpu_time_ws=4_rk=1.json",
        folder + "gpu_time_ws=4_rk=2.json",
        folder + "gpu_time_ws=4_rk=3.json",
    ]
    for iteration in process_iterations:
        extract_time_excel_from_json(folder, file_paths, iteration, mode="gpu")
    file_paths = [
        folder + "python_time_ws=1_rk=0.json",
        folder + "python_time_ws=4_rk=0.json",
        folder + "python_time_ws=4_rk=1.json",
        folder + "python_time_ws=4_rk=2.json",
        folder + "python_time_ws=4_rk=3.json",
    ]
    for iteration in process_iterations:
        extract_time_excel_from_json(folder, file_paths, iteration, mode="python")

    extract_all_memory_json_from_log(folder)

def adjust2(folder):
    suffix_list = [
        # "ws=1_rk=0",
        # "ws=2_rk=0",
        # "ws=2_rk=1",
        "ws=4_rk=0",
        "ws=4_rk=1",
        "ws=4_rk=2",
        "ws=4_rk=3",
    ]
    data = {}
    for suffix in suffix_list:
        file_path = folder + f"gpu_time_{suffix}.log"
        gpu_time_json = extract_json_from_gpu_time_log(file_path)
        file_path = folder + f"python_time_{suffix}.log"
        python_time_json = extract_json_from_python_time_log(file_path)
        data[suffix] = {"gpu_time": gpu_time_json, "python_time": python_time_json}

    file_paths = [
        # folder + "gpu_time_ws=1_rk=0.json",
        # folder + "gpu_time_ws=2_rk=0.json",
        # folder + "gpu_time_ws=2_rk=1.json",
        folder + "gpu_time_ws=4_rk=0.json",
        folder + "gpu_time_ws=4_rk=1.json",
        folder + "gpu_time_ws=4_rk=2.json",
        folder + "gpu_time_ws=4_rk=3.json",
    ]
    iterations_to_process = [250*i+1 for i in range(100, 120)]
    for iteration in iterations_to_process:
        extract_time_excel_from_json(folder, file_paths, iteration, mode="gpu")


    file_paths = [folder + f"gpu_time_it={it}.csv" for it in iterations_to_process]
    merge_csv_which_have_same_columns(file_paths, folder + f"merged_gpu_time.csv")
    # delete all file_paths
    for file_path in file_paths:
        os.remove(file_path)

    file_paths = [
        # folder + "python_time_ws=1_rk=0.json",
        # folder + "python_time_ws=2_rk=0.json",
        # folder + "python_time_ws=2_rk=1.json",
        folder + "python_time_ws=4_rk=0.json",
        folder + "python_time_ws=4_rk=1.json",
        folder + "python_time_ws=4_rk=2.json",
        folder + "python_time_ws=4_rk=3.json",
    ]

    for iteration in iterations_to_process:
        extract_time_excel_from_json(folder, file_paths, iteration, mode="python")
    
    file_paths = [folder + f"python_time_it={it}.csv" for it in iterations_to_process]
    merge_csv_which_have_same_columns(file_paths, folder + f"merged_python_time.csv")
    # delete all file_paths
    for file_path in file_paths:
        os.remove(file_path)


def adjust(folder):
    suffix_list = get_suffix_in_folder(folder)
    print("processing suffix_list: ", suffix_list)
    # return 

    data = {}
    for suffix in suffix_list:
        file_path = folder + f"gpu_time_{suffix}.log"
        gpu_time_json = extract_json_from_gpu_time_log(file_path, load_genereated_json=False)
        file_path = folder + f"python_time_{suffix}.log"
        python_time_json = extract_json_from_python_time_log(file_path, load_genereated_json=False)
        data[suffix] = {"gpu_time": gpu_time_json, "python_time": python_time_json}

    file_paths = [folder + f"gpu_time_{suffix}.json" for suffix in suffix_list]

    iterations_to_process = [250*i+1 for i in range(1, 120, 2)]
    for iteration in iterations_to_process:
        extract_time_excel_from_json(folder, file_paths, iteration, mode="gpu")


    file_paths = [folder + f"gpu_time_it={it}.csv" for it in iterations_to_process]
    merge_csv_which_have_same_columns(file_paths, folder + f"merged_gpu_time.csv")
    # delete all file_paths
    for file_path in file_paths:
        os.remove(file_path)

    file_paths = [folder + f"python_time_{suffix}.json" for suffix in suffix_list]

    for iteration in iterations_to_process:
        extract_time_excel_from_json(folder, file_paths, iteration, mode="python")
    
    file_paths = [folder + f"python_time_it={it}.csv" for it in iterations_to_process]
    merge_csv_which_have_same_columns(file_paths, folder + f"merged_python_time.csv")
    # delete all file_paths
    for file_path in file_paths:
        os.remove(file_path)


def adjust_analyze_optimal(folder):
    gpu_time_df = pd.read_csv(folder + "merged_gpu_time.csv")
    python_time_df = pd.read_csv(folder + "merged_python_time.csv")

    # filter by ws=4
    gpu_time_df = gpu_time_df[gpu_time_df["ws"] == 4]
    python_time_df = python_time_df[python_time_df["ws"] == 4]

    #print len of gpu_time_df and python_time_df
    # print("len(gpu_time_df): ", len(gpu_time_df))
    # print("len(python_time_df): ", len(python_time_df))

    # get the total time. save in a list. 
    iterations_to_process = [250*i+1 for i in range(1, 120, 2)]

    gpu_forward_columns1 = [
        "10 preprocess time",
    ]
    gpu_forward_columns2 = [
        "24 updateDistributedStatLocally.updateTileTouched time",
        "30 InclusiveSum time",
        "40 duplicateWithKeys time",
        "50 SortPairs time",
        "60 identifyTileRanges time",
        "70 render time",
        "81 sum_n_render time",
        "82 sum_n_consider time",
        "83 sum_n_contrib time",
    ]
    gpu_backward_columns1 = [
        "b10 render time",
    ]
    gpu_backward_columns2 = [
        "b20 preprocess time",
    ]
    python_gpu_columns = {
        "forward": [gpu_forward_columns1, gpu_forward_columns2],
        "backward": [gpu_backward_columns1, gpu_backward_columns2],
    }

    def max_subtract_mean_time(all_time):
        max_time = max(all_time)
        mean_time = np.mean(all_time)
        return max_time - mean_time

    current_time = []
    estimated_optimal_time = []
    for idx, iteration in enumerate(iterations_to_process):
        # get the current time spent from statistics
        cur_time_sum = 0
        for column in [
                "forward",
                "[loss]prepare_for_distributed_loss_computation",
                "gt_image_load_to_gpu",
                "local_loss_computation",
                "backward",
                "optimizer_step"
            ]:
            cur_time_sum += python_time_df[column][idx*4:(idx+1)*4].max()
        current_time.append(round(cur_time_sum, 6))

        # get the optimal time estimated from statistics.
        estimated_time_sum = cur_time_sum

        # reduce time by averaging some components' time.
        for python_col in python_gpu_columns:
            for gpu_col_group in python_gpu_columns[python_col]:
                gpu_time_all = []
                for j in range(idx*4, (idx+1)*4):
                    # print(j)
                    gpu_time_sum = 0
                    for col in gpu_col_group:
                        gpu_time_sum += gpu_time_df[col][j:j+1].sum()
                    gpu_time_all.append(gpu_time_sum)
                estimated_time_sum -= max_subtract_mean_time(gpu_time_all)

        # We should also subtract the unbalanced time of forward loss and backward loss. 
        # However, current measure of local_loss_computation is wrong. 

        # Let me estimate it. local loss computation is about 25ms per GPU in ws=4 case. 
        # The unbalance range could be (20ms~30ms), potentially optimizable unbalance time is 5ms. 10ms for both forward and backward.
        # 10ms is 5% optimization opportunity. 

        # These are abandoned code.
        # loss_forward_all_time = (
        #     python_time_df["local_loss_computation"][idx*4:(idx+1)*4] + 
        #     python_time_df["gt_image_load_to_gpu"][idx*4:(idx+1)*4]
        # ).to_list()
        # estimated_time_sum -= max_subtract_mean_time(loss_forward_all_time)
        # forward loss: Current measure of local_loss_computation contains the unbalanced of: 
        #     - local loss computation + load gt-image loading + merge_image_tiles_by_pos
        #     - Basically, everything after the all2all computation.

        # estimated_time_sum -= max_subtract_mean_time(loss_forward_all_time)
        # backward loss. I need to only substruct 1 copy of the unbalanced time. 
        # Ignore the unbalance of local_loss_computation backward time for now. because it is too complicated to estimate.

        estimated_optimal_time.append(round(estimated_time_sum, 6))
    
    speed_up = [current_time[i]/estimated_optimal_time[i] for i in range(len(current_time))]

    # create a dataframe with iteration list, current_time list, estimated_optimal_time list, speed_up list as columns
    df = pd.DataFrame({
        'Iteration': iterations_to_process,
        'Current Time Spent': current_time,
        'Estimated Optimal Time Spent': estimated_optimal_time,
        'Speed Up': speed_up
    })
    df.to_csv(folder + "current_vs_optimal_time.csv", index=False)


    fig, ax1 = plt.subplots(figsize=(36, 6))

    # 绘制当前时间和估计的最佳时间
    color_current = 'tab:red'
    color_estimated = 'tab:green'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Time')
    ax1.plot(df['Iteration'], df['Current Time Spent'], color=color_current, label='Current Time')
    ax1.plot(df['Iteration'], df['Estimated Optimal Time Spent'], color=color_estimated, label='Estimated Optimal Time')
    ax1.tick_params(axis='y')
    ax1.xaxis.set_ticks(iterations_to_process)  # 确保迭代显示在X轴上
    ax1.legend(loc='upper left')

    # 实例化另一个Y轴共享相同的X轴
    ax2 = ax1.twinx()  
    color_speed_up = 'tab:blue'
    ax2.set_ylabel('Speed Up', color=color_speed_up)  # 我们已经处理了y轴的标签
    ax2.plot(df['Iteration'], df['Speed Up'], linestyle='--', color=color_speed_up)
    ax2.tick_params(axis='y', labelcolor=color_speed_up)
    ax2.legend(loc='upper right')
    fig.tight_layout()  # 为了布局不重叠
    # save the figure
    plt.savefig(folder + "current_vs_optimal_time.png")


def adjust3(folder):
    suffix_list = get_suffix_in_folder(folder)
    print("processing suffix_list: ", suffix_list)

    data = {}
    for suffix in suffix_list:
        file_path = folder + f"gpu_time_{suffix}.log"
        gpu_time_json = extract_json_from_gpu_time_log(file_path, load_genereated_json=True)
        file_path = folder + f"python_time_{suffix}.log"
        python_time_json = extract_json_from_python_time_log(file_path, load_genereated_json=True)
        data[suffix] = {"gpu_time": gpu_time_json, "python_time": python_time_json}

    file_paths = [folder + f"gpu_time_{suffix}.json" for suffix in suffix_list]

    iterations_to_process = [10*i+1 for i in range(1, 100, 1)]
    for iteration in iterations_to_process:
        extract_time_excel_from_json(folder, file_paths, iteration, mode="gpu")

    file_paths = [folder + f"gpu_time_it={it}.csv" for it in iterations_to_process]
    merge_csv_which_have_same_columns(file_paths, folder + f"merged_gpu_time.csv")
    # delete all file_paths
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)

    file_paths = [folder + f"python_time_{suffix}.json" for suffix in suffix_list]

    for iteration in iterations_to_process:
        extract_time_excel_from_json(folder, file_paths, iteration, mode="python")
    
    file_paths = [folder + f"python_time_it={it}.csv" for it in iterations_to_process]
    merge_csv_which_have_same_columns(file_paths, folder + f"merged_python_time.csv")
    # delete all file_paths
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)
    
    gpu_time_df = pd.read_csv(folder + "merged_gpu_time.csv")
    python_time_df = pd.read_csv(folder + "merged_python_time.csv")
    gpu_time_df = gpu_time_df[gpu_time_df["ws"] == 4]
    python_time_df = python_time_df[python_time_df["ws"] == 4]

    df_stats = None
    for rk in range(4):
        gpu_time_df_rk_i = gpu_time_df[gpu_time_df["rk"] == rk]

        # calculate the mean, std and coeff_of_var of gpu_time_df_rk_i, and save in df_stats
        mean_row = {"rk": rk, "mode": "mean"}
        std_row = {"rk": rk, "mode": "std"}
        coeff_of_var_row = {"rk": rk, "mode": "coeff_of_var"}


        for column in gpu_time_df.columns:
            if column in ["file_path", "ws", "rk"]:
                continue
            # print(column)
            data = gpu_time_df_rk_i[column].to_list()
            # print(data)
            mean = np.mean(data)
            std = np.std(data)
            coeff_of_var = std/mean
            mean_row[column] = round(mean, 6)
            std_row[column] = round(std, 6)
            coeff_of_var_row[column] = round(coeff_of_var, 6)


        if df_stats is None:
            df_stats = pd.DataFrame([mean_row])
        else:
            df_stats = df_stats._append(pd.DataFrame([mean_row]))
        df_stats = df_stats._append(pd.DataFrame([std_row]))
        df_stats = df_stats._append(pd.DataFrame([coeff_of_var_row]))

    # print(df_stats)
    df_stats.to_csv(folder + "gpu_time_stats.csv", index=False)

def compare_end2end_stats(save_folder, file_paths=None):
    # print(get_end2end_stats("experiments/adjust_baseline_garden4k_1/python_ws=1_rk=0.log"))

    if file_paths is None:
        file_paths = [
            "experiments/adjust_baseline_garden4k_1/python_ws=1_rk=0.log",
            "experiments/adjust_baseline_garden4k_1/python_ws=4_rk=0.log",
            "experiments/adjust_baseline_garden4k_1/python_ws=4_rk=1.log",
            "experiments/adjust_baseline_garden4k_1/python_ws=4_rk=2.log",
            "experiments/adjust_baseline_garden4k_1/python_ws=4_rk=3.log",
            "experiments/iteration_memory_4k_adjust2_1/python_ws=4_rk=0.log",
            "experiments/iteration_memory_4k_adjust2_1/python_ws=4_rk=1.log",
            "experiments/iteration_memory_4k_adjust2_1/python_ws=4_rk=2.log",
            "experiments/iteration_memory_4k_adjust2_1/python_ws=4_rk=3.log",
            # "experiments/repeat/adjust_2_garden4k_1/python_ws=4_rk=0.log",
            "experiments/repeat/adjust_2_garden4k_2/python_ws=4_rk=0.log",
            "experiments/repeat/adjust_2_garden4k_3/python_ws=4_rk=0.log",
            "experiments/adjust_baseline_garden4k_2/python_ws=1_rk=0.log",
            "experiments/adjust_baseline_garden4k_2/python_ws=4_rk=0.log",
            # "experiments/adjust_baseline_garden4k_2/python_ws=4_rk=1.log",
            # "experiments/adjust_baseline_garden4k_2/python_ws=4_rk=2.log", 
            # "experiments/adjust_baseline_garden4k_2/python_ws=4_rk=3.log",               
            "experiments/adjust_baseline_room4k_1/python_ws=1_rk=0.log",
            "experiments/adjust_baseline_room4k_1/python_ws=4_rk=0.log",
            "experiments/adjust_baseline_room4k_2/python_ws=1_rk=0.log",
            "experiments/adjust_baseline_room4k_2/python_ws=4_rk=0.log",
            "experiments/adjust_baseline_bicycle4k_1/python_ws=1_rk=0.log",
            "experiments/adjust_baseline_bicycle4k_1/python_ws=4_rk=0.log",
            "experiments/adjust_baseline_bicycle4k_2/python_ws=1_rk=0.log",
            "experiments/adjust_baseline_bicycle4k_2/python_ws=4_rk=0.log",
        ]

    df = None
    for file_path in file_paths:
        print("Processing: ", file_path)
        data = get_end2end_stats(file_path)
        if data is None:
            continue
        if df is None:
            df = pd.DataFrame([data])
        else:
            df = df._append(pd.DataFrame([data]))

    df.to_csv(save_folder + "compare_end2end_stats.csv", index=False)
    # print(df)

def compare_garden_adjust_mode(save_folder):
    folder = [
        "experiments/adjust_baseline_garden4k_1/",
        "experiments/repeat/adjust_2_garden4k_1/",
        "experiments/repeat/adjust_2_garden4k_2/",
        "experiments/repeat/adjust_2_garden4k_3/",
    ]
    color = [
        "tab:red",
        "tab:blue",
        "tab:green",
        "tab:orange",
    ]
    all_name = []

    df = None
    for f in folder:
        expe_name = f.split("/")[-2]
        all_name.append(expe_name)
        new_df = pd.read_csv(f + "current_vs_optimal_time.csv")
        if df is None:
            df = new_df[["Iteration"]]
            iterations_to_process = new_df["Iteration"].to_list()
            df[expe_name+"_time"] = new_df["Current Time Spent"]
        else:
            df[expe_name+"_time"] = new_df["Current Time Spent"]

    df.to_csv(save_folder+"compare_garden_adjust_mode.csv", index=False)


    fig, ax1 = plt.subplots(figsize=(42, 6))

    # 绘制当前时间和估计的最佳时间
    # color_current = 'tab:red'
    # color_estimated = 'tab:green'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Time')
    for expe_name, c in zip(all_name, color):
        ax1.plot(df['Iteration'], df[expe_name+"_time"], color=c, label=expe_name)
    ax1.tick_params(axis='y')
    ax1.xaxis.set_ticks(iterations_to_process)  # 确保迭代显示在X轴上
    ax1.legend(loc='upper left')

    # save the figure
    plt.savefig(save_folder + "compare_garden_adjust_mode.png")

def redistribute_analyze_comm_and_count3dgs(folders):
    dict_i2jsend_stats = {}
    dict_count3dgs_stats = {}
    for folder in folders:
        expe_name = folder.split("/")[-2]


        i2jsend_stats, i2jsend_iterations = extract_comm_count_from_i2jsend_log(folder)
        if dict_i2jsend_stats == {}:
            dict_i2jsend_stats["iterations"] = i2jsend_iterations
        # merge i2jsend_stats into dict_i2jsend_stats
        dict_i2jsend_stats[expe_name+":total_comm_count"] = i2jsend_stats["total_comm_count"]
        

        count3dgs_stats, count3dgs_iterations = extract_3dgs_count_from_python_log(folder)
        if dict_count3dgs_stats == {}:
            dict_count3dgs_stats["iterations"] = count3dgs_iterations
        for key in count3dgs_stats:
            dict_count3dgs_stats[expe_name+":"+key] = count3dgs_stats[key]

    df_i2jsend_stats = pd.DataFrame(dict_i2jsend_stats)
    df_i2jsend_stats.to_csv(folders[0]+"compare_i2jsend_stats.csv", index=False)
    df_count3dgs_stats = pd.DataFrame(dict_count3dgs_stats)
    df_count3dgs_stats.to_csv(folders[0]+"compare_count3dgs_stats.csv", index=False)



def analyze_heuristics(folder, image_count=6, working_image_ids=None):
    if folder[-1] != "/":
        folder += "/"
    suffix_list = [
        "ws=4_rk=0",
        "ws=4_rk=1",
        "ws=4_rk=2",
        "ws=4_rk=3",
    ]
    print("processing suffix_list: ", suffix_list)

    # strategy_history_ws=4_rk=0.json
    data = json.load(open(folder + "strategy_history_ws=4_rk=0.json", "r"))
    sampled_image_id = list(range(0, image_count))
    if working_image_ids is not None:
        sampled_image_id = working_image_ids

    df = pd.DataFrame(columns=["image_id", "iteration", "n_tiles_0", "time_0", "n_tiles_1", "time_1", "n_tiles_2", "time_2", "n_tiles_3", "time_3"])
    for image_id in sampled_image_id:
        history_for_one_image = data[str(image_id)]
        for tmp in history_for_one_image:
            iteration = tmp["iteration"]
            global_strategy_str = tmp["strategy"]["gloabl_strategy_str"]
            global_strategy = [int(x) for x in global_strategy_str.split(",")]
            global_n_tiles = [global_strategy[i+1]-global_strategy[i] for i in range(len(global_strategy)-1)]
            global_time = tmp["strategy"]["global_running_times"]
            df = df._append({
                "image_id": int(image_id),
                "iteration": int(iteration),
                "n_tiles_0": global_n_tiles[0],
                "time_0": round(global_time[0], 5),
                "n_tiles_1": global_n_tiles[1],
                "time_1": round(global_time[1], 5),
                "n_tiles_2": global_n_tiles[2],
                "time_2": round(global_time[2], 5),
                "n_tiles_3": global_n_tiles[3],
                "time_3": round(global_time[3], 5),
            }, ignore_index=True)

    df.to_csv(folder + "heuristics.csv", index=False)

    def get_df_for_one_image_id(df, image_id):
        return df[df["image_id"] == image_id]

    for image_id in sampled_image_id:
        one_image_id_df = get_df_for_one_image_id(df, image_id)
        epochs = range(0, len(one_image_id_df))

        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(35, 18))
        four_colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange']

        # 在第一个子图上绘制y1
        # axs[0].plot(x, y1, 'r')  # 'r'是红色的意思
        axs[0].set_title('N_tiles', fontsize=20)
        # axs[0].set_xlabel('epochs')
        axs[0].set_ylabel('N_tiles', fontsize=16)
        axs[0].plot(epochs, one_image_id_df['n_tiles_0'], color=four_colors[0], label='rk=0')
        axs[0].plot(epochs, one_image_id_df['n_tiles_1'], color=four_colors[1], label='rk=1')
        axs[0].plot(epochs, one_image_id_df['n_tiles_2'], color=four_colors[2], label='rk=2')
        axs[0].plot(epochs, one_image_id_df['n_tiles_3'], color=four_colors[3], label='rk=3')
        axs[0].tick_params(axis='y')
        axs[0].xaxis.set_ticks(epochs)  # 确保迭代显示在X轴上
        # make it larger
        axs[0].legend(loc='upper left', fontsize=16)


        # 在第二个子图上绘制y2
        axs[1].set_title('N_tiles_time', fontsize=20)
        # axs[1].set_xlabel('epochs')
        axs[1].set_ylabel('time', fontsize=16)
        axs[1].plot(epochs, one_image_id_df['time_0'], linestyle='-', color=four_colors[0])
        axs[1].plot(epochs, one_image_id_df['time_1'], linestyle='-', color=four_colors[1])
        axs[1].plot(epochs, one_image_id_df['time_2'], linestyle='-', color=four_colors[2])
        axs[1].plot(epochs, one_image_id_df['time_3'], linestyle='-', color=four_colors[3])
        axs[1].tick_params(axis='y')
        axs[1].xaxis.set_ticks(epochs)  # 确保迭代显示在X轴上
        # axs[1].legend(loc='upper left')

        def per_tile_time(n_tiles, time):
            # n_tiles, time are both column of a dataframe
            return time / n_tiles

        axs[2].set_title('per_tile_time', fontsize=20)
        axs[2].set_xlabel('epochs', fontsize=16)
        axs[2].set_ylabel('time', fontsize=16)
        per_tile_time_0 = per_tile_time(one_image_id_df['n_tiles_0'], one_image_id_df['time_0'])
        per_tile_time_1 = per_tile_time(one_image_id_df['n_tiles_1'], one_image_id_df['time_1'])
        per_tile_time_2 = per_tile_time(one_image_id_df['n_tiles_2'], one_image_id_df['time_2'])
        per_tile_time_3 = per_tile_time(one_image_id_df['n_tiles_3'], one_image_id_df['time_3'])
        axs[2].plot(epochs, per_tile_time_0, linestyle='--', color=four_colors[0], label='rk=0')
        axs[2].plot(epochs, per_tile_time_1, linestyle='--', color=four_colors[1], label='rk=1')
        axs[2].plot(epochs, per_tile_time_2, linestyle='--', color=four_colors[2], label='rk=2')
        axs[2].plot(epochs, per_tile_time_3, linestyle='--', color=four_colors[3], label='rk=3')
        axs[2].tick_params(axis='y')
        axs[2].xaxis.set_ticks(epochs)  # 确保迭代显示在X轴上
        # axs[2].legend(loc='upper left')

        plt.savefig(folder + "analyze_heuristics_"+str(image_id)+".png")

    # tiles_stats analysis.
    df = pd.DataFrame(columns=["camera_id", "epoch", "sum_n_render_0", "sum_n_render_1", "sum_n_render_2", "sum_n_render_3", "sum_n_consider_0", "sum_n_consider_1", "sum_n_consider_2", "sum_n_consider_3", "sum_n_contrib_0", "sum_n_contrib_1", "sum_n_contrib_2", "sum_n_contrib_3"])
    data = []
    for rk, suffix in enumerate(suffix_list):
        data_rki = json.load(open(folder + f"strategy_history_ws=4_rk={rk}.json", "r"))
        # only keep image_id in sampled_image_id
        data_rki = {k: v for k, v in data_rki.items() if int(k) in sampled_image_id}
        data.append(data_rki)
    for image_id in sampled_image_id:
        sum_n_render = []
        sum_n_consider = []
        sum_n_contrib = []
        for rk, suffix in enumerate(suffix_list):
            data_rki = data[rk]
            sum_n_render_rki = []
            sum_n_consider_rki = []
            sum_n_contrib_rki = []
            history_for_one_image = data_rki[str(image_id)]
            
            for tmp in history_for_one_image:
                iteration = tmp["iteration"]
                sum_n_render_rki.append(tmp["strategy"]["sum_n_render"])
                sum_n_consider_rki.append(tmp["strategy"]["sum_n_consider"])
                sum_n_contrib_rki.append(tmp["strategy"]["sum_n_contrib"])
            sum_n_render.append(sum_n_render_rki)
            sum_n_consider.append(sum_n_consider_rki)
            sum_n_contrib.append(sum_n_contrib_rki)
            # print(len(sum_n_render_rki))

        for epoch in range(len(sum_n_render[0])):
            df = df._append({
                "image_id": int(image_id),
                "epoch": epoch,
                "sum_n_render_0": sum_n_render[0][epoch],
                "sum_n_render_1": sum_n_render[1][epoch],
                "sum_n_render_2": sum_n_render[2][epoch],
                "sum_n_render_3": sum_n_render[3][epoch],
                "sum_n_consider_0": sum_n_consider[0][epoch],
                "sum_n_consider_1": sum_n_consider[1][epoch],
                "sum_n_consider_2": sum_n_consider[2][epoch],
                "sum_n_consider_3": sum_n_consider[3][epoch],
                "sum_n_contrib_0": sum_n_contrib[0][epoch],
                "sum_n_contrib_1": sum_n_contrib[1][epoch],
                "sum_n_contrib_2": sum_n_contrib[2][epoch],
                "sum_n_contrib_3": sum_n_contrib[3][epoch],
            }, ignore_index=True)

    # save df in the file.
    df.to_csv(folder + "tiles_stats.csv", index=False)

    for image_id in sampled_image_id:
        one_image_id_df = get_df_for_one_image_id(df, image_id)
        epochs = range(0, len(one_image_id_df))
        print(len(one_image_id_df))

        fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(50, 18))
        four_colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange']

        # render on one graph;
        axs[0].set_title('n_render', fontsize=20)
        # axs[0].set_xlabel('epochs')
        axs[0].set_ylabel('count', fontsize=16)
        axs[0].plot(epochs, one_image_id_df['sum_n_render_0'], color=four_colors[0], label='rk=0')
        axs[0].plot(epochs, one_image_id_df['sum_n_render_1'], color=four_colors[1], label='rk=1')
        axs[0].plot(epochs, one_image_id_df['sum_n_render_2'], color=four_colors[2], label='rk=2')
        axs[0].plot(epochs, one_image_id_df['sum_n_render_3'], color=four_colors[3], label='rk=3')
        axs[0].tick_params(axis='y')
        axs[0].xaxis.set_ticks(epochs)

        # consider and contrib on another graph. 
        axs[1].set_title('n_consider', fontsize=20)
        # axs[1].set_xlabel('epochs')
        axs[1].set_ylabel('count', fontsize=16)
        axs[1].plot(epochs, one_image_id_df['sum_n_consider_0'], color=four_colors[0], label='n_consider_0')
        axs[1].plot(epochs, one_image_id_df['sum_n_consider_1'], color=four_colors[1], label='n_consider_1')
        axs[1].plot(epochs, one_image_id_df['sum_n_consider_2'], color=four_colors[2], label='n_consider_2')
        axs[1].plot(epochs, one_image_id_df['sum_n_consider_3'], color=four_colors[3], label='n_consider_3')
        axs[1].tick_params(axis='y')
        axs[1].xaxis.set_ticks(epochs)

        # consider and contrib on another graph. 
        axs[2].set_title('n_contrib', fontsize=20)
        # axs[2].set_xlabel('epochs')
        axs[2].set_ylabel('count', fontsize=16)
        axs[2].plot(epochs, one_image_id_df['sum_n_contrib_0'], color=four_colors[0], label='n_contrib_0')
        axs[2].plot(epochs, one_image_id_df['sum_n_contrib_1'], color=four_colors[1], label='n_contrib_1')
        axs[2].plot(epochs, one_image_id_df['sum_n_contrib_2'], color=four_colors[2], label='n_contrib_2')
        axs[2].plot(epochs, one_image_id_df['sum_n_contrib_3'], color=four_colors[3], label='n_contrib_3')
        axs[2].tick_params(axis='y')
        axs[2].xaxis.set_ticks(epochs)


        # contrib/consider on the third graph.
        axs[3].set_title('contrib/consider', fontsize=20)
        axs[3].set_xlabel('epochs')
        axs[3].set_ylabel('ratio', fontsize=16)
        ratio_0 = one_image_id_df['sum_n_contrib_0'] / one_image_id_df['sum_n_consider_0']
        ratio_1 = one_image_id_df['sum_n_contrib_1'] / one_image_id_df['sum_n_consider_1']
        ratio_2 = one_image_id_df['sum_n_contrib_2'] / one_image_id_df['sum_n_consider_2']
        ratio_3 = one_image_id_df['sum_n_contrib_3'] / one_image_id_df['sum_n_consider_3']
        axs[3].plot(epochs, ratio_0, linestyle='--', color=four_colors[0], label='rk=0')
        axs[3].plot(epochs, ratio_1, linestyle='--', color=four_colors[1], label='rk=1')
        axs[3].plot(epochs, ratio_2, linestyle='--', color=four_colors[2], label='rk=2')
        axs[3].plot(epochs, ratio_3, linestyle='--', color=four_colors[3], label='rk=3')
        axs[3].tick_params(axis='y')
        axs[3].xaxis.set_ticks(epochs)

        plt.savefig(folder + "analyze_tiles_stats_"+str(image_id)+".png")


def check_GPU_utilization(folder):
    if folder[-1] != "/":
        folder += "/"
    suffix_list = get_suffix_in_folder(folder)

    # assert "ws=1_rk=0" in suffix_list and "ws=4_rk=0" in suffix_list, "ws=1_rk=0 and ws=4_rk=0 must be in the folder."
    print("processing suffix_list: ", suffix_list)

    data = {}
    for suffix in suffix_list:
        file_path = folder + f"gpu_time_{suffix}.log"
        gpu_time_json = extract_json_from_gpu_time_log(file_path, load_genereated_json=True)
        data[suffix] = gpu_time_json

    file_paths = [folder + f"gpu_time_{suffix}.json" for suffix in suffix_list]

    iterations_to_process = [250*i+1 for i in range(1, 80, 2)]
    compare_sum_time_df_ws1 = pd.DataFrame(columns=["iteration", "ws", "b10 render time", "b20 preprocess time", "70 render time", "10 preprocess time"])
    compare_sum_time_df_ws4 = pd.DataFrame(columns=["iteration", "ws", "b10 render time", "b20 preprocess time", "70 render time", "10 preprocess time"])
    for iteration in iterations_to_process:
        extract_time_excel_from_json(folder, file_paths, iteration, mode="gpu")
        save_path = folder + f"gpu_time_it={iteration}.csv"
        # df.to_csv(folder + f"{mode}_time_it="+ str(iteration) +".csv", index=False)
        df = pd.read_csv(save_path)
        # 4 columns: "b10 render time", "b20 preprocess time", "70 render time", "10 preprocess time"
        ws1_df = df[df["ws"] == 1]
        if not ws1_df.empty:
            compare_sum_time_df_ws1 = compare_sum_time_df_ws1._append({
                "iteration": iteration,
                "ws": 1,
                "b10 render time": round(ws1_df["b10 render time"].sum(), 6),
                "b20 preprocess time": round(ws1_df["b20 preprocess time"].sum(), 6),
                "70 render time": round(ws1_df["70 render time"].sum(), 6),
                "10 preprocess time": round(ws1_df["10 preprocess time"].sum(), 6),
            }, ignore_index=True)
        ws4_df = df[df["ws"] == 4]
        compare_sum_time_df_ws4 = compare_sum_time_df_ws4._append({
            "iteration": iteration,
            "ws": 4,
            "b10 render time": round(ws4_df["b10 render time"].sum(), 6),
            "b20 preprocess time": round(ws4_df["b20 preprocess time"].sum(), 6),
            "70 render time": round(ws4_df["70 render time"].sum(), 6),
            "10 preprocess time": round(ws4_df["10 preprocess time"].sum(), 6),
        }, ignore_index=True)
        # remove save_path
        if os.path.exists(save_path):
            os.remove(save_path)
    compare_sum_time_df_ws1.to_csv(folder + "compare_sum_time_ws=1.csv", index=False)
    compare_sum_time_df_ws4.to_csv(folder + "compare_sum_time_ws=4.csv", index=False)

def compare_GPU_utilization(save_folder, file_paths):
    # compare `b10 render time` for different df file_paths. 
    # draw a df with iteration as x-axis, file_paths as y-axis. 

    all_df = None
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        if all_df is None:
            all_df = df[["iteration", "b10 render time"]]
            # rename "b10 render time" to "baseline"
            all_df = all_df.rename(columns={"b10 render time": "baseline"})
        all_df[file_path] = df["b10 render time"] / all_df["baseline"]
    all_df.to_csv(save_folder + "compare_multiple_GPU_utilization.csv", index=False)

def draw_epoch_loss(file_paths):
    epoch_losses = []
    for file_path in file_paths:
        epoch_loss = []
        lines = open(file_path, "r").readlines()
        for line in lines:
            #epoch 2 loss: 0.17376218013391145
            if line.startswith("epoch "):
                epoch_loss.append(float(line.split(" ")[-1]))
        epoch_losses.append(epoch_loss)
    
    fig, ax = plt.subplots(figsize=(20, 10))
    for i, epoch_loss in enumerate(epoch_losses):
        ax.plot(range(len(epoch_loss)), epoch_loss, label=file_paths[i])
    ax.legend(loc='upper right')
    folder = "/".join(file_paths[0].split("/")[:-1]) + "/"
    plt.savefig(folder+"compare_epoch_loss.png")

def draw_evaluation_results(file_paths):
    eval_tests_PSNR = []
    eval_trains_PSNR = []
    iterations = []
    # Evaluating test: 
    for file_path in file_paths:
        lines = open(file_path, "r").readlines()
        eval_test_PSNR = []
        eval_train_PSNR = []
        for line in lines:
            # [ITER 30000] Evaluating test: L1 0.058287687942777805 PSNR 21.94811627739354
            # [ITER 30000] Evaluating train: L1 0.03144958354532719 PSNR 26.123293685913087
            if "Evaluating test: " in line:
                eval_test_PSNR.append(float(line.split(" ")[-1]))
                if len(eval_tests_PSNR) == 0:
                    iterations.append(int(line.split(" ")[1][:-1]))
            if "Evaluating train: " in line:
                eval_train_PSNR.append(float(line.split(" ")[-1]))
        eval_tests_PSNR.append(eval_test_PSNR)
        eval_trains_PSNR.append(eval_train_PSNR)

    # draw the two figures on the same graph.
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))
    for i, eval_test_PSNR in enumerate(eval_tests_PSNR):
        # x-axis is iteration
        # y-axis is PSNR
        ax[0].plot(iterations, eval_test_PSNR, label=file_paths[i])
    
    ax[0].set_ylabel('PSNR')
    secax = ax[0].secondary_yaxis('right')
    secax.set_ylabel('PSNR')
    ax[0].legend(loc='lower right')
    ax[0].set_title("Evaluating test PSNR")

    for i, eval_train_PSNR in enumerate(eval_trains_PSNR):
        # x-axis is iteration
        # y-axis is PSNR
        ax[1].plot(iterations, eval_train_PSNR, label=file_paths[i])

    ax[1].set_ylabel('PSNR')
    secax = ax[1].secondary_yaxis('right')
    secax.set_ylabel('PSNR')
    ax[1].legend(loc='lower right')
    ax[1].set_title("Evaluating train PSNR")

    folder = "/".join(file_paths[0].split("/")[:-1]) + "/"
    plt.savefig(folder+"compare_evaluation_results.png")


def i2jsend_size_(folder, working_image_ids=[0], draw1=True, draw2=True, draw3=True):
    if folder[-1] != "/":
        folder += "/"

    suffix_list = [
        "ws=4_rk=0",
        "ws=4_rk=1",
        "ws=4_rk=2",
        "ws=4_rk=3",
    ]

    strategy_history_rk0 = json.load(open(folder + "strategy_history_ws=4_rk=0.json", "r"))

    # get map_iteration_to_epoch
    map_iter_to_epoch = {}
    epoch = 0
    iteration = 0
    log_file = open(folder + "python_ws=4_rk=0.log", "r").readlines()
    for line in log_file:
        if line.startswith("iteration "):
            iteration += 1 # iteration starts from 1. 
            map_iter_to_epoch[iteration] = epoch  # epoch starts from 0. 
        if line.startswith("epoch "):
            epoch += 1

    # get map_iteration_to_n_3dgs for each worker
    def get_map_to_n_3dgs_for_a_worker(file_path):
        data = open(file_path, "r").readlines()
        map_iteration_to_n_3dgs = {}
        map_epoch_to_n_3dgs = []
        for line in data:
            if line.startswith("xyz shape: torch.Size(["):
                # xyz shape: torch.Size([13569, 3])
                cur_n_3dgs = int(line.split(",")[0].split("[")[-1])
            if line.startswith("iteration ") and " loss: " in line:
                iteration = int(line.split(" ")[1])
                map_iteration_to_n_3dgs[iteration] = cur_n_3dgs
            if line.startswith("epoch "):
                map_epoch_to_n_3dgs.append(cur_n_3dgs)
            if "densify_and_prune. " in line:
                cur_n_3dgs = int(line.split("Now num of 3dgs: ")[1].split(".")[0])
        return map_iteration_to_n_3dgs, map_epoch_to_n_3dgs
    rki_map_iter_to_n_3dgs = []
    rki_map_epoch_to_n_3dgs = []
    for suffix in suffix_list:
        file_path = folder + f"python_{suffix}.log"
        map_iteration_to_n_3dgs, map_epoch_to_n_3dgs = get_map_to_n_3dgs_for_a_worker(file_path)
        rki_map_iter_to_n_3dgs.append(map_iteration_to_n_3dgs)
        rki_map_epoch_to_n_3dgs.append(map_epoch_to_n_3dgs)


    def draw_3():
        # if file exists
        if os.path.exists(folder + "n_3dgs.png"):
            return
        
        # change y axis to be log mode.

        max_size = np.array(rki_map_epoch_to_n_3dgs).max()
        max_size = max_size // 10000 * 10000 + 10000
        y_axis_interval = max_size // 20 // 1000 * 1000

        print("Generating ", folder + "n_3dgs.png")
        # draw the figure.
        fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(20, 25))
        for rk in range(4):
            ax[0].plot(
                range(len(rki_map_epoch_to_n_3dgs[rk])),
                rki_map_epoch_to_n_3dgs[rk],
                label=f"rk={rk}'s n_3dgs'",
            )

        ax[0].set_xlabel('epoch', fontsize=25)
        ax[0].set_ylabel('count', fontsize=25)
        ax[0].legend(loc='lower right', fontsize=25)
        ax[0].xaxis.set_major_locator(MultipleLocator(5))
        ax[0].yaxis.set_major_locator(MultipleLocator(y_axis_interval))
        ax[0].set_title("n_3dgs Changes During Training", fontsize=25)

        # black, very thick line. sum among 4 workers.
        ax[1].plot(
            range(len(rki_map_epoch_to_n_3dgs[0])),
            np.array(rki_map_epoch_to_n_3dgs).sum(axis=0),
            label="sum 3dgs among 4 workers",
            color="black",
            linewidth=5,
        )
        ax[1].set_xlabel('epoch', fontsize=25)
        ax[1].set_ylabel('count', fontsize=25)
        ax[1].legend(loc='lower right', fontsize=25)
        ax[1].xaxis.set_major_locator(MultipleLocator(5))
        ax[1].set_title("sum 3dgs among 4 workers", fontsize=25)

        rki_map_epoch_to_n_3dgs_percent = np.array(rki_map_epoch_to_n_3dgs) / np.array(rki_map_epoch_to_n_3dgs).sum(axis=0, keepdims=True)

        for rk in range(4):
            ax[2].plot(
                range(len(rki_map_epoch_to_n_3dgs[rk])),
                rki_map_epoch_to_n_3dgs_percent[rk],
                label=f"rk={rk}'s n_3dgs percent",
            )
        max_y = rki_map_epoch_to_n_3dgs_percent.max()
        ax[2].set_xlabel('epoch', fontsize=25)
        ax[2].set_ylabel('percentage', fontsize=25)
        ax[2].legend(loc='lower right', fontsize=25)
        ax[2].xaxis.set_major_locator(MultipleLocator(5))
        ax[2].yaxis.set_major_locator(MultipleLocator(round(max_y/10, 2)))
        ax[2].set_ylim(0, max_y*2)
        ax[2].set_title("(local n_3dgs) / (global n_edgs) Changes During Training", fontsize=25)

        plt.savefig(folder + "n_3dgs.png")
    if draw3:
        draw_3()

    def draw_1(image_id, all_i2jsend_size):
        # if file exists
        if os.path.exists(folder + f"i2j_send_size_image_{image_id}.png"):
            return
        print("Generating ", folder + f"i2j_send_size_image_{image_id}.png")
        
        max_size = np.array(all_i2jsend_size).max()
        max_size = max_size // 10000 * 10000 + 10000
        y_axis_interval = max_size // 20 // 1000 * 1000

        # 4 by 4 grid and each subplot is a line chart.
        fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(40, 60))
        for i in range(4):
            for j in range(4):
                # make the line thicker
                ax[i][j].plot(
                    range(len(all_i2jsend_size)),
                    list(map(lambda t: t[i][j], all_i2jsend_size)),
                    linewidth=3,
                    label=f"gpu_{i}_send_to_gpu_{j}",
                )
                ax[i][j].set_title(f"gpu_{i}_send_to_gpu_{j}", fontsize=30)
                if i == 3:
                    ax[i][j].set_xlabel("epoch", fontsize=24)
                if j == 0:
                    ax[i][j].set_ylabel("n_3dgs", fontsize=24)
                ax[i][j].xaxis.set_major_locator(MultipleLocator(10))
                ax[i][j].yaxis.set_major_locator(MultipleLocator(y_axis_interval))
                ax[i][j].tick_params(axis='both', which='major', labelsize=22)
                ax[i][j].set_ylim(0, max_size)

        plt.savefig(folder + f"i2j_send_size_image_{image_id}.png")

    def draw_2(image_id, all_i2jsend_size):
        # if file exists
        if os.path.exists(folder + f"i2j_send_size_image_{image_id}_distribution.png"):
            return
        print("Generating ", folder + f"i2j_send_size_image_{image_id}_distribution.png")

        # 4 by 4 grid and each subplot is a line chart.
        fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(40, 20))
        # seperate the first row and the second row by a blank line.
        fig.subplots_adjust(hspace=1)


        all_i2jsend_size_np = np.array(all_i2jsend_size)

        # first row is the send volume distribution of i2j_send_size.
        for i in range(4):
            # = all_i2jsend_size_np / all_i2jsend_size_np.sum(axis=-1, keepdims=True)
            distribution_along_row = all_i2jsend_size_np / all_i2jsend_size_np.sum(axis=-1, keepdims=True)
            ax[0][i].set_title(f"gpu_{i}_send_volume_distribution", fontsize=30)
            ax[0][i].set_xlabel("epoch", fontsize=24)
            if i == 0:
                ax[0][i].set_ylabel("precentage", fontsize=24)
            for j in range(4):
                ax[0][i].plot(
                    range(len(all_i2jsend_size)),
                    distribution_along_row[:, i, j],
                    linewidth=2,
                    label=f"send_to_gpu_{j}",
                )

            ax[0][i].xaxis.set_major_locator(MultipleLocator(10))
            ax[0][i].yaxis.set_major_locator(MultipleLocator(0.1))
            ax[0][i].set_ylim(0, 1.0)
            ax[0][i].tick_params(axis='both', which='major', labelsize=22)
            ax[0][i].legend(loc='upper right')

        # Second row is the receive volume distribution of i2j_send_size.
        for i in range(4):
            distribution_along_column = all_i2jsend_size_np / all_i2jsend_size_np.sum(axis=-2, keepdims=True)
            ax[1][i].set_title(f"gpu_{i}_recv_volume_distribution", fontsize=30)
            ax[1][i].set_xlabel("epoch", fontsize=24)
            if i == 0:
                ax[1][i].set_ylabel("precentage", fontsize=24)
            for j in range(4):
                ax[1][i].plot(
                    range(len(all_i2jsend_size)),
                    distribution_along_column[:, j, i],
                    linewidth=2,
                    label=f"recv_from_gpu_{j}",
                )

            ax[1][i].xaxis.set_major_locator(MultipleLocator(10))
            ax[1][i].yaxis.set_major_locator(MultipleLocator(0.1))
            ax[1][i].set_ylim(0, 1.0)
            ax[1][i].tick_params(axis='both', which='major', labelsize=22)
            ax[1][i].legend(loc='upper right', fontsize=20)

        plt.savefig(folder + f"i2j_send_size_image_{image_id}_distribution.png")

    for image_id in working_image_ids:
        cur_strategy_history_rk0 = strategy_history_rk0[str(image_id)]
        all_i2jsend_size = []
        for epoch_id in range(len(cur_strategy_history_rk0)):
            all_i2jsend_size.append(cur_strategy_history_rk0[epoch_id]["strategy"]["i2j_send_size"])

        if draw1:
            draw_1(image_id, all_i2jsend_size)
        if draw2:
            draw_2(image_id, all_i2jsend_size)
        pass
    


def analyze_3dgs_change(folder):
    if folder[-1] != "/":
        folder += "/"

    suffix_list = [
        "ws=4_rk=0",
        "ws=4_rk=1",
        "ws=4_rk=2",
        "ws=4_rk=3",
    ]

    data = []
    for rk in range(4):
        file_path = folder + f"python_{suffix_list[rk]}.log"
        data.append(open(file_path, "r").readlines())

    iterations = []
    epochs = []

    n_3dgs_every_epoch = []
    visible_3dgs_every_epoch = []
    ave_3dgs_radii_every_epoch = []
    ave_3dgs_grad_norm_every_epoch = []

    cur_n_3dgs = 0
    cur_visible_3dgs = 0
    cur_ave_3dgs_radii = 0
    cur_ave_3dgs_grad_norm = 0
    n_image_in_this_epoch = 0

    for line_id, line in enumerate(data[0]):
        if "iteration " in line and " loss: " in line:
            iteration = int(line.split(" ")[1])
            iterations.append(iteration)
            n_image_in_this_epoch += 1

        if line.startswith("epoch "):
            epoch = int(line.split(" ")[1])
            epochs.append(epoch)

            assert len(iterations) % n_image_in_this_epoch == 0

            n_3dgs_every_epoch.append(cur_n_3dgs/n_image_in_this_epoch)
            visible_3dgs_every_epoch.append(cur_visible_3dgs/n_image_in_this_epoch)
            ave_3dgs_radii_every_epoch.append(cur_ave_3dgs_radii/n_image_in_this_epoch)
            ave_3dgs_grad_norm_every_epoch.append(cur_ave_3dgs_grad_norm/n_image_in_this_epoch)
            n_image_in_this_epoch = 0
            cur_n_3dgs = 0
            cur_visible_3dgs = 0
            cur_ave_3dgs_radii = 0
            cur_ave_3dgs_grad_norm = 0

        if line.startswith("local_n_3dgs: "):
            local_n_3dgs = 0
            local_visible_3dgs = 0
            local_sum_3dgs_radii = 0
            local_sum_3dgs_grad_norm = 0
            for rk in range(4):
                cur_line = data[rk][line_id]
                local_n_3dgs += int(cur_line.split("local_n_3dgs: ")[1].split(";")[0])
                local_visible_3dgs += int(cur_line.split("local_visible_3dgs: ")[1].split(";")[0])
                local_sum_3dgs_radii += float(cur_line.split("local_sum_3dgs_radii: ")[1].split(";")[0])
                local_sum_3dgs_grad_norm += float(cur_line.split("local_sum_3dgs_grad_norm: ")[1].split(";")[0])

            cur_n_3dgs += local_n_3dgs
            cur_visible_3dgs += local_visible_3dgs
            cur_ave_3dgs_radii += local_sum_3dgs_radii/local_visible_3dgs
            cur_ave_3dgs_grad_norm += local_sum_3dgs_grad_norm/local_visible_3dgs
        
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(20, 20))
    fig.subplots_adjust(hspace=0.5)
    ax[0].plot(epochs, n_3dgs_every_epoch, label="n_of_3dgs")
    ax[0].plot(epochs, visible_3dgs_every_epoch, label="n_of_visible_3dgs")
    # set y axis to be log mode.
    # ax[0].set_yscale('log')
    ax[0].set_title("n_of_3dgs and n_of_visible_3dgs", fontsize=24)
    ax[0].set_xlabel('epoch', fontsize=20)
    ax[0].set_ylabel('count', fontsize=20)
    ax[0].xaxis.set_major_locator(MultipleLocator(3))
    ax[0].legend(loc='lower right', fontsize=20)

    ax[1].plot(epochs, ave_3dgs_radii_every_epoch, label="ave_3dgs_radii")
    ax[1].set_title("average radii of visible 3dgs' covered area", fontsize=24)
    ax[1].set_xlabel('epoch', fontsize=20)
    ax[1].set_ylabel('radii', fontsize=20)
    ax[1].xaxis.set_major_locator(MultipleLocator(3))
    ax[1].legend(loc='upper right', fontsize=20)

    ax[2].plot(epochs, ave_3dgs_grad_norm_every_epoch, label="ave_3dgs_grad_norm")
    ax[2].set_title("average grad_norm of visible 3dgs position on the screen", fontsize=24)
    ax[2].set_xlabel('epoch', fontsize=20)
    ax[2].set_ylabel('grad_norm', fontsize=20)
    ax[2].xaxis.set_major_locator(MultipleLocator(3))
    ax[2].legend(loc='upper right', fontsize=20)

    plt.savefig(folder + "analyze_3dgs_change.png")

def redis_mode(folder):
    suffix_list = [
        "ws=4_rk=0",
        "ws=4_rk=1",
        "ws=4_rk=2",
        "ws=4_rk=3",
    ]

    time_data = {}
    for suffix in suffix_list:
        file_path = folder + f"gpu_time_{suffix}.log"
        gpu_time_json = extract_json_from_gpu_time_log(file_path)
        file_path = folder + f"python_time_{suffix}.log"
        python_time_json = extract_json_from_python_time_log(file_path)
        time_data[suffix] = {"gpu_time": gpu_time_json, "python_time": python_time_json}

    process_iterations = [i for i in range(251, 30000, 500)]

    file_paths = [folder + f"gpu_time_{suffix}.json" for suffix in suffix_list]
    for iteration in process_iterations:
        extract_time_excel_from_json(folder, file_paths, iteration, mode="gpu")
    file_paths = [folder + f"gpu_time_it={it}.csv" for it in process_iterations]
    merge_csv_which_have_same_columns(file_paths, folder + f"merged_gpu_time.csv")
    delete_all_file_paths(file_paths)

    file_paths = [folder + f"python_time_ws=4_rk={i}.json" for i in range(4)]
    for iteration in process_iterations:
        extract_time_excel_from_json(folder, file_paths, iteration, mode="python")
    file_paths = [folder + f"python_time_it={it}.csv" for it in process_iterations]
    merge_csv_which_have_same_columns(file_paths, folder + f"merged_python_time.csv")
    delete_all_file_paths(file_paths)

def compare_total_communication_volume_and_time(save_folder, folders):
    dict_i2jsend_stats = {
        "all_to_all_ave" : {},
        "all_to_all_max_ave" : {},
        "all_to_all_volume_sum" : {},
    }
    for folder in folders:
        expe_name = folder.split("/")[-2]

        file_path = folder + "strategy_history_ws=4_rk=0.json"
        data = json.load(open(file_path, "r"))

        all_i2jsend_size = 0
        for image_id in data:
            cur_strategy_history_rk0 = data[image_id]
            for epoch_id in range(len(cur_strategy_history_rk0)):
                for x in range(4):
                    for y in range(4):
                        if x == y:
                            continue
                        all_i2jsend_size += cur_strategy_history_rk0[epoch_id]["strategy"]["i2j_send_size"][x][y]


        python_time_csv = folder + "merged_python_time.csv"
        python_time_df = pd.read_csv(python_time_csv)
        python_time_df = python_time_df[python_time_df["ws"] == 4]
        # get the column `forward_all_to_all_communication`
        forward_all_to_all_communication = python_time_df["forward_all_to_all_communication"].to_list()
        forward_all_to_all_communication_ave = sum(forward_all_to_all_communication) / len(forward_all_to_all_communication)
        forward_all_to_all_communication_max = []
        for i in range(0, len(forward_all_to_all_communication), 4):
            forward_all_to_all_communication_max.append(max(forward_all_to_all_communication[i:i+4]))
        forward_all_to_all_communication_max_ave = sum(forward_all_to_all_communication_max) / len(forward_all_to_all_communication_max)
        dict_i2jsend_stats["all_to_all_ave"][expe_name] = forward_all_to_all_communication_ave
        dict_i2jsend_stats["all_to_all_max_ave"][expe_name] = forward_all_to_all_communication_max_ave
        dict_i2jsend_stats["all_to_all_volume_sum"][expe_name] = all_i2jsend_size

    json.dump(dict_i2jsend_stats, open(save_folder + "compare_communication_stats.json", "w"), indent=4)

def average_csv(file_path, save_path):
    df_all_ws = pd.read_csv(file_path)

    all_ws_in_df = []
    for x in df_all_ws["ws"].unique():
        # if x is not NaN
        if x == x:
            all_ws_in_df.append(int(x))

    data_to_save = {}
    for ws in all_ws_in_df:
        data_to_save_ws = {}
        df = df_all_ws[df_all_ws["ws"] == ws]
        for column in df.columns:
            if column == "rk" or column == "ws" or column == "file_path":
                continue
            col_data = df[column]
            line_id = 0
            new_col_data = []
            while line_id < len(df):
                new_col_data.append(col_data[line_id:line_id+ws].sum())
                line_id += ws
            # average
            data_to_save_ws[column] = round(np.mean(new_col_data), 3)
        data_to_save[f"ws={ws}"] = data_to_save_ws
    # json.dump(data_to_save, open(save_path, "w"), indent=4)
    return data_to_save

def average_gpu_python_time_csv(gpu_file_path, python_file_path, save_path):
    if os.path.exists(gpu_file_path):
        gpu_data = average_csv(gpu_file_path, save_path)
    else:
        gpu_data = None

    if os.path.exists(python_file_path):
        python_data = average_csv(python_file_path, save_path)
    else:
        python_data = None

    data = {
        "gpu": gpu_data,
        "python": python_data,
    }
    json.dump(data, open(save_path, "w"), indent=4)
        

def analyze_time(folder, process_iterations = [i for i in range(51, 7000, 50)], no_gpu_time=False, no_python_time=False):
    # suffix_list = [
    #     "ws=1_rk=0",
    #     "ws=4_rk=0",
    #     "ws=4_rk=1",
    #     "ws=4_rk=2",
    #     "ws=4_rk=3",
    # ]

    if folder[-1] != "/":
        folder += "/"
    suffix_list = get_suffix_in_folder(folder)

    time_data = {}
    for suffix in suffix_list:
        file_path = folder + f"gpu_time_{suffix}.log"
        gpu_time_json = extract_json_from_gpu_time_log(file_path)
        file_path = folder + f"python_time_{suffix}.log"
        python_time_json = extract_json_from_python_time_log(file_path)
        time_data[suffix] = {"gpu_time": gpu_time_json, "python_time": python_time_json}

    if not no_gpu_time:
        file_paths = [folder + f"gpu_time_{suffix}.json" for suffix in suffix_list]
        for iteration in process_iterations:
            extract_time_excel_from_json(folder, file_paths, iteration, mode="gpu")
        file_paths = [folder + f"gpu_time_it={it}.csv" for it in process_iterations]
        merge_csv_which_have_same_columns(file_paths, folder + f"merged_gpu_time.csv")
        delete_all_file_paths(file_paths)
    # average_csv(folder + f"merged_gpu_time.csv", folder + f"averaged_gpu_time.json")

    if not no_python_time:
        file_paths = [folder + f"python_time_{suffix}.json" for suffix in suffix_list]
        for iteration in process_iterations:
            extract_time_excel_from_json(folder, file_paths, iteration, mode="python")
        file_paths = [folder + f"python_time_it={it}.csv" for it in process_iterations]
        merge_csv_which_have_same_columns(file_paths, folder + f"merged_python_time.csv")
        delete_all_file_paths(file_paths)
    # average_csv(folder + f"merged_python_time.csv", folder + f"averaged_python_time.json")
    average_gpu_python_time_csv(folder + f"merged_gpu_time.csv", folder + f"merged_python_time.csv", folder + f"averaged_time.json")

def compare_1gpu_and_4gpu_time(file1, file2, column, save_folder):
    if not os.path.exists(file1) or not os.path.exists(file2):
        return

    if os.path.exists(save_folder+f"/compare_1gpu_and_4gpu_{column}.csv"):
        return
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    if column == "local_loss_computation":
        column1 = "loss"
        column4 = "local_loss_computation"
    else:
        column1 = column
        column4 = column


    df1_line_start = 0
    df4_line_start = 0

    # df_compare = pd.DataFrame(columns=["iteration", "1gpu", "4gpu.sum", "4gpu.max", "1gpu/(4gpu.sum)", "1gpu/(4gpu.max)", "1gpu/(4gpu.ave)"])
    df_compare = pd.DataFrame(columns=["iteration", "1gpu", "4gpu.sum", "4gpu.max", "1gpu/(4gpu.max)", "1gpu/(4gpu.ave)"])
    iterations = []
    while df1_line_start<len(df1) and df4_line_start<len(df2):
        # experiments/analyze_time_train/gpu_time_it=251.csv
        cur_iteration = int(df1.iloc[df1_line_start]["file_path"].split("/")[-1].split("=")[-1].split(".")[0])
        iterations.append(cur_iteration)
        _1gpu = df1.iloc[df1_line_start][column1]
        _4gpu_sum = df2.iloc[df4_line_start:df4_line_start+4][column4].sum()
        _4gpu_max = df2.iloc[df4_line_start:df4_line_start+4][column4].max()
        _4gpu_ave = df2.iloc[df4_line_start:df4_line_start+4][column4].mean()
        # round to 2 digits
        df_compare = df_compare._append({
            "iteration": cur_iteration,
            "1gpu": round(_1gpu, 2),
            "4gpu.sum": round(_4gpu_sum, 2),
            "4gpu.max": round(_4gpu_max, 2),
            # "1gpu/(4gpu.sum)": round(_1gpu/_4gpu_sum, 2),
            "1gpu/(4gpu.max)": round(_1gpu/_4gpu_max, 2),
            "1gpu/(4gpu.ave)": round(_1gpu/_4gpu_ave, 2),
        }, ignore_index=True)

        df1_line_start += 1 + 1
        df4_line_start += 4 + 1

    # the last row df_compare is average of all rows in df_compare
    df_compare = df_compare._append({
        "iteration": "average",
        "1gpu": round(df_compare["1gpu"].mean(), 2),
        "4gpu.sum": round(df_compare["4gpu.sum"].mean(), 2),
        "4gpu.max": round(df_compare["4gpu.max"].mean(), 2),
        # "1gpu/(4gpu.sum)": round(df_compare["1gpu/(4gpu.sum)"].mean(), 2),
        "1gpu/(4gpu.max)": round(df_compare["1gpu/(4gpu.max)"].mean(), 2),
        "1gpu/(4gpu.ave)": round(df_compare["1gpu/(4gpu.ave)"].mean(), 2),
    }, ignore_index=True)

    df_compare.to_csv(save_folder+f"/compare_1gpu_and_4gpu_{column}.csv", index=False)

def merged_compare_1gpu_and_4gpu_time(columns, folder):
    df = pd.DataFrame(columns=["name", "1gpu", "4gpu.sum", "4gpu.max", "1gpu/(4gpu.max)", "1gpu/(4gpu.ave)"])

    # average scaling for different components in the training process.
    for column in columns:
        csv_path = folder + f"/compare_1gpu_and_4gpu_{column}.csv"
        if not os.path.exists(csv_path):
            continue
        df_compare = pd.read_csv(csv_path)
        df = df._append({
            "name": column,
            "1gpu": df_compare.iloc[-1]["1gpu"],
            "4gpu.sum": df_compare.iloc[-1]["4gpu.sum"],
            "4gpu.max": df_compare.iloc[-1]["4gpu.max"],
            "1gpu/(4gpu.max)": df_compare.iloc[-1]["1gpu/(4gpu.max)"],
            "1gpu/(4gpu.ave)": df_compare.iloc[-1]["1gpu/(4gpu.ave)"],
        }, ignore_index=True)
    
    df.to_csv(folder + "/merged_compare_1gpu_and_4gpu.csv", index=False)

def compare_different_block_sizes(baseline_folder, folders):
    step_names_to_compare = [
        "10 preprocess time",
        "50 SortPairs time",
        "70 render time",
        "b10 render time",
        "b20 preprocess time"
    ]

    def get_BLOCK_sizes_and_end2end_throughput(folder):
        # This is not for WS=4
        lines = open(folder + "python_ws=4_rk=0.log", "r").readlines()
        # This is by default
        BLOCK_X = 16
        BLOCK_Y = 16
        ONE_DIM_BLOCK_SIZE = 256
        end2end_throughput = -1
        L1 = -1
        PSNR = -1
        for line in lines:
            # cuda_block_x: 8; cuda_block_y: 8; one_dim_block_size: 256;
            if "cuda_block_x" in line:
                BLOCK_X = int(line.split("cuda_block_x: ")[1].split(";")[0])
            if "cuda_block_y" in line:
                BLOCK_Y = int(line.split("cuda_block_y: ")[1].split(";")[0])
            if "one_dim_block_size" in line:
                ONE_DIM_BLOCK_SIZE = int(line.split("one_dim_block_size: ")[1].split(";")[0])
            # end2end total_time: 687.932720 ms, iterations: 30000, throughput 43.61 it/s
            if ", throughput " in line:
                end2end_throughput = float(line.split(", throughput ")[1].split(" it/s")[0])
            # [ITER 30000] Evaluating train: L1 0.036896323785185814 PSNR 24.66712532043457
            if "Evaluating train: L1" in line:
                L1 = round(float(line.split("Evaluating train: L1 ")[1].split(" PSNR")[0]), 3)
                PSNR = round(float(line.split(" PSNR ")[1].strip("\n")), 3)
        return BLOCK_X, BLOCK_Y, ONE_DIM_BLOCK_SIZE, end2end_throughput, L1, PSNR

    def get_step_times(folder, step_names_to_compare):
        #folder+averaged_time.json
        json_data = json.load(open(folder + "averaged_time.json", "r"))
        step_times = []
        for name in step_names_to_compare:
            if name in json_data["python"]["ws=4"]:
                step_times.append(json_data["python"]["ws=4"][name])
            elif name in json_data["gpu"]["ws=4"]:
                step_times.append(json_data["gpu"]["ws=4"][name])
        return step_times

    df = pd.DataFrame(columns=
                ["experiment_name", "BLOCK_X", "BLOCK_Y", "ONE_DIM_BLOCK_SIZE", "end2end_throughput", "L1", "PSNR"]
                +step_names_to_compare)
    all_folders = [baseline_folder] + folders
    for folder in all_folders:
        BLOCK_X, BLOCK_Y, ONE_DIM_BLOCK_SIZE, end2end_throughput, L1, PSNR = get_BLOCK_sizes_and_end2end_throughput(folder)
        step_times = get_step_times(folder, step_names_to_compare)
        df = df._append({
            "experiment_name": folder.split("/")[-2],
            "BLOCK_X": BLOCK_X,
            "BLOCK_Y": BLOCK_Y,
            "ONE_DIM_BLOCK_SIZE": ONE_DIM_BLOCK_SIZE,
            "end2end_throughput": end2end_throughput,
            "L1": L1,
            "PSNR": PSNR,
            **{step_names_to_compare[i]: step_times[i] for i in range(len(step_names_to_compare))}
        }, ignore_index=True)
    df.to_csv(baseline_folder + "/compare_different_block_sizes.csv", index=False)

if __name__ == "__main__":
    # NOTE: folder_path must end with "/" !!!


    # python_timer_0()
    # python_timer_1()
    # python_timer_sync_sparse_grad()
    # end2end_timer_0()
    # analyze_sparse_grad_speed_up()

    # bench_train_rows("experiments/bench_train_rows0/")
    # bench_train_rows("experiments/bench_train_rows1/")
    # bench_train_rows("experiments/bench_train_rows2/")
    # bench_train_rows("experiments/bench_train_rows3/")
    # bench_train_rows("experiments/bench_train_rows4/")
    # bench_train_rows("experiments/bench_train_rows5/")

    # memory_distribution_4("experiments/memory_distribution_4/")
    # memory_distribution_4_no("experiments/memory_distribution_4_no/")
    # memory_distribution_4_no_sep_render_ws1("experiments/memory_distribution_4_no_sep_render_ws1/")
    # bench_sklearn_dataset("/scratch/hz3496/sklearn/sklearn_dataset/")

    # div_stra_5_adjust("experiments/div_stra_5_adjust_none/")
    # div_stra_5_adjust("experiments/div_stra_5_adjust_n_contrib/")
    # merge_csv_for_div_stra_5_adjust()

    # mem_dist_stats_3("experiments/mem_dist_stats_3/")
    # mem_dist_stats_4("experiments/mem_dist_stats_4/")

    # mem_dist_stats_4k_garden_2("experiments/mem_dist_stats_4k_garden_2/")
    # mem_dist_stats_4k_garden_3("experiments/mem_dist_stats_4k_garden_3/")
    # mem_dist_stats_4k_garden_3("experiments/mem_dist_stats_4k_garden_3/")

    # adjust2("experiments/adjust2_1/")
    # adjust2("experiments/adjust2_2/")
    # adjust2("experiments/adjust2_3/")
    # adjust2("experiments/adjust2_4/")

    # adjust2("experiments/adjust2_4k_1/")
    # adjust2("experiments/adjust2_4k_2/")
    # adjust2("experiments/adjust2_4k_3/")
    # adjust2("experiments/time_stats_4k_30000its/")

    # adjust("experiments/adjust_baseline_garden4k_1/")
    # adjust("experiments/adjust_baseline_garden4k_2/")
    # adjust_analyze_optimal("experiments/adjust_baseline_garden4k_1/")
    # adjust_analyze_optimal("experiments/adjust_baseline_garden4k_2/")

    # adjust("experiments/adjust_baseline_bicycle4k_1/")
    # adjust("experiments/adjust_baseline_bicycle4k_2/")
    # adjust_analyze_optimal("experiments/adjust_baseline_bicycle4k_1/")
    # adjust_analyze_optimal("experiments/adjust_baseline_bicycle4k_2/")

    # adjust("experiments/adjust_baseline_room4k_1/")
    # adjust("experiments/adjust_baseline_room4k_2/")
    # adjust_analyze_optimal("experiments/adjust_baseline_room4k_1/")
    # adjust_analyze_optimal("experiments/adjust_baseline_room4k_2/")

    # adjust3("experiments/adjust3_1/")
    # adjust3("experiments/adjust3_1_1/")
    # adjust3("experiments/adjust3_1_2/")
    # adjust3("experiments/adjust3_1_3/")
    # adjust3("experiments/adjust3_1_4/")

    # adjust("experiments/adjust_2_garden4k_1/")
    # adjust_analyze_optimal("experiments/adjust_2_garden4k_1/")
    # adjust("experiments/adjust_2_garden4k_2/")
    # adjust_analyze_optimal("experiments/adjust_2_garden4k_2/")
    # adjust("experiments/adjust_2_garden4k_3/")
    # adjust_analyze_optimal("experiments/adjust_2_garden4k_3/")

    # adjust("experiments/repeat/adjust_2_garden4k_1/")
    # adjust_analyze_optimal("experiments/repeat/adjust_2_garden4k_1/")
    # adjust("experiments/repeat/adjust_2_garden4k_2/")
    # adjust_analyze_optimal("experiments/repeat/adjust_2_garden4k_2/")
    # adjust("experiments/repeat/adjust_2_garden4k_3/")
    # adjust_analyze_optimal("experiments/repeat/adjust_2_garden4k_3/")


    # compare_end2end_stats("experiments/adjust_baseline_garden4k_1/")
    # compare_garden_adjust_mode("experiments/adjust_baseline_garden4k_1/")

    # redistribute_analyze_comm_and_count3dgs([
    #     # "experiments/redistribute_1/",
    #     # "experiments/redistribute_1_baseline/",
    #     "experiments/redistribute_4k/",
    #     "experiments/redistribute_4k_baseline/",
    # ])

    # compare_end2end_stats(
    #     save_folder="experiments/iteration_memory_4k_adjust1/",
    #     file_paths=[
    #         "experiments/iteration_memory_4k_adjust1/python_ws=1_rk=0.log",
    #         "experiments/iteration_memory_4k_adjust1/python_ws=4_rk=0.log",
    #         "experiments/iteration_memory_4k_adjust1/python_ws=4_rk=1.log",
    #         "experiments/iteration_memory_4k_adjust1/python_ws=4_rk=2.log",
    #         "experiments/iteration_memory_4k_adjust1/python_ws=4_rk=3.log",
    #         "experiments/iteration_memory_4k_adjust2_1/python_ws=4_rk=0.log",
    #         "experiments/iteration_memory_4k_adjust2_1/python_ws=4_rk=1.log",
    #         "experiments/iteration_memory_4k_adjust2_1/python_ws=4_rk=2.log",
    #         "experiments/iteration_memory_4k_adjust2_1/python_ws=4_rk=3.log",
    #         "experiments/iter_mem_4k_adjust2_1_no/python_ws=4_rk=0.log",
    #         "experiments/iter_mem_4k_adjust2_1_no/python_ws=4_rk=1.log",
    #         "experiments/iter_mem_4k_adjust2_1_no/python_ws=4_rk=2.log",
    #         "experiments/iter_mem_4k_adjust2_1_no/python_ws=4_rk=3.log",
    #         "experiments/iteration_memory_4k_adjust2_4/python_ws=4_rk=0.log",
    #         "experiments/iteration_memory_4k_adjust2_4/python_ws=4_rk=1.log",
    #         "experiments/iteration_memory_4k_adjust2_4/python_ws=4_rk=2.log",
    #         "experiments/iteration_memory_4k_adjust2_4/python_ws=4_rk=3.log",
    #     ])

    # compare_end2end_stats(
    #     save_folder="experiments/iter_mem_4k_adjust2_5/",
    #     file_paths=[
    #         "experiments/iteration_memory_4k_adjust1/python_ws=1_rk=0.log",
    #         "experiments/iteration_memory_4k_adjust1/python_ws=4_rk=0.log",
    #         "experiments/iteration_memory_4k_adjust2_1/python_ws=4_rk=0.log",
    #         "experiments/iter_mem_4k_adjust2_1_no/python_ws=4_rk=0.log",
    #         "experiments/iteration_memory_4k_adjust2_4/python_ws=4_rk=0.log",
    #         "experiments/iter_mem_4k_adjust2_5/python_ws=4_rk=0.log",
    #         "experiments/iter_mem_4k_adjust2_5/python_ws=4_rk=1.log",
    #         "experiments/iter_mem_4k_adjust2_5/python_ws=4_rk=2.log",
    #         "experiments/iter_mem_4k_adjust2_5/python_ws=4_rk=3.log",
    #     ])

    # adjust("experiments/iteration_memory_4k_adjust1/")
    # adjust_analyze_optimal("experiments/iteration_memory_4k_adjust1/")
    # adjust("experiments/iteration_memory_4k_adjust2_1/")
    # adjust_analyze_optimal("experiments/iteration_memory_4k_adjust2_1/")
    # adjust("experiments/iter_mem_4k_adjust2_1_no/")
    # adjust_analyze_optimal("experiments/iter_mem_4k_adjust2_1_no/")
    # adjust("experiments/iteration_memory_4k_adjust2_4/")
    # adjust_analyze_optimal("experiments/iteration_memory_4k_adjust2_4/")


    # compare_end2end_stats(
    #     save_folder="experiments/iter_mem_bi_adjust1/",
    #     file_paths=[
    #         "experiments/iter_mem_bi_adjust1/python_ws=1_rk=0.log",
    #         "experiments/iter_mem_bi_adjust1/python_ws=4_rk=0.log",
    #         "experiments/iter_mem_bi_adjust1/python_ws=4_rk=1.log",
    #         "experiments/iter_mem_bi_adjust1/python_ws=4_rk=2.log",
    #         "experiments/iter_mem_bi_adjust1/python_ws=4_rk=3.log",
    #         "experiments/iter_mem_bi_adjust2_1/python_ws=4_rk=0.log",
    #         "experiments/iter_mem_bi_adjust2_1/python_ws=4_rk=1.log",
    #         "experiments/iter_mem_bi_adjust2_1/python_ws=4_rk=2.log",
    #         "experiments/iter_mem_bi_adjust2_1/python_ws=4_rk=3.log",
    #         "experiments/iter_mem_bi_adjust2_1n/python_ws=4_rk=0.log",
    #         "experiments/iter_mem_bi_adjust2_1n/python_ws=4_rk=1.log",
    #         "experiments/iter_mem_bi_adjust2_1n/python_ws=4_rk=2.log",
    #         "experiments/iter_mem_bi_adjust2_1n/python_ws=4_rk=3.log",
    #         "experiments/iter_mem_bi_adjust2_4/python_ws=4_rk=0.log",
    #         "experiments/iter_mem_bi_adjust2_4/python_ws=4_rk=1.log",
    #         "experiments/iter_mem_bi_adjust2_4/python_ws=4_rk=2.log",
    #         "experiments/iter_mem_bi_adjust2_4/python_ws=4_rk=3.log",
    #     ])

    # compare_end2end_stats(
    #     save_folder="experiments/iter_mem_bi_adjust2_5/",
    #     file_paths=[
    #         "experiments/iter_mem_bi_adjust1/python_ws=1_rk=0.log",
    #         "experiments/iter_mem_bi_adjust1/python_ws=4_rk=0.log",
    #         "experiments/iter_mem_bi_adjust2_1/python_ws=4_rk=0.log",
    #         "experiments/iter_mem_bi_adjust2_1n/python_ws=4_rk=0.log",
    #         "experiments/iter_mem_bi_adjust2_4/python_ws=4_rk=0.log",
    #         "experiments/iter_mem_bi_adjust2_5/python_ws=4_rk=0.log",
    #         "experiments/iter_mem_bi_adjust2_5/python_ws=4_rk=1.log",
    #         "experiments/iter_mem_bi_adjust2_5/python_ws=4_rk=2.log",
    #         "experiments/iter_mem_bi_adjust2_5/python_ws=4_rk=3.log",            
    #     ])


    # adjust("experiments/iter_mem_bi_adjust1/")
    # adjust_analyze_optimal("experiments/iter_mem_bi_adjust1/")
    # adjust("experiments/iter_mem_bi_adjust2_1/")
    # adjust_analyze_optimal("experiments/iter_mem_bi_adjust2_1/")
    # adjust("experiments/iter_mem_bi_adjust2_1n/")
    # adjust_analyze_optimal("experiments/iter_mem_bi_adjust2_1n/")
    # adjust("experiments/iter_mem_bi_adjust2_4/")
    # adjust_analyze_optimal("experiments/iter_mem_bi_adjust2_4/")


    # compare_end2end_stats(
    #     save_folder="experiments/iter_mem_bi15_adjust1/",
    #     file_paths=[
    #         "experiments/iter_mem_bi15_adjust1/python_ws=1_rk=0.log",
    #         "experiments/iter_mem_bi15_adjust1/python_ws=4_rk=0.log",
    #         "experiments/iter_mem_bi15_adjust1/python_ws=4_rk=1.log",
    #         "experiments/iter_mem_bi15_adjust1/python_ws=4_rk=2.log",
    #         "experiments/iter_mem_bi15_adjust1/python_ws=4_rk=3.log",
    #         "experiments/iter_mem_bi15_adjust2_1/python_ws=4_rk=0.log",
    #         "experiments/iter_mem_bi15_adjust2_1/python_ws=4_rk=1.log",
    #         "experiments/iter_mem_bi15_adjust2_1/python_ws=4_rk=2.log",
    #         "experiments/iter_mem_bi15_adjust2_1/python_ws=4_rk=3.log",
    #         "experiments/iter_mem_bi15_adjust2_1n/python_ws=4_rk=0.log",
    #         "experiments/iter_mem_bi15_adjust2_1n/python_ws=4_rk=1.log",
    #         "experiments/iter_mem_bi15_adjust2_1n/python_ws=4_rk=2.log",
    #         "experiments/iter_mem_bi15_adjust2_1n/python_ws=4_rk=3.log",
    #         "experiments/iter_mem_bi15_adjust2_4/python_ws=4_rk=0.log",
    #         "experiments/iter_mem_bi15_adjust2_4/python_ws=4_rk=1.log",
    #         "experiments/iter_mem_bi15_adjust2_4/python_ws=4_rk=2.log",
    #         "experiments/iter_mem_bi15_adjust2_4/python_ws=4_rk=3.log",
    #     ])
    
    # adjust("experiments/iter_mem_bi15_adjust1/")
    # adjust_analyze_optimal("experiments/iter_mem_bi15_adjust1/")
    # adjust("experiments/iter_mem_bi15_adjust2_1/")
    # adjust_analyze_optimal("experiments/iter_mem_bi15_adjust2_1/")
    # adjust("experiments/iter_mem_bi15_adjust2_1n/")
    # adjust_analyze_optimal("experiments/iter_mem_bi15_adjust2_1n/")
    # adjust("experiments/iter_mem_bi15_adjust2_4/")
    # adjust_analyze_optimal("experiments/iter_mem_bi15_adjust2_4/")

    # heuristics check.

    # image_count=6 by default
    # analyze_heuristics("experiments/heu_che_gar_adjust1/", image_count=20)
    # analyze_heuristics("experiments/heu_che_gar_adjust2_1/")
    # analyze_heuristics("experiments/heu_che_bi_adjust1/")
    # analyze_heuristics("experiments/heu_che_bi_adjust2_1/")

    ########################################################################
    #### analyze heu_tiles_stats_
    # analyze_heuristics("experiments/heu_tiles_stats_bi_adjust1/")#image_count=6
    # analyze_heuristics("experiments/heu_tiles_stats_bi_adjust2_1/")
    # analyze_heuristics("experiments/heu_tiles_stats_bi_adjust2_1n/")
    # analyze_heuristics("experiments/heu_tiles_stats_bi_adjust2_4/")
    # analyze_heuristics("experiments/heu_tiles_stats_bi_adjust2_5/")
    # analyze_heuristics("experiments/heu_tiles_stats_bi_adjust4/")

    # analyze_heuristics("experiments/heu_tiles_stats_gar_adjust1/")
    # analyze_heuristics("experiments/heu_tiles_stats_gar_adjust2_1/")
    # analyze_heuristics("experiments/heu_tiles_stats_gar_adjust2_1n/")
    # analyze_heuristics("experiments/heu_tiles_stats_gar_adjust2_4/")
    # analyze_heuristics("experiments/heu_tiles_stats_gar_adjust2_5/")
    # analyze_heuristics("experiments/heu_tiles_stats_gar_adjust4/")


    # compare_end2end_stats(
    #     save_folder="experiments/heu_tiles_stats_bi_adjust1/",
    #     file_paths=[
    #         "experiments/heu_tiles_stats_bi_adjust1/python_ws=1_rk=0.log",
    #         "experiments/heu_tiles_stats_bi_adjust1/python_ws=4_rk=0.log",
    #         "experiments/heu_tiles_stats_bi_adjust2_1/python_ws=4_rk=0.log",
    #         "experiments/heu_tiles_stats_bi_adjust2_1n/python_ws=4_rk=0.log",
    #         "experiments/heu_tiles_stats_bi_adjust2_4/python_ws=4_rk=0.log",
    #         "experiments/heu_tiles_stats_bi_adjust2_5/python_ws=4_rk=0.log",
    #         "experiments/heu_tiles_stats_bi_adjust4/python_ws=4_rk=0.log",
    #     ])
    # compare_end2end_stats(
    #     save_folder="experiments/heu_tiles_stats_gar_adjust1/",
    #     file_paths=[
    #         "experiments/heu_tiles_stats_gar_adjust1/python_ws=1_rk=0.log",
    #         "experiments/heu_tiles_stats_gar_adjust1/python_ws=4_rk=0.log",
    #         "experiments/heu_tiles_stats_gar_adjust2_1/python_ws=4_rk=0.log",
    #         "experiments/heu_tiles_stats_gar_adjust2_1n/python_ws=4_rk=0.log",
    #         "experiments/heu_tiles_stats_gar_adjust2_4/python_ws=4_rk=0.log",
    #         "experiments/heu_tiles_stats_gar_adjust2_5/python_ws=4_rk=0.log",
    #         "experiments/heu_tiles_stats_gar_adjust4/python_ws=4_rk=0.log",
    #     ])

    # check_GPU_utilization("experiments/heu_tiles_stats_gar_adjust1/")
    # check_GPU_utilization("experiments/heu_tiles_stats_gar_adjust2_1/")
    # check_GPU_utilization("experiments/heu_tiles_stats_gar_adjust2_1n/")
    # check_GPU_utilization("experiments/heu_tiles_stats_gar_adjust2_4/")
    # check_GPU_utilization("experiments/heu_tiles_stats_gar_adjust2_5/")
    # check_GPU_utilization("experiments/heu_tiles_stats_gar_adjust4/")
    # compare_GPU_utilization(save_folder="experiments/heu_tiles_stats_gar_adjust1/",
    #                         file_paths=
    #     [
    #         "experiments/heu_tiles_stats_gar_adjust1/compare_sum_time_ws=1.csv",
    #         "experiments/heu_tiles_stats_gar_adjust1/compare_sum_time_ws=4.csv",
    #         "experiments/heu_tiles_stats_gar_adjust2_1/compare_sum_time_ws=4.csv",
    #         "experiments/heu_tiles_stats_gar_adjust2_1n/compare_sum_time_ws=4.csv",
    #         "experiments/heu_tiles_stats_gar_adjust2_4/compare_sum_time_ws=4.csv",
    #         "experiments/heu_tiles_stats_gar_adjust2_5/compare_sum_time_ws=4.csv",
    #         "experiments/heu_tiles_stats_gar_adjust4/compare_sum_time_ws=4.csv",
    #     ]
    # )

    # check_GPU_utilization("experiments/heu_tiles_stats_bi_adjust1/")
    # check_GPU_utilization("experiments/heu_tiles_stats_bi_adjust2_1/")
    # check_GPU_utilization("experiments/heu_tiles_stats_bi_adjust2_1n/")
    # check_GPU_utilization("experiments/heu_tiles_stats_bi_adjust2_4/")
    # check_GPU_utilization("experiments/heu_tiles_stats_bi_adjust2_5/")
    # check_GPU_utilization("experiments/heu_tiles_stats_bi_adjust4/")
    # compare_GPU_utilization(save_folder="experiments/heu_tiles_stats_bi_adjust1/",
    #                         file_paths=
    #     [
    #         "experiments/heu_tiles_stats_bi_adjust1/compare_sum_time_ws=1.csv",
    #         "experiments/heu_tiles_stats_bi_adjust1/compare_sum_time_ws=4.csv",
    #         "experiments/heu_tiles_stats_bi_adjust2_1/compare_sum_time_ws=4.csv",
    #         "experiments/heu_tiles_stats_bi_adjust2_1n/compare_sum_time_ws=4.csv",
    #         "experiments/heu_tiles_stats_bi_adjust2_4/compare_sum_time_ws=4.csv",
    #         "experiments/heu_tiles_stats_bi_adjust2_5/compare_sum_time_ws=4.csv",
    #         "experiments/heu_tiles_stats_bi_adjust4/compare_sum_time_ws=4.csv",
    #     ]
    # )

    ########################################################################



    # draw_epoch_loss(
    #     [
    #         "experiments/bsz1/python_ws=4_rk=0.log",
    #         "experiments/bsz2/python_ws=4_rk=0.log",
    #         "experiments/bsz4/python_ws=4_rk=0.log",
    #     ]
    # )

    # draw_epoch_loss(
    #     [
    #         "experiments/bsz1_2/python_ws=4_rk=0.log",
    #         "experiments/bsz2_2/python_ws=4_rk=0.log",
    #         "experiments/bsz4_2/python_ws=4_rk=0.log",
    #     ]
    # )

    # draw_epoch_loss(
    #     [
    #         "experiments/bsz1/python_ws=4_rk=0.log",
    #         "experiments/bsz2/python_ws=4_rk=0.log",
    #         "experiments/bsz4/python_ws=4_rk=0.log",
    #         "experiments/bsz1_2/python_ws=4_rk=0.log",
    #         "experiments/bsz2_2/python_ws=4_rk=0.log",
    #         "experiments/bsz4_2/python_ws=4_rk=0.log",
    #     ]
    # )

    # draw_evaluation_results(
    #     [
    #         "experiments/bsz1_perf/python_ws=4_rk=0.log",
    #         "experiments/bsz2_perf/python_ws=4_rk=0.log",
    #         "experiments/bsz4_perf/python_ws=4_rk=0.log",
    #         "experiments/bsz8_perf/python_ws=4_rk=0.log",
    #         "experiments/bsz16_perf/python_ws=4_rk=0.log",
    #         "experiments/bsz32_perf/python_ws=4_rk=0.log",
    #     ]
    # )

    # analyze_heuristics("experiments/i2jsend_size_train_adjust_1/", working_image_ids=[0,10,20,30,40])
    # analyze_heuristics("experiments/i2jsend_size_train_adjust_2/", working_image_ids=[0,10,20,30,40])
    # analyze_heuristics("experiments/i2jsend_size_garden_adjust_1/", working_image_ids=[0,10,20,30,40])
    # analyze_heuristics("experiments/i2jsend_size_garden_adjust_2/", working_image_ids=[0,10,20,30,40])
    # analyze_heuristics("experiments/i2jsend_size_bicycle_adjust_1/", working_image_ids=[0,10,20,30,40])
    # analyze_heuristics("experiments/i2jsend_size_bicycle_adjust_2/", working_image_ids=[0,10,20,30,40])
    # i2jsend_size_("experiments/i2jsend_size_train_adjust_1/", [0,10,20,30,40])
    # i2jsend_size_("experiments/i2jsend_size_train_adjust_2/", [0,10,20,30,40])
    # i2jsend_size_("experiments/i2jsend_size_garden_adjust_1/", [0,10,20,30,40])
    # i2jsend_size_("experiments/i2jsend_size_garden_adjust_2/", [0,10,20,30,40])
    # i2jsend_size_("experiments/i2jsend_size_bicycle_adjust_1/", [0,10,20,30,40])
    # i2jsend_size_("experiments/i2jsend_size_bicycle_adjust_2/", [0,10,20,30,40])
    # draw_epoch_loss(["experiments/i2jsend_size_train_adjust_1/python_ws=4_rk=0.log",])
    # draw_epoch_loss(["experiments/i2jsend_size_train_adjust_2/python_ws=4_rk=0.log",])
    # draw_epoch_loss(["experiments/i2jsend_size_garden_adjust_1/python_ws=4_rk=0.log",])
    # draw_epoch_loss(["experiments/i2jsend_size_garden_adjust_2/python_ws=4_rk=0.log",])
    # draw_epoch_loss(["experiments/i2jsend_size_bicycle_adjust_1/python_ws=4_rk=0.log",])
    # draw_epoch_loss(["experiments/i2jsend_size_bicycle_adjust_2/python_ws=4_rk=0.log",])

    # analyze_3dgs_change("experiments/analyze_3dgs_change_train/")
    # analyze_3dgs_change("experiments/analyze_3dgs_change_garden/")
    # analyze_3dgs_change("experiments/analyze_3dgs_change_bicycle/")

    # i2jsend_size_("experiments/redis_mode_0/", [0,10,20,30,40])
    # i2jsend_size_("experiments/redis_mode_1/", [0,10,20,30,40])
    # i2jsend_size_("experiments/redis_mode_2/", [0,10,20,30,40])
    # compare_end2end_stats(
    #     save_folder="experiments/redis_mode_0/",
    #     file_paths=[
    #         "experiments/redis_mode_0/python_ws=4_rk=0.log",
    #         "experiments/redis_mode_0/python_ws=4_rk=1.log",
    #         "experiments/redis_mode_0/python_ws=4_rk=2.log",
    #         "experiments/redis_mode_0/python_ws=4_rk=3.log",
    #         "experiments/redis_mode_1/python_ws=4_rk=0.log",
    #         "experiments/redis_mode_1/python_ws=4_rk=1.log",
    #         "experiments/redis_mode_1/python_ws=4_rk=2.log",
    #         "experiments/redis_mode_1/python_ws=4_rk=3.log",
    #         "experiments/redis_mode_2/python_ws=4_rk=0.log",
    #         "experiments/redis_mode_2/python_ws=4_rk=1.log",
    #         "experiments/redis_mode_2/python_ws=4_rk=2.log",
    #         "experiments/redis_mode_2/python_ws=4_rk=3.log",
    #     ])

    # redis_mode("experiments/redis_mode_0/")
    # redis_mode("experiments/redis_mode_1/")


    # def mode_str(mode):
    #     if mode < 4:
    #         return str(mode)
    #     elif mode == 4:
    #         return "1_adjust2"

    # for scene in ["train", "garden", "bicycle"]:
    #     for mode in range(5):
    #         i2jsend_size_(f"experiments/redis_mode_{scene}_{mode_str(mode)}/", [0,10,20,30,40])
    #         redis_mode(f"experiments/redis_mode_{scene}_{mode_str(mode)}/")

    #     file_paths_for_compare = []
    #     for mode in range(5):
    #         for rk in range(4):
    #             file_paths_for_compare.append(f"experiments/redis_mode_{scene}_{mode_str(mode)}/python_ws=4_rk={rk}.log")
    #     compare_end2end_stats(
    #         save_folder=f"experiments/redis_mode_{scene}_0/",
    #         file_paths=file_paths_for_compare)
    #     compare_total_communication_volume_and_time(
    #         save_folder=f"experiments/redis_mode_{scene}_0/",
    #         folders=[f"experiments/redis_mode_{scene}_{mode_str(mode)}/" for mode in range(5)]
    #     )






    # for file_path in ["experiments/analyze_time_train/",
    #              "experiments/analyze_time_garden/",
    #              "experiments/analyze_time_bicycle/"]:
    #     process_iterations = [i for i in range(251, 30000, 500)]
    #     analyze_time(
    #         file_path,
    #         process_iterations
    #     )

    columns_python = ["forward",
                    "forward_prepare_gaussians",
                    "forward_preprocess_gaussians",
                    "forward_compute_locally",
                    "forward_render_gaussians",
                    "gt_image_load_to_gpu",
                    "local_loss_computation",
                    "optimizer_step",
                    "backward"]
    columns_gpu = ["70 render time",
                    "50 SortPairs time",
                    "b10 render time",
                    "b20 preprocess time"]
    # for col in columns_python:
    #     compare_1gpu_and_4gpu_time(
    #         "experiments/analyze_time_train/merged_python_time.csv",
    #         "experiments/redis_mode_train_1_adjust2/merged_python_time.csv",
    #         col,
    #         "experiments/analyze_time_train/"
    #     )
    # for col in columns_gpu:
    #     compare_1gpu_and_4gpu_time(
    #         "experiments/analyze_time_train/merged_gpu_time.csv",
    #         "experiments/redis_mode_train_1_adjust2/merged_gpu_time.csv",
    #         col,
    #         "experiments/analyze_time_train/"
    #     )

    # # compare 4 GPU training using different block sizes to 1 GPU training, and then see the difference.
    # for folder in ["analyze_time_train_bx16by16_64",
    #                "analyze_time_train_bx16by16_128",
    #                "analyze_time_train_bx16by8_256",
    #                "analyze_time_train_bx8by8_256",
    #                "analyze_time_train_bx32by8_256",
    #                "analyze_time_train_only_python_time",
    #                "redis_mode_train_1_adjust2",
    #                "faster_image_distri_train",
    #                "allreduce_image_distri_train",
    #                "func_allreduce_image_distri_train",
    #                "fast_less_comm_image_distri_train"]:
    #     analyze_time(
    #         f"experiments/{folder}/",
    #         [i for i in range(251, 30000, 500)]
    #     )
    #     for col in columns_python:
    #         compare_1gpu_and_4gpu_time(
    #             "experiments/analyze_time_train/merged_python_time.csv",
    #             f"experiments/{folder}/merged_python_time.csv",
    #             col,
    #             f"experiments/{folder}/"
    #         )
    #     for col in columns_gpu:
    #         compare_1gpu_and_4gpu_time(
    #             "experiments/analyze_time_train/merged_gpu_time.csv",
    #             f"experiments/{folder}/merged_gpu_time.csv",
    #             col,
    #             f"experiments/{folder}/"
    #         )

    #     merged_compare_1gpu_and_4gpu_time(
    #         columns = columns_python+columns_gpu,
    #         folder = f"experiments/{folder}/"
    #     )

    # analyze_time(
    #     "experiments/analyze_time_train1_bx8by8_256/",
    #     [i for i in range(251, 30000, 500)]
    # )
    # for col in columns_python:
    #     compare_1gpu_and_4gpu_time(
    #         "experiments/analyze_time_train1_bx8by8_256/merged_python_time.csv",
    #         "experiments/analyze_time_train_bx8by8_256/merged_python_time.csv",
    #         col,
    #         "experiments/analyze_time_train1_bx8by8_256/"
    #     )
    # for col in columns_gpu:
    #     compare_1gpu_and_4gpu_time(
    #         "experiments/analyze_time_train1_bx8by8_256/merged_gpu_time.csv",
    #         "experiments/analyze_time_train_bx8by8_256/merged_gpu_time.csv",
    #         col,
    #         "experiments/analyze_time_train1_bx8by8_256/"
    #     )


    # compare_different_block_sizes(
    #     baseline_folder="experiments/redis_mode_train_1_adjust2/",
    #     folders=[
    #         "experiments/analyze_time_train_bx16by16_128/",
    #         "experiments/analyze_time_train_bx16by16_64/",
    #         "experiments/analyze_time_train_bx16by8_256/",
    #         "experiments/analyze_time_train_bx8by8_256/",
    #         "experiments/analyze_time_train_bx32by8_256/",
    #     ]
    # )

    #dist_stra5_1
    # analyze_heuristics("experiments/dist_stra5_1/", working_image_ids=[0,10,20,30,40])


    # for folder in ["no_avoid_pixel_all2all_train",
    #                "avoid_pixel_all2all_train",
    #                "avoid_pixel_all2all_tr_flc",
    #                "avoid_pixel_all2all_train_2",
    #                "avoid_pixel_all2all_train_flcnal"]:
    #     analyze_time(
    #         f"experiments/{folder}/",
    #         [i for i in range(251, 30000, 500)]
    #     )
    #     analyze_heuristics(f"experiments/{folder}/", working_image_ids=[0,10,20,30,40])

    # for folder in [
    #                 "bicycle_mode1",
    #                 "bicycle_mode2",
    #                 "bicycle_mode4",
    #                 "garden_mode1",
    #                 "garden_mode2",
    #                 "garden_mode4",
    #                 "train_mode1",
    #                 "train_mode2",
    #                 "train_mode4",
    #                 ]:
    #     analyze_time(
    #         f"experiments/{folder}/",
    #         [i for i in range(251, 30000, 500)]
    #     )
    #     # analyze_heuristics(f"experiments/{folder}/", working_image_ids=[0,10,20,30,40])
    #     pass
    # for scene in ["train", "garden", "bicycle"]:
    #     compare_end2end_stats(
    #         save_folder=f"experiments/{scene}_mode1/",
    #         file_paths=[
    #             f"experiments/{scene}_mode1/python_ws=4_rk=0.log",
    #             f"experiments/{scene}_mode1/python_ws=4_rk=1.log",
    #             f"experiments/{scene}_mode1/python_ws=4_rk=2.log",
    #             f"experiments/{scene}_mode1/python_ws=4_rk=3.log",
    #             f"experiments/{scene}_mode2/python_ws=4_rk=0.log",
    #             f"experiments/{scene}_mode2/python_ws=4_rk=1.log",
    #             f"experiments/{scene}_mode2/python_ws=4_rk=2.log",
    #             f"experiments/{scene}_mode2/python_ws=4_rk=3.log",
    #             f"experiments/{scene}_mode4/python_ws=4_rk=0.log",
    #             f"experiments/{scene}_mode4/python_ws=4_rk=1.log",
    #             f"experiments/{scene}_mode4/python_ws=4_rk=2.log",
    #             f"experiments/{scene}_mode4/python_ws=4_rk=3.log",
    #         ])

    #     compare_total_communication_volume_and_time(
    #         save_folder=f"experiments/{scene}_mode1/",
    #         folders=[f"experiments/{scene}_mode{mode}/" for mode in [1,2,4]]
    #     )

    # for folder in ["bench_train_1gpu",
    #                 "bench_bicycle_1gpu",
    #                 "bench_garden_1gpu"]:
    #     analyze_time(
    #         f"experiments/{folder}/",
    #         [i for i in range(251, 30000, 500)]
    #     )
    #     scene = folder.split("_")[1]
    #     compare_end2end_stats(
    #         save_folder=f"experiments/bench_{scene}_1gpu/",
    #         file_paths=[
    #             f"experiments/bench_{scene}_1gpu/python_ws=1_rk=0.log",
    #             f"experiments/{scene}_mode1/python_ws=4_rk=0.log",
    #         ])

    dp_system_debug_expes = ["debug_dp_1gpu",
        "debug_dpsize1_memdis0_1",
        "debug_dpsize1_memdis1_1",
        "debug_dpsize2_memdis0_1",
        "debug_dpsize2_memdis0_adj5_1",
        "debug_dpsize2_memdis0_ws2_1",
        "debug_dpsize2_memdis1_1",
        "debug_dpsize2_memdis1_adj5_1",
        "debug_dpsize2_memdis2_1",
        "debug_dpsize2_memdis2_adj5_1",
        "debug_dpsize4_memdis0_1",
        "debug_dpsize4_memdis0_adj5_1",
        "debug_dpsize4_memdis1_1",
        "debug_dpsize4_memdis1_adj5_1",
        "debug_dpsize4_memdis2_1",
        "debug_dpsize4_memdis2_adj5_1"]
    for folder in dp_system_debug_expes:
        analyze_time(
            f"experiments/{folder}/",
            [i for i in range(51, 7000, 100)],
            no_gpu_time=True
        )
    compare_end2end_stats(
        save_folder=f"experiments/debug_dp_1gpu/",
        file_paths=[
            f"experiments/debug_dp_1gpu/python_ws=1_rk=0.log",
        ] + [f"experiments/{folder}/python_ws=4_rk=0.log" for folder in dp_system_debug_expes[1:]]
    )

    sync_grad_mode_expes = ["debug_dpsize2_memdis2_adj5_1",
        "debug_dpsize2_memdis2_adj5_spg_1",
        "debug_dpsize2_memdis2_adj5_fude_1"]
    for folder in sync_grad_mode_expes:
        analyze_time(
            f"experiments/{folder}/",
            [i for i in range(51, 7000, 100)],
            no_gpu_time=True
        )
    compare_end2end_stats(
        save_folder=f"experiments/debug_dpsize2_memdis2_adj5_1/",
        file_paths=[f"experiments/{folder}/python_ws=4_rk=0.log" for folder in sync_grad_mode_expes] +
                   [f"experiments/debug_dp_1gpu/python_ws=1_rk=0.log"]
    )
        