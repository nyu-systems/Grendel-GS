import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import json

# TODO: delete them later
folder = None
file_names = None
num_render_file_names = None

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



def analyze_heuristics(folder, image_count=6):
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







    pass


