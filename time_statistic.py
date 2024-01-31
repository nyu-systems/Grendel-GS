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

        assert data is not None, "Queried iteration statistics should be in the log file."

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
    df.to_csv(output_file_path, index=False)

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

    # if os.path.exists(file_path.removesuffix(".log") + ".json"):
    #     with open(file_path.removesuffix(".log") + ".json", 'r') as f:
    #         return json.load(f)

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
    return stats

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

def extract_json_from_gpu_time_log(file_path):
    
    file_name = file_path.split("/")[-1]
    ws, rk = file_name.split("_")[2].split("=")[1], file_name.split("_")[3].split("=")[1].split(".")[0]
    ws, rk = int(ws), int(rk)

    # if os.path.exists(file_path.removesuffix(".log") + ".json"):
    #     with open(file_path.removesuffix(".log") + ".json", 'r') as f:
    #         return json.load(f)

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
    mem_dist_stats_4k_garden_3("experiments/mem_dist_stats_4k_garden_3/")


    pass


