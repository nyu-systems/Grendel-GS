import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

all_loss_path = []

def change_log2json():
    for file in all_loss_path:
        with open(file, 'r') as f:
            lines = f.readlines()
            loss = []
            for line in lines:
                if line.startswith("iteration ") and "densify" not in line:
                    loss.append(round(float(line.split(' ')[-1]), 6))
            # print(file, len(loss), loss[:10])
            # save loss
            save_file_name = file.removesuffix(".log") + ".json"
            with open(save_file_name, 'w') as f:
                json.dump(loss, f)

def compare_loss2(save_file_name = None):
    # save in csv file
    loss = []
    for file in all_loss_path:
        with open(file.removesuffix(".log") + ".json", 'r') as f:
            loss.append({"name": file, "loss": json.load(f)})

    max_length = max([len(x["loss"]) for x in loss])

    for i in range(1, len(loss)):
        print(loss[i]["name"])
        # fill with zero
        loss[i]["loss"] = loss[i]["loss"] + [0] * (max_length - len(loss[i]["loss"]))

        loss_diff_ratio = list(abs(np.array(loss[i]["loss"]) - np.array(loss[0]["loss"])) / np.array(loss[i]["loss"]))
        # loss[i]["loss_diff_ratio_to_0"] = loss_diff_ratio
        # keep 8 digits
        loss[i]["loss_diff_ratio_to_0"] = [str(round(x*100, 4))+"%" for x in loss_diff_ratio]

    # save statistics
    # get the folder name from all_loss_path[0]
    path = all_loss_path[0].split("/python_")
    path = path[0]
    if save_file_name is None:
        save_file_name = path + "/loss_statistics.csv"

    if os.path.exists(save_file_name):
        os.remove(save_file_name)

    # create a dataframe
    df = pd.DataFrame()
    iterations = list(range(len(loss[0]["loss"])))
    df["iteration"] = iterations
    for i in range(len(loss)):
        df[loss[i]["name"]] = loss[i]["loss"]
        if i > 0:
            df[loss[i]["name"] + "_diff_ratio_to_0"] = loss[i]["loss_diff_ratio_to_0"]
    df.to_csv(save_file_name)

def expe_debug_dist_loss2():
    global all_loss_path
    all_loss_path = [
        'experiments/debug_dist_loss2/python_ws=1_rk=0.log',
        'experiments/debug_dist_loss2/python_ws=2_rk=0.log',
        'experiments/debug_dist_loss2/python_ws=2_rk=1.log',
        'experiments/debug_dist_loss2_again/python_ws=1_rk=0.log',
    ]

    change_log2json()
    compare_loss2()

def expe_debug_dist_loss3():

    global all_loss_path
    all_loss_path = [
        'experiments/debug_dist_loss3/python_ws=1_rk=0.log',
        'experiments/debug_dist_loss3/python_ws=2_rk=0.log',
        'experiments/debug_dist_loss3/python_ws=2_rk=1.log',
        'experiments/debug_dist_loss3_again/python_ws=1_rk=0.log',
    ]

    change_log2json()
    compare_loss2()

# expe_debug_dist_loss3()

def expe_debug_dist_loss4():

    global all_loss_path
    all_loss_path = [
        'experiments/debug_dist_loss4/python_ws=1_rk=0.log',
        'experiments/debug_dist_loss4/python_ws=2_rk=0.log',
        'experiments/debug_dist_loss4/python_ws=2_rk=1.log',
        'experiments/debug_dist_loss4_again/python_ws=1_rk=0.log',
    ]

    change_log2json()
    compare_loss2()

def expe_debug_dist_loss5():

    global all_loss_path
    all_loss_path = [
        'experiments/debug_dist_loss5/python_ws=1_rk=0.log',
        'experiments/debug_dist_loss5/python_ws=2_rk=0.log',
        'experiments/debug_dist_loss5/python_ws=2_rk=1.log',
        'experiments/debug_dist_loss5_again/python_ws=1_rk=0.log',
    ]

    change_log2json()
    compare_loss2()

# expe_debug_dist_loss5()

def expe_debug_dist_loss7():

    global all_loss_path
    all_loss_path = [
        'experiments/debug_dist_loss7/python_ws=1_rk=0.log',
        'experiments/debug_dist_loss7/python_ws=2_rk=0.log',
        'experiments/debug_dist_loss7/python_ws=2_rk=1.log',
        'experiments/debug_dist_loss7_again/python_ws=1_rk=0.log',
    ]

    change_log2json()
    compare_loss2()

# expe_debug_dist_loss7()

def expe_debug_dist_loss8():

    global all_loss_path
    all_loss_path = [
        'experiments/debug_dist_loss8/python_ws=1_rk=0.log',
        'experiments/debug_dist_loss8/python_ws=2_rk=0.log',
        'experiments/debug_dist_loss8/python_ws=2_rk=1.log',
        'experiments/debug_dist_loss8_again/python_ws=1_rk=0.log',
    ]

    change_log2json()
    compare_loss2()

# expe_debug_dist_loss8()
    
def expe_debug_dist_loss10():
    global all_loss_path
    all_loss_path = [
        'experiments/debug_dist_loss10/python_ws=1_rk=0.log',
        'experiments/debug_dist_loss10/python_ws=2_rk=0.log',
        'experiments/debug_dist_loss10/python_ws=2_rk=1.log',
        'experiments/debug_dist_loss10_again/python_ws=1_rk=0.log',
    ]

    change_log2json()
    compare_loss2()

# expe_debug_dist_loss10()

def expe_debug_dist_loss12():
    global all_loss_path
    all_loss_path = [
        'experiments/debug_dist_loss12_0/python_ws=1_rk=0.log',
        'experiments/debug_dist_loss12_1/python_ws=1_rk=0.log',
        'experiments/debug_dist_loss12_2/python_ws=1_rk=0.log',
        'experiments/debug_dist_loss12_3/python_ws=1_rk=0.log',
        'experiments/debug_dist_loss12_4/python_ws=1_rk=0.log',
    ]

    change_log2json()
    compare_loss2()

# expe_debug_dist_loss12()
    
def expe_debug_dist_loss13():
    global all_loss_path
    all_loss_path = [
        'experiments/debug_dist_loss13_0/python_ws=1_rk=0.log',
        'experiments/debug_dist_loss13_1/python_ws=1_rk=0.log',
        'experiments/debug_dist_loss13_2/python_ws=1_rk=0.log',
        'experiments/debug_dist_loss13_3/python_ws=1_rk=0.log',
        'experiments/debug_dist_loss13_4/python_ws=1_rk=0.log',
    ]

    change_log2json()
    compare_loss2()

# expe_debug_dist_loss13()

def expe_debug_dist_loss14():
    global all_loss_path
    all_loss_path = [
        'experiments/debug_dist_loss14_0/python_ws=1_rk=0.log',
        'experiments/debug_dist_loss14_1/python_ws=1_rk=0.log',
        'experiments/debug_dist_loss14_2/python_ws=1_rk=0.log',
        'experiments/debug_dist_loss14_3/python_ws=1_rk=0.log',
        'experiments/debug_dist_loss14_4/python_ws=1_rk=0.log',
    ]

    change_log2json()
    compare_loss2()

# expe_debug_dist_loss14()

def expe_debug_dist_loss15():
    global all_loss_path
    all_loss_path = [
        'experiments/debug_dist_loss15_0/python_ws=1_rk=0.log',
        'experiments/debug_dist_loss15_1/python_ws=1_rk=0.log',
        'experiments/debug_dist_loss15_2/python_ws=1_rk=0.log',
        'experiments/debug_dist_loss15_3/python_ws=1_rk=0.log',
        'experiments/debug_dist_loss15_4/python_ws=1_rk=0.log',
    ]

    change_log2json()
    compare_loss2()

# expe_debug_dist_loss15()

def expe_debug_dist_loss16():
    global all_loss_path
    all_loss_path = [
        'experiments/debug_dist_loss16_0/python_ws=1_rk=0.log',
        'experiments/debug_dist_loss16_1/python_ws=1_rk=0.log',
        'experiments/debug_dist_loss16_2/python_ws=1_rk=0.log',
        'experiments/debug_dist_loss16_3/python_ws=1_rk=0.log',
        'experiments/debug_dist_loss16_4/python_ws=1_rk=0.log',
    ]

    change_log2json()
    compare_loss2()

# expe_debug_dist_loss16()

def expe_debug_dist_loss17():
    global all_loss_path
    all_loss_path = [
        'experiments/debug_dist_loss17_0/python_ws=1_rk=0.log',
        'experiments/debug_dist_loss17_1/python_ws=1_rk=0.log',
        'experiments/debug_dist_loss17_2/python_ws=1_rk=0.log',
        'experiments/debug_dist_loss17_3/python_ws=1_rk=0.log',
        'experiments/debug_dist_loss17_4/python_ws=1_rk=0.log',
    ]

    change_log2json()
    compare_loss2()

# expe_debug_dist_loss17()

def expe_debug_dist_loss19():
    global all_loss_path
    all_loss_path = [
        'experiments/debug_dist_loss19_0/python_ws=1_rk=0.log',
        'experiments/debug_dist_loss19_1/python_ws=1_rk=0.log',
        'experiments/debug_dist_loss19_2/python_ws=1_rk=0.log',
        'experiments/debug_dist_loss19_3/python_ws=1_rk=0.log',
        'experiments/debug_dist_loss19_4/python_ws=1_rk=0.log',
    ]

    change_log2json()
    compare_loss2()

# expe_debug_dist_loss19()
    
def expe_debug_dist_loss20():
    global all_loss_path
    all_loss_path = [
        'experiments/debug_dist_loss13_0/python_ws=1_rk=0.log',
        'experiments/debug_dist_loss20_0/python_ws=2_rk=0.log',
        'experiments/debug_dist_loss20_0/python_ws=2_rk=1.log',
        'experiments/debug_dist_loss20_1/python_ws=2_rk=0.log',
        'experiments/debug_dist_loss20_1/python_ws=2_rk=1.log',
        'experiments/debug_dist_loss20_2/python_ws=2_rk=0.log',
        'experiments/debug_dist_loss20_2/python_ws=2_rk=1.log',
        'experiments/debug_dist_loss20_3/python_ws=2_rk=0.log',
        'experiments/debug_dist_loss20_3/python_ws=2_rk=1.log',
        'experiments/debug_dist_loss20_4/python_ws=2_rk=0.log',
        'experiments/debug_dist_loss20_4/python_ws=2_rk=1.log',
    ]

    change_log2json()
    compare_loss2('experiments/debug_dist_loss20_0/loss_statistics.csv')

# expe_debug_dist_loss20()

def expe_debug_dist_loss21():
    global all_loss_path
    all_loss_path = [
        'experiments/debug_dist_loss13_0/python_ws=1_rk=0.log',
        'experiments/debug_dist_loss21_0/python_ws=2_rk=0.log',
        'experiments/debug_dist_loss21_0/python_ws=2_rk=1.log',
        'experiments/debug_dist_loss21_1/python_ws=2_rk=0.log',
        'experiments/debug_dist_loss21_1/python_ws=2_rk=1.log',
        'experiments/debug_dist_loss21_2/python_ws=2_rk=0.log',
        'experiments/debug_dist_loss21_2/python_ws=2_rk=1.log',
        'experiments/debug_dist_loss21_3/python_ws=2_rk=0.log',
        'experiments/debug_dist_loss21_3/python_ws=2_rk=1.log',
        'experiments/debug_dist_loss21_4/python_ws=2_rk=0.log',
        'experiments/debug_dist_loss21_4/python_ws=2_rk=1.log',
    ]

    change_log2json()
    compare_loss2('experiments/debug_dist_loss21_0/loss_statistics.csv')

expe_debug_dist_loss21()