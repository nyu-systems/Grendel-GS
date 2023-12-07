import numpy as np
import json

def work1():
    folder = "expe11"

    # Load the data
    data01 = np.loadtxt(folder + "/_xyz_grad_0_1.txt")
    data02 = np.loadtxt(folder + "/_xyz_grad_0_2.txt")
    data12 = np.loadtxt(folder + "/_xyz_grad_1_2.txt")

    # Compute the difference between data01 and data02+data12
    data2_sum = data02 + data12

    data_diff = data01 - data2_sum
    # how to deal with the case when data01 is 0?
    data_diff_percent = abs(data01 - data2_sum)/abs(data01) * 100

    data_diff_over_threshood = data_diff_percent > 1

    # take out rows that are true in data_diff

    cnt = 0
    print(len(data_diff))
    for i in range(len(data_diff)):
        if data_diff_over_threshood[i, 0] or data_diff_over_threshood[i, 1] or data_diff_over_threshood[i, 2]:
            print("data id: ", i)
            print("data01: ", data01[i])
            print("data02: ", data02[i])
            print("data12: ", data12[i])
            print("data_diff: ", data_diff[i])
            print("data_diff_percent: ", np.around(data_diff_percent[i], 2))
            print("\n")

            cnt += 1
            if cnt > 10:
                break

    # cnt = 0
    # for i in [0,1,2,3,4,13,17]:
    #     print("data id: ", i)
    #     print("data01: ", data01[i])
    #     print("data02: ", data02[i])
    #     print("data12: ", data12[i])
    #     print("data_diff: ", data_diff[i])
    #     print("data_diff_percent: ", np.around(data_diff_percent[i], 2))
    #     print("\n")

    # output first 10 for debugging
    # print(data_diff[:100])

# work1()

def work2():
    folder1 = "expe5"
    folder2 = "expe6"

    # Load the data
    data1 = np.loadtxt(folder1 + "/_xyz_grad_0_1.txt")
    data2 = np.loadtxt(folder2 + "/_xyz_grad_0_1.txt")

    data_diff = data1 - data2
    # how to deal with the case when data01 is 0?
    data_diff_percent = abs(data_diff)/abs(data1) *100

    data_diff_over_threshood = data_diff_percent > 10

    cnt = 0
    for i in range(len(data_diff)):
        if data_diff_over_threshood[i, 0] or data_diff_over_threshood[i, 1] or data_diff_over_threshood[i, 2]:
            print("data id: ", i)
            print("data1: ", data1[i])
            print("data2: ", data2[i])
            print("data_diff: ", data_diff[i])
            print("data_diff_percent: ", np.around(data_diff_percent[i], 2))
            print("\n")

            cnt += 1
            if cnt > 10:
                break

# work2()

def work3():
    folder = "expe11"
    file0 = folder + "/image_0_1_1.json"
    file1 = folder + "/image_0_2_1.json"
    file2 = folder + "/image_1_2_1.json"
    
    # load data from file
    json0 = np.array(json.load(open(file0)))
    json1 = np.array(json.load(open(file1)))
    json2 = np.array(json.load(open(file2)))
    print(type(json1))

    print(json0[0, 0:10, 0:10])
    print(json1[0, 0:10, 0:10]) # many values around 0.23369782;
    print(json2[0, 0:10, 0:10]) # all zero

    print()

    print(json0[0, -10:, -10:])
    print(json1[0, -10:, -10:]) # many values around 0.23369782;
    print(json2[0, -10:, -10:]) # all zero

    # we expect that: json0 = json1 + json2, check if it is true
    json_sum = json1 + json2
    json_diff = json0 - json_sum
    json_diff_percent = abs(json_diff)/abs(json0) *100
    json_diff_over_threshood = json_diff_percent > 1
    # count how many points are over the threshold
    cnt = np.sum(json_diff_over_threshood)
    print("# of points are over the threshold: ", cnt)
    max_diff = np.max(json_diff_percent)# 0
    print("max diff: ", max_diff)# 0

# work3()

