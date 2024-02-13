import numpy as np
import matplotlib.pyplot as plt
import os
import json

IMAGE_WIDTH = 980
IMAGE_HEIGHT = 545
BLOCK_X = 16
BLOCK_Y = 16
TILE_WIDTH = (IMAGE_WIDTH + BLOCK_X - 1) // BLOCK_X # 62
TILE_HEIGHT = (IMAGE_HEIGHT + BLOCK_Y - 1) // BLOCK_Y # 35

def pixel_to_tile_index(x, y):
    # return the tile index of the pixel (x, y) in the form of (tile_x, tile_y)
    return (x / BLOCK_X, y / BLOCK_Y)

def tile_index_to_pixel(tile_x, tile_y):
    # return the pixel index of the left up cornor of tile (tile_x, tile_y) in the form of (x, y)
    return (tile_x * BLOCK_X, tile_y * BLOCK_Y)

def get_rect(p, max_radius, grid):
    rect_min = (
        min(grid[0], max(0, int((p[0] - max_radius) / BLOCK_X))), 
        min(grid[1], max(0, int((p[1] - max_radius) / BLOCK_Y)))
    )
    rect_max = (
        min(grid[0], max(0, int((p[0] + max_radius + BLOCK_X - 1) / BLOCK_X))),
        min(grid[1], max(0, int((p[1] + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
    )
    return rect_min, rect_max

def read_rectangle(path):
    all_rectangle = []
    all_radius = []
    all_mean = []

    with open(path, 'r') as f:
        cnt = 0
        for line in f:
            if line.startswith("number of rendered:"):
                continue
            cnt += 1
            tmp = line.strip().split(" ")
            assert len(tmp) == 7, "line " + str(cnt) + " has wrong format"

            rectange = list(map(int, tmp[3:])) # rect_min.x << " " << rect_min.y << " " << rect_max.x << " " << rect_max.y
            radius = int(tmp[2])
            mean = list(map(float, tmp[:2]))

            rect_min, rect_max = get_rect(mean, radius, (TILE_WIDTH, TILE_HEIGHT))
            if rect_min[0] == rectange[0] and rect_min[1] == rectange[1] and rect_max[0] == rectange[2] and rect_max[1] == rectange[3]:
                pass
            else:
                pass
                # print("line " + str(cnt) + " has wrong rectangle. " + str(rect_min) + " " + str(rect_max) + " : " + str(rectange))

                # line 30446 has different rectangle. (17, 16) (19, 18) : [17, 16, 18, 18]
                # line 41703 has different rectangle. (19, 23) (21, 24) : [19, 22, 21, 24]
                # line 48324 has different rectangle. (15, 14) (19, 17) : [15, 13, 19, 17]
                # ...
                # the error might be rounding error; does not matter

            all_mean.append(mean)
            all_rectangle.append(rectange)
            all_radius.append(radius)

    return all_rectangle, all_radius

# rectangle, radius = read_rectangle("expe16/rectangle_iter=151.txt")

def hit_map(path):
    rectangle, radius = read_rectangle(path)
    hit = np.zeros((TILE_HEIGHT, TILE_WIDTH))

    for i in range(len(rectangle)):
        for x in range(rectangle[i][0], rectangle[i][2]):
            for y in range(rectangle[i][1], rectangle[i][3]):
                hit[y, x] += 1
    return hit

def draw_hit_map(path, iteration):
    hit = hit_map(path)
    print(hit.shape)
    # plot heatmap and save the hit map
    # refesh 

    plt.clf()
    plt.imshow(hit, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.savefig(folder+"/figures/tile_hit_map_iter=" + str(iteration) + ".png")
    
    # write this in file
    with open(folder+"/figures/tile_hit_map_iter=" + str(iteration) + ".txt", 'w') as f:
        f.write("num_rendered: " + str(np.sum(hit)) + "\n")
        for i in range(TILE_HEIGHT):
            for j in range(TILE_WIDTH):
                # make output to be fixed length for each number
                f.write(str(int(hit[i, j])).rjust(4))
                f.write(" ")
            f.write("\n")
    
def draw_many_hit_map():
    files = os.listdir(folder)
    os.makedirs(folder+"/figures", exist_ok=True)
    # file example: rectangle_iter=801.txt
    for file in files:
        if file.startswith("rectangle_iter="):
            print(file[15:-4])
            iteration = int(file[15:-4])
            path = folder+"/rectangle_iter=" + str(iteration) + ".txt"
            data = draw_hit_map(path, iteration)

def affected_tile(path):
    # return the number of affected tiles
    rectangle, radius  = read_rectangle(path)
    stat = np.zeros((TILE_WIDTH*TILE_HEIGHT+1, ), dtype=int)
    cnt = 0
    for i in range(len(rectangle)):
        stat[(rectangle[i][2] - rectangle[i][0]) * (rectangle[i][3] - rectangle[i][1])] += 1
        cnt += (rectangle[i][2] - rectangle[i][0]) * (rectangle[i][3] - rectangle[i][1])
    
    print("num_rendered: ", cnt)
    return stat

def draw_affected_tile(path, iteration):
    stat = affected_tile(path)
    os.makedirs(folder+"/figures", exist_ok=True)
    print(stat.shape) # (2171, )
    print(stat[:100])

    # plot the histogram to know the distribution of stat 1d array where y=stat[x]
    plt.clf() # clear the figure
    plt.figure(figsize=(10, 6))
    plt.yscale('log')  # Set the y-axis to a logarithmic scale
    plt.plot(stat, label='log(cnt)', color='blue')  # Plot A
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.title('Distribution of Affected Tiles for one 3dgs')
    plt.xlabel('Number of Affected Tiles by one 3dgs')
    plt.ylabel('Number of 3dgs(log scale)')
    plt.legend()
    plt.savefig(folder+"/figures/affected_tile_iter=" + str(iteration) + ".png")

    # save the stat in file
    with open(folder+"/figures/affected_tile_iter=" + str(iteration) + ".txt", 'w') as f:
        num_rendered = 0
        for i in range(len(stat)):
            num_rendered += stat[i]*i
        f.write("num_rendered: " + str(num_rendered) + "\n")
        for i in range(len(stat)):
            f.write(str(stat[i]))
            f.write("\n")

def draw_many_affected_tile():
    files = os.listdir(folder)
    os.makedirs(folder+"/figures", exist_ok=True)
    # file example: rectangle_iter=801.txt
    for file in files:
        if file.startswith("rectangle_iter="):
            print(file[15:-4])
            iteration = int(file[15:-4])
            path = folder+"/rectangle_iter=" + str(iteration) + ".txt"
            data = draw_affected_tile(path, iteration)

def sparsity_of_data_method1():
    path = folder+"/rectangle_iter=151.txt"
    rectangle, radius = read_rectangle(path)
    not_considered_data_points = 0
    for i in range(len(rectangle)):
        if (rectangle[i][2] - rectangle[i][0]) * (rectangle[i][3] - rectangle[i][1]) == 0 or radius[i] == 0:
            not_considered_data_points += 1

    print("not_considered_data_points: ", not_considered_data_points)
    print("total_data_points: ", len(rectangle))
    print("sparsity: ", not_considered_data_points / len(rectangle)*100, "%")


def sparsity_of_data_method2(path):
    
    # readlines of the file
    with open(path, 'r') as f:
        lines = f.readlines()

    # print("num_lines: ", len(lines))

    # get the number of data points
    not_considered_data_points = 0
    for line in lines:
        tmp = line.strip().split(" ")
        x_g = float(tmp[0])
        y_g = float(tmp[1])
        z_g = float(tmp[2])
        grad = np.array([x_g, y_g, z_g])
        if np.linalg.norm(grad) > 0:
            not_considered_data_points += 1

    # path
    print("path: ", path)
    print("not_considered_data_points: ", not_considered_data_points)
    print("total_data_points: ", len(lines))
    print("sparsity: ", not_considered_data_points / len(lines)*100, "%")
    return not_considered_data_points / len(lines)*100
        

def sparsity_of_data_method2_all():
    files = os.listdir(folder)
    # file example: rectangle_iter=801.txt
    sparsities = []
    for file in files:
        if file.startswith("_xyz_grad_"):
            path = folder+"/"+file
            sparsities.append(sparsity_of_data_method2(path))
    print("average sparsity: ", np.mean(sparsities), "%")


def num_of_3dgs_across_border(path):
    rectangle, radius  = read_rectangle(path)
    border_x = 31
    cnt = 0
    for i in range(len(rectangle)):
        if rectangle[i][0] <= border_x and border_x < rectangle[i][2]:
            cnt += 1

    print("num of rectangles across border_x: ", cnt)
    print("total num of rectangles: ", len(rectangle))
    print("ratio: ", cnt / len(rectangle)*100, "%")
    return cnt / len(rectangle)*100

def num_of_3dgs_across_border_all():
    files = os.listdir(folder)
    ratios = []
    for file in files:
        if file.startswith("rectangle_iter="):
            path = folder+"/"+file
            ratios.append(num_of_3dgs_across_border(path))
    print("average ratio: ", np.mean(ratios), "%")

def num_of_3dgs_for_different_location(path):
    rectangle, radius  = read_rectangle(path)
    border_x = 31
    shared = 0
    gpu0 = 0
    gpu1 = 0
    n = len(rectangle)
    for i in range(n):
        if rectangle[i][0] <= border_x and border_x < rectangle[i][2]:
            shared += 1
        elif rectangle[i][2] <= border_x:
            gpu0 += 1
        else:
            gpu1 += 1

    print("ratio of rectangles across border_x: ", shared / n *100, "%")
    print("ratio of rectangles in gpu0: ", gpu0 / n * 100, "%")
    print("ratio of rectangles in gpu1: ", gpu1 / n * 100, "%")

def num_of_3dgs_for_different_location_all():
    files = os.listdir(folder)
    ratios = []
    for file in files:
        if file.startswith("rectangle_iter="):
            path = folder+"/"+file
            print(path)
            num_of_3dgs_for_different_location(path)

folder = "expe17"
# draw_many_hit_map()
# draw_many_affected_tile()

# sparsity_of_data_method2()
# sparsity_of_data_method2_all()
# num_of_3dgs_across_border_all()
num_of_3dgs_for_different_location_all()