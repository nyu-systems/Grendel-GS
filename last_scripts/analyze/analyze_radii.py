import os
import matplotlib.pyplot as plt
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 生成指数衰减分布数据
np.random.seed(42)  # 为了结果可重复
data = np.random.exponential(scale=1.0, size=1000)  # 规模参数为1.0，生成1000个样本

# 绘制密度图
plt.figure(figsize=(10, 6))
sns.kdeplot(data, bw_adjust=0.5)
plt.title('Density Plot of Exponential Distribution')
plt.xlabel('Value')
plt.ylabel('Density')
plt.grid(True)

plt.savefig('exponential_density_plot.png')

scenes = [
    "Train",
    "Bicycle",
    "Rubble"
]

paths = [
    "/pscratch/sd/j/jy-nyu/last_scripts/tandb/4g4b/train",
    "/pscratch/sd/j/jy-nyu/last_scripts/mip360_1080p/4g_4b/e_bicycle",
    "/pscratch/sd/j/jy-nyu/last_scripts/rub/rub_16g_c"
]

widths = [
    980,
    1920,
    4591
]

save_folder = "/pscratch/sd/j/jy-nyu/last_scripts/radii_analyze"

all_radii = []
all_radii_over_width = []
for scene, path, width in zip(scenes, paths, widths):
    radii_path = os.path.join(path, "one_radii.json")
    with open(radii_path, "r") as f:
        radii = json.load(f)
    all_radii.append(radii)
    all_radii_over_width.append(np.array(radii) / width)


plt.figure(figsize=(10, 6))
for scene, radii, radii_over_width in zip(scenes, all_radii, all_radii_over_width):
    # use sns.kdeplot to draw thecumulative percent curve
    # x-axis: radii, y-axis: percent
    print(scene)
    radii_non_zero = [r for r in radii if r > 0]
    radii_non_zero_over_width = [r for r in radii_over_width if r > 0 and r < 1]
    # print(len(radii_non_zero))
    # print(len(radii))
    # print(len(radii) - len(radii_non_zero))
    # sns.kdeplot(radii_non_zero_over_width, cumulative=True, label=scene)
    sns.ecdfplot(radii_non_zero_over_width, label=scene)
    # set the x-axis to be log scale
    

plt.title('Cumulative Percent Curve of Radius Magnitude', fontsize=24)
plt.xlabel('Gaussian Footprint Radius / Image Width', fontsize=20)
# plt.xlabel(r'$\frac{\text{Gaussian Footprint Radius}}{\text{Image Width}}$', fontsize=15)
plt.xscale('log')
# set the x-axis font size
plt.xticks(fontsize=17)
# set the y-axis font size
plt.yticks(fontsize=17)
plt.ylabel('Cumulative Percent', fontsize=20)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.legend(fontsize=20)
plt.grid(True)
plt.savefig(save_folder + "/radii_cumulative_percent_curve.png")
plt.savefig(save_folder + "/radii_cumulative_percent_curve.svg")
plt.savefig(save_folder + "/radii_cumulative_percent_curve.pdf")

