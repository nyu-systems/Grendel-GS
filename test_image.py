import json

image10_p = "experiments/debug_dist_loss1/image_ws=1_rk=0_it=51.json"
image20_p = "experiments/debug_dist_loss1/image_ws=2_rk=0_it=51.json"
image21_p = "experiments/debug_dist_loss1/image_ws=2_rk=1_it=51.json"

with open(image10_p, "r") as f:
    image10 = json.load(f)

with open(image20_p, "r") as f:
    image20 = json.load(f)

with open(image21_p, "r") as f:
    image21 = json.load(f)

_, H, W = len(image10), len(image10[0]), len(image10[0][0])
print(H, W)

cnt = 0
for i in range(H):
    for j in range(W):
        # print
        # print(i, j, image10[0][i][j], image20[0][i][j], image21[0][i][j], image20[0][i][j] + image21[0][i][j])
        if abs(image10[0][i][j] - image20[0][i][j] - image21[0][i][j]) > 1e-3:
            cnt += 1
            print(i, j, image10[0][i][j], image20[0][i][j], image21[0][i][j], image20[0][i][j] + image21[0][i][j])
            # nothing output; that means the render forward is in the correct way.


