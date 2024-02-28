import torch
import random

def w_1_to_n(n):
    # generate 1,2,3...n tensor
    w = torch.arange(1, n+1, dtype=torch.float32, device="cuda")
    return w

def w_1_to_n_to_1(n):
    assert n % 2 == 0
    n = n // 2
    # generate 1,2,3...n,n-1,n-2...1 tensor
    w = torch.cat([torch.arange(1, n+1, dtype=torch.float32, device="cuda"), torch.arange(n, 0, -1, dtype=torch.float32, device="cuda")])
    return w

def w_gaussian(n):
    mean = n / 2
    std = n / 6
    x = torch.arange(1, n+1, dtype=torch.float32, device="cuda")
    w = torch.exp(-0.5 * ((x - mean) / std) ** 2)
    return w

def w_2_gaussian(n):
    mean1 = n / 3
    mean2 = 2 * n / 3
    std = n / 6
    x = torch.arange(1, n+1, dtype=torch.float32, device="cuda")
    w = torch.exp(-0.5 * ((x - mean1) / std) ** 2) + torch.exp(-0.5 * ((x - mean2) / std) ** 2)
    return w

def w_3_gaussian(n):
    mean1 = n / 4
    mean2 = n / 2
    mean3 = 3 * n / 4
    std = n / 6
    x = torch.arange(1, n+1, dtype=torch.float32, device="cuda")
    w = torch.exp(-0.5 * ((x - mean1) / std) ** 2) + torch.exp(-0.5 * ((x - mean2) / std) ** 2) + torch.exp(-0.5 * ((x - mean3) / std) ** 2)
    return w

def some_1_at_random_pos(n, portion_1=0.01, seed=0):
    random.seed(seed)
    w = torch.zeros(n, device="cuda")
    for i in range(int(n*portion_1)):
        pos = random.randint(0, n-1)
        w[pos] = 1
    return w

def w_sharp_gaussian(n, std):
    mean = n / 2
    x = torch.arange(1, n+1, dtype=torch.float32, device="cuda")
    w = torch.exp(-0.5 * ((x - mean) / std) ** 2)
    return w

def one_gaussian(n, mean, std):
    x = torch.arange(1, n+1, dtype=torch.float32, device="cuda")
    w = torch.exp(-0.5 * ((x - mean) / std) ** 2)
    return w

def some_gaussian(n, random_seed):
    random.seed(random_seed)
    w = torch.zeros(n, device="cuda")
    for i in range(5):
        mean = random.randint(0, n-1)
        std = random.randint(1, n//4)
        w += one_gaussian(n, mean, std)
    return w

def get_division_pos(w, n_groups):
    w_prefix_sum = torch.cumsum(w, dim=0)
    w_sum = w_prefix_sum[-1]
    w_per_group = w_sum / n_groups

    thresholds = torch.arange(1, n_groups, device="cuda") * w_per_group

    # use searchsorted to find the positions
    division_indices = torch.searchsorted(w_prefix_sum, thresholds)

    division_pos = [0] + division_indices.tolist() + [len(w)]
    return division_pos

def get_sum_portion(w, division_pos, m):
    w_sum = w.sum().item()
    sum_portion_for_different_parts = [w[division_pos[i]:division_pos[i+1]].sum().item()/w_sum for i in range(m)]
    return sum_portion_for_different_parts

def find_optimal_division(w_ground_truth, m, balance_threshold=0.01):

    # get ground truth division pos
    division_pos_gt = get_division_pos(w_ground_truth, m)
    sum_portions_gt = get_sum_portion(w_ground_truth, division_pos_gt, m)
    print("ground truth division pos: ", division_pos_gt)
    print("ground truth sum portion: ", sum_portions_gt)

    # initilize with 1
    w = torch.full_like(w_ground_truth, 1.0)

    iteration = 0
    while True:
        print("")
        iteration += 1
        if iteration > 50:
            print("iteration exceeds 50, break")
            break
        division_pos = get_division_pos(w, m)
        sum_portions = get_sum_portion(w_ground_truth, division_pos, m)
        # min and max is less than 0.05*max
        print("iteration: ", iteration)
        print("division pos: ", division_pos)
        print("sum portions: ", sum_portions)

        if max(sum_portions) - min(sum_portions) < balance_threshold * max(sum_portions):
            break
        w_new = torch.zeros_like(w)
        for i in range(m):
            w_new[division_pos[i]:division_pos[i+1]] = w_ground_truth[division_pos[i]:division_pos[i+1]].mean()
        w = w_new
    
    print("finished!\n")


if __name__ == "__main__":
    n = 100000
    m = 2

    # find_optimal_division(w_1_to_n(n), m)
    # find_optimal_division(w_1_to_n_to_1(n), m)
    # find_optimal_division(w_gaussian(n), m)
    # find_optimal_division(w_2_gaussian(n), m)
    # find_optimal_division(w_3_gaussian(n), m)
    # find_optimal_division(some_1_at_random_pos(n, 0.5), m)
    # find_optimal_division(some_1_at_random_pos(n, 0.1), m)
    # find_optimal_division(some_1_at_random_pos(n, 0.05), m)
    # find_optimal_division(some_1_at_random_pos(n, 0.01), m)
    # find_optimal_division(some_1_at_random_pos(n, 0.005), m)
    
    # find_optimal_division(some_1_at_random_pos(n, 0.001, seed=0), 2)
    # seed=0 needs 152 iterations. 
    # find_optimal_division(some_1_at_random_pos(n, 0.001, seed=127), 2)
    # seed=127 will never converge. because the optimal could not achieve 0.001 difference.
    # find_optimal_division(some_1_at_random_pos(n, 0.005, seed=44), 2)


    # find_optimal_division(w_sharp_gaussian(n, n/16), m)
    # find_optimal_division(w_sharp_gaussian(n, n/32), m)
    # find_optimal_division(w_sharp_gaussian(n, n/64), m)

    # find_optimal_division(some_gaussian(n, 0), m)
    # find_optimal_division(some_gaussian(n, 1), m)
    # find_optimal_division(some_gaussian(n, 2), m)
    # find_optimal_division(some_gaussian(n, 7), m)
    # it converges very slow. 50 iterations is not enough for it to converge. 
    find_optimal_division(some_gaussian(n, 13), m)
    pass
