from scene.cameras import Camera
import torch.distributed as dist
import torch
import time
import utils.general_utils as utils

########################## Utility Functions ##########################

def strategy_str_to_interval(strategy_str):
    # strategy_str: `T:$l,$r`
    # return: (l, r)
    # print(strategy_str)
    l = int(strategy_str.split(":")[1].split(",")[0])
    r = int(strategy_str.split(":")[1].split(",")[1])
    return (l, r)

def interval_to_strategy_str(interval):
    # print(interval)
    return f"T:{interval[0]},{interval[1]}"

def get_tile_pixel_range(j, i, image_width, image_height):
    pix_minx = i * utils.BLOCK_X
    pix_miny = j * utils.BLOCK_Y
    pix_maxx = min((i+1) * utils.BLOCK_X, image_width)
    pix_maxy = min((j+1) * utils.BLOCK_Y, image_height)
    return pix_minx, pix_miny, pix_maxx, pix_maxy

def get_tile_pixel_cnt(j, i, image_width, image_height):
    pix_minx, pix_miny, pix_maxx, pix_maxy = get_tile_pixel_range(j, i, image_width, image_height)
    return (pix_maxx - pix_minx) * (pix_maxy - pix_miny)

def division_pos_to_global_strategy_str(division_pos):
    # division_pos: [0, d1, d2, ..., tile_num]
    # return: "0,100,200,300,400,500,600,700,800,900,1000"
    return ",".join(map(str, division_pos))

def get_evenly_division_pos(camera):
    tile_x = (camera.image_width + utils.BLOCK_X - 1) // utils.BLOCK_X
    tile_y = (camera.image_height + utils.BLOCK_Y - 1) // utils.BLOCK_Y
    tile_num = tile_x * tile_y

    # return division_pos # format:[0, d1, d2, ..., tile_num]
    if tile_num % utils.MP_GROUP.size() == 0:
        cnt = tile_num // utils.MP_GROUP.size()
    else:
        cnt = tile_num // utils.MP_GROUP.size() + 1
    division_pos = [cnt * i for i in range(utils.MP_GROUP.size())] + [tile_num]
    return division_pos

def get_evenly_global_strategy_str(camera):
    division_pos = get_evenly_division_pos(camera)
    return division_pos_to_global_strategy_str(division_pos)

def check_division_indices_globally_same(division_indices):
    recevie = [None for _ in range(utils.MP_GROUP.size())]
    torch.distributed.all_gather_object(recevie, division_indices, group=utils.MP_GROUP)
    for i in range(utils.MP_GROUP.size()):
        for j in range(utils.MP_GROUP.size()):
            assert recevie[i][j] == division_indices[j], f"check_division_indices_globally_save failed: {i} {j}"

def division_pos_heuristic(cur_heuristic, tile_num, world_size):
    assert cur_heuristic.shape[0] == tile_num, "the length of heuristics should be the same as the number of tiles."
    heuristic_prefix_sum = torch.cumsum(cur_heuristic, dim=0)
    heuristic_sum = heuristic_prefix_sum[-1]
    heuristic_per_worker = heuristic_sum / world_size

    thresholds = torch.arange(1, world_size, device="cuda") * heuristic_per_worker
    division_pos = [0]

    # Use searchsorted to find the positions
    division_indices = torch.searchsorted(heuristic_prefix_sum, thresholds)

    # check_division_indices_globally_same(division_indices)

    # Convert to a Python list and prepend the initial division at 0.
    division_pos = [0] + division_indices.cpu().tolist() + [tile_num]

    return division_pos

def get_local_running_time_by_modes(stats_collector):
    args = utils.get_args()

    if args.render_distribution_adjust_mode == "1":
        return stats_collector["backward_render_time"]

    if args.render_distribution_adjust_mode == "2":
        if args.loss_distribution_mode == "fast_less_comm_noallreduceloss":
            return (
                stats_collector["backward_render_time"]+
                2*stats_collector["forward_loss_time"]
            )
        else:
            return stats_collector["backward_render_time"]

    if args.render_distribution_adjust_mode == "3":
        return stats_collector["backward_render_time"]

    if args.render_distribution_adjust_mode == "4":
        return stats_collector["backward_render_time"]

    if args.render_distribution_adjust_mode == "5":
        return (
            stats_collector["forward_render_time"]+
            stats_collector["backward_render_time"]+
            2*stats_collector["forward_loss_time"]
        )
        # return stats_collector["pixelwise_workloads_time"]

    if args.render_distribution_adjust_mode == "6":
        return (
            stats_collector["forward_render_time"]+
            stats_collector["backward_render_time"]+
            2*stats_collector["forward_loss_time"]
        )

    raise ValueError(f"Unknown render_distribution_adjust_mode: {args.render_distribution_adjust_mode}")

########################## DivisionStrategy ##########################
class DivisionStrategy:

    def __init__(self, camera, world_size, rank, tile_x, tile_y, division_pos, render_distribution_adjust_mode):
        self.camera = camera
        self.world_size = world_size
        self.rank = rank
        self.tile_x = tile_x
        self.tile_y = tile_y
        self.division_pos = division_pos
        self.render_distribution_adjust_mode = render_distribution_adjust_mode

    def get_compute_locally(self):
        tile_ids_l, tile_ids_r = self.division_pos[self.rank], self.division_pos[self.rank+1]
        compute_locally = torch.zeros(self.tile_y*self.tile_x, dtype=torch.bool, device="cuda")
        compute_locally[tile_ids_l:tile_ids_r] = True
        compute_locally = compute_locally.view(self.tile_y, self.tile_x)
        return compute_locally

    def is_avoid_pixel_all2all(self):
        # by default, each gpu compute it local tiles in forward and use all2all to fetch pixels near border on other GPUs, for later loss computation.
        if self.render_distribution_adjust_mode in  [None, "1", "2", "3", "4", "evaluation"]:
            return False
        if self.render_distribution_adjust_mode in ["5", "6"]:
            return True
        raise ValueError(f"Unknown render_distribution_adjust_mode: {self.render_distribution_adjust_mode}")

    def update_stats(self, stats_collector, n_render, n_consider, n_contrib, i2j_send_size):
        pass

    def get_global_strategy_str(self):
        return division_pos_to_global_strategy_str(self.division_pos)

    def to_json(self):
        # convert to json format
        data = {}
        data["gloabl_strategy_str"] = self.get_global_strategy_str()
        data["local_strategy"] = (self.division_pos[self.rank], self.division_pos[self.rank+1])
        return data

class DivisionStrategyWS1(DivisionStrategy):

    def __init__(self, camera, world_size, rank, tile_x, tile_y):
        division_pos = [0, tile_x*tile_y]
        super().__init__(camera, world_size, rank, tile_x, tile_y, division_pos, None)

    def update_stats(self, stats_collector, n_render, n_consider, n_contrib, i2j_send_size):
        self.local_running_time = stats_collector["backward_render_time"]
        self.sum_n_render = n_render.sum().item()
        self.sum_n_consider = n_consider.sum().item()
        self.sum_n_contrib = n_contrib.sum().item()

    def to_json(self):
        # convert to json format
        data = {}
        data["local_running_time"] = self.local_running_time
        data["sum_n_render"] = self.sum_n_render
        data["sum_n_consider"] = self.sum_n_consider
        data["sum_n_contrib"] = self.sum_n_contrib
        return data

class DivisionStrategyNoRenderDistribution(DivisionStrategyWS1):
    # when MP_GROUP.size() == 1, we do not have render distribution. Then, we should use it. 

    def update_stats(self, stats_collector):
        self.local_running_time = stats_collector["backward_render_time"]

    def to_json(self):
        # convert to json format
        data = {}
        data["local_running_time"] = self.local_running_time
        return data

class DivisionStrategyNoRenderDistribution_simplified(DivisionStrategyWS1):
    # when MP_GROUP.size() == 1, we do not have render distribution. Then, we should use it. 

    def update_stats(self, local_running_time):
        self.local_running_time = local_running_time

    def to_json(self):
        # convert to json format
        data = {}
        data["local_running_time"] = self.local_running_time
        return data

class DivisionStrategy_1(DivisionStrategy):

    def __init__(self, camera, world_size, rank, tile_x, tile_y, division_pos, render_distribution_adjust_mode):
        super().__init__(camera, world_size, rank, tile_x, tile_y, division_pos, render_distribution_adjust_mode)

    def update_stats(self, stats_collector, n_render, n_consider, n_contrib, i2j_send_size):
        local_running_time = stats_collector["backward_render_time"] # For now, use the heaviest part as the running time.
        gloabl_running_times = [None for _ in range(self.world_size)]
        torch.distributed.all_gather_object(gloabl_running_times, local_running_time, group=utils.MP_GROUP)
        self.local_running_time = local_running_time
        self.global_running_times = gloabl_running_times
        self.sum_n_render = n_render.sum().item()
        self.sum_n_consider = n_consider.sum().item()
        self.sum_n_contrib = n_contrib.sum().item()
        self.i2j_send_size = i2j_send_size

    def to_json(self):
        # convert to json format
        data = {}
        data["gloabl_strategy_str"] = self.get_global_strategy_str()
        data["local_strategy"] = (self.division_pos[self.rank], self.division_pos[self.rank+1])
        data["global_running_times"] = self.global_running_times
        data["local_running_time"] = self.local_running_time
        data["sum_n_render"] = self.sum_n_render
        data["sum_n_consider"] = self.sum_n_consider
        data["sum_n_contrib"] = self.sum_n_contrib
        data["i2j_send_size"] = self.i2j_send_size
        return data

class DivisionStrategy_1_simplified(DivisionStrategy):

    def __init__(self, camera, world_size, rank, tile_x, tile_y, division_pos, render_distribution_adjust_mode):
        super().__init__(camera, world_size, rank, tile_x, tile_y, division_pos, render_distribution_adjust_mode)

    def update_stats(self, stats_collector):
        local_running_time = stats_collector["backward_render_time"] # For now, use the heaviest part as the running time.
        gloabl_running_times = [None for _ in range(self.world_size)]
        torch.distributed.all_gather_object(gloabl_running_times, local_running_time, group=utils.MP_GROUP)
        self.local_running_time = local_running_time
        self.global_running_times = gloabl_running_times

    def to_json(self):
        # convert to json format
        data = {}
        data["gloabl_strategy_str"] = self.get_global_strategy_str()
        data["local_strategy"] = (self.division_pos[self.rank], self.division_pos[self.rank+1])
        data["global_running_times"] = self.global_running_times
        data["local_running_time"] = self.local_running_time
        return data


class DivisionStrategy_2(DivisionStrategy):

    def __init__(self, camera, world_size, rank, tile_x, tile_y, division_pos, render_distribution_adjust_mode):
        super().__init__(camera, world_size, rank, tile_x, tile_y, division_pos, render_distribution_adjust_mode)

        self.local_running_time = None
        self.global_running_times = None
        self.heuristic = None

    def update_stats(self, stats_collector, n_render, n_consider, n_contrib, i2j_send_size):
        local_running_time = get_local_running_time_by_modes(stats_collector)
        timers = utils.get_timers()
        gloabl_running_times = [None for _ in range(self.world_size)]
        timers.start("[strategy.update_stats]all_gather_object")
        torch.distributed.all_gather_object(gloabl_running_times, local_running_time, group=utils.MP_GROUP)
        timers.stop("[strategy.update_stats]all_gather_object")
        self.local_running_time = local_running_time
        self.global_running_times = gloabl_running_times
        self.sum_n_render = n_render.sum().item()
        self.sum_n_consider = n_consider.sum().item()
        self.sum_n_contrib = n_contrib.sum().item()
        self.i2j_send_size = i2j_send_size

        timers.start("[strategy.update_stats]update_heuristic")
        self.update_heuristic()
        timers.stop("[strategy.update_stats]update_heuristic")

    def update_heuristic(self):
        assert self.global_running_times is not None, "You should call update_stats first."
        assert self.local_running_time is not None, "You should call update_stats first."

        with torch.no_grad():
            tile_ids_l, tile_ids_r = self.division_pos[self.rank], self.division_pos[self.rank+1]
            gather_heuristic = [torch.full((self.division_pos[i+1]-self.division_pos[i],),
                                        self.global_running_times[i] / (self.division_pos[i+1]-self.division_pos[i]),
                                        dtype=torch.float32,
                                        device="cuda",
                                        requires_grad=False)
                                for i in range(self.world_size)]
            self.heuristic = torch.cat(gather_heuristic, dim=0)
    
    def well_balanced(self, threshold=0.06):
        max_time = max(self.global_running_times)
        min_time = min(self.global_running_times)
        
        if max_time - min_time > max_time * threshold:
            return False

        return True

    def to_json(self):
        # convert to json format
        data = {}
        data["gloabl_strategy_str"] = self.get_global_strategy_str()
        data["local_strategy"] = (self.division_pos[self.rank], self.division_pos[self.rank+1])
        data["global_running_times"] = self.global_running_times
        data["local_running_time"] = self.local_running_time
        data["sum_n_render"] = self.sum_n_render
        data["sum_n_consider"] = self.sum_n_consider
        data["sum_n_contrib"] = self.sum_n_contrib
        data["i2j_send_size"] = self.i2j_send_size
        return data

class DivisionStrategy_2_simplified(DivisionStrategy_2):

    def update_stats(self, stats_collector, i2j_send_size=None):
        local_running_time = get_local_running_time_by_modes(stats_collector)
        timers = utils.get_timers()
        gloabl_running_times = [None for _ in range(self.world_size)]
        timers.start("[strategy.update_stats]all_gather_object")
        torch.distributed.all_gather_object(gloabl_running_times, local_running_time, group=utils.MP_GROUP)
        timers.stop("[strategy.update_stats]all_gather_object")
        self.local_running_time = local_running_time
        self.global_running_times = gloabl_running_times
        # self.i2j_send_size = i2j_send_size

        timers.start("[strategy.update_stats]update_heuristic")
        self.update_heuristic()
        timers.stop("[strategy.update_stats]update_heuristic")

    def to_json(self):
        # convert to json format
        data = {}
        data["gloabl_strategy_str"] = self.get_global_strategy_str()
        data["local_strategy"] = (self.division_pos[self.rank], self.division_pos[self.rank+1])
        data["global_running_times"] = self.global_running_times
        data["local_running_time"] = self.local_running_time
        # data["i2j_send_size"] = self.i2j_send_size
        return data

class DivisionStrategy_2_most_simplified(DivisionStrategy_2):

    def update_stats(self, global_running_times):
        self.global_running_times = global_running_times
        self.local_running_time = global_running_times[self.rank]

        timers = utils.get_timers()
        timers.start("[strategy.update_stats]update_heuristic")
        self.update_heuristic()
        timers.stop("[strategy.update_stats]update_heuristic")

    def to_json(self):
        # convert to json format
        data = {}
        data["gloabl_strategy_str"] = self.get_global_strategy_str()
        data["local_strategy"] = (self.division_pos[self.rank], self.division_pos[self.rank+1])
        data["global_running_times"] = self.global_running_times
        data["local_running_time"] = self.local_running_time
        # data["i2j_send_size"] = self.i2j_send_size
        return data

class DivisionStrategyFixedByUser(DivisionStrategy):
    def __init__(self, camera, world_size, rank, tile_x, tile_y, global_division_pos_str, render_distribution_adjust_mode):
        division_pos = list(map(int, global_division_pos_str.split(",")))
        super().__init__(camera, world_size, rank, tile_x, tile_y, division_pos, render_distribution_adjust_mode)

class DivisionStrategy_4(DivisionStrategy):

    def __init__(self, camera, world_size, rank, tile_x, tile_y, division_pos, render_distribution_adjust_mode):
        super().__init__(camera, world_size, rank, tile_x, tile_y, division_pos, render_distribution_adjust_mode)

    def update_stats(self, stats_collector, n_render, n_consider, n_contrib, i2j_send_size):
        local_running_time = stats_collector["backward_render_time"]
        gloabl_running_times = [None for _ in range(self.world_size)]
        torch.distributed.all_gather_object(gloabl_running_times, local_running_time, group=utils.MP_GROUP)
        self.local_running_time = local_running_time
        self.global_running_times = gloabl_running_times
        self.sum_n_render = n_render.sum().item()
        self.sum_n_consider = n_consider.sum().item()
        self.sum_n_contrib = n_contrib.sum().item()
        self.i2j_send_size = i2j_send_size

        # assert n_contrib is 1-d array
        assert len(n_contrib.shape) == 1
        self.n_contrib = n_contrib

        self.update_heuristic()

    def update_heuristic(self):
        assert self.global_running_times is not None, "You should call update_stats first."
        assert self.local_running_time is not None, "You should call update_stats first."

        with torch.no_grad():
            self.heuristic = self.n_contrib
            torch.distributed.all_reduce(self.heuristic, op=dist.ReduceOp.SUM, group=utils.MP_GROUP)


    def to_json(self):
        # convert to json format
        data = {}
        data["gloabl_strategy_str"] = self.get_global_strategy_str()
        data["local_strategy"] = (self.division_pos[self.rank], self.division_pos[self.rank+1])
        data["global_running_times"] = self.global_running_times
        data["local_running_time"] = self.local_running_time
        data["sum_n_render"] = self.sum_n_render
        data["sum_n_consider"] = self.sum_n_consider
        data["sum_n_contrib"] = self.sum_n_contrib
        data["i2j_send_size"] = self.i2j_send_size
        return data



class DivisionStrategy_6(DivisionStrategy):

    def __init__(self, camera, world_size, rank, tile_x, tile_y, division_pos, render_distribution_adjust_mode):
        super().__init__(camera, world_size, rank, tile_x, tile_y, division_pos, render_distribution_adjust_mode)
        # option 1: I should fisrt divide the x-axis(images are wide) and then divide the y-axis.
        # division_pos = (division_pos_ys, division_pos_xs)
        ## division_pos_xs = [0, ..., tile_x], shape is (self.grid_size_x,)
        ## division_pos_ys = shape is (self.grid_size_x, self.grid_size_y+1)
        ## example, suppose we have rank=8, [[0, 8, 16], [0, 9, 16], [0, 6, 16], [0, 12, 16]]
        self.grid_size_y, self.grid_size_x = DivisionStrategy_6.get_grid_size(world_size)
        self.grid_y_rank = self.rank // self.grid_size_x
        self.grid_x_rank = self.rank % self.grid_size_x

        # option 2
        # division_pos = (division_pos_ys, division_pos_xs)
        ## division_pos_ys = [0, ..., tile_y]
        ## division_pos_xs = [0, ..., tile_x]


    @staticmethod
    def get_grid_size(world_size):
        # return (grid_size_y, grid_size_x)
        if world_size == 2:
            return 1, 2
        elif world_size == 4:
            return 2, 2
        elif world_size == 8:
            return 2, 4
        elif world_size == 16:
            #  4, 4
            raise NotImplementedError
        else:
            raise NotImplementedError

    @staticmethod
    def get_default_division_pos(camera, world_size, rank, tile_x, tile_y):
        grid_size_y, grid_size_x = DivisionStrategy_6.get_grid_size(world_size)

        tile_x = (camera.image_width + utils.BLOCK_X - 1) // utils.BLOCK_X
        tile_y = (camera.image_height + utils.BLOCK_Y - 1) // utils.BLOCK_Y

        if tile_x % grid_size_x == 0:
            x_chunk_size = tile_x // grid_size_x
        else:
            x_chunk_size = tile_x // grid_size_x + 1
        division_pos_xs = [x_chunk_size * i for i in range(grid_size_x)] + [tile_x]

        if tile_y % grid_size_y == 0:
            y_chunk_size = tile_y // grid_size_y
        else:
            y_chunk_size = tile_y // grid_size_y + 1
        one_division_pos_ys = [y_chunk_size * i for i in range(grid_size_y)] + [tile_y]
        division_pos_ys = [one_division_pos_ys.copy() for i in range(grid_size_x)]
        assert len(division_pos_ys)*(len(division_pos_ys[0])-1) == world_size, "Each rank should have one rectangle."
        division_pos = (division_pos_xs, division_pos_ys)
        return division_pos

    def get_local_strategy(self):
        division_pos_xs, division_pos_ys = self.division_pos

        local_tile_x_l, local_tile_x_r = division_pos_xs[self.grid_x_rank], division_pos_xs[self.grid_x_rank+1]
        local_tile_y_l, local_tile_y_r = division_pos_ys[self.grid_x_rank][self.grid_y_rank], division_pos_ys[self.grid_x_rank][self.grid_y_rank+1]

        return ((local_tile_y_l, local_tile_y_r), (local_tile_x_l, local_tile_x_r) )

    def get_compute_locally(self):
        ((local_tile_y_l, local_tile_y_r), (local_tile_x_l, local_tile_x_r) ) = self.get_local_strategy()

        compute_locally = torch.zeros( (self.tile_y, self.tile_x), dtype=torch.bool, device="cuda")
        compute_locally[local_tile_y_l:local_tile_y_r, local_tile_x_l:local_tile_x_r] = True
        return compute_locally

    def update_stats(self, global_running_times):
        self.global_running_times = global_running_times
        self.local_running_time = global_running_times[self.rank]

        timers = utils.get_timers()
        timers.start("[strategy.update_stats]update_heuristic")
        self.update_heuristic()
        timers.stop("[strategy.update_stats]update_heuristic")

    def get_global_strategy_str(self):
        # for adjust_mode == "6", we do not change it into string.
        return self.division_pos
        # division_pos_xs, division_pos_ys = self.division_pos
        # division_pos_xs_str = ",".join(division_pos_xs)

        # one_division_pos_ys_str_list = []
        # for one_division_pos_ys in division_pos_ys:
        #     one_division_pos_ys_str = ",".join(one_division_pos_ys)
        #     one_division_pos_ys_str_list.append(one_division_pos_ys_str)
        # division_pos_ys_str = "$".join(one_division_pos_ys_str_list)

        # global_strategy_str = division_pos_xs_str+"@"+division_pos_ys_str

        # return global_strategy_str

    def to_json(self):
        # convert to json format
        data = {}
        data["gloabl_strategy_str"] = self.get_global_strategy_str()
        data["local_strategy"] = self.get_local_strategy()
        data["global_running_times"] = self.global_running_times
        data["local_running_time"] = self.local_running_time
        return data

    def update_heuristic(self):
        assert self.global_running_times is not None, "You should call update_stats first."
        assert self.local_running_time is not None, "You should call update_stats first."

        # TODO: maybe write a kernel to do this. 
        with torch.no_grad():
            self.heuristic = torch.zeros((self.tile_y, self.tile_x), dtype=torch.float32, device="cuda", requires_grad=False)
            # self.heuristic = torch.empty((self.tile_y, self.tile_x), dtype=torch.float32, device="cuda", requires_grad=False)

            division_pos_xs, division_pos_ys = self.division_pos
            rk_i = 0
            for rk_i_y in range(self.grid_size_y):
                for rk_i_x in range(self.grid_size_x):
                    local_tile_x_l, local_tile_x_r = division_pos_xs[rk_i_x], division_pos_xs[rk_i_x+1]
                    local_tile_y_l, local_tile_y_r = division_pos_ys[rk_i_x][rk_i_y], division_pos_ys[rk_i_x][rk_i_y+1]
                    running_time = self.global_running_times[rk_i]
                    self.heuristic[local_tile_y_l: local_tile_y_r, local_tile_x_l: local_tile_x_r] = running_time / ((local_tile_y_r-local_tile_y_l)*(local_tile_x_r-local_tile_x_l))
                    rk_i += 1
            assert (self.heuristic > 0).all(), "every element should be touched here."
    
    def well_balanced(self, threshold=0.06):
        max_time = max(self.global_running_times)
        min_time = min(self.global_running_times)
        
        if max_time - min_time > max_time * threshold:
            return False

        return True



########################## DivisionStrategyHistory ##########################
class DivisionStrategyHistory:
    def __init__(self, camera, world_size, rank, render_distribution_adjust_mode):
        self.camera = camera
        self.world_size = world_size
        self.rank = rank
        self.render_distribution_adjust_mode = render_distribution_adjust_mode

        self.tile_x = (camera.image_width + utils.BLOCK_X - 1) // utils.BLOCK_X
        self.tile_y = (camera.image_height + utils.BLOCK_Y - 1) // utils.BLOCK_Y
        self.tile_num = self.tile_x * self.tile_y

        self.history = []

        self.working_strategy = None
        self.working_iteration = None

    def add(self, iteration, strategy):
        self.history.append({"iteration": iteration, "strategy": strategy})
    
    def start_strategy(self):
        raise NotImplementedError

    def finish_strategy(self):
        raise NotImplementedError

    def to_json(self):
        # change to json format
        json = []
        for item in self.history:
            data = {
                "iteration": item["iteration"],
                "strategy": item["strategy"].to_json(),
            }
            json.append(data)
        return json

class DivisionStrategyHistoryWS1(DivisionStrategyHistory):
    def __init__(self, camera, world_size, rank, render_distribution_adjust_mode=None):
        super().__init__(camera, world_size, rank, render_distribution_adjust_mode)

    def start_strategy(self):
        self.working_strategy = DivisionStrategyWS1(self.camera, self.world_size, self.rank, self.tile_x, self.tile_y)
        self.working_iteration = utils.get_cur_iter()
        return self.working_strategy

    def finish_strategy(self):
        self.add(self.working_iteration, self.working_strategy)
        pass

class DivisionStrategyHistoryNoRenderDistribution(DivisionStrategyHistoryWS1):
    # when MP_GROUP.size() == 1, we do not have render distribution. Then, we should use it.

    def start_strategy(self):
        # DivisionStrategyNoRenderDistribution_simplified
        self.working_strategy = DivisionStrategyNoRenderDistribution_simplified(self.camera, self.world_size, self.rank, self.tile_x, self.tile_y)
        # self.working_strategy = DivisionStrategyNoRenderDistribution(self.camera, self.world_size, self.rank, self.tile_x, self.tile_y)
        self.working_iteration = utils.get_cur_iter()
        return self.working_strategy


class DivisionStrategyHistory_1(DivisionStrategyHistory):
    def __init__(self, camera, world_size, rank, render_distribution_adjust_mode):
        super().__init__(camera, world_size, rank, render_distribution_adjust_mode)

    def start_strategy(self):
        # self.working_strategy = DivisionStrategy_1(self.camera, self.world_size, self.rank, self.tile_x, self.tile_y, 
        #                                            get_evenly_division_pos(self.camera),
        #                                            self.render_distribution_adjust_mode)
        self.working_strategy = DivisionStrategy_1_simplified(self.camera, self.world_size, self.rank, self.tile_x, self.tile_y, 
                                                              get_evenly_division_pos(self.camera),
                                                              self.render_distribution_adjust_mode)

        self.working_iteration = utils.get_cur_iter()
        return self.working_strategy

    def finish_strategy(self):
        self.add(self.working_iteration, self.working_strategy)
        pass

class DivisionStrategyHistory_2(DivisionStrategyHistory):
    def __init__(self, camera, world_size, rank, render_distribution_adjust_mode):
        super().__init__(camera, world_size, rank, render_distribution_adjust_mode)

        self.cur_heuristic = None

    def update_heuristic(self, strategy=None):# XXX: local_running_time is now render backward running time for benchmarking method. 
        if strategy is None:
            strategy = self.working_strategy

        new_heuristic = strategy.heuristic
        if self.cur_heuristic is None:
            self.cur_heuristic = new_heuristic
            return

        args = utils.get_args()

        if args.stop_adjust_if_workloads_well_balanced and strategy.well_balanced(args.render_distribution_unbalance_threshold):
            # if the new strategy has already been well balanced, we do not have to update the heuristic.
            
            # FIXME: Maybe we should always update the heuristic, but we do not need to create a new distribution strategy. 
            # But that will lead to larger scheduling overhead. 

            return

        # update self.cur_heuristic
        self.cur_heuristic = self.cur_heuristic * args.heuristic_decay + new_heuristic * (1-args.heuristic_decay)

    def division_pos_heuristic(self):
        division_pos = division_pos_heuristic(self.cur_heuristic, self.tile_num, self.world_size)
        return division_pos

    def start_strategy(self):
        with torch.no_grad():
            if len(self.history) == 0:
                division_pos = get_evenly_division_pos(self.camera)
            else:
                division_pos = self.division_pos_heuristic()

            # DivisionStrategy_2_most_simplified
            self.working_strategy = DivisionStrategy_2_most_simplified(self.camera, self.world_size, self.rank, self.tile_x, self.tile_y, division_pos, self.render_distribution_adjust_mode)
            # self.working_strategy = DivisionStrategy_2_simplified(self.camera, self.world_size, self.rank, self.tile_x, self.tile_y, division_pos, self.render_distribution_adjust_mode)
            # self.working_strategy = DivisionStrategy_2(self.camera, self.world_size, self.rank, self.tile_x, self.tile_y, division_pos, self.render_distribution_adjust_mode)
            self.working_iteration = utils.get_cur_iter()
        return self.working_strategy

    def finish_strategy(self):
        with torch.no_grad():
            self.update_heuristic()
            if utils.get_args().benchmark_stats:
                self.working_strategy.heuristic = None
                # Because the heuristic is of size (# of tiles, ) and takes up lots of memory if we keep it for every iteration.
            self.add(self.working_iteration, self.working_strategy)

class DivisionStrategyHistoryFixedByUser(DivisionStrategyHistory):
    def start_strategy(self):
        args = utils.get_args()
        self.working_strategy = DivisionStrategyFixedByUser(self.camera, self.world_size, self.rank, self.tile_x, self.tile_y, args.dist_global_strategy, self.render_distribution_adjust_mode)
        self.working_iteration = utils.get_cur_iter()
        return self.working_strategy

    def finish_strategy(self):
        self.add(self.working_iteration, self.working_strategy)
        pass

class DivisionStrategyHistory_4(DivisionStrategyHistory):
    def __init__(self, camera, world_size, rank, render_distribution_adjust_mode):
        super().__init__(camera, world_size, rank, render_distribution_adjust_mode)
        self.cur_heuristic = None

    def update_heuristic(self, strategy=None):
        if strategy is None:
            strategy = self.working_strategy
        self.cur_heuristic = strategy.heuristic

    def division_pos_heuristic(self):
        division_pos = division_pos_heuristic(self.cur_heuristic, self.tile_num, self.world_size)
        return division_pos

    def start_strategy(self):
        with torch.no_grad():
            if len(self.history) == 0:
                division_pos = get_evenly_division_pos(self.camera)
            else:
                division_pos = self.division_pos_heuristic()

            self.working_strategy = DivisionStrategy_4(self.camera, self.world_size, self.rank, self.tile_x, self.tile_y, division_pos, self.render_distribution_adjust_mode)
            self.working_iteration = utils.get_cur_iter()
        return self.working_strategy

    def finish_strategy(self):
        with torch.no_grad():
            self.update_heuristic()
            if utils.get_args().benchmark_stats:
                self.working_strategy.heuristic = None
                # Because the heuristic is of size (# of tiles, ) and takes up lots of memory. 
            self.add(self.working_iteration, self.working_strategy)


class DivisionStrategyHistory_6(DivisionStrategyHistory):
    def __init__(self, camera, world_size, rank, render_distribution_adjust_mode):
        super().__init__(camera, world_size, rank, render_distribution_adjust_mode)

        self.cur_heuristic = None
        self.grid_size_y, self.grid_size_x = DivisionStrategy_6.get_grid_size(world_size)
        self.grid_y_rank = self.rank // self.grid_size_x
        self.grid_x_rank = self.rank % self.grid_size_x

    def update_heuristic(self, strategy=None):# XXX: local_running_time is now render backward running time for benchmarking method. 
        if strategy is None:
            strategy = self.working_strategy

        new_heuristic = strategy.heuristic
        if self.cur_heuristic is None:
            self.cur_heuristic = new_heuristic
            return

        args = utils.get_args()

        if args.stop_adjust_if_workloads_well_balanced and strategy.well_balanced(args.render_distribution_unbalance_threshold):
            # if the new strategy has already been well balanced, we do not have to update the heuristic.
            
            # FIXME: Maybe we should always update the heuristic, but we do not need to create a new distribution strategy. 
            # But that will lead to larger scheduling overhead. 

            return

        # update self.cur_heuristic
        if args.heuristic_decay == 0:
            self.cur_heuristic = new_heuristic
        else:
            self.cur_heuristic = self.cur_heuristic * args.heuristic_decay + new_heuristic * (1-args.heuristic_decay)

    def division_pos_heuristic(self):
        # division_pos_xs, division_pos_ys = self.division_pos
        cur_heuristic_along_x = self.cur_heuristic.sum(dim=0)
        division_pos_xs = division_pos_heuristic(cur_heuristic_along_x, self.tile_x, self.grid_size_x)

        division_pos_ys = []
        for i in range(self.grid_size_x):
            sliced_cur_heuristic_along_y = self.cur_heuristic[:, division_pos_xs[i]:division_pos_xs[i+1]].sum(1)
            one_division_pos_ys = division_pos_heuristic(sliced_cur_heuristic_along_y, self.tile_y, self.grid_size_y)
            division_pos_ys.append(one_division_pos_ys)

        division_pos = (division_pos_xs, division_pos_ys)
        return division_pos

    def start_strategy(self):
        with torch.no_grad():
            if len(self.history) == 0:
                division_pos = DivisionStrategy_6.get_default_division_pos(self.camera, self.world_size, self.rank, self.tile_x, self.tile_y)
            else:
                division_pos = self.division_pos_heuristic()

            self.working_strategy = DivisionStrategy_6(self.camera, self.world_size, self.rank, self.tile_x, self.tile_y, division_pos, self.render_distribution_adjust_mode)
            self.working_iteration = utils.get_cur_iter()
        return self.working_strategy

    def finish_strategy(self):
        with torch.no_grad():
            self.update_heuristic()
            self.working_strategy.heuristic = None
            # Because the heuristic is of size (# of tiles, ) and takes up lots of memory if we keep it for every iteration.
            self.add(self.working_iteration, self.working_strategy)


########################## Create DivisionStrategyHistory ##########################

def create_division_strategy_history(viewpoint_cam, render_distribution_adjust_mode):

    if utils.MP_GROUP.size() == 1:
        # return DivisionStrategyHistoryWS1(viewpoint_cam, utils.MP_GROUP.size(), utils.MP_GROUP.rank())
        return DivisionStrategyHistoryNoRenderDistribution(viewpoint_cam, utils.MP_GROUP.size(), utils.MP_GROUP.rank())
    elif render_distribution_adjust_mode == "1":
        return DivisionStrategyHistory_1(viewpoint_cam, utils.MP_GROUP.size(), utils.MP_GROUP.rank(), render_distribution_adjust_mode)
    elif render_distribution_adjust_mode == "2":
        return DivisionStrategyHistory_2(viewpoint_cam, utils.MP_GROUP.size(), utils.MP_GROUP.rank(), render_distribution_adjust_mode)
    elif render_distribution_adjust_mode == "3":
        assert False, "have not modified for the DP mode."
        return DivisionStrategyHistoryFixedByUser(viewpoint_cam, utils.MP_GROUP.size(), utils.MP_GROUP.rank(), render_distribution_adjust_mode)
    elif render_distribution_adjust_mode == "4":
        assert False, "have not modified for the DP mode."
        return DivisionStrategyHistory_4(viewpoint_cam, utils.MP_GROUP.size(), utils.MP_GROUP.rank(), render_distribution_adjust_mode)
    elif render_distribution_adjust_mode == "5":
        return DivisionStrategyHistory_2(viewpoint_cam, utils.MP_GROUP.size(), utils.MP_GROUP.rank(), render_distribution_adjust_mode)
    elif render_distribution_adjust_mode == "6":
        return DivisionStrategyHistory_6(viewpoint_cam, utils.MP_GROUP.size(), utils.MP_GROUP.rank(), render_distribution_adjust_mode)
    elif render_distribution_adjust_mode == "evaluation":
        args = utils.get_args()
        if args.render_distribution_adjust_mode == "6":
            return DivisionStrategyHistory_6(viewpoint_cam, utils.MP_GROUP.size(), utils.MP_GROUP.rank(), "evaluation")
        else:
            return DivisionStrategyHistory_1(viewpoint_cam, utils.MP_GROUP.size(), utils.MP_GROUP.rank(), render_distribution_adjust_mode)

    raise ValueError(f"Unknown render_distribution_adjust_mode: {render_distribution_adjust_mode}")

