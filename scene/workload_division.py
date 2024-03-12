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
    if tile_num % utils.WORLD_SIZE == 0:
        cnt = tile_num // utils.WORLD_SIZE
    else:
        cnt = tile_num // utils.WORLD_SIZE + 1
    division_pos = [cnt * i for i in range(utils.WORLD_SIZE)] + [tile_num]
    return division_pos

def get_evenly_global_strategy_str(camera):
    division_pos = get_evenly_division_pos(camera)
    return division_pos_to_global_strategy_str(division_pos)

def check_division_indices_globally_same(division_indices):
    recevie = [None for _ in range(utils.WORLD_SIZE)]
    torch.distributed.all_gather_object(recevie, division_indices)
    for i in range(utils.WORLD_SIZE):
        for j in range(utils.WORLD_SIZE):
            assert recevie[i][j] == division_indices[j], f"check_division_indices_globally_save failed: {i} {j}"

def division_pos_heuristic(cur_heuristic, tile_num, world_size):
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

    if args.render_distribution_mode == "1":
        return stats_collector["backward_render_time"]

    if args.render_distribution_mode == "2":
        if args.loss_distribution_mode == "fast_less_comm_noallreduceloss":
            return (
                stats_collector["backward_render_time"]+
                2*stats_collector["forward_loss_time"]
            )
        else:
            return stats_collector["backward_render_time"]

    if args.render_distribution_mode == "3":
        return stats_collector["backward_render_time"]

    if args.render_distribution_mode == "4":
        return stats_collector["backward_render_time"]

    if args.render_distribution_mode == "5":
        return (
            stats_collector["forward_render_time"]+
            stats_collector["backward_render_time"]+
            2*stats_collector["forward_loss_time"]
        )

    raise ValueError(f"Unknown render_distribution_mode: {args.render_distribution_mode}")

########################## DivisionStrategy ##########################
class DivisionStrategy:

    def __init__(self, camera, world_size, rank, tile_x, tile_y, division_pos, render_distribution_mode):
        self.camera = camera
        self.world_size = world_size
        self.rank = rank
        self.tile_x = tile_x
        self.tile_y = tile_y
        self.division_pos = division_pos
        self.render_distribution_mode = render_distribution_mode

    def get_compute_locally(self):
        tile_ids_l, tile_ids_r = self.division_pos[self.rank], self.division_pos[self.rank+1]
        compute_locally = torch.zeros(self.tile_y*self.tile_x, dtype=torch.bool, device="cuda")
        compute_locally[tile_ids_l:tile_ids_r] = True
        compute_locally = compute_locally.view(self.tile_y, self.tile_x)
        return compute_locally

    def is_avoid_pixel_all2all(self):
        # by default, each gpu compute it local tiles in forward and use all2all to fetch pixels near border on other GPUs, for later loss computation.
        if self.render_distribution_mode in  [None, "1", "2", "3", "4", "evaluation"]:
            return False
        if self.render_distribution_mode == "5":
            return True
        raise ValueError(f"Unknown render_distribution_mode: {self.render_distribution_mode}")

    def update_stats(self, stats_collector, n_render, n_consider, n_contrib, i2j_send_size):
        pass

    def get_gloabl_strategy_str(self):
        return division_pos_to_global_strategy_str(self.division_pos)

    def to_json(self):
        # convert to json format
        data = {}
        data["gloabl_strategy_str"] = self.get_gloabl_strategy_str()
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

class DivisionStrategy_1(DivisionStrategy):

    def __init__(self, camera, world_size, rank, tile_x, tile_y, division_pos, render_distribution_mode):
        super().__init__(camera, world_size, rank, tile_x, tile_y, division_pos, render_distribution_mode)

    def update_stats(self, stats_collector, n_render, n_consider, n_contrib, i2j_send_size):
        local_running_time = stats_collector["backward_render_time"] # For now, use the heaviest part as the running time.
        gloabl_running_times = [None for _ in range(self.world_size)]
        torch.distributed.all_gather_object(gloabl_running_times, local_running_time)
        self.local_running_time = local_running_time
        self.global_running_times = gloabl_running_times
        self.sum_n_render = n_render.sum().item()
        self.sum_n_consider = n_consider.sum().item()
        self.sum_n_contrib = n_contrib.sum().item()
        self.i2j_send_size = i2j_send_size

    def to_json(self):
        # convert to json format
        data = {}
        data["gloabl_strategy_str"] = self.get_gloabl_strategy_str()
        data["local_strategy"] = (self.division_pos[self.rank], self.division_pos[self.rank+1])
        data["global_running_times"] = self.global_running_times
        data["local_running_time"] = self.local_running_time
        data["sum_n_render"] = self.sum_n_render
        data["sum_n_consider"] = self.sum_n_consider
        data["sum_n_contrib"] = self.sum_n_contrib
        data["i2j_send_size"] = self.i2j_send_size
        return data

class DivisionStrategy_2(DivisionStrategy):

    def __init__(self, camera, world_size, rank, tile_x, tile_y, division_pos, render_distribution_mode):
        super().__init__(camera, world_size, rank, tile_x, tile_y, division_pos, render_distribution_mode)

        self.local_running_time = None
        self.global_running_times = None
        self.heuristic = None

    def update_stats(self, stats_collector, n_render, n_consider, n_contrib, i2j_send_size):
        local_running_time = get_local_running_time_by_modes(stats_collector)
        timers = utils.get_timers()
        gloabl_running_times = [None for _ in range(self.world_size)]
        timers.start("[strategy.update_stats]all_gather_object")
        torch.distributed.all_gather_object(gloabl_running_times, local_running_time)
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
        data["gloabl_strategy_str"] = self.get_gloabl_strategy_str()
        data["local_strategy"] = (self.division_pos[self.rank], self.division_pos[self.rank+1])
        data["global_running_times"] = self.global_running_times
        data["local_running_time"] = self.local_running_time
        data["sum_n_render"] = self.sum_n_render
        data["sum_n_consider"] = self.sum_n_consider
        data["sum_n_contrib"] = self.sum_n_contrib
        data["i2j_send_size"] = self.i2j_send_size
        return data

class DivisionStrategyFixedByUser(DivisionStrategy):
    def __init__(self, camera, world_size, rank, tile_x, tile_y, global_division_pos_str, render_distribution_mode):
        division_pos = list(map(int, global_division_pos_str.split(",")))
        super().__init__(camera, world_size, rank, tile_x, tile_y, division_pos, render_distribution_mode)

class DivisionStrategy_4(DivisionStrategy):

    def __init__(self, camera, world_size, rank, tile_x, tile_y, division_pos, render_distribution_mode):
        super().__init__(camera, world_size, rank, tile_x, tile_y, division_pos, render_distribution_mode)

    def update_stats(self, stats_collector, n_render, n_consider, n_contrib, i2j_send_size):
        local_running_time = stats_collector["backward_render_time"]
        gloabl_running_times = [None for _ in range(self.world_size)]
        torch.distributed.all_gather_object(gloabl_running_times, local_running_time)
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
            torch.distributed.all_reduce(self.heuristic, op=dist.ReduceOp.SUM)


    def to_json(self):
        # convert to json format
        data = {}
        data["gloabl_strategy_str"] = self.get_gloabl_strategy_str()
        data["local_strategy"] = (self.division_pos[self.rank], self.division_pos[self.rank+1])
        data["global_running_times"] = self.global_running_times
        data["local_running_time"] = self.local_running_time
        data["sum_n_render"] = self.sum_n_render
        data["sum_n_consider"] = self.sum_n_consider
        data["sum_n_contrib"] = self.sum_n_contrib
        data["i2j_send_size"] = self.i2j_send_size
        return data




########################## DivisionStrategyHistory ##########################
class DivisionStrategyHistory:
    def __init__(self, camera, world_size, rank, render_distribution_mode):
        self.camera = camera
        self.world_size = world_size
        self.rank = rank
        self.render_distribution_mode = render_distribution_mode

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
    def __init__(self, camera, world_size, rank, render_distribution_mode=None):
        super().__init__(camera, world_size, rank, render_distribution_mode)

    def start_strategy(self):
        self.working_strategy = DivisionStrategyWS1(self.camera, self.world_size, self.rank, self.tile_x, self.tile_y)
        self.working_iteration = utils.get_cur_iter()
        return self.working_strategy

    def finish_strategy(self):
        self.add(self.working_iteration, self.working_strategy)
        pass

class DivisionStrategyHistory_1(DivisionStrategyHistory):
    def __init__(self, camera, world_size, rank, render_distribution_mode):
        super().__init__(camera, world_size, rank, render_distribution_mode)

    def start_strategy(self):
        self.working_strategy = DivisionStrategy_1(self.camera, self.world_size, self.rank, self.tile_x, self.tile_y, 
                                                   get_evenly_division_pos(self.camera),
                                                   self.render_distribution_mode)
        self.working_iteration = utils.get_cur_iter()
        return self.working_strategy

    def finish_strategy(self):
        self.add(self.working_iteration, self.working_strategy)
        pass

class DivisionStrategyHistory_2(DivisionStrategyHistory):
    def __init__(self, camera, world_size, rank, render_distribution_mode):
        super().__init__(camera, world_size, rank, render_distribution_mode)

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

            self.working_strategy = DivisionStrategy_2(self.camera, self.world_size, self.rank, self.tile_x, self.tile_y, division_pos, self.render_distribution_mode)
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
        self.working_strategy = DivisionStrategyFixedByUser(self.camera, self.world_size, self.rank, self.tile_x, self.tile_y, args.dist_global_strategy, self.render_distribution_mode)
        self.working_iteration = utils.get_cur_iter()
        return self.working_strategy

    def finish_strategy(self):
        self.add(self.working_iteration, self.working_strategy)
        pass

class DivisionStrategyHistory_4(DivisionStrategyHistory):
    def __init__(self, camera, world_size, rank, render_distribution_mode):
        super().__init__(camera, world_size, rank, render_distribution_mode)
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

            self.working_strategy = DivisionStrategy_4(self.camera, self.world_size, self.rank, self.tile_x, self.tile_y, division_pos, self.render_distribution_mode)
            self.working_iteration = utils.get_cur_iter()
        return self.working_strategy

    def finish_strategy(self):
        with torch.no_grad():
            self.update_heuristic()
            if utils.get_args().benchmark_stats:
                self.working_strategy.heuristic = None
                # Because the heuristic is of size (# of tiles, ) and takes up lots of memory. 
            self.add(self.working_iteration, self.working_strategy)





########################## Create DivisionStrategyHistory ##########################

def create_division_strategy_history(viewpoint_cam, render_distribution_mode):
    if utils.WORLD_SIZE == 1:
        return DivisionStrategyHistoryWS1(viewpoint_cam, utils.WORLD_SIZE, utils.LOCAL_RANK)
    elif render_distribution_mode == "1":
        return DivisionStrategyHistory_1(viewpoint_cam, utils.WORLD_SIZE, utils.LOCAL_RANK, render_distribution_mode)
    elif render_distribution_mode == "2":
        return DivisionStrategyHistory_2(viewpoint_cam, utils.WORLD_SIZE, utils.LOCAL_RANK, render_distribution_mode)
    elif render_distribution_mode == "3":
        return DivisionStrategyHistoryFixedByUser(viewpoint_cam, utils.WORLD_SIZE, utils.LOCAL_RANK, render_distribution_mode)
    elif render_distribution_mode == "4":
        return DivisionStrategyHistory_4(viewpoint_cam, utils.WORLD_SIZE, utils.LOCAL_RANK, render_distribution_mode)
    elif render_distribution_mode == "5":
        return DivisionStrategyHistory_2(viewpoint_cam, utils.WORLD_SIZE, utils.LOCAL_RANK, render_distribution_mode)
    elif render_distribution_mode == "evaluation":
        return DivisionStrategyHistory_1(viewpoint_cam, utils.WORLD_SIZE, utils.LOCAL_RANK, render_distribution_mode)

    raise ValueError(f"Unknown render_distribution_mode: {render_distribution_mode}")

