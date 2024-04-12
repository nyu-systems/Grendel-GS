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

def division_pos_heuristic(heuristic, tile_num, world_size):
    assert heuristic.shape[0] == tile_num, "the length of heuristics should be the same as the number of tiles."
    heuristic_prefix_sum = torch.cumsum(heuristic, dim=0)
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
    local_running_time = 0
    for mode in args.image_distribution_config.local_running_time_mode:
        local_running_time += stats_collector[mode]
    return local_running_time

########################## DivisionStrategy ##########################
class DivisionStrategy:

    def __init__(self, camera, world_size, rank, tile_x, tile_y, heuristic, workloads_division_mode):
        self.camera = camera
        self.world_size = world_size
        self.rank = rank
        self.tile_x = tile_x
        self.tile_y = tile_y

        heuristic = heuristic.view(-1)
        self.division_pos = division_pos_heuristic(heuristic, self.tile_x*self.tile_y, self.world_size)
        self.workloads_division_mode = workloads_division_mode

    def get_compute_locally(self):
        tile_ids_l, tile_ids_r = self.division_pos[self.rank], self.division_pos[self.rank+1]
        compute_locally = torch.zeros(self.tile_y*self.tile_x, dtype=torch.bool, device="cuda")
        compute_locally[tile_ids_l:tile_ids_r] = True
        compute_locally = compute_locally.view(self.tile_y, self.tile_x)
        return compute_locally
    
    def get_extended_compute_locally(self):
        tile_ids_l, tile_ids_r = self.division_pos[self.rank], self.division_pos[self.rank+1]
        tile_l = max(tile_ids_l-self.tile_x-1, 0)
        tile_r = min(tile_ids_r+self.tile_x+1, self.tile_y*self.tile_x)
        extended_compute_locally = torch.zeros(self.tile_y*self.tile_x, dtype=torch.bool, device="cuda")
        extended_compute_locally[tile_l:tile_r] = True
        extended_compute_locally = extended_compute_locally.view(self.tile_y, self.tile_x)
        return extended_compute_locally

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

class DivisionStrategyUniform(DivisionStrategy):

    def __init__(self, camera, world_size, rank, tile_x, tile_y, heuristic, workloads_division_mode):
        super().__init__(camera, world_size, rank, tile_x, tile_y, heuristic, workloads_division_mode)

    def update_stats(self, global_running_times):
        self.global_running_times = global_running_times
        self.local_running_time = global_running_times[self.rank]

        self.heuristic = torch.ones((self.tile_y*self.tile_x, ), dtype=torch.float32, device="cuda", requires_grad=False)

    def need_adjustment(self, threshold=0.06):
        return False

    def to_json(self):
        # convert to json format
        data = {}
        data["global_strategy_str"] = self.get_global_strategy_str()
        data["local_strategy"] = (self.division_pos[self.rank], self.division_pos[self.rank+1])
        data["global_running_times"] = self.global_running_times
        data["local_running_time"] = self.local_running_time
        return data


class DivisionStrategyDynamicAdjustment(DivisionStrategy):

    def __init__(self, camera, world_size, rank, tile_x, tile_y, heuristic, workloads_division_mode):
        super().__init__(camera, world_size, rank, tile_x, tile_y, heuristic, workloads_division_mode)

        self.local_running_time = None
        self.global_running_times = None

    def update_stats(self, global_running_times):
        self.global_running_times = global_running_times
        self.local_running_time = global_running_times[self.rank]

        timers = utils.get_timers()
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

    def need_adjustment(self, threshold=0.06):
        max_time = max(self.global_running_times)
        min_time = min(self.global_running_times)
        if max_time - min_time > max_time * threshold:
            return True

        return False

    def to_json(self):
        # convert to json format
        data = {}
        data["global_strategy_str"] = self.get_global_strategy_str()
        data["local_strategy"] = (self.division_pos[self.rank], self.division_pos[self.rank+1])
        data["global_running_times"] = self.global_running_times
        data["local_running_time"] = self.local_running_time
        return data


# TODO: maybe implement this later.
# class DivisionStrategyFixedByUser(DivisionStrategy):
#     def __init__(self, camera, world_size, rank, tile_x, tile_y, global_division_pos_str, workloads_division_mode):
#         division_pos = list(map(int, global_division_pos_str.split(",")))
#         super().__init__(camera, world_size, rank, tile_x, tile_y, division_pos, workloads_division_mode)


class DivisionStrategyAsGrid:

    def __init__(self, camera, world_size, rank, tile_x, tile_y, heuristic, workloads_division_mode):
        self.camera = camera
        self.world_size = world_size
        self.rank = rank
        self.tile_x = tile_x
        self.tile_y = tile_y

        # I should fisrt divide the x-axis(images are wide) and then divide the y-axis.
        # division_pos = (division_pos_ys, division_pos_xs)
        ## division_pos_xs = [0, ..., tile_x], shape is (self.grid_size_x,)
        ## division_pos_ys = shape is (self.grid_size_x, self.grid_size_y+1)
        ## example, suppose we have rank=8, [[0, 8, 16], [0, 9, 16], [0, 6, 16], [0, 12, 16]]
        self.grid_size_y, self.grid_size_x = DivisionStrategyAsGrid.get_grid_size(world_size)
        self.grid_y_rank = self.rank // self.grid_size_x
        self.grid_x_rank = self.rank % self.grid_size_x

        self.division_pos = self.division_pos_heuristic(heuristic)

    def division_pos_heuristic(self, heuristic):
        cur_heuristic_along_x = heuristic.sum(dim=0)
        division_pos_xs = division_pos_heuristic(cur_heuristic_along_x, self.tile_x, self.grid_size_x)

        division_pos_ys = []
        for i in range(self.grid_size_x):
            sliced_cur_heuristic_along_y = heuristic[:, division_pos_xs[i]:division_pos_xs[i+1]].sum(1)
            one_division_pos_ys = division_pos_heuristic(sliced_cur_heuristic_along_y, self.tile_y, self.grid_size_y)
            division_pos_ys.append(one_division_pos_ys)

        division_pos = (division_pos_xs, division_pos_ys)
        return division_pos

    @staticmethod
    def get_grid_size(world_size):
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
        grid_size_y, grid_size_x = DivisionStrategyAsGrid.get_grid_size(world_size)

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

    def get_extended_compute_locally(self):
        ((local_tile_y_l, local_tile_y_r), (local_tile_x_l, local_tile_x_r) ) = self.get_local_strategy()

        extended_compute_locally = torch.zeros((self.tile_y, self.tile_x), dtype=torch.bool, device="cuda")
        extended_compute_locally[max(local_tile_y_l-1,0):min(local_tile_y_r+1, self.tile_y),
                                 max(local_tile_x_l-1,0):min(local_tile_x_r+1, self.tile_x)] = True

        return extended_compute_locally

    def update_stats(self, global_running_times):
        self.global_running_times = global_running_times
        self.local_running_time = global_running_times[self.rank]

        timers = utils.get_timers()
        timers.start("[strategy.update_stats]update_heuristic")
        self.update_heuristic()
        timers.stop("[strategy.update_stats]update_heuristic")

    def get_global_strategy_str(self):
        # we do not change it into string here
        return self.division_pos

    def to_json(self):
        data = {}
        data["global_strategy_str"] = self.get_global_strategy_str()
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
    
    def need_adjustment(self, threshold=0.06):
        max_time = max(self.global_running_times)
        min_time = min(self.global_running_times)
        
        if max_time - min_time > max_time * threshold:
            return False

        return True


name2DivisionStrategy = {
    "DivisionStrategyUniform": DivisionStrategyUniform,
    "DivisionStrategyDynamicAdjustment": DivisionStrategyDynamicAdjustment,
    "DivisionStrategyAsGrid": DivisionStrategyAsGrid,
    "evaluation": DivisionStrategyUniform,
}
########################## DivisionStrategyHistory ##########################

class DivisionStrategyHistory:
    def __init__(self, camera, world_size, rank, workloads_division_mode):
        self.camera = camera
        self.world_size = world_size
        self.rank = rank
        self.workloads_division_mode = workloads_division_mode

        self.tile_x = (camera.image_width + utils.BLOCK_X - 1) // utils.BLOCK_X
        self.tile_y = (camera.image_height + utils.BLOCK_Y - 1) // utils.BLOCK_Y
        self.tile_num = self.tile_x * self.tile_y

        self.history = []

        self.working_strategy = None
        self.working_iteration = None
        self.accum_heuristic = torch.ones((self.tile_y, self.tile_x), dtype=torch.float32, device="cuda", requires_grad=False)
        self.current_heuristic = None

    def add(self, iteration, strategy):
        self.history.append({"iteration": iteration, "strategy": strategy})
    
    def update_heuristic(self):
        args = utils.get_args()

        if args.heuristic_decay == 0:
            self.accum_heuristic = self.working_strategy.heuristic
            return

        # TODO: Does gradually increasing heuristic_decay work?  Quick results show it does not. 
        # if self.working_iteration < 1500:
        #     heuristic_decay = 0
        # elif self.working_iteration < 3000:
        #     heuristic_decay = 0.2
        # else:
        #     heuristic_decay = 0.5
        heuristic_decay = args.heuristic_decay

        # update accummulated heuristic
        self.accum_heuristic = self.accum_heuristic * heuristic_decay + self.working_strategy.heuristic.view((self.tile_y, self.tile_x)) * (1-heuristic_decay)

    def start_strategy(self):
        args = utils.get_args()
        with torch.no_grad():
            if args.stop_adjust_if_workloads_well_balanced and len(self.history) > 0 and not self.working_strategy.need_adjustment(args.image_distribution_unbalance_threshold):
                # now, the self.working_strategy is actually the last strategy.
                heuristic2use = self.current_heuristic
            else:
                heuristic2use = self.accum_heuristic
                self.current_heuristic = self.accum_heuristic

            self.working_strategy = name2DivisionStrategy[self.workloads_division_mode](self.camera,
                                                                                   self.world_size, self.rank,
                                                                                   self.tile_x, self.tile_y,
                                                                                   heuristic2use,
                                                                                   self.workloads_division_mode)
            self.working_iteration = utils.get_cur_iter()
        return self.working_strategy

    def finish_strategy(self):
        with torch.no_grad():
            self.update_heuristic()
            if utils.get_args().benchmark_stats:
                self.working_strategy.heuristic = None
                # Because the heuristic is of size (# of tiles, ) and takes up lots of memory if we keep it for every iteration.
            self.add(self.working_iteration, self.working_strategy)

    def to_json(self):
        json = []
        for item in self.history:
            data = {
                "iteration": item["iteration"],
                "strategy": item["strategy"].to_json(),
            }
            json.append(data)
        return json

########################## Create DivisionStrategyHistory ##########################

def get_division_strategy_history(cameraId2StrategyHistory, viewpoint_cam, workloads_division_mode):
    args = utils.get_args()
    if viewpoint_cam.uid not in cameraId2StrategyHistory:
        cameraId2StrategyHistory[viewpoint_cam.uid] = DivisionStrategyHistory(viewpoint_cam, utils.MP_GROUP.size(), utils.MP_GROUP.rank(), workloads_division_mode)
    return cameraId2StrategyHistory[viewpoint_cam.uid]

