from scene.cameras import Camera
import torch.distributed as dist
import torch
import time
import utils.general_utils as utils


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

# class DivisionStrategy:

#     def __init__(self, camera, world_size, rank, tile_x, tile_y, division_pos, adjust_mode):
#         self.camera = camera
#         self.world_size = world_size
#         self.rank = rank
#         self.tile_x = tile_x
#         self.tile_y = tile_y
#         self.division_pos = division_pos
#         self.adjust_mode = adjust_mode

#         # results
#         self.tile_n_render = None # number of 3dgs that are rendered on each tile.
#         self.tile_n_consider = None # sum of number of 3dgs considered by each pixel.
#         self.tile_n_contrib = None # sum of number of 3dgs that contribute to each pixel.
#         self.forward_time = None # forward time for each worker.
#         self.backward_time = None # backward time for each worker.

#     @property
#     def local_strategy(self):
#         return (self.division_pos[self.rank], self.division_pos[self.rank+1])

#     @property
#     def global_strategy(self):
#         return self.division_pos

#     @property
#     def local_strategy_str(self):
#         return interval_to_strategy_str(self.local_strategy)
    
#     @property
#     def global_strategy_str(self):
#         return division_pos_to_global_strategy_str(self.division_pos)
#         # example: "0,100,200,300,400,500,600,700,800,900,1000"

#     def update_result(self, n_render, n_consider, n_contrib, forward_time, backward_time):
#         self.tile_n_render = n_render
#         self.tile_n_consider = n_consider
#         self.tile_n_contrib = n_contrib

#         self.forward_time = forward_time
#         self.backward_time = backward_time

#         self.process_result_stats()

#     def process_result_stats(self):
#         # self.presum_n_render[i] = sum(self.tile_n_render[:i]) # it does not include self.tile_n_render[i]. This is for code simplicity.
#         self.presum_n_render = [0] # prefix sum of self.tile_n_render.
#         self.presum_n_consider = [0] # prefix sum of self.tile_n_consider.
#         self.presum_n_contrib = [0] # prefix sum of self.tile_n_contrib.
#         self.worker_n_render = [] # number of rendered 3dgs that for each worker.
#         self.worker_n_consider = [] # sum of number of considered 3dgs for each worker.
#         self.worker_n_contrib = [] # sum of number of contributed 3dgs for each worker.
#         self.worker_tile_num = [] # number of tiles for each worker.

#         for j in range(self.tile_y):
#             for i in range(self.tile_x):
#                 idx = j * self.tile_x + i
#                 pix_cnt = get_tile_pixel_cnt(j, i, self.camera.image_width, self.camera.image_height)
#                 tile_n_render = self.tile_n_render[idx] * pix_cnt
#                 tile_n_consider = self.tile_n_consider[idx]
#                 tile_n_contrib = self.tile_n_contrib[idx]

#                 # TODO: why do we need presum_? maybe we could delete it to make the code simpler.
#                 self.presum_n_render.append(tile_n_render + self.presum_n_render[-1])
#                 self.presum_n_consider.append(tile_n_consider + self.presum_n_consider[-1])
#                 self.presum_n_contrib.append(tile_n_contrib + self.presum_n_contrib[-1])

#         for rk in range(self.world_size):
#             self.worker_n_render.append(self.presum_n_render[self.division_pos[rk+1]] - self.presum_n_render[self.division_pos[rk]])
#             self.worker_n_consider.append(self.presum_n_consider[self.division_pos[rk+1]] - self.presum_n_consider[self.division_pos[rk]])
#             self.worker_n_contrib.append(self.presum_n_contrib[self.division_pos[rk+1]] - self.presum_n_contrib[self.division_pos[rk]])
#             self.worker_tile_num.append(self.division_pos[rk+1] - self.division_pos[rk])

#         self.global_forward_time = sum(self.forward_time)
#         self.global_backward_time = sum(self.backward_time)
#         self.local_forward_time = self.forward_time[self.rank]
#         self.local_backward_time = self.backward_time[self.rank]

#     def workload_is_balanced(self, threshold=1):
#         # threshold: 1 means 1ms.
#         # This function should return the same result for all workers.

#         self.mean_forward_time = self.global_forward_time / self.world_size
#         self.mean_backward_time = self.global_backward_time / self.world_size
        
#         for one_forward_time, one_backward_time in zip(self.forward_time, self.backward_time):
#             if abs(self.mean_backward_time - one_backward_time + self.mean_forward_time - one_forward_time) > threshold:
#                 return False

#         return True
    
#     @staticmethod
#     def synchronize_stats(n_render, n_consider, n_contrib, timers=None):
#         dist.all_reduce(n_render, op=dist.ReduceOp.SUM)
#         dist.all_reduce(n_consider, op=dist.ReduceOp.SUM)
#         dist.all_reduce(n_contrib, op=dist.ReduceOp.SUM)
#         if timers is not None:
#             timers.start("synchronize_stats: *.cpu().tolist()")
#         # move to cpu and to list
#         n_render = n_render.cpu().tolist()
#         n_consider = n_consider.cpu().tolist()
#         n_contrib = n_contrib.cpu().tolist()
#         if timers is not None:
#             timers.stop("synchronize_stats: *.cpu().tolist()")
#         return n_render, n_consider, n_contrib

#     @staticmethod
#     def synchronize_time(world_size, rank, forward_time, backward_time):# TODO: could I do synchronize_time between cpu processes? Or set device="cpu"? Or torch.distributed.all_gather_object? 
#         buffer = torch.zeros(world_size*2, dtype=torch.float32, device="cuda")
#         buffer[rank] = forward_time
#         buffer[world_size + rank] = backward_time
#         dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
#         forward_time = buffer[:world_size].cpu().tolist()
#         backward_time = buffer[world_size:].cpu().tolist()
#         return forward_time, backward_time

#     def to_json(self, log_stats = False):
#         # convert to json format
#         data = {}
#         data["strategy"] = self.local_strategy
        
#         if log_stats:
#             for j in range(self.tile_y):
#                 for i in range(self.tile_x):
#                     idx = j * self.tile_x + i
#                     if idx < self.local_strategy[0] or self.local_strategy[1] <= idx:
#                         continue
                    
#                     pix_cnt = get_tile_pixel_cnt(j, i, self.camera.image_width, self.camera.image_height)
                    
#                     data_str = ""
#                     data_str += f"n_render: {self.tile_n_render[idx]}, "
#                     data_str += f"n_consider: {self.tile_n_consider[idx]}, "
#                     data_str += f"n_contrib: {self.tile_n_contrib[idx]}, "
#                     data_str += f"n_consider_per_pixal: {round(self.tile_n_consider[idx]/pix_cnt, 6)}, "
#                     data_str += f"n_contrib_per_pixal: {round(self.tile_n_contrib[idx]/pix_cnt, 6)}, "
#                     data[f"({j},{i})"] = data_str
        
#         return data




class DivisionStrategy:

    def __init__(self, camera, world_size, rank, tile_x, tile_y, division_pos, adjust_mode):
        self.camera = camera
        self.world_size = world_size
        self.rank = rank
        self.tile_x = tile_x
        self.tile_y = tile_y
        self.division_pos = division_pos
        self.adjust_mode = adjust_mode

    def get_compute_locally(self):
        tile_ids_l, tile_ids_r = self.division_pos[self.rank], self.division_pos[self.rank+1]
        compute_locally = torch.zeros(self.tile_y*self.tile_x, dtype=torch.bool, device="cuda")
        compute_locally[tile_ids_l:tile_ids_r] = True
        compute_locally = compute_locally.view(self.tile_y, self.tile_x)
        return compute_locally
    
    def get_gloabl_strategy_str(self):
        return division_pos_to_global_strategy_str(self.division_pos)

    def to_json(self):
        # convert to json format
        data = {}
        data["gloabl_strategy_str"] = self.get_gloabl_strategy_str()
        data["local_strategy"] = (self.division_pos[self.rank], self.division_pos[self.rank+1])
        return data

class DivisionStrategy_2(DivisionStrategy):

    def __init__(self, camera, world_size, rank, tile_x, tile_y, division_pos, adjust_mode):
        super().__init__(camera, world_size, rank, tile_x, tile_y, division_pos, adjust_mode)

        self.local_running_time = None
        self.global_running_times = None
        self.heuristic = None
    
    def update_stats(self, local_running_time):
        if self.global_running_times is not None and self.heuristic is not None:
            return

        gloabl_running_times = [None for _ in range(self.world_size)]
        torch.distributed.all_gather_object(gloabl_running_times, local_running_time)
        self.local_running_time = local_running_time
        self.global_running_times = gloabl_running_times

        self.update_heuristic()

    def update_heuristic(self):
        assert self.global_running_times is not None, "You should call update_stats first."
        assert self.local_running_time is not None, "You should call update_stats first."

        tile_ids_l, tile_ids_r = self.division_pos[self.rank], self.division_pos[self.rank+1]
        gather_heuristic = [torch.full((self.division_pos[i+1]-self.division_pos[i],),
                                       self.global_running_times[i] / (self.division_pos[i+1]-self.division_pos[i]),
                                       dtype=torch.float32,
                                       device="cuda",
                                       requires_grad=False)
                            for i in range(self.world_size)]
        self.heuristic = torch.cat(gather_heuristic, dim=0)
    
    def well_balanced(self, threshold=0.06):
        # threshold: 0.1 means 10% error is allowed. 

        max_time = max(self.global_running_times)
        min_time = min(self.global_running_times)
        
        if max_time - min_time > max_time * threshold:
            return False

        return True



class DivisionStrategy_1(DivisionStrategy):

    def __init__(self, camera, world_size, rank, tile_x, tile_y, division_pos, adjust_mode):
        super().__init__(camera, world_size, rank, tile_x, tile_y, division_pos, adjust_mode)

class DivisionStrategyWS1(DivisionStrategy):

    def __init__(self, camera, world_size, rank, tile_x, tile_y):
        division_pos = [0, tile_x*tile_y]
        super().__init__(camera, world_size, rank, tile_x, tile_y, division_pos, None)



    # @property
    # def local_strategy(self):
    #     return (self.division_pos[self.rank], self.division_pos[self.rank+1])

    # @property
    # def global_strategy(self):
    #     return self.division_pos

    # @property
    # def local_strategy_str(self):
    #     return interval_to_strategy_str(self.local_strategy)
    
    # @property
    # def global_strategy_str(self):
    #     return division_pos_to_global_strategy_str(self.division_pos)
    #     # example: "0,100,200,300,400,500,600,700,800,900,1000"

    # def update_result(self, n_render, n_consider, n_contrib, forward_time, backward_time):
    #     self.tile_n_render = n_render
    #     self.tile_n_consider = n_consider
    #     self.tile_n_contrib = n_contrib

    #     self.forward_time = forward_time
    #     self.backward_time = backward_time

    #     self.process_result_stats()

    # def process_result_stats(self):
    #     # self.presum_n_render[i] = sum(self.tile_n_render[:i]) # it does not include self.tile_n_render[i]. This is for code simplicity.
    #     self.presum_n_render = [0] # prefix sum of self.tile_n_render.
    #     self.presum_n_consider = [0] # prefix sum of self.tile_n_consider.
    #     self.presum_n_contrib = [0] # prefix sum of self.tile_n_contrib.
    #     self.worker_n_render = [] # number of rendered 3dgs that for each worker.
    #     self.worker_n_consider = [] # sum of number of considered 3dgs for each worker.
    #     self.worker_n_contrib = [] # sum of number of contributed 3dgs for each worker.
    #     self.worker_tile_num = [] # number of tiles for each worker.

    #     for j in range(self.tile_y):
    #         for i in range(self.tile_x):
    #             idx = j * self.tile_x + i
    #             pix_cnt = get_tile_pixel_cnt(j, i, self.camera.image_width, self.camera.image_height)
    #             tile_n_render = self.tile_n_render[idx] * pix_cnt
    #             tile_n_consider = self.tile_n_consider[idx]
    #             tile_n_contrib = self.tile_n_contrib[idx]

    #             # TODO: why do we need presum_? maybe we could delete it to make the code simpler.
    #             self.presum_n_render.append(tile_n_render + self.presum_n_render[-1])
    #             self.presum_n_consider.append(tile_n_consider + self.presum_n_consider[-1])
    #             self.presum_n_contrib.append(tile_n_contrib + self.presum_n_contrib[-1])

    #     for rk in range(self.world_size):
    #         self.worker_n_render.append(self.presum_n_render[self.division_pos[rk+1]] - self.presum_n_render[self.division_pos[rk]])
    #         self.worker_n_consider.append(self.presum_n_consider[self.division_pos[rk+1]] - self.presum_n_consider[self.division_pos[rk]])
    #         self.worker_n_contrib.append(self.presum_n_contrib[self.division_pos[rk+1]] - self.presum_n_contrib[self.division_pos[rk]])
    #         self.worker_tile_num.append(self.division_pos[rk+1] - self.division_pos[rk])

    #     self.global_forward_time = sum(self.forward_time)
    #     self.global_backward_time = sum(self.backward_time)
    #     self.local_forward_time = self.forward_time[self.rank]
    #     self.local_backward_time = self.backward_time[self.rank]

    # def workload_is_balanced(self, threshold=1):
    #     # threshold: 1 means 1ms.
    #     # This function should return the same result for all workers.

    #     self.mean_forward_time = self.global_forward_time / self.world_size
    #     self.mean_backward_time = self.global_backward_time / self.world_size
        
    #     for one_forward_time, one_backward_time in zip(self.forward_time, self.backward_time):
    #         if abs(self.mean_backward_time - one_backward_time + self.mean_forward_time - one_forward_time) > threshold:
    #             return False

    #     return True
    
    # @staticmethod
    # def synchronize_stats(n_render, n_consider, n_contrib, timers=None):
    #     dist.all_reduce(n_render, op=dist.ReduceOp.SUM)
    #     dist.all_reduce(n_consider, op=dist.ReduceOp.SUM)
    #     dist.all_reduce(n_contrib, op=dist.ReduceOp.SUM)
    #     if timers is not None:
    #         timers.start("synchronize_stats: *.cpu().tolist()")
    #     # move to cpu and to list
    #     n_render = n_render.cpu().tolist()
    #     n_consider = n_consider.cpu().tolist()
    #     n_contrib = n_contrib.cpu().tolist()
    #     if timers is not None:
    #         timers.stop("synchronize_stats: *.cpu().tolist()")
    #     return n_render, n_consider, n_contrib

    # @staticmethod
    # def synchronize_time(world_size, rank, forward_time, backward_time):# TODO: could I do synchronize_time between cpu processes? Or set device="cpu"? Or torch.distributed.all_gather_object? 
    #     buffer = torch.zeros(world_size*2, dtype=torch.float32, device="cuda")
    #     buffer[rank] = forward_time
    #     buffer[world_size + rank] = backward_time
    #     dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
    #     forward_time = buffer[:world_size].cpu().tolist()
    #     backward_time = buffer[world_size:].cpu().tolist()
    #     return forward_time, backward_time

    # def to_json(self, log_stats = False):
    #     # convert to json format
    #     data = {}
    #     data["strategy"] = self.local_strategy
        
    #     if log_stats:
    #         for j in range(self.tile_y):
    #             for i in range(self.tile_x):
    #                 idx = j * self.tile_x + i
    #                 if idx < self.local_strategy[0] or self.local_strategy[1] <= idx:
    #                     continue
                    
    #                 pix_cnt = get_tile_pixel_cnt(j, i, self.camera.image_width, self.camera.image_height)
                    
    #                 data_str = ""
    #                 data_str += f"n_render: {self.tile_n_render[idx]}, "
    #                 data_str += f"n_consider: {self.tile_n_consider[idx]}, "
    #                 data_str += f"n_contrib: {self.tile_n_contrib[idx]}, "
    #                 data_str += f"n_consider_per_pixal: {round(self.tile_n_consider[idx]/pix_cnt, 6)}, "
    #                 data_str += f"n_contrib_per_pixal: {round(self.tile_n_contrib[idx]/pix_cnt, 6)}, "
    #                 data[f"({j},{i})"] = data_str
        
    #     return data























class DivisionStrategyHistory:
    def __init__(self, camera, world_size, rank, adjust_mode):
        self.camera = camera
        self.world_size = world_size
        self.rank = rank
        self.adjust_mode = adjust_mode

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


class DivisionStrategyHistory_1(DivisionStrategyHistory):
    def __init__(self, camera, world_size, rank, adjust_mode):
        super().__init__(camera, world_size, rank, adjust_mode)

    def start_strategy(self):
        self.working_strategy = DivisionStrategy_1(self.camera, self.world_size, self.rank, self.tile_x, self.tile_y, get_evenly_division_pos(self.camera), self.adjust_mode)
        self.working_iteration = utils.get_cur_iter()
        return self.working_strategy

    def finish_strategy(self):
        self.add(self.working_iteration, self.working_strategy)
        pass


def check_division_indices_globally_same(division_indices):
    recevie = [None for _ in range(utils.WORLD_SIZE)]
    torch.distributed.all_gather_object(recevie, division_indices)
    for i in range(utils.WORLD_SIZE):
        for j in range(utils.WORLD_SIZE):
            assert recevie[i][j] == division_indices[j], f"check_division_indices_globally_save failed: {i} {j}"
        

class DivisionStrategyHistory_2(DivisionStrategyHistory):
    def __init__(self, camera, world_size, rank, adjust_mode, heuristic_decay):
        super().__init__(camera, world_size, rank, adjust_mode)

        self.cur_heuristic = None
        self.heuristic_decay = heuristic_decay

    def update_heuristic(self, strategy=None):# XXX: local_running_time is now render backward running time for benchmarking method. 
        if strategy is None:
            strategy = self.working_strategy

        new_heuristic = strategy.heuristic
        if self.cur_heuristic is None:
            self.cur_heuristic = new_heuristic
            return
        
        if strategy.well_balanced():
            # if the new strategy has already been well balanced, we do not have to update the heuristic.
            return

        # update self.cur_heuristic
        self.cur_heuristic = self.cur_heuristic * self.heuristic_decay + new_heuristic * (1-self.heuristic_decay)

    def division_pos_heuristic(self):
        heuristic_prefix_sum = torch.cumsum(self.cur_heuristic, dim=0)
        heuristic_sum = heuristic_prefix_sum[-1]
        heuristic_per_worker = heuristic_sum / self.world_size

        thresholds = torch.arange(1, self.world_size, device="cuda") * heuristic_per_worker
        division_pos = [0]

        # Use searchsorted to find the positions
        division_indices = torch.searchsorted(heuristic_prefix_sum, thresholds)

        # check_division_indices_globally_same(division_indices)

        # Convert to a Python list and prepend the initial division at 0.
        division_pos = [0] + division_indices.cpu().tolist() + [self.tile_num]
        # TODO: use communication collective to check division_pos is the same for all GPUs.

        return division_pos

    def start_strategy(self):
        if len(self.history) == 0:
            division_pos = get_evenly_division_pos(self.camera)
        else:
            division_pos = self.division_pos_heuristic()

        self.working_strategy = DivisionStrategy_2(self.camera, self.world_size, self.rank, self.tile_x, self.tile_y, division_pos, self.adjust_mode)
        self.working_iteration = utils.get_cur_iter()
        return self.working_strategy

    def finish_strategy(self):
        self.update_heuristic()
        self.add(self.working_iteration, self.working_strategy)





# class OldDivisionStrategyHistory(DivisionStrategyHistory):
#     def __init__(self, camera, world_size, rank, adjust_mode):
#         super().__init__(camera, world_size, rank, adjust_mode)

#     def get_next_division_pos(self):
#         # return division_pos # format:[0, d1, d2, ..., tile_num]
#         # This is the core function of workload division.
#         if len(self.history) == 0:
#              # Initialize the division_pos if it is the first iteration for this camera. Use evenly division Strategy.
#             return get_evenly_division_pos(self.camera)
        
#         last_strategy = self.history[-1]["strategy"]

#         if self.adjust_mode == "none":# none, heuristic
#             return last_strategy.division_pos
#         elif self.adjust_mode == "history_heuristic":

#             if last_strategy.workload_is_balanced():
#                 return last_strategy.division_pos

#             division_pos = [0]

#             heuristics = last_strategy.presum_n_contrib # TODO: support other heuristics.

#             # TODO: why do we need presum_? maybe we could delete it to make the code simpler.
#             heuristics_per_worker = heuristics[-1] // self.world_size + 1
#             for j in range(self.tile_y):
#                 for i in range(self.tile_x):
#                     idx = j * self.tile_x + i
#                     if heuristics[idx] > heuristics_per_worker * len(division_pos):
#                         # that will make the last worker's heuristic smaller than others.
#                         division_pos.append(idx)
#             division_pos.append(self.tile_num)
#             # print(self.rank, division_pos)
#             return division_pos # format:[0, d1, d2, ..., tile_num]
    
#     def get_next_strategy(self):
#         division_pos = self.get_next_division_pos()
#         return DivisionStrategy(self.camera, self.world_size, self.rank, self.tile_x, self.tile_y, division_pos, self.adjust_mode)

#     def to_json(self):
#         # change to json format
#         json = []
#         for item in self.history:
#             data = {
#                 "iteration": item["iteration"],
#                 "forward_time": item["strategy"].forward_time,
#                 "backward_time": item["strategy"].backward_time,
#                 "strategy": item["strategy"].to_json(),
#             }
#             if self.adjust_mode in ["history_heuristic", "none"]:
#                 data["worker_n_render"] = item["strategy"].worker_n_render
#                 data["worker_n_consider"] = item["strategy"].worker_n_consider
#                 data["worker_n_contrib"] = item["strategy"].worker_n_contrib
#                 data["worker_tile_num"] = item["strategy"].worker_tile_num
#             json.append(data)
#         return json


class WorkloadDivisionTimer:

    def __init__(self):
        self.timers = {}

    def start(self, key):
        """Start timer for the given key"""
        if key not in self.timers:
            self.timers[key] = {'start_time': None, 'stop_time': None}
        torch.cuda.synchronize()
        self.timers[key]['start_time'] = time.time()

    def stop(self, key):
        torch.cuda.synchronize()
        self.timers[key]['stop_time'] = time.time()

    def elapsed(self, key):
        """Get the elapsed time for the given key without stopping the timer"""
        assert key in self.timers
        assert self.timers[key]['start_time'] is not None
        assert self.timers[key]['stop_time'] is not None

        return (self.timers[key]['stop_time'] - self.timers[key]['start_time'])*1000 # the unit is ms
