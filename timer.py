import time
import utils.general_utils as utils
import torch

class Timer:
    def __init__(self, args, file=None):
        self.timers = {}
        self.args = args
        if args.zhx_python_time:
            self.file = open(args.log_folder+"/python_time_ws="+str(utils.WORLD_SIZE)+"_rk="+str(utils.LOCAL_RANK)+".log", 'a')
        else:
            self.file = None

    def start(self, key):
        if not self.args.zhx_python_time:
            return
        """Start timer for the given key"""
        if key not in self.timers:
            self.timers[key] = {'start_time': None, "cnt": 0, 'all_time': []}

        if utils.WORLD_SIZE > 1 and self.args.global_timer:
            torch.distributed.barrier()
        torch.cuda.synchronize()

        self.timers[key]['start_time'] = time.time()

    def stop(self, key, print_elapsed=False):
        if not self.args.zhx_python_time:
            return

        """Stop the timer for the given key, and report the elapsed time"""
        if key not in self.timers or self.timers[key]['start_time'] is None:
            raise ValueError(f"Timer with key '{key}' is not running.")

        if utils.WORLD_SIZE > 1 and self.args.global_timer:
            torch.distributed.barrier()
        torch.cuda.synchronize()

        cur_time = time.time()
        duration = cur_time - self.timers[key]['start_time']
        self.timers[key]['cnt'] += 1
        self.timers[key]['all_time'].append(duration)
        self.timers[key]['start_time'] = None
        if print_elapsed:
            print(f"Time for '{key}': {duration:.6f} seconds")
        return duration

    def elapsed(self, iteration, mode="this_iteration"):# this_iteration, average
        """Get the elapsed time for the given key without stopping the timer"""
        if not self.args.zhx_python_time:
            return

        for key in self.timers:
            if mode == 'this_iteration':
                print(f"iter {iteration}, TimeFor '{key}': {self.timers[key]['all_time'][-1]*1000:.6f} ms")
                self.file.write(f"iter {iteration}, TimeFor '{key}': {self.timers[key]['all_time'][-1]*1000:.6f} ms\n")
            else:
                average_time = sum(self.timers[key]['all_time']) / self.timers[key]['cnt']
                print(f"iter {iteration}, AverageTimeFor '{key}': {average_time*1000:.6f} ms")
                self.file.write(f"iter {iteration}, AverageTimeFor '{key}': {average_time*1000:.6f} ms\n")
        self.file.write("\n")
