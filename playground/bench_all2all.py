import torch
import time
import argparse
import os
import json

def benchmark_all2all(rank, size, tensor_size, num_iterations):
    """
    Benchmark the all-2-all operation
    """
    all_i2j_size = [
        [
            [0,1,1,1],
            [1,0,1,1],
            [1,1,0,1],
            [1,1,1,0]
        ],
        [
            [0,1,1,0],
            [1,0,0,2],
            [1,0,0,2],
            [0,2,2,0]
        ], 
        [
            [0,1,1,0],
            [1,0,0,4],
            [1,0,0,4],
            [0,4,4,0]
        ], 
        [
            [0,1,1,1],
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0]
        ], 
        [
            [0,1,1,1],
            [1,0,0,0],
            [1,0,0,0],
            [1,0,0,0]
        ], 

        [
            [0,1,1,1],
            [0,0,1,1],
            [0,0,0,1],
            [0,0,0,0]
        ],
        [
            [0,0,0,0],
            [1,0,0,0],
            [1,1,0,0],
            [1,1,1,0]
        ],
        [
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1],
            [0,0,0,0]
        ], 
        [
            [0,1,0,0],
            [0,0,0,0],
            [0,0,0,1],
            [0,0,0,0]
        ], 
        [
            [0, 8, 4, 2],
            [8, 0, 2, 1],
            [4, 2, 0, 1],
            [2, 1, 1, 0]
        ],
        [
            [0, 1, 1, 2],
            [1, 0, 2, 4],
            [1, 2, 0, 8],
            [2, 4, 8, 0]
        ],
        # [
        #     [0, 1, 1, 1],
        #     [2, 0, 2, 2],
        #     [3, 3, 0, 3],
        #     [4, 4, 4, 0]
        # ],
        [
            [0, 1, 1, 1],
            [1, 0, 2, 2],
            [1, 2, 0, 4],
            [1, 2, 4, 0]
        ],
        [
            [0,10,0,0],
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0]
        ],

        # [
        #     [0,10,0,0],
        #     [10,0,0,0],
        #     [0,0,0,0],
        #     [0,0,0,0]
        # ]
    ]

    if rank == 0:
        os.makedirs("/home/hz3496/gaussian-splatting/experiments/bench_all2all", exist_ok=True)
        file = open("/home/hz3496/gaussian-splatting/experiments/bench_all2all/all2all.log", "w")

    for test_case_idx, i2j_size in enumerate(all_i2j_size):
        if rank == 0:
            print(f"Test case: {test_case_idx}")

        for i in range(4):
            i2j_size[i][i] = 0

        i2j_tensors = []
        for i in range(4):
            a_row = []
            for j in range(4):
                a_row.append(torch.full((i2j_size[i][j]*tensor_size,), rank, dtype=torch.int, device='cuda'))
            i2j_tensors.append(a_row)

        local2j_tensors = i2j_tensors[rank]
        j2local_tensors = [i2j_tensors[i][rank] for i in range(4)]

        # Warm up
        for _ in range(10):
            torch.distributed.all_to_all(j2local_tensors, local2j_tensors)

        # Benchmark
        torch.distributed.barrier()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)


        # torch.cuda.synchronize()    
        # start_time = time.time()

        start_event.record()
        for _ in range(num_iterations):
            torch.distributed.all_to_all(j2local_tensors, local2j_tensors)
        end_event.record()
        # torch.distributed.barrier()
        # torch.cuda.synchronize()
        # end_time = time.time()

        # Calculate the average time per operation
        torch.cuda.synchronize()
        avg_time = start_event.elapsed_time(end_event) / num_iterations

        local_send_size = sum(i2j_size[rank]) * tensor_size
        local_recv_size = sum([i2j_size[i][rank] for i in range(4)]) * tensor_size
        global_send_size = sum([sum(i2j_size[i]) for i in range(4)]) * tensor_size

        all_local_send_size = [None for _ in range(4)]
        all_local_recv_size = [None for _ in range(4)]
        all_ave_time = [None for _ in range(4)]
        torch.distributed.all_gather_object(all_local_send_size, local_send_size)
        torch.distributed.all_gather_object(all_local_recv_size, local_recv_size)
        torch.distributed.all_gather_object(all_ave_time, avg_time)

        # if rank == 0:
        #     print(all_local_send_size, all_local_recv_size, all_ave_time)
        # continue

        global_finish_time = max(all_ave_time)

        if rank == 0:
            output_str = f"Test case {test_case_idx}: {json.dumps(i2j_size)}\n"
            output_str += "i send to j size matrix:\n"
            for i in range(4):
                for j in range(4):
                    output_str += f"{i2j_size[i][j]:2d} "
                output_str += "\n"

            output_str += f"global_send_size: {global_send_size *4 / (1024**3):.6f} GB, \
global_send_bandwidth: {global_send_size / (global_finish_time * 1e6):.6f} GB/s\n"

            for rk in range(4):
                local_send_size = all_local_send_size[rk]
                local_recv_size = all_local_recv_size[rk]
                avg_time = all_ave_time[rk]
                
                output_str += f"Rank: {rk}, \
avg_time: {avg_time} ms, \
local_send_size: {local_send_size *4 / (1024**3):.6f} GB, \
local_recv_size: {local_recv_size *4 / (1024**3):.6f} GB, \
local_send_bandwidth: {local_send_size / (avg_time * 1e6):.6f} GB/s, \
local_recv_bandwidth: {local_recv_size / (avg_time * 1e6):.6f} GB/s\n"
            file.write(output_str + "\n\n")

        torch.distributed.barrier()

def init_process(rank, size, tensor_size, num_iterations, fn, backend='nccl', **args):
    """
    Initialize the distributed environment and call the benchmark function
    """
    torch.distributed.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, tensor_size, num_iterations, **args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensor-size', type=float, default=128, help='Size of the tensor for all-reduce; in MB')
    parser.add_argument('--num-iterations', type=int, default=1, help='Number of iterations for benchmarking')
    parser.add_argument('--backend', type=str, default='nccl', help='Distributed backend')
    parser.add_argument('--mode', type=str, default='all2all')
    args = parser.parse_args()

    # get environment variables
    print("NCCL_ALGO: ", os.environ.get("NCCL_ALGO", None))
    print("NCCL_DEBUG: ", os.environ.get("NCCL_DEBUG", None))

    # print memory occupy of the tensor in GB
    print("tensor size: ", args.tensor_size, "MB")
    print("num elements: ", int(args.tensor_size * 1024 * 1024 / 4))

    tensor_size = int(args.tensor_size * 1024 * 1024 / 4)
    num_iterations = args.num_iterations
    backend = args.backend

    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(torch.device("cuda", LOCAL_RANK))

    if args.mode == 'all2all':
        init_process(LOCAL_RANK, WORLD_SIZE, tensor_size, num_iterations, benchmark_all2all, backend)

# torchrun --standalone --nnodes=1 --nproc-per-node=4 bench_all2all.py