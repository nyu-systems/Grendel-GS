import torch
import torch.distributed as dist
import time
import argparse
import os

def benchmark_all_reduce(rank, size, tensor_size, num_iterations):
    """
    Benchmark the all-reduce operation
    """
    # Create a tensor of the specified size
    tensor = torch.rand(tensor_size).cuda(rank) # type: float32

    # Warm up
    for _ in range(10):
        dist.all_reduce(tensor)

    # Benchmark
    torch.distributed.barrier()
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_iterations):
        dist.all_reduce(tensor)
    torch.distributed.barrier()
    torch.cuda.synchronize()
    end_time = time.time()

    # Calculate the average time per operation
    avg_time = (end_time - start_time) / num_iterations
    print(f"Rank: {rank}, Average all-reduce time for {tensor_size} elements: {avg_time:.6f} seconds, bandwidth: {tensor_size * 4 / avg_time / 1024 / 1024 / 1024:.6f} GB/s")

def benchmark_all_reduce_unbalance(rank, size, tensor_size, num_iterations):
    """
    Benchmark the all-reduce operation to test unbalance speed among ranks
    """
    # Create a tensor of the specified size
    tensor = torch.rand(tensor_size).cuda(rank) # type: float32

    # Warm up
    for _ in range(10):
        dist.all_reduce(tensor)
        torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_iterations):
        dist.all_reduce(tensor)
        torch.cuda.synchronize()
    torch.cuda.synchronize()
    end_time = time.time()

    # Calculate the average time per operation
    avg_time = (end_time - start_time) / num_iterations
    print(f"Rank: {rank}, Average all-reduce time for {tensor_size} elements: {avg_time:.6f} seconds, bandwidth: {tensor_size * 4 / avg_time / 1024 / 1024 / 1024:.6f} GB/s")

def benchmark_broadcast(rank, size, tensor_size, num_iterations, root=0):
    """
    Benchmark the broadcast operation
    """
    # Create a tensor of the specified size
    tensor = torch.rand(tensor_size).cuda(rank) # type: float32

    # Warm up
    for _ in range(10):
        dist.broadcast(tensor, root)

    # Benchmark
    torch.distributed.barrier()
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_iterations):
        dist.broadcast(tensor, root)
    torch.distributed.barrier()
    torch.cuda.synchronize()
    end_time = time.time()

    # Calculate the average time per operation
    avg_time = (end_time - start_time) / num_iterations
    print(f"Rank: {rank}, Average broadcast time for {tensor_size} elements: {avg_time:.6f} seconds, bandwidth: {tensor_size * 4 / avg_time / 1024 / 1024 / 1024:.6f} GB/s")

def benchmark_send_recv(rank, size, tensor_size, num_iterations, sender=0, receiver=1):
    """
    Benchmark the send/recv operation
    """

    if rank not in [sender, receiver]:
        return
    
    # Create a tensor of the specified size
    tensor = torch.rand(tensor_size).cuda(rank) # type: float32

    # Warm up
    for _ in range(10):
        if rank == sender:
            dist.send(tensor, dst=receiver)
        elif rank == receiver:
            dist.recv(tensor, src=sender)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    # Benchmark
    # torch.distributed.barrier()
    # torch.cuda.synchronize()
    # start_time = time.time()
    start_event.record()
    for _ in range(num_iterations):
        if rank == sender:
            dist.send(tensor, dst=receiver)
        elif rank == receiver:
            dist.recv(tensor, src=sender)
    end_event.record()
    # torch.distributed.barrier()
    # torch.cuda.synchronize()
    # end_time = time.time()

    # Calculate the average time per operation
    # avg_time = (end_time - start_time) / num_iterations
    torch.cuda.synchronize()
    avg_time = start_event.elapsed_time(end_event) / num_iterations
    print(f"Rank: {rank}, Average send/recv time for {tensor_size} elements: {avg_time:.6f} ms, bandwidth: {tensor_size * 4 / avg_time / 1024 / 1024 :.6f} GB/s")

def benchmark_parallel_send_recv(rank, size, tensor_size, num_iterations):
    """
    Benchmark the parallel send/recv operation
    """
    
    # Create a tensor of the specified size
    tensor = torch.rand(tensor_size).cuda(rank) # type: float32

    # Warm up
    for _ in range(10):
        if rank in [0, 1]:
            dist.send(tensor, dst=rank+2)
        elif rank in [2, 3]:
            dist.recv(tensor, src=rank-2)

    # Benchmark
    # torch.distributed.barrier()
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_iterations):
        if rank in [0, 1]:
            dist.send(tensor, dst=rank+2)
        elif rank in [2, 3]:
            dist.recv(tensor, src=rank-2)
    # torch.distributed.barrier()
    torch.cuda.synchronize()
    end_time = time.time()

    # Calculate the average time per operation
    avg_time = (end_time - start_time) / num_iterations
    print(f"Rank: {rank}, Average parallel send/recv time for {tensor_size} elements: {avg_time:.6f} seconds, bandwidth: {tensor_size * 4 / avg_time / 1024 / 1024 / 1024:.6f} GB/s")

def benchmark_parallel_send_recv_2(rank, size, tensor_size, num_iterations):
    """
    Benchmark the parallel send/recv operation
    """
    
    # Create a tensor of the specified size
    tensor1 = torch.rand(tensor_size).cuda(rank) # type: float32
    tensor2 = torch.rand(tensor_size).cuda(rank) # type: float32

    # Warm up
    for _ in range(10):
        if rank == 0:
            dist.send(tensor1, dst=1)
        if rank == 1:
            dist.recv(tensor1, src=0)
            dist.send(tensor2, dst=2)
        if rank == 2:
            dist.recv(tensor2, src=1)


    # Benchmark
    # torch.distributed.barrier()
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_iterations):
        if rank == 0:
            dist.send(tensor1, dst=1)
        if rank == 1:
            dist.recv(tensor1, src=0)
            dist.send(tensor2, dst=2)
        if rank == 2:
            dist.recv(tensor2, src=1)
    # torch.distributed.barrier()
    torch.cuda.synchronize()
    end_time = time.time()

    # Calculate the average time per operation
    avg_time = (end_time - start_time) / num_iterations
    print(f"Rank: {rank}, Average parallel send/recv time for {tensor_size} elements: {avg_time:.6f} seconds, bandwidth: {tensor_size * 4 / avg_time / 1024 / 1024 / 1024:.6f} GB/s")

def benchmark_parallel_send_recv_3(rank, size, tensor_size, num_iterations):
    """
    Benchmark the parallel send/recv operation
    """
    
    # Create a tensor of the specified size
    tensor1 = torch.rand(tensor_size).cuda(rank) # type: float32
    tensor2 = torch.rand(tensor_size).cuda(rank) # type: float32

    # Warm up
    for _ in range(10):
        if rank == 0:
            dist.send(tensor1, dst=1)
        if rank == 1:
            dist.recv(tensor1, src=0)
        if rank == 1:
            dist.send(tensor2, dst=0)
        if rank == 0:
            dist.recv(tensor2, src=1)


    # Benchmark
    # torch.distributed.barrier()
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_iterations):
        if rank == 0:
            dist.send(tensor1, dst=1)
        if rank == 1:
            dist.recv(tensor1, src=0)
        if rank == 1:
            dist.send(tensor2, dst=0)
        if rank == 0:
            dist.recv(tensor2, src=1)

    # torch.distributed.barrier()
    torch.cuda.synchronize()
    end_time = time.time()

    # Calculate the average time per operation
    avg_time = (end_time - start_time) / num_iterations
    print(f"Rank: {rank}, Average parallel send/recv time for {tensor_size} elements: {avg_time:.6f} seconds, bandwidth: {tensor_size * 4 / avg_time / 1024 / 1024 / 1024:.6f} GB/s")

def benchmark_parallel_send_recv_4(rank, size, tensor_size, num_iterations):
    """
    Benchmark the parallel send/recv operation
    """
    
    # Create a tensor of the specified size
    tensors = [torch.rand(tensor_size).cuda(rank) for _ in range(size)] # type: float32

    # Warm up
    for _ in range(10):
        if rank != 3:
            dist.send(tensors[rank], dst=(rank+1)%size)
        if rank != 0:
            dist.recv(tensors[(rank-1+size)%size], src=(rank-1+size)%size)

    # Benchmark
    # torch.distributed.barrier()
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_iterations):
        if rank != 3:
            dist.send(tensors[rank], dst=(rank+1)%size)
        if rank != 0:
            dist.recv(tensors[(rank-1+size)%size], src=(rank-1+size)%size)

    # torch.distributed.barrier()
    torch.cuda.synchronize()
    end_time = time.time()

    # Calculate the average time per operation
    avg_time = (end_time - start_time) / num_iterations
    print(f"Rank: {rank}, Average parallel send/recv time for {tensor_size} elements: {avg_time:.6f} seconds, bandwidth: {tensor_size * 4 / avg_time / 1024 / 1024 / 1024:.6f} GB/s")

def init_process(rank, size, tensor_size, num_iterations, fn, backend='nccl', **args):
    """
    Initialize the distributed environment and call the benchmark function
    """
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, tensor_size, num_iterations, **args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensor-size', type=float, default=128, help='Size of the tensor for all-reduce; in MB')
    parser.add_argument('--num-iterations', type=int, default=1, help='Number of iterations for benchmarking')
    parser.add_argument('--backend', type=str, default='nccl', help='Distributed backend')
    parser.add_argument('--mode', type=str, default='allreduce')
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
    if args.mode == 'allreduce':
        init_process(LOCAL_RANK, WORLD_SIZE, tensor_size, num_iterations, benchmark_all_reduce, backend)
    elif args.mode == 'allreduce_unbalance':
        init_process(LOCAL_RANK, WORLD_SIZE, tensor_size, num_iterations, benchmark_all_reduce_unbalance, backend)
    elif args.mode == 'sendrecv':
        init_process(LOCAL_RANK, WORLD_SIZE, tensor_size, num_iterations, benchmark_send_recv, backend, sender=1, receiver=0)
    elif args.mode == 'broadcast':
        init_process(LOCAL_RANK, WORLD_SIZE, tensor_size, num_iterations, benchmark_broadcast, backend, root=0)
    elif args.mode == 'parallel_sendrecv':
        init_process(LOCAL_RANK, WORLD_SIZE, tensor_size, num_iterations, benchmark_parallel_send_recv)
    elif args.mode == 'parallel_sendrecv_2':
        init_process(LOCAL_RANK, WORLD_SIZE, tensor_size, num_iterations, benchmark_parallel_send_recv_2)
    elif args.mode == 'parallel_sendrecv_3':
        init_process(LOCAL_RANK, WORLD_SIZE, tensor_size, num_iterations, benchmark_parallel_send_recv_3)
    elif args.mode == 'parallel_sendrecv_4':
        init_process(LOCAL_RANK, WORLD_SIZE, tensor_size, num_iterations, benchmark_parallel_send_recv_4)

# torchrun --standalone --nnodes=1 --nproc-per-node=4 bench_communication.py --mode sendrecv --tensor-size 1024
# Rank: 1, Average send/recv time for 268435456 elements: 0.002048 ms, bandwidth: 500000.001262 GB/s
# Rank: 0, Average send/recv time for 268435456 elements: 93.270180 ms, bandwidth: 10.978857 GB/s