import torch
import time
import argparse
import os



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensor-size', type=float, default=128, help='Size of the tensor for all-reduce; in MB')
    parser.add_argument('--num-iterations', type=int, default=30, help='Number of iterations for benchmarking')
    args = parser.parse_args()

    # print memory occupy of the tensor in GB
    print("tensor size: ", args.tensor_size, "MB")
    print("num elements: ", int(args.tensor_size * 1024 * 1024 / 4))

    tensor_size = int(args.tensor_size * 1024 * 1024 / 4)
    num_iterations = args.num_iterations

    # cpu2gpu

    # Create a tensor of the specified size
    tensor_cpu = torch.rand(tensor_size) # type: float32

    tensor_gpu = []
    # Warm up
    for _ in range(10):
        tensor_gpu.append(tensor_cpu.to('cuda:0'))
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_iterations):
        tensor_gpu.append(tensor_cpu.to('cuda:0'))
    torch.cuda.synchronize()
    end_time = time.time()

    # Calculate the average time per operation
    avg_time = (end_time - start_time) / num_iterations
    print(f"cpu2gpu -> Average time for {tensor_size} elements: {avg_time:.6f} seconds, bandwidth: {tensor_size * 4 / avg_time / 1024 / 1024 / 1024:.6f} GB/s")


    # Create a tensor of the specified size
    tensor_gpu = torch.rand(tensor_size).cuda(0) # type: float32

    tensor_cpu = []
    # Warm up
    for _ in range(10):
        tensor_cpu.append(tensor_gpu.cpu())
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_iterations):
        tensor_cpu.append(tensor_gpu.cpu())
    torch.cuda.synchronize()
    end_time = time.time()

    # Calculate the average time per operation
    avg_time = (end_time - start_time) / num_iterations
    print(f"gpu2cpu -> Average time for {tensor_size} elements: {avg_time:.6f} seconds, bandwidth: {tensor_size * 4 / avg_time / 1024 / 1024 / 1024:.6f} GB/s")







