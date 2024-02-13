import torch
import torch.distributed as dist
from datetime import timedelta
import os
import time

LOCAL_RANK, WORLD_SIZE = 0, 1

def init_distributed():
    global LOCAL_RANK, WORLD_SIZE
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    print (f"LOCAL_RANK: {LOCAL_RANK}, WORLD_SIZE: {WORLD_SIZE}")
    if WORLD_SIZE > 1:
        dist.init_process_group("nccl", rank=LOCAL_RANK, world_size=WORLD_SIZE)
        assert torch.cuda.is_available(), "Distributed mode requires CUDA"
        assert torch.distributed.is_initialized(), "Distributed mode requires init_distributed() to be called first"

if __name__ == "__main__":

    init_distributed()

    if LOCAL_RANK == 0:
        # Run on process 1 (server)
        server_store = dist.TCPStore("127.0.0.1", 6666, 2, True, timedelta(seconds=30))
        # Use any of the store methods from either the client or server after initialization

        # obj = "aaaa" * (1024 * 1024 * 16) # 16MB
        # obj = "aaaa" * 1 # 1 B => 0.0639ms
        # obj = "aaaa" * 64 # 64 B => 0.0646ms
        obj = "aaaa" * 256 # 64 B => 0.0646ms
        print("memory consumption of obj: ", len(obj))
        print(LOCAL_RANK, "Server set at ", time.time())
        server_store.set("key", obj)
    else:
        client_store = dist.TCPStore("127.0.0.1", 6666, 2, False)
        obj = client_store.get("key")
        print(LOCAL_RANK, "client get at ", time.time()) # time unit is second
        print("memory consumption of obj: ", len(obj))

# torchrun --standalone --nnodes=1 --nproc-per-node=2 test_dist_store.py