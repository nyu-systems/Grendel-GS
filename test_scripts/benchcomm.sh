#!/bin/bash

echo Running benchcomm.sh
echo MASTER_ADDR: $MASTER_ADDR
echo port: $SLURM_STEP_RESV_PORTS
echo rid: $rid

torchrun \
    --nnodes 2 \
    --nproc_per_node 4 \
    --rdzv_id $rid \
    --rdzv_backend c10d \
    --rdzv_endpoint $MASTER_ADDR:$SLURM_STEP_RESV_PORTS \
    playground/bench_communication.py \
    --mode allreduce \
    --tensor-size 1024 \
    --num-iterations 10

# torchrun \
#     --nnodes 2 \
#     --nproc_per_node 4 \
#     --rdzv_id $rid \
#     --rdzv_backend c10d \
#     --rdzv_endpoint $MASTER_ADDR:$port \
#     playground/bench_communication.py \
#     --mode allreduce \
#     --tensor-size 1024 \
#     --num-iterations 10



# reference: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#initialize-ddp-with-torch-distributed-run-torchrun