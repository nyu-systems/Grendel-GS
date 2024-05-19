#!/bin/bash

echo "Running benchcomm.sh"


torchrun \
    --nnodes 2 \
    --nproc_per_node 4 \
    --master_addr=$MASTER_ADDR \
    --master_port=$port \
    playground/bench_communication.py \
    --mode allreduce \
    --tensor-size 1024 \
    --num-iterations 10

# reference: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#initialize-ddp-with-torch-distributed-run-torchrun