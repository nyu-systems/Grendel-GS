#!/bin/bash

torchrun --standalone --nnodes=1 --nproc-per-node=4 bench_communication.py --mode allreduce --tensor-size 1024 --num-iterations 10

