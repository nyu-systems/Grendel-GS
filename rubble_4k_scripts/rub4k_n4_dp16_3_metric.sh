#!/bin/bash

python metrics.py \
    --mode test \
    --model_paths /pscratch/sd/j/jy-nyu/running_expes/rub4k_n4_dp16_3

python metrics.py \
    --mode train \
    --model_paths /pscratch/sd/j/jy-nyu/running_expes/rub4k_n4_dp16_3



