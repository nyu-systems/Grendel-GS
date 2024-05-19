#!/bin/bash

# bash gaussian-splatting/rubble_4k_scripts/rub4k_n4_dp16_3_render.sh
# source ~/zhx.sh

# expe_name="rub4k_n4_dp16_3"

# EXPE_NAME=""
# CHECKPOINTS=(199985)

# for iter in ${CHECKPOINTS[@]}; do


# python render.py \
#     -s /pscratch/sd/j/jy-nyu/datasets/mill19/rubble-4k \
#     --model_path /pscratch/sd/j/jy-nyu/running_expes/rub4k_n4_dp16_3 \
#     --iteration 199985 \
#     --sample_freq 5 \
#     --num_train_cameras 50 \
#     --distributed_load \
#     --llffhold 83 \
#     --skip_test


python render.py \
    -s /pscratch/sd/j/jy-nyu/datasets/mill19/rubble-4k \
    --model_path /pscratch/sd/j/jy-nyu/running_expes/rub4k_n4_dp16_3 \
    --iteration 199985 \
    --num_train_cameras 1 \
    --distributed_load \
    --llffhold 83 \
    --skip_train
