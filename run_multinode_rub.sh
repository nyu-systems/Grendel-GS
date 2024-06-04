
#!/bin/bash

expe_name="run_multinode_rub"

# Set paths
data_path="/pscratch/sd/j/jy-nyu/datasets/mill19/rubble-pixsfm-from-fred"
llffhold=83
log_folder="/pscratch/sd/j/jy-nyu/release_expes/$expe_name"
model_path="/pscratch/sd/j/jy-nyu/release_expes/$expe_name"

# Training configurations
train_config_opts="--iterations 10000 \
--log_interval 250 \
--bsz 8 \
--test_iterations 7000 50000 100000 150000 200000 \
--save_iterations 7000 50000 100000 150000 200000 \
--checkpoint_iterations 7000 50000 100000 150000 200000"

# Densification parameters
densify_opts="--densify_until_iter 50000 \
--densify_grad_threshold 0.0002 \
--percent_dense 0.01 \
--opacity_reset_interval 9000"

# Distributions options
distribution_opts="--image_distribution \
--gaussians_distribution \
--redistribute_gaussians_mode random_redistribute \
--distributed_dataset_storage \
--distributed_save"

# Monitoring Settings
monitor_opts="--enable_timer \
--end2end_time \
--check_gpu_memory \
--check_cpu_memory"

# Run the training script with grouped parameters
python /global/homes/j/jy-nyu/gaussian-splatting/train.py \
    -s $data_path \
    --llffhold $llffhold \
    --log_folder $log_folder \
    --model_path $model_path \
    $train_config_opts \
    $distribution_opts \
    $densify_opts \
    $resource_monitor \
    $iteration_steps \
    --auto_start_checkpoint \
    --eval \
    --num_train_cameras 100






