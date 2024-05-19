
# torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py \
#     -s /pscratch/sd/j/jy-nyu/datasets/360_v2/garden \
#     --iterations 7000 \
#     --log_interval 50 \
#     --log_folder experiments/garden_mp4_dist \
#     --model_path experiments/garden_mp4_dist \
#     --redistribute_gaussians_mode "1" \
#     --gaussians_distribution \
#     --image_distribution_mode "2" \
#     --dp_size 1 \
#     --bsz 1 \
#     --benchmark_stats \
#     --test_iterations 200 2000 7000

# torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py \
#     -s /pscratch/sd/j/jy-nyu/datasets/360_v2/garden \
#     --iterations 500 \
#     --log_interval 50 \
#     --log_folder experiments/garden_debug3 \
#     --model_path experiments/garden_debug3 \
#     --redistribute_gaussians_mode "1" \
#     --gaussians_distribution \
#     --image_distribution_mode "2" \
#     --dp_size 1 \
#     --bsz 1 \
#     --benchmark_stats \
#     --test_iterations 200 2000 7000

torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py \
    -s /pscratch/sd/j/jy-nyu/datasets/360_v2/garden \
    --iterations 500 \
    --log_interval 50 \
    --log_folder experiments/garden_mp \
    --model_path experiments/garden_mp \
    --redistribute_gaussians_mode "1" \
    --gaussians_distribution \
    --image_distribution_mode "2" \
    --dp_size 1 \
    --bsz 1 \
    --zhx_python_time \
    --log_iteration_memory_usage \
    --check_memory_usage \
    --end2end_time \
    --test_iterations 200 2000 7000



# torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py \
#     -s /pscratch/sd/j/jy-nyu/datasets/mill19/rubble-pixsfm-from-fred \
#     --iterations 7000 \
#     --log_interval 250 \
#     --log_folder experiments/rubble_dp4_2 \
#     --model_path experiments/rubble_dp4_2 \
#     --redistribute_gaussians_mode "1" \
#     --gaussians_distribution \
#     --image_distribution_mode "2" \
#     --dp_size 4 \
#     --bsz 4 \
#     --benchmark_stats \
#     --lr_scale_mode "sqrt" \
#     --num_train_cameras 400 \
#     --test_iterations 200 2000 7000

# python train.py \
#     -s /pscratch/sd/j/jy-nyu/datasets/mill19/rubble-pixsfm-from-fred \
#     --iterations 2000 \
#     --log_interval 50 \
#     --log_folder experiments/rubble_test1 \
#     --model_path experiments/rubble_test1 \
#     --bsz 1 \
#     --benchmark_stats \
#     --num_train_cameras 100


# torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py \
#     -s /pscratch/sd/j/jy-nyu/datasets/mill19/rubble-pixsfm-from-fred \
#     --iterations 2000 \
#     --log_interval 50 \
#     --log_folder experiments/rubble_test2 \
#     --model_path experiments/rubble_test2 \
#     --redistribute_gaussians_mode "1" \
#     --gaussians_distribution \
#     --image_distribution_mode "2" \
#     --dp_size 4 \
#     --bsz 4 \
#     --benchmark_stats

# torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py \
#     -s /pscratch/sd/j/jy-nyu/datasets/mill19/rubble-pixsfm-from-fred \
#     --iterations 2000 \
#     --log_interval 50 \
#     --log_folder experiments/rubble_test3 \
#     --model_path experiments/rubble_test3 \
#     --redistribute_gaussians_mode "1" \
#     --gaussians_distribution \
#     --image_distribution_mode "2" \
#     --dp_size 1 \
#     --bsz 1 \
#     --benchmark_stats
