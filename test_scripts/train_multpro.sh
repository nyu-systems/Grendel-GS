
    # -s /pscratch/sd/j/jy-nyu/datasets/360_v2/bicycle \

# torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py \
#     -s /pscratch/sd/j/jy-nyu/datasets/tandt_db/tandt/train \
#     --iterations 7000 \
#     --log_interval 50 \
#     --log_folder experiments/debug_5_2 \
#     --model_path experiments/debug_5_2 \
#     --redistribute_gaussians_mode "1" \
#     --gaussians_distribution \
#     --image_distribution_mode "2" \
#     --dp_size 1 \
#     --bsz 1 \
#     --benchmark_stats \
#     --distributed_dataset_storage

torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py \
    -s /pscratch/sd/j/jy-nyu/datasets/tandt_db/tandt/train \
    --iterations 7000 \
    --log_interval 50 \
    --log_folder experiments/debug_9 \
    --model_path experiments/debug_9 \
    --redistribute_gaussians_mode "1" \
    --gaussians_distribution \
    --image_distribution_mode "3" \
    --dp_size 1 \
    --bsz 1 \
    --benchmark_stats \
    --test_iterations 200 1000 7000 \
    --distributed_dataset_storage



torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py \
    -s /pscratch/sd/j/jy-nyu/datasets/tandt_db/tandt/train \
    --iterations 7000 \
    --log_interval 50 \
    --log_folder experiments/debug_9_gt \
    --model_path experiments/debug_9_gt \
    --redistribute_gaussians_mode "1" \
    --gaussians_distribution \
    --image_distribution_mode "3" \
    --dp_size 1 \
    --bsz 1 \
    --benchmark_stats \
    --test_iterations 200 1000 7000

    
#  \
    # --distributed_dataset_storage

# torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py \
#     -s /pscratch/sd/j/jy-nyu/datasets/tandt_db/tandt/train \
#     --iterations 7000 \
#     --log_interval 50 \
#     --log_folder experiments/debug_1_gt \
#     --model_path experiments/debug_1_gt \
#     --redistribute_gaussians_mode "1" \
#     --gaussians_distribution \
#     --image_distribution_mode "2" \
#     --dp_size 4 \
#     --bsz 4 \
#     --benchmark_stats

# torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py \
#     -s /pscratch/sd/j/jy-nyu/datasets/tandt_db/tandt/train \
#     --iterations 7000 \
#     --log_interval 50 \
#     --log_folder experiments/debug_0_gt \
#     --model_path experiments/debug_0_gt \
#     --redistribute_gaussians_mode "1" \
#     --gaussians_distribution \
#     --image_distribution_mode "2" \
#     --dp_size 1 \
#     --bsz 1 \
#     --benchmark_stats



# torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py \
#     -s /pscratch/sd/j/jy-nyu/datasets/360_v2/bicycle \
#     --iterations 10 \
#     --log_interval 50 \
#     --log_folder experiments/time_image_loading \
#     --model_path experiments/time_image_loading \
#     --redistribute_gaussians_mode "1" \
#     --gaussians_distribution \
#     --image_distribution_mode "3" \
#     --dp_size 1 \
#     --bsz 1 \
#     --benchmark_stats \
#     --time_image_loading

# torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py \
#     -s /pscratch/sd/j/jy-nyu/datasets/tandt_db/tandt/train \
#     --iterations 7000 \
#     --log_interval 50 \
#     --log_folder experiments/test_mp_1 \
#     --model_path experiments/test_mp_1 \
#     --redistribute_gaussians_mode "1" \
#     --gaussians_distribution \
#     --image_distribution_mode "2" \
#     --dp_size 1 \
#     --bsz 1 \
#     --benchmark_stats \
#     --eval

# I should try on garden, train scenes.
