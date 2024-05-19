

python train.py \
    -s /pscratch/sd/j/jy-nyu/datasets/tandt_db/tandt/train \
    --iterations 7000 \
    --log_interval 50 \
    --log_folder experiments/test \
    --model_path experiments/test \
    --bsz 1 \
    --benchmark_stats

# python train.py \
#     -s /pscratch/sd/j/jy-nyu/datasets/mill19/rubble-pixsfm-from-fred \
#     --iterations 100 \
#     --log_interval 50 \
#     --log_folder experiments/multipro_load0_gt \
#     --model_path experiments/multipro_load0_gt \
#     --bsz 1 \
#     --benchmark_stats

# python train.py \
#     -s /pscratch/sd/j/jy-nyu/datasets/mill19/rubble-pixsfm-from-fred \
#     --iterations 100 \
#     --log_interval 50 \
#     --log_folder experiments/multipro_load0 \
#     --model_path experiments/multipro_load0 \
#     --bsz 1 \
#     --benchmark_stats \
#     --multiprocesses_image_loading

# torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py \
#     -s /pscratch/sd/j/jy-nyu/datasets/mill19/rubble-pixsfm-from-fred \
#     --iterations 2000 \
#     --log_interval 50 \
#     --log_folder experiments/rubble_test \
#     --model_path experiments/rubble_test \
#     --redistribute_gaussians_mode "1" \
#     --gaussians_distribution \
#     --image_distribution_mode "2" \
#     --dp_size 4 \
#     --bsz 4 \
#     --benchmark_stats

