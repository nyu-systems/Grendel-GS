

python train.py \
    -s /pscratch/sd/j/jy-nyu/datasets/tandt_db/tandt/train \
    --iterations 200 \
    --log_interval 50 \
    --log_folder experiments/test \
    --model_path experiments/test \
    --bsz 1 \
    --test_iterations 1000 7000 \
    --benchmark_stats

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



# I should try on garden, train scenes.
