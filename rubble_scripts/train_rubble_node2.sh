

# python train.py \
#     -s /pscratch/sd/j/jy-nyu/datasets/mill19/rubble-pixsfm-from-fred \
#     --iterations 2000 \
#     --log_interval 50 \
#     --log_folder experiments/rubble_test1 \
#     --model_path experiments/rubble_test1 \
#     --bsz 1 \
#     --benchmark_stats \
#     --num_train_cameras 100


torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py \
    -s /pscratch/sd/j/jy-nyu/datasets/mill19/rubble-pixsfm-from-fred \
    --iterations 2000 \
    --log_interval 50 \
    --log_folder experiments/rubble_dp4 \
    --model_path experiments/rubble_dp4 \
    --redistribute_gaussians_mode "1" \
    --gaussians_distribution \
    --image_distribution_mode "2" \
    --dp_size 4 \
    --bsz 4 \
    --benchmark_stats \
    --lr_scale_mode "sqrt" \
    --num_train_cameras 400

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
