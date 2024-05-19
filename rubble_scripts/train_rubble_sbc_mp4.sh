

torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py \
    -s /pscratch/sd/j/jy-nyu/datasets/mill19/rubble-pixsfm-from-fred \
    --iterations 90000 \
    --log_interval 250 \
    --log_folder /pscratch/sd/j/jy-nyu/running_expes/rubble_mp4_4_90k_sbc \
    --model_path /pscratch/sd/j/jy-nyu/running_expes/rubble_mp4_4_90k_sbc \
    --redistribute_gaussians_mode "1" \
    --gaussians_distribution \
    --image_distribution_mode "3" \
    --dp_size 1 \
    --bsz 1 \
    --densification_interval 300 \
    --opacity_reset_interval 9000 \
    --densify_from_iter 1500 \
    --densify_until_iter 45000 \
    --zhx_python_time \
    --log_iteration_memory_usage \
    --check_memory_usage \
    --end2end_time \
    --test_iterations 200 7000 15000 20000 30000 40000 50000 60000 70000 80000 90000 \
    --save_iterations 7000 15000 30000 50000 80000 \
    --distributed_dataset_storage

        # self.densification_interval = 100
        # self.opacity_reset_interval = 3000
        # self.densify_from_iter = 500
        # self.densify_until_iter = 15_000
        # self.densify_grad_threshold = 0.0002