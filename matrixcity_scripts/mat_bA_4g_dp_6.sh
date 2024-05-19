


expe_name="mat_bA_4g_dp_6"

torchrun --standalone --nnodes=1 --nproc-per-node=4 /global/homes/j/jy-nyu/gaussian-splatting/train.py \
    -s /pscratch/sd/j/jy-nyu/datasets/matrixcity_small/bdaibdai___MatrixCity/small_city/aerial/pose/block_A_my \
    --iterations 300000 \
    --log_interval 250 \
    --log_folder /pscratch/sd/j/jy-nyu/mat_expes/$expe_name \
    --model_path /pscratch/sd/j/jy-nyu/mat_expes/$expe_name \
    --redistribute_gaussians_mode "1" \
    --gaussians_distribution \
    --image_distribution_mode "2" \
    --dp_size 4 \
    --bsz 4 \
    --densify_from_iter 6000 \
    --densification_interval 300 \
    --densify_until_iter 80000 \
    --densify_grad_threshold 0.0002 \
    --opacity_reset_until_iter -1 \
    --percent_dense 0.005 \
    --opacity_reset_interval 9000 \
    --zhx_python_time \
    --log_iteration_memory_usage \
    --check_memory_usage \
    --end2end_time \
    --test_iterations 200 7000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 110000 120000 130000 140000 150000 160000 170000 180000 190000 200000 \
    --save_iterations 7000 15000 20000 30000 50000 80000 100000 120000 150000 170000 200000 \
    --checkpoint_iterations 7000 15000 20000 30000 50000 70000 85000 100000 120000 150000 170000 200000 \
    --auto_start_checkpoint \
    --distributed_dataset_storage \
    --distributed_save \
    --check_cpu_memory \
    --eval \
    --lr_scale_mode "sqrt"







