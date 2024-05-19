


expe_name="definal_bA_4g_5_final2"

torchrun --standalone --nnodes=1 --nproc-per-node=4 /global/homes/j/jy-nyu/gaussian-splatting/train.py \
    -s /pscratch/sd/j/jy-nyu/datasets/matrixcity_small/bdaibdai___MatrixCity/small_city/aerial/pose/block_A_my \
    --llffhold 10 \
    --num_train_cameras -1 \
    --num_test_cameras -1 \
    --iterations 10000 \
    --log_interval 250 \
    --log_folder /pscratch/sd/j/jy-nyu/definal_expes/$expe_name \
    --model_path /pscratch/sd/j/jy-nyu/definal_expes/$expe_name \
    --redistribute_gaussians_mode "1" \
    --gaussians_distribution \
    --image_distribution_mode "2" \
    --bsz 4 \
    --densify_from_iter 3000 \
    --densification_interval 300 \
    --densify_until_iter 50000 \
    --densify_grad_threshold 0.0002 \
    --percent_dense 0.01 \
    --opacity_reset_interval 9000 \
    --opacity_reset_until_iter 30000 \
    --zhx_python_time \
    --log_iteration_memory_usage \
    --check_memory_usage \
    --end2end_time \
    --test_iterations 200 1000 2000 3000 4000 5000 6000 7000 8000 10000 \
    --save_iterations 200 7000 \
    --checkpoint_iterations 200 7000 \
    --auto_start_checkpoint \
    --distributed_dataset_storage \
    --distributed_save \
    --check_cpu_memory \
    --eval \
    --use_final_system2 \
    --lr_scale_mode "sqrt"







