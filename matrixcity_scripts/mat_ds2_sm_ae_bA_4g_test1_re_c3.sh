
# bash matrixcity_scripts/mat_ds2_sm_ae_bA_4g_test1_re_c3.sh

expe_name="mat_ds2_sm_ae_bA_4g_test1_re_c3"

torchrun --standalone --nnodes=1 --nproc-per-node=4 /global/homes/j/jy-nyu/gaussian-splatting/train.py \
    -s /pscratch/sd/j/jy-nyu/datasets/matrixcity_small/bdaibdai___MatrixCity/small_city/aerial/pose/block_A_my_ds2 \
    --iterations 250000 \
    --log_interval 250 \
    --log_folder /pscratch/sd/j/jy-nyu/mat_expes/$expe_name \
    --model_path /pscratch/sd/j/jy-nyu/mat_expes/$expe_name \
    --redistribute_gaussians_mode "1" \
    --gaussians_distribution \
    --image_distribution_mode "2" \
    --dp_size 4 \
    --bsz 4 \
    --densify_until_iter 15000 \
    --densify_grad_threshold 0.0002 \
    --percent_dense 0.01 \
    --opacity_reset_interval 5000 \
    --zhx_python_time \
    --log_iteration_memory_usage \
    --check_memory_usage \
    --end2end_time \
    --test_iterations 7000 15000 15500 16000 16500 17000 17500 18000 15000 19000 19500 20000 30000 40000 50000 60000 70000 80000 90000 100000 110000 120000 130000 140000 150000 160000 170000 180000 190000 200000 \
    --save_iterations 7000 15000 15500 16000 16500 17000 17500 18000 15000 19000 19500 20000 30000 40000 50000 80000 100000 120000 150000 170000 200000 210000 220000 230000 240000 250000 \
    --checkpoint_iterations 7000 15000 20000 30000 40000 50000 80000 100000 120000 150000 170000 200000 230000 250000 \
    --start_checkpoint /pscratch/sd/j/jy-nyu/mat_expes/mat_ds2_sm_ae_bA_4g_test1_re/checkpoints/199997 \
    --distributed_dataset_storage \
    --check_cpu_memory \
    --eval \
    --lr_scale_mode "sqrt" \
    --clear_floaters \
    --prune_based_on_opacity_interval 4000







