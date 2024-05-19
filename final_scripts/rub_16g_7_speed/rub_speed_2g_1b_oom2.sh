
# bash gaussian-splatting/final_scripts/rub_16g_7_speed/rub_speed_2g_1b_oom.sh

source ~/zhx.sh

expe_name="rub_speed_2g_1b_oom2"
log_file="/pscratch/sd/j/jy-nyu/final_expes/rub_16g_7_speed/$expe_name.log"
err_file="/pscratch/sd/j/jy-nyu/final_expes/rub_16g_7_speed/$expe_name.err"

torchrun --standalone --nnodes=1 --nproc-per-node=2 /global/homes/j/jy-nyu/gaussian-splatting/train.py \
    -s /pscratch/sd/j/jy-nyu/datasets/mill19/rubble-pixsfm-from-fred \
    --llffhold 83 \
    --iterations 250000 \
    --log_interval 250 \
    --log_folder /pscratch/sd/j/jy-nyu/final_expes/rub_16g_7_speed/$expe_name \
    --model_path /pscratch/sd/j/jy-nyu/final_expes/rub_16g_7_speed/$expe_name \
    --redistribute_gaussians_mode "1" \
    --gaussians_distribution \
    --image_distribution_mode "2" \
    --bsz 1 \
    --densify_from_iter 201000 \
    --densify_until_iter 210000 \
    --densify_grad_threshold 0.0001 \
    --percent_dense 0.002 \
    --opacity_reset_interval 9000 \
    --zhx_python_time \
    --log_iteration_memory_usage \
    --check_memory_usage \
    --end2end_time \
    --test_iterations 201000 202000 203000 210000 220000 230000 240000 250000 \
    --save_iterations 201000 202000 203000 210000 220000 230000 240000 250000 \
    --checkpoint_iterations 201000 202000 203000 210000 220000 230000 240000 250000 \
    --start_checkpoint /pscratch/sd/j/jy-nyu/final_expes/rub_16g_7/checkpoints/199985 \
    --distributed_dataset_storage \
    --distributed_save \
    --check_cpu_memory \
    --eval \
    --num_train_cameras 50 \
    --lr_scale_mode "sqrt" \
    --use_final_system2 \





