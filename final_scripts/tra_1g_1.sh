
# source ~/zhx.sh

# bash /global/homes/j/jy-nyu/gaussian-splatting/final_scripts/tra_1g_1.sh

expe_name="tra_1g_1"

torchrun /global/homes/j/jy-nyu/gaussian-splatting/train.py \
    -s /pscratch/sd/j/jy-nyu/datasets/tandt_db/tandt/train \
    --llffhold 8 \
    --iterations 30000 \
    --log_interval 250 \
    --log_folder /pscratch/sd/j/jy-nyu/final_expes/$expe_name \
    --model_path /pscratch/sd/j/jy-nyu/final_expes/$expe_name \
    --bsz 1 \
    --densify_until_iter 15000 \
    --densify_grad_threshold 0.0002 \
    --percent_dense 0.01 \
    --opacity_reset_interval 3000 \
    --zhx_python_time \
    --zhx_time \
    --end2end_time \
    --test_iterations 7000 15000 30000 \
    --save_iterations 7000 15000 30000 \
    --checkpoint_iterations 30000 \
    --eval \
    --lr_scale_mode "sqrt" \
    --use_final_system2 \
    --preload_dataset_gpu




