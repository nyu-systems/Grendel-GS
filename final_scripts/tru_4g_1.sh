
# source ~/zhx.sh

# bash /global/homes/j/jy-nyu/gaussian-splatting/final_scripts/tru_4g_1.sh

expe_name="tru_4g_1"

torchrun --standalone --nnodes=1 --nproc-per-node=4 /global/homes/j/jy-nyu/gaussian-splatting/train.py \
    -s /pscratch/sd/j/jy-nyu/datasets/tandt_db/tandt/truck \
    --llffhold 8 \
    --iterations 30000 \
    --log_interval 250 \
    --log_folder /pscratch/sd/j/jy-nyu/final_expes/$expe_name \
    --model_path /pscratch/sd/j/jy-nyu/final_expes/$expe_name \
    --redistribute_gaussians_mode "1" \
    --gaussians_distribution \
    --bsz 4 \
    --densify_until_iter 15000 \
    --densify_grad_threshold 0.0002 \
    --percent_dense 0.01 \
    --opacity_reset_interval 3000 \
    --zhx_python_time \
    --end2end_time \
    --test_iterations 7000 15000 30000 \
    --save_iterations 7000 15000 30000 \
    --checkpoint_iterations 30000 \
    --eval \
    --lr_scale_mode "sqrt" \
    --use_final_system2 \
    --preload_dataset_gpu \
    --no_heuristics_update




