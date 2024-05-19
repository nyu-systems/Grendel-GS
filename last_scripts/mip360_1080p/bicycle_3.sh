

# bash /global/homes/j/jy-nyu/gaussian-splatting/last_scripts/mip360_1080p/bicycle_3.sh


source ~/zhx.sh
expe_name="bicycle_3"

torchrun --standalone --nnodes=1 --nproc-per-node=4 /global/homes/j/jy-nyu/gaussian-splatting/train.py \
    -s /pscratch/sd/j/jy-nyu/datasets/360_v2/bicycle1080p \
    --llffhold 8 \
    --iterations 50000 \
    --log_interval 250 \
    --log_folder /pscratch/sd/j/jy-nyu/last_scripts/bicycle/$expe_name \
    --model_path /pscratch/sd/j/jy-nyu/last_scripts/bicycle/$expe_name \
    --redistribute_gaussians_mode "1" \
    --gaussians_distribution \
    --bsz 4 \
    --densify_until_iter 15000 \
    --densify_grad_threshold 0.00015 \
    --percent_dense 0.005 \
    --opacity_reset_interval 3000 \
    --zhx_python_time \
    --log_iteration_memory_usage \
    --check_memory_usage \
    --end2end_time \
    --test_iterations 7000 15000 30000 40000 50000 \
    --save_iterations 7000 15000 30000 50000 \
    --checkpoint_iterations 50000 \
    --auto_start_checkpoint \
    --check_cpu_memory \
    --eval \
    --lr_scale_mode "sqrt" \
    --use_final_system2 \
    --preload_dataset_gpu


