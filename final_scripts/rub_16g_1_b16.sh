
source ~/zhx.sh

rid=32
head_node_ip="nid001164"
# 1264, 1916, 8593
port=277$rid
expe_name="rub_16g_1_b16"

echo "connecting to head_node_ip: $head_node_ip, port: $port"

# bash gaussian-splatting/final_scripts/rub_16g_1_b16.sh

export FI_MR_CACHE_MONITOR=disabled

torchrun \
    --nnodes 4 \
    --nproc_per_node 4 \
    --rdzv_id $rid \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node_ip:$port \
    /global/homes/j/jy-nyu/gaussian-splatting/train.py \
    -s /pscratch/sd/j/jy-nyu/datasets/mill19/rubble-pixsfm-from-fred \
    --llffhold 83 \
    --iterations 200000 \
    --log_interval 250 \
    --log_folder /pscratch/sd/j/jy-nyu/final_expes/$expe_name \
    --model_path /pscratch/sd/j/jy-nyu/final_expes/$expe_name \
    --redistribute_gaussians_mode "1" \
    --gaussians_distribution \
    --image_distribution_mode "2" \
    --bsz 16 \
    --densify_until_iter 50000 \
    --densify_grad_threshold 0.0001 \
    --percent_dense 0.002 \
    --opacity_reset_interval 9000 \
    --zhx_python_time \
    --log_iteration_memory_usage \
    --check_memory_usage \
    --end2end_time \
    --test_iterations 200 7000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 110000 120000 130000 140000 150000 160000 170000 180000 190000 200000 \
    --save_iterations 7000 30000 50000 60000 70000 80000 900000 100000 110000 120000 130000 140000 150000 160000 170000 180000 190000 200000 \
    --checkpoint_iterations 7000 30000 50000 60000 70000 80000 900000 100000 110000 120000 130000 140000 150000 160000 170000 180000 190000 200000 \
    --auto_start_checkpoint \
    --distributed_dataset_storage \
    --distributed_save \
    --check_cpu_memory \
    --eval \
    --lr_scale_mode "sqrt" \
    --use_final_system2




