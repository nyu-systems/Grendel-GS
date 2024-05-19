# launched
source ~/zhx.sh

rid=11
head_node_ip="nid003621"

port=277$rid
expe_name="rub_16g_e"

echo "connecting to head_node_ip: $head_node_ip, port: $port"

# bash /global/homes/j/jy-nyu/gaussian-splatting/last_scripts/rub/rub_16g_e.sh

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
    --log_folder /pscratch/sd/j/jy-nyu/last_scripts/rub/$expe_name \
    --model_path /pscratch/sd/j/jy-nyu/last_scripts/rub/$expe_name \
    --redistribute_gaussians_mode "1" \
    --gaussians_distribution \
    --image_distribution_mode "2" \
    --bsz 16 \
    --densify_until_iter 5000 \
    --densify_grad_threshold 0.0002 \
    --percent_dense 0.01 \
    --opacity_reset_interval 9000 \
    --zhx_python_time \
    --log_iteration_memory_usage \
    --check_memory_usage \
    --end2end_time \
    --test_iterations 200 7000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 110000 120000 130000 140000 150000 160000 170000 180000 190000 200000 \
    --save_iterations 7000 50000 100000 150000 200000 \
    --checkpoint_iterations 7000 50000 100000 150000 200000 \
    --auto_start_checkpoint \
    --distributed_dataset_storage \
    --distributed_save \
    --check_cpu_memory \
    --eval \
    --lr_scale_mode "sqrt" \
    --use_final_system2




