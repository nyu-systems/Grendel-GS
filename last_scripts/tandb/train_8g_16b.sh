
source ~/zhx.sh

rid=82
head_node_ip="nid003105"
port=277$rid
expe_name="train_8g_16b"

echo "connecting to head_node_ip: $head_node_ip, port: $port"

# bash /global/homes/j/jy-nyu/gaussian-splatting/last_scripts/tandb/train_8g_16b.sh

export FI_MR_CACHE_MONITOR=disabled

torchrun \
    --nnodes 2 \
    --nproc_per_node 4 \
    --rdzv_id $rid \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node_ip:$port \
    /global/homes/j/jy-nyu/gaussian-splatting/train.py \
    -s /pscratch/sd/j/jy-nyu/datasets/tandt_db/tandt/train \
    --llffhold 8 \
    --iterations 30000 \
    --log_interval 250 \
    --log_folder /pscratch/sd/j/jy-nyu/tmp/scalability/$expe_name \
    --model_path /pscratch/sd/j/jy-nyu/tmp/scalability/$expe_name \
    --redistribute_gaussians_mode "1" \
    --gaussians_distribution \
    --bsz 16 \
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




