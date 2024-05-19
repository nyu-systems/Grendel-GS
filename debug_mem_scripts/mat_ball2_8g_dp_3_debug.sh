

# bash gaussian-splatting/debug_mem_scripts/mat_ball2_8g_dp_3_debug.sh

source ~/zhx.sh

rid=18
head_node_ip="nid002753"
port=277$rid
echo "connecting to head_node_ip: $head_node_ip, port: $port, rid: $rid"

expe_name="mat_ball2_8g_dp_3_debug"

export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:2048'

export FI_MR_CACHE_MONITOR=disabled

torchrun \
    --nnodes 2 \
    --nproc_per_node 4 \
    --rdzv_id $rid \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node_ip:$port \
    /global/homes/j/jy-nyu/gaussian-splatting/train.py \
    -s /pscratch/sd/j/jy-nyu/datasets/matrixcity_small/bdaibdai___MatrixCity/small_city/aerial/pose/block_all_my2 \
    --llffhold 10 \
    --iterations 300000 \
    --log_interval 250 \
    --log_folder /pscratch/sd/j/jy-nyu/demem_expes/$expe_name \
    --model_path /pscratch/sd/j/jy-nyu/demem_expes/$expe_name \
    --redistribute_gaussians_mode "1" \
    --gaussians_distribution \
    --image_distribution_mode "2" \
    --dp_size 8 \
    --bsz 8 \
    --densify_from_iter 10000 \
    --densification_interval 500 \
    --densify_until_iter 150000 \
    --densify_grad_threshold 0.0002 \
    --percent_dense 0.01 \
    --opacity_reset_interval 30000 \
    --zhx_python_time \
    --log_iteration_memory_usage \
    --check_memory_usage \
    --end2end_time \
    --test_iterations 200 7000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 110000 120000 130000 140000 150000 160000 170000 180000 190000 200000 \
    --save_iterations 7000 20000 50000 80000 100000 120000 150000 170000 200000 \
    --checkpoint_iterations 7000 20000 50000 80000 85000 89000 100000 120000 150000 170000 200000 \
    --start_checkpoint /pscratch/sd/j/jy-nyu/mat_expes/mat_ball2_8g_dp_2/checkpoints/79993 \
    --distributed_dataset_storage \
    --distributed_save \
    --check_cpu_memory \
    --eval \
    --lr_scale_mode "sqrt" \
    --empty_cache_more

    # --log_memory_summary

    # --auto_start_checkpoint \
        # --start_checkpoint /pscratch/sd/j/jy-nyu/mat_expes/mat_ball2_8g_dp_2/checkpoints/79993 \

    # --num_test_cameras 100 \




