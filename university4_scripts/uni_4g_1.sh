

# bash /global/homes/j/jy-nyu/gaussian-splatting/building_scripts/bui2k_1g_test.sh


# source ~/zhx.sh

# head_node_ip="nid003080"
# port=27709
# rid=104

expe_name="uni_4g_1"

# echo "connecting to head_node_ip: $head_node_ip, port: $port"

# export FI_MR_CACHE_MONITOR=disabled

torchrun --standalone --nnodes=1 --nproc-per-node=4 /global/homes/j/jy-nyu/gaussian-splatting/train.py \
    -s /pscratch/sd/j/jy-nyu/datasets/university4/university4 \
    --iterations 80000 \
    --log_interval 250 \
    --log_folder /pscratch/sd/j/jy-nyu/uni_expes/$expe_name \
    --model_path /pscratch/sd/j/jy-nyu/uni_expes/$expe_name \
    --llffhold 30 \
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
    --test_iterations 200 7000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 110000 120000 130000 140000 150000 160000 170000 180000 190000 200000 \
    --save_iterations 7000 30000 50000 80000 100000 120000 150000 170000 200000 \
    --checkpoint_iterations 7000 30000 50000 70000 85000 100000 120000 150000 170000 200000 \
    --auto_start_checkpoint \
    --distributed_dataset_storage \
    --distributed_save \
    --check_cpu_memory \
    --eval \
    --lr_scale_mode "sqrt"







