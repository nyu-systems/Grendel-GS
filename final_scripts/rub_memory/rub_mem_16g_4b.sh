
source ~/zhx.sh

# bash gaussian-splatting/final_scripts/rub_memory/rub_mem_16g_4b.sh > /pscratch/sd/j/jy-nyu/final_expes/rub_memory/rub_mem_16g_4b_0.log 2> /pscratch/sd/j/jy-nyu/final_expes/rub_memory/rub_mem_16g_4b_0.err
# bash gaussian-splatting/final_scripts/rub_memory/rub_mem_16g_4b.sh > /pscratch/sd/j/jy-nyu/final_expes/rub_memory/rub_mem_16g_4b_1.log 2> /pscratch/sd/j/jy-nyu/final_expes/rub_memory/rub_mem_16g_4b_1.err
# bash gaussian-splatting/final_scripts/rub_memory/rub_mem_16g_4b.sh > /pscratch/sd/j/jy-nyu/final_expes/rub_memory/rub_mem_16g_4b_2.log 2> /pscratch/sd/j/jy-nyu/final_expes/rub_memory/rub_mem_16g_4b_2.err
# bash gaussian-splatting/final_scripts/rub_memory/rub_mem_16g_4b.sh > /pscratch/sd/j/jy-nyu/final_expes/rub_memory/rub_mem_16g_4b_3.log 2> /pscratch/sd/j/jy-nyu/final_expes/rub_memory/rub_mem_16g_4b_3.err

expe_name="rub_mem_16g_4b"
log_file="/pscratch/sd/j/jy-nyu/final_expes/rub_memory/$expe_name.log"
err_file="/pscratch/sd/j/jy-nyu/final_expes/rub_memory/$expe_name.err"

rid=71
head_node_ip="nid001529"
port=277$rid

echo "connecting to head_node_ip: $head_node_ip, port: $port"

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
    --log_folder /pscratch/sd/j/jy-nyu/final_expes/rub_memory/$expe_name \
    --model_path /pscratch/sd/j/jy-nyu/final_expes/rub_memory/$expe_name \
    --redistribute_gaussians_mode "1" \
    --gaussians_distribution \
    --image_distribution_mode "2" \
    --bsz 4 \
    --densify_until_iter 100000 \
    --densify_grad_threshold 0.00005 \
    --percent_dense 0.001 \
    --opacity_reset_interval 100000 \
    --zhx_python_time \
    --log_iteration_memory_usage \
    --check_memory_usage \
    --end2end_time \
    --test_iterations 200 7000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 110000 120000 130000 140000 150000 160000 170000 180000 190000 200000 210000 220000 230000 240000 250000 \
    --save_iterations 7000 30000 50000 60000 70000 80000 900000 100000 110000 120000 130000 140000 150000 160000 170000 180000 190000 200000 210000 220000 230000 240000 250000 \
    --checkpoint_iterations 7000 30000 50000 60000 70000 80000 900000 100000 110000 120000 130000 140000 150000 160000 170000 180000 190000 200000 210000 220000 230000 240000 250000 \
    --distributed_dataset_storage \
    --distributed_save \
    --check_cpu_memory \
    --eval \
    --lr_scale_mode "sqrt" \
    --use_final_system2 \
    --no_heuristics_update


# # bash gaussian-splatting/final_scripts/rub_16g_7_speed/rub_speed_4g_1b.sh



# source ~/zhx.sh

# expe_name="rub_speed_8g_2b"
# log_file="/pscratch/sd/j/jy-nyu/final_expes/rub_16g_7_speed/$expe_name.log"
# err_file="/pscratch/sd/j/jy-nyu/final_expes/rub_16g_7_speed/$expe_name.err"

# rid=61
# head_node_ip="nid003925"
# port=277$rid

# echo "connecting to head_node_ip: $head_node_ip, port: $port"

# export FI_MR_CACHE_MONITOR=disabled

# torchrun \
#     --nnodes 2 \
#     --nproc_per_node 4 \
#     --rdzv_id $rid \
#     --rdzv_backend c10d \
#     --rdzv_endpoint $head_node_ip:$port \
#     /global/homes/j/jy-nyu/gaussian-splatting/train.py \
#     -s /pscratch/sd/j/jy-nyu/datasets/mill19/rubble-pixsfm-from-fred \
#     --llffhold 83 \
#     --iterations 250000 \
#     --log_interval 250 \
#     --log_folder /pscratch/sd/j/jy-nyu/final_expes/rub_16g_7_speed/$expe_name \
#     --model_path /pscratch/sd/j/jy-nyu/final_expes/rub_16g_7_speed/$expe_name \
#     --redistribute_gaussians_mode "1" \
#     --gaussians_distribution \
#     --image_distribution_mode "2" \
#     --bsz 4 \
#     --densify_until_iter 53000 \
#     --densify_grad_threshold 0.0001 \
#     --percent_dense 0.002 \
#     --opacity_reset_interval 9000 \
#     --zhx_python_time \
#     --log_iteration_memory_usage \
#     --check_memory_usage \
#     --end2end_time \
#     --test_iterations 200 7000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 110000 120000 130000 140000 150000 160000 170000 180000 190000 200000 210000 220000 230000 240000 250000 \
#     --save_iterations 7000 30000 50000 60000 70000 80000 900000 100000 110000 120000 130000 140000 150000 160000 170000 180000 190000 200000 210000 220000 230000 240000 250000 \
#     --checkpoint_iterations 7000 30000 50000 60000 70000 80000 900000 100000 110000 120000 130000 140000 150000 160000 170000 180000 190000 200000 210000 220000 230000 240000 250000 \
#     --start_checkpoint /pscratch/sd/j/jy-nyu/final_expes/rub_16g_7/checkpoints/199985 \
#     --distributed_dataset_storage \
#     --distributed_save \
#     --check_cpu_memory \
#     --eval \
#     --lr_scale_mode "sqrt" \
#     --use_final_system2 \
#     --final_system_bench_time




