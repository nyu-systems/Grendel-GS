
source ~/zhx.sh

expe_name="rub_mem_1g_4b"
log_file="/pscratch/sd/j/jy-nyu/final_expes/rub_memory/$expe_name.log"
err_file="/pscratch/sd/j/jy-nyu/final_expes/rub_memory/$expe_name.err"

torchrun /global/homes/j/jy-nyu/gaussian-splatting/train.py \
    -s /pscratch/sd/j/jy-nyu/datasets/mill19/rubble-pixsfm-from-fred \
    --llffhold 83 \
    --iterations 200000 \
    --log_interval 250 \
    --log_folder /pscratch/sd/j/jy-nyu/final_expes/rub_memory/$expe_name \
    --model_path /pscratch/sd/j/jy-nyu/final_expes/rub_memory/$expe_name \
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
    --check_cpu_memory \
    --eval \
    --lr_scale_mode "sqrt" \
    --use_final_system2 > $log_file 2> $err_file




