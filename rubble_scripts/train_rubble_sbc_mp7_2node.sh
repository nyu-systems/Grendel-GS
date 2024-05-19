

IP_ADDR=$1
NODE_RANK=$2

NODE_COUNT=2
PORT=27576

echo "IP_ADDR="$IP_ADDR
echo "NODE_RANK="$NODE_RANK
echo "PORT="$PORT

torchrun --nnodes=$NODE_COUNT --node_rank=$NODE_RANK --nproc-per-node=4 \
    --master_addr=$IP_ADDR --master_port=$PORT train.py \
    -s /tmp/rubble \
    --iterations 120000 \
    --log_interval 450 \
    --log_folder /tmp/rubble_mp4_7_sbc_2node \
    --model_path /tmp/rubble_mp4_7_sbc_2node \
    --redistribute_gaussians_mode "1" \
    --gaussians_distribution \
    --image_distribution_mode "3" \
    --dp_size 1 \
    --bsz 1 \
    --densify_grad_threshold 0.00005 \
    --percent_dense 0.001 \
    --zhx_python_time \
    --densification_interval 200 \
    --densify_until_iter 60000 \
    --log_iteration_memory_usage \
    --check_memory_usage \
    --end2end_time \
    --test_iterations 200 7000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 110000 120000 \
    --save_iterations 200 7000 20000 30000 40000 50000 60000 70000 80000 90000 100000 110000 120000 \
    --distributed_dataset_storage \
    --distributed_save \
    --check_cpu_memory \
    --num_train_cameras 100

    # --log_folder /pscratch/sd/j/jy-nyu/running_expes/rubble_mp4_7_sbc_2node \
    # --model_path /pscratch/sd/j/jy-nyu/running_expes/rubble_mp4_7_sbc_2node \