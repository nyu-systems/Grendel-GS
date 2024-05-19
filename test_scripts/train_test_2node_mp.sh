

head_node_ip="nid008200"
port=27797

# export TORCH_CPP_LOG_LEVEL=INFO
# export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export NCCL_DEBUG_SUBSYS=COLL

# export NCCL_LAUNCH_MODE=GROUP

torchrun \
    --nnodes 2 \
    --nproc_per_node 4 \
    --rdzv_id 12 \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node_ip:$port \
    train.py \
    -s /pscratch/sd/j/jy-nyu/datasets/360_v2/garden \
    --iterations 100000 \
    --log_interval 250 \
    --log_folder experiments/test_2node_debug_mp_6 \
    --model_path experiments/test_2node_debug_mp_6 \
    --redistribute_gaussians_mode "1" \
    --gaussians_distribution \
    --image_distribution_mode "3" \
    --dp_size 1 \
    --bsz 1 \
    --zhx_python_time \
    --log_iteration_memory_usage \
    --check_memory_usage \
    --end2end_time \
    --test_iterations 200 7000 15000 20000 30000 40000 50000 60000 70000 80000 \
    --save_iterations 7000 15000 30000 50000 80000 \
    --distributed_dataset_storage \
    --distributed_save \
    --check_cpu_memory \
    --eval

# bash train_test_2node_mp.sh > debug_node0.log 2> debug_node0.err
# bash train_test_2node_mp.sh > debug_node1.log 2> debug_node1.err



