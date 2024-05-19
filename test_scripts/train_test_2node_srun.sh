

head_node_ip="nid001284"
port=27710

srun torchrun \
    --nnodes 2 \
    --nproc_per_node 4 \
    --rdzv_id "191" \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node_ip:$port \
    train.py \
    -s /pscratch/sd/j/jy-nyu/datasets/tandt_db/tandt/train \
    --iterations 15000 \
    --log_interval 50 \
    --log_folder experiments/test_2node_srun \
    --model_path experiments/test_2node_srun \
    --redistribute_gaussians_mode "1" \
    --gaussians_distribution \
    --image_distribution_mode "2" \
    --dp_size 8 \
    --bsz 8 \
    --zhx_python_time \
    --log_iteration_memory_usage \
    --check_memory_usage \
    --end2end_time \
    --test_iterations 200 7000 15000 20000 30000 40000 50000 60000 70000 80000 \
    --save_iterations 7000 15000 30000 50000 80000 \
    --distributed_dataset_storage \
    --distributed_save \
    --check_cpu_memory \
    --eval \
    --lr_scale_mode "sqrt"


    # --checkpoint_iterations 5161 \
    # --start_checkpoint experiments/test_2node_c/checkpoints/5161/ \
# head_node_ip=$1
# port=27778

# torchrun \
#     --nnodes 2 \
#     --nproc_per_node 4 \
#     --rdzv_id "111" \
#     --rdzv_backend c10d \
#     --rdzv_endpoint $head_node_ip:$port \
#     train.py \
#     -s /pscratch/sd/j/jy-nyu/datasets/tandt_db/tandt/train \
#     --iterations 15000 \
#     --log_interval 50 \
#     --log_folder experiments/test_2node_c2 \
#     --model_path experiments/test_2node_c2 \
#     --redistribute_gaussians_mode "1" \
#     --gaussians_distribution \
#     --image_distribution_mode "2" \
#     --dp_size 8 \
#     --bsz 8 \
#     --zhx_python_time \
#     --log_iteration_memory_usage \
#     --check_memory_usage \
#     --end2end_time \
#     --test_iterations 200 7000 15000 20000 30000 40000 50000 60000 70000 80000 \
#     --save_iterations 7000 15000 30000 50000 80000 \
#     --checkpoint_iterations 5161 \
#     --start_checkpoint experiments/test_2node_c/checkpoints/5161/ \
#     --distributed_dataset_storage \
#     --distributed_save \
#     --check_cpu_memory \
#     --eval \
#     --lr_scale_mode "sqrt"

