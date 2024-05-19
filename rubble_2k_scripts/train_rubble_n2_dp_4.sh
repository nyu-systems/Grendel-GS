
head_node_ip="nid002469"
port=27738
rid=28

torchrun \
    --nnodes 2 \
    --nproc_per_node 4 \
    --rdzv_id $rid \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node_ip:$port \
    /global/homes/j/jy-nyu/gaussian-splatting/train.py \
    -s /pscratch/sd/j/jy-nyu/datasets/mill19/rubble-2k \
    --iterations 200000 \
    --log_interval 250 \
    --log_folder /pscratch/sd/j/jy-nyu/running_expes/rubble_2k_n2_dp_4 \
    --model_path /pscratch/sd/j/jy-nyu/running_expes/rubble_2k_n2_dp_4 \
    --redistribute_gaussians_mode "1" \
    --gaussians_distribution \
    --image_distribution_mode "2" \
    --dp_size 8 \
    --bsz 8 \
    --densify_until_iter 50000 \
    --densify_grad_threshold 0.0001 \
    --percent_dense 0.002 \
    --opacity_reset_interval 9000 \
    --min_opacity 0.0032 \
    --zhx_python_time \
    --log_iteration_memory_usage \
    --check_memory_usage \
    --end2end_time \
    --test_iterations 200 7000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 110000 120000 130000 140000 150000 160000 170000 180000 190000 200000 \
    --save_iterations 7000 30000 50000 80000 100000 120000 130000 150000 180000 200000 \
    --checkpoint_iterations 200 30000 50000 80000 100000 120000 150000 180000 200000 \
    --distributed_dataset_storage \
    --distributed_save \
    --check_cpu_memory \
    --eval \
    --lr_scale_mode "sqrt" \
    --my_all2all

# bash rubble_2k_scripts/train_rubble_n2_dp_4.sh > rubble_2k_n2_dp_4_node0.log 2> rubble_2k_n2_dp_4_node0.err
# bash rubble_2k_scripts/train_rubble_n2_dp_4.sh > rubble_2k_n2_dp_4_node1.log 2> rubble_2k_n2_dp_4_node1.err


