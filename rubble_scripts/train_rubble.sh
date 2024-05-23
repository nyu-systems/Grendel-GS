

torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py \
    -s /pscratch/sd/j/jy-nyu/datasets/mill19/rubble-pixsfm-from-fred \
    --iterations 100 \
    --log_interval 250 \
    --log_folder experiments/rubble_test \
    --model_path experiments/rubble_test \
    --redistribute_gaussians_mode "1" \
    --gaussians_distribution \
    --image_distribution_mode "2" \
    --dp_size 1 \
    --bsz 1 \
    --zhx_python_time \
    --log_iteration_memory_usage \
    --check_memory_usage \
    --end2end_time \
    --test_iterations 200 7000 15000 20000 30000 40000 50000 \
    --save_iterations 200 7000 15000 20000 30000 40000 50000 \
    --distributed_dataset_storage \
    --multiprocesses_image_loading