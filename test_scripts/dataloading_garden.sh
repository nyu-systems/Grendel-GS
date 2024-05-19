

torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py \
    -s /pscratch/sd/j/jy-nyu/datasets/360_v2/garden \
    --iterations 1000 \
    --log_interval 50 \
    --log_folder experiments/dataloading_test \
    --model_path experiments/dataloading_test \
    --redistribute_gaussians_mode "1" \
    --gaussians_distribution \
    --image_distribution_mode "2" \
    --dp_size 1 \
    --bsz 1 \
    --zhx_python_time \
    --log_iteration_memory_usage \
    --check_memory_usage \
    --end2end_time \
    --test_iterations 200 2000 7000 \
    --distributed_dataset_storage
