

torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py \
    -s /pscratch/sd/j/jy-nyu/datasets/tandt_db/tandt/train \
    --iterations 600 \
    --log_interval 50 \
    --log_folder experiments/distr_save_test \
    --model_path experiments/distr_save_test \
    --redistribute_gaussians_mode "1" \
    --gaussians_distribution \
    --image_distribution_mode "2" \
    --dp_size 1 \
    --bsz 1 \
    --zhx_python_time \
    --log_iteration_memory_usage \
    --check_memory_usage \
    --end2end_time \
    --save_iterations 100 150 200 250 300 250 400 450 500 550 600 \
    --distributed_dataset_storage \
    --distributed_save \
    --check_cpu_memory

