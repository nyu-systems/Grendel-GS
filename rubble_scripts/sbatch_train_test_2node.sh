#!/bin/bash
#SBATCH -N 2
#SBATCH -n 8
#SBATCH -t 00:10:00
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH -c 2
#SBATCH -G 8
#SBATCH --gpus-per-task=1
#SBATCH --account=m4243_g
#SBATCH --output=sbatch_outputs/twonode_out_%j.log
#SBATCH --error=sbatch_outputs/twonode_err_%j.log

module load pytorch/2.0.1

echo "Starting sbatch script"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"

# nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
# nodes_array=($nodes)
# head_node_ip=${nodes_array[0]}
# MASTER_ADDR=$(hostname)

export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
port=29446

echo "MASTER_ADDR: $MASTER_ADDR"
echo "port: $port"


srun --nodes=2 torchrun \
    --nnodes 2 \
    --nproc_per_node 4 \
    --rdzv_id 100 \
    --rdzv_backend c10d \
    --rdzv_endpoint $MASTER_ADDR:$port \
    train.py \
    -s /pscratch/sd/j/jy-nyu/datasets/tandt_db/tandt/train \
    --iterations 7000 \
    --log_interval 50 \
    --log_folder experiments/test_2node_sbc \
    --model_path experiments/test_2node_sbc \
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
    --eval

# https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#initialize-ddp-with-torch-distributed-run-torchrun