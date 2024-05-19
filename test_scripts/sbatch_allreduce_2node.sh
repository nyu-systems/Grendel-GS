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
#SBATCH --output=sbatch_outputs/allreduce2node_out_%j.log
#SBATCH --error=sbatch_outputs/allreduce2node_err_%j.log

module load pytorch/2.0.1

echo "Starting sbatch script"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"

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
    playground/bench_communication.py \
    --mode allreduce \
    --tensor-size 1024 \
    --num-iterations 10

# https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#initialize-ddp-with-torch-distributed-run-torchrun