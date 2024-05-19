#!/bin/bash
#SBATCH -N 2
#SBATCH -n 8
#SBATCH --qos=debug
#SBATCH -t 0:03:00
#SBATCH --constraint=gpu
#SBATCH -c 2
#SBATCH -G 8
#SBATCH --gpus-per-task=1
#SBATCH --account=m4243_g
#SBATCH --output=sbatch_outputs/output_%j.log
#SBATCH --error=sbatch_outputs/error_%j.log


module load pytorch/2.1.0-cu12

# nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
# nodes_array=($nodes)
# head_node_ip=${nodes_array[0]}# "nid001144"

export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
export MASTER_PORT=29512
export rid=2

srun --nodes=2 torchrun \
    --nnodes 2 \
    --nproc_per_node 4 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    /global/homes/j/jy-nyu/gaussian-splatting/playground/bench_communication.py \
    --mode allreduce \
    --tensor-size 1024 \
    --num-iterations 10
