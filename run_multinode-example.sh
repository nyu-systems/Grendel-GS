#!/bin/bash -l
#SBATCH --time=00:05:00
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --account=m4243_g
#SBATCH -o run_multinode_%j.out
#SBATCH -e run_multinode_%j.err

module load pytorch/2.1.0-cu12

export MASTER_ADDR=$(hostname)

cmd="bash run_multinode_rub.sh"

set -x
srun -l \
    bash -c "
    source export_DDP_vars.sh
    $cmd
    " 

