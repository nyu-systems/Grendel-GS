#!/bin/bash
#SBATCH -N 2
#SBATCH -n 8
#SBATCH --qos=debug
#SBATCH -t 0:10:00
#SBATCH --constraint=gpu
#SBATCH -c 2
#SBATCH -G 8
#SBATCH --gpus-per-task=1
#SBATCH --account=m4243_g
#SBATCH --output=sbatch_outputs/output_%j.log
#SBATCH --error=sbatch_outputs/error_%j.log


# Load modules or source environments if required
module load pytorch/2.0.1

# Execute the training script
bash train_rubble.sh
