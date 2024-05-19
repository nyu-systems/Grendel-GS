#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 0:30:00
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH -c 2
#SBATCH -G 4
#SBATCH --gpus-per-task=1
#SBATCH --account=m4243_g
#SBATCH --output=sbatch_outputs/final_out_%j.log
#SBATCH --error=sbatch_outputs/final_err_%j.log

# Load modules or source environments if required
module load pytorch/2.1.0-cu12

# Execute the training script
bash /global/homes/j/jy-nyu/gaussian-splatting/final_scripts/360v21080_4g_1_b4_stump_b1.sh
