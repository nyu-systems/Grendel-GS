#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 4:00:00
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH -c 2
#SBATCH -G 4
#SBATCH --gpus-per-task=1
#SBATCH --account=m4243_g
#SBATCH --output=sbatch_outputs/train_mat_out_%j.log
#SBATCH --error=sbatch_outputs/train_mat_err_%j.log

# Load modules or source environments if required
module load pytorch/2.1.0-cu12

# Execute the training script
bash /global/homes/j/jy-nyu/gaussian-splatting/matrixcity_scripts/mat_ds2_bA_4g_mp_2.sh
