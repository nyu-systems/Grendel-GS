#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 8:00:00
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH -c 2
#SBATCH -G 4
#SBATCH --gpus-per-task=1
#SBATCH --account=m4243_g
#SBATCH --output=sbatch_outputs/train_rubble_out_%j.log
#SBATCH --error=sbatch_outputs/train_rubble_err_%j.log

# Load modules or source environments if required
module load pytorch/2.0.1

# Execute the training script
bash /global/homes/j/jy-nyu/gaussian-splatting/rubble_2k_scripts/train_rubble_mp_11.sh
