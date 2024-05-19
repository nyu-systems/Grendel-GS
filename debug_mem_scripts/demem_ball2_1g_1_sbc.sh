#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 16:00:00
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH -c 2
#SBATCH -G 1
#SBATCH --gpus-per-task=1
#SBATCH --account=m4243_g
#SBATCH --output=sbatch_outputs/demem_out_%j.log
#SBATCH --error=sbatch_outputs/demem_err_%j.log

# Load modules or source environments if required
module load pytorch/2.1.0-cu12

# Execute the training script
bash /global/homes/j/jy-nyu/gaussian-splatting/debug_mem_scripts/demem_ball2_1g_1.sh
