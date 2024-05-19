#!/bin/bash
#SBATCH -N 1                 # Number of nodes
#SBATCH -n 4                 # Total number of tasks
#SBATCH --qos=interactive_ss11  # Quality of Service
#SBATCH -t 1:00:00           # Time limit hh:mm:ss
#SBATCH --constraint=gpu     # Constraint for GPU
#SBATCH -c 2                 # Number of CPUs per task
#SBATCH -G 4                 # Total number of GPUs
#SBATCH --gpus-per-task=1    # Number of GPUs per task
#SBATCH --account=m4243_g    # Account to charge

# Load modules or source environments if required
module load pytorch/2.0.1

# Execute the training script
bash train_rubble.sh
