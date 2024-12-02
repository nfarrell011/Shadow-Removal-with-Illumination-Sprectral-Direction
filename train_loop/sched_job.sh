#!/bin/bash
#SBATCH --job-name=vision_transformer      # Job name
#SBATCH --output=logs/%x_%j.out            # Output log file (%x = job name, %j = job ID)
#SBATCH --error=logs/%x_%j.err             # Error log file
#SBATCH --partition=gpu                    # GPU partition (adjust as per your cluster)
#SBATCH --gres=gpu:1                       # Number of GPUs per node
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --ntasks=1                         # Number of tasks (processes)
#SBATCH --cpus-per-task=4                  # Number of CPU cores per task
#SBATCH --mem=32G                          # Memory per node
#SBATCH --time=24:00:00                    # Time limit (hh:mm:ss)
#SBATCH --mail-user=$USER@northeastern.edu  # Email
#SBATCH --mail-type=ALL                     # Type of email notifications


# Activate your Conda environment
source ~/miniconda3/bin/activate isd_vit

# Run your Python script
srun python /isd_ViT/train_driver.py