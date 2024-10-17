#!/bin/bash

#SBATCH --job-name=my_job            # Job name
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks-per-node=1          # Tasks per node
#SBATCH --gpus-per-task=1            # GPUs per task
#SBATCH --cpus-per-task=32           # CPUs per task
#SBATCH --mem=128G                   # Memory per node
#SBATCH --partition=gpu              # Partition to use
#SBATCH --time=02:00:00              # Time limit (hh:mm:ss)
#SBATCH --output=job_output_%j.txt   # Standard output and error log

# Run your job
srun python run.py                 # Replace with your actual command
