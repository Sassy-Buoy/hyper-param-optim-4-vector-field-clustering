#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./job.out.%j
#SBATCH -e ./job.err.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J 2D_clustering
#
#SBATCH --nodes=1            # Request 1 or more full nodes
#SBATCH --constraint="gpu"   #   providing GPUs.
#SBATCH --gres=gpu:a100:4    # Request 4 GPUs per node.
#SBATCH --ntasks-per-node=4  # Run one task per GPU
#SBATCH --cpus-per-task=18   #   using 18 cores each.
#SBATCH --time=12:00:00

module purge
module load intel/21.2.0 impi/2021.2 cuda/11.2

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

srun .venv/bin/python run.py