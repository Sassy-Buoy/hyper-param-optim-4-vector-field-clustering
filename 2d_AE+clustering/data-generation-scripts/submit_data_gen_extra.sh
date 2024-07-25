#!/bin/bash

# submit this script as: sbatch --array=min-max submit.sh
# min, max are integers

#SBATCH -o ./job.%A_%a.out
#SBATCH -e ./job.%A_%a.out
#SBATCH -D ./
#SBATCH -J disk_sim
#SBATCH -p public

#SBATCH --ntasks=16
#SBATCH --mem=24000

# Adjust time limit if necessary
#SBATCH -t 20:00:00

# module purge
# module load intel/21.2.0 impi/2021.2 anaconda/3/2021.11

# source activate ubermagdev

conda info

python -u data_gen_disk_extra.py
