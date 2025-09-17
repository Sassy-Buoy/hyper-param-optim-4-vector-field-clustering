#!/bin/bash

# Set the number of CUDA devices
num_devices=4

# Set the path to your Python script
python_script="/home/tummalas/hyper-param-optim-4-vector-field-clustering/hpo_optuna.py"

# Loop through the CUDA devices and run the Python script
for ((i=0; i<num_devices; i++)); do
    CUDA_VISIBLE_DEVICES=$i python $python_script &
done

# Wait for all the processes to finish
wait