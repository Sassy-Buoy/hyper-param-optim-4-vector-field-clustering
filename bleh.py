import os
import torch

# Print number of CPUs
num_cpus = os.cpu_count()
print("Number of CPUs:", num_cpus)

# Print number of GPUs
num_gpus = torch.cuda.device_count()
print("Number of GPUs:", num_gpus)
