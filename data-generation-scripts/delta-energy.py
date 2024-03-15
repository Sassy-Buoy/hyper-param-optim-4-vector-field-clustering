import json
import pathlib as pl
import multiprocessing as mp
import numpy as np


simulation_folders = [pl.Path(f"../sims/Configs_{i}/") for i in range(351)]

num_init_states = 10


def populate_delta_E(path):
    energies = np.empty(num_init_states)
    externel_fields = np.empty(num_init_states)
    for i in range(num_init_states):
        file_path = path/f"drive-{i}"/"parameters.json"
        with open(file_path, mode="r") as file_handle:
            parameters_dict = json.load(file_handle)
            energies[i] = parameters_dict["E"]
            externel_fields[i] = parameters_dict["H"]
    delta_energies = energies - energies.min()
    for j in range(num_init_states):
        new_file_path = path/f"drive-{j}"/"parameters_DE.json"
        with open(new_file_path, mode="w") as new_file_handle:
            new_parameters_dict = {"E": delta_energies[j], "H": externel_fields[j]}
            json.dump(new_parameters_dict, new_file_handle)


with mp.Pool(16) as p:
    p.map(populate_delta_E, simulation_folders)
