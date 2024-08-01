""" Load the data from the simulations. 
"""

import pathlib as pl
import json
import numpy as np
import discretisedfield as df
import torch


def load():
    """Load the data from the simulations."""
    simulation_file_paths = list(
        pl.Path("data/sims").glob("Configs_*/drive-[0-9]/Configs_*.omf")
    )

    sim_arr = np.array([df.Field.from_file(file).orientation.sel(
        "z").array for file in simulation_file_paths])

    parameters_dict = {}
    for path in simulation_file_paths:
        json_file_path = path.parent / "parameters_DE.json"
        with open(json_file_path, "r", encoding="utf-8") as f_handle:
            parameters_dict[str(path)] = json.load(f_handle)

    return sim_arr, parameters_dict, simulation_file_paths


if __name__ == "__main__":
    sim_arr, parameters_dict, simulation_file_paths = load()
    sim_arr = sim_arr.reshape(sim_arr.shape[0], sim_arr.shape[3],
                              sim_arr.shape[1], sim_arr.shape[2])
    sim_arr_tensor = torch.tensor(sim_arr, dtype=torch.float32)
    torch.save(sim_arr_tensor, "data/sim_arr_tensor.pt")
    with open("data/parameters_dict.json", "w", encoding="utf-8") as f:
        json.dump(parameters_dict, f)
    with open("data/simulation_file_paths.json", "w", encoding="utf-8") as f:
        json.dump([str(path) for path in simulation_file_paths], f)
