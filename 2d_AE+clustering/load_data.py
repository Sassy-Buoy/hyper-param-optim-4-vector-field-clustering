""" Load the data from the simulations. 
"""

import pathlib as pl
import json
import numpy as np
import discretisedfield as df


def load():
    """Load the data from the simulations."""
    simulation_file_paths = list(
        pl.Path("sims").glob("Configs_*/drive-[0-9]/Configs_*.omf")
    )

    sim_arr = np.array([df.Field.from_file(file).orientation.sel(
        "z").array for file in simulation_file_paths])

    parameters_dict = {}
    for path in simulation_file_paths:
        json_file_path = path.parent / "parameters_DE.json"
        with open(json_file_path, "r", encoding="utf-8") as f_handle:
            parameters_dict[str(path)] = json.load(f_handle)

    return sim_arr, parameters_dict, simulation_file_paths


if 'sim_arr' not in locals() or 'parameters_dict' not in locals():
    sim_arr, parameters_dict, simulation_file_paths = load()
