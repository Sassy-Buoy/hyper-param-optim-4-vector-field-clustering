import discretisedfield as df
import micromagneticmodel as mm
import oommfc as oc
import numpy as np
import os
import micromagneticdata as mdata
import json

rand_gen = np.random.default_rng()

d = 160e-9  # m
h_min, h_max = 0.8 / mm.consts.mu0, 1.1 / mm.consts.mu0
h = rand_gen.uniform(h_min, h_max)

Ms = 3.84e5  # A/m
A = 8.78e-12  # J/m
D = 1.58e-3  # J/m^2

# Region params
p1 = (-d / 2.0, -d / 2.0, 0)
p2 = (d / 2.0, d / 2.0, 10e-9)

# Mesh params
cell = (2e-9, 2e-9, 2e-9)

region = df.Region(p1=p1, p2=p2)
mesh = df.Mesh(region=region, cell=cell)


def disk(point):
    x, y, z = point
    if x**2 + y**2 <= (d / 2.0) ** 2:
        return Ms
    else:
        return 0


system = mm.System(name=f"Configs_{os.getenv('SLURM_ARRAY_TASK_ID')}")
system.energy = (
    mm.Exchange(A=A) + mm.DMI(D=D, crystalclass="T") +
    mm.Zeeman(H=(0, 0, h)) + mm.Demag()
)
for init in ["sat.omf", "sk1.omf", "sk4.omf"]:
    system.m = df.Field.from_file(init)
    # RUN SIMULATION
    minimizer = oc.MinDriver()
    minimizer.drive(system, n_threads=16)
    total_energy = system.table.data["E"].tolist()[-1]
    print(total_energy)

    # save material parameters
    parameter_file = (
        mdata.Data(system.name)[-1].drive_path.absolute() / "parameters.json"
    )
    with parameter_file.open("w", encoding="utf-8") as f:
        json.dump({"d": d, "H": (h * mm.consts.mu0), "E": total_energy}, f)
