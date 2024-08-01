import discretisedfield as df
import micromagneticmodel as mm
import oommfc as oc
import numpy as np
import os
import micromagneticdata as mdata
import json
from scipy.optimize import bisect

rand_gen = np.random.default_rng()

d = 160e-9  # m
h_min, h_max = 0.0, 1.2 / mm.consts.mu0
h = rand_gen.uniform(h_min, h_max)

Ms = 3.84e5  # A/m
A = 8.78e-12  # J/m
D = 1.58e-3  # J/m^2

# Init configurations


def init_m_sk(k):
    def _init(pos):
        x, y, _ = pos
        rho = (x**2 + y**2) ** 0.5
        phi = np.arctan2(y, x)

        m_phi = np.sin(k * rho)
        m_z = -np.cos(k * rho)

        return (-m_phi * np.sin(phi), m_phi * np.cos(phi), m_z)

    return _init


def init_m_helical(k_h):
    def _init(pos):
        x, _, _ = pos
        return (0, np.cos(k_h * x), np.sin(k_h * x))

    return _init


def g(x):
    return -2 * np.sin(x) ** 2 - np.sin(2 * x) / (2 * x) + 1


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
for i in range(11):
    if i in range(6):
        x_min = i * 0.5 + 0.5
        k = bisect(g, x_min * np.pi, (x_min + 0.5) * np.pi) / d * 2
        system.m = df.Field(
            nvdim=3,
            mesh=mesh,
            value=init_m_sk(k),
            norm=disk,
        )
    elif i in range(6, 9):
        lambda_h = 2 * d / ((i - 6) + 2)
        system.m = df.Field(
            nvdim=3,
            mesh=mesh,
            value=init_m_helical(2 * np.pi / lambda_h),
            norm=disk,
        )
    elif i == 9:
        system.m = df.Field(
            nvdim=3,
            mesh=mesh,
            value=(0, 0, 1),
            norm=disk,
        )
    else:
        system.m = df.Field(
            nvdim=3,
            mesh=mesh,
            value=lambda point: (
                rand_gen.uniform(-1.0, 1.0),
                rand_gen.uniform(-1.0, 1.0),
                rand_gen.uniform(-1.0, 1.0),
            ),
            norm=disk,
        )
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
