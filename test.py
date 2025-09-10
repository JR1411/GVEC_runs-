import gvec
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

from xarray.core.indexes import _wrap_index_equals
os.environ["OMP_NUM_THREADS"] = "1"

params = {}

params["ProjectName"] = "test_run"
params["PhiEdge"] = -276
params["which_hmap"] = 1
params["nfp"] = 1
params["X1_mn_max"] = [3, 3]
params["X2_mn_max"] = [3, 3]
params["LA_mn_max"] = [3, 3]
params["sgrid_nelems"] = 5
params["X1X2_deg"] = 3
params["LA_deg"] = 3
params["totalIter"] = 10000
params["minimize_tol"] = 1e-06
params["iota"] = {
    "type": "interpolation",
    "rho2": [0.0, 0.25, 0.5, 0.75, 1.0],
    "vals": [-0.8, -0.75625, -0.725, -0.70625, -0.7],
    "scale": 1,
}
params["pres"] = {
    "type": "interpolation",
    "rho2": [0.0, 0.25, 0.5, 0.75, 1.0],
    "vals": [0.8, 0.738125, 0.652500, 0.343125, 0.1],
    "scale": 1e6,
}

# params["geometry"] = gvec.util.read_parameters(
# "/Users/josefreiterer/Desktop/Bachelor-Thesis/coding/workshop-material/mars/modes2.toml")

params["X1_b_cos"] = {(0, 0): 9.057,
                      (1, 0): -1.27,
                      (2, 0): 0.042,

                      }
params["X2_b_cos"] = {(0, 0): 0.09,
                      (1, 0): -0.02,
                      (2, 0): 0.01,

                      }

params["X1_b_sin"] = {(0, 0): 0,
                      (1, 0): 0.024,
                      (2, 0): 0.0024}
params["X2_b_sin"] = {(0, 0): 0,
                      (1, 0): -3.11,
                      (2, 0): -0.023, }


params["init_average_axis"] = True
runpath = Path("test_run")

run = gvec.run(params, runpath=runpath, keep_intermediates="all")

state = run.state

rho = np.linspace(0, 1, 20)
theta = np.linspace(0, 2*np.pi, 50)
zeta = np.linspace(0, 2*np.pi, 25)

ev = state.evaluate("X1", "X2", "LA", "p", "iota", "pos",
                    rho=rho, theta=theta, zeta=zeta)

R = ev.X1[:, :, 0]
Z = ev.X2[:, :, 0]

rho_vals = [0.1, 0.25, 0.05, 0.75, 1.0]

rho_vis = R*0 + ev.rho

vis_lvl = np.linspace(0, 1, 10)

fig, ax = plt.subplots()
ax.contourf(R, Z, rho_vis, vis_lvl, cmap="viridis")
ax.axis("equal")
plt.show()
