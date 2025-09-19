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
params["X1_mn_max"] = [5, 1]
params["X2_mn_max"] = [5, 1]
params["LA_mn_max"] = [5, 1]
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

params["X1_b_cos"] = {(0, 0): 9.58,
                      (1, 0): -2.2457,
                      (2, 0): 0.1525,
                      (3, 0): 0.383,
                      (4, 0): 0.09,
                      (0, 1): 0.479,
                      (1, 1): -0.112,
                      (2, 1): 7.8e-3,
                      (3, 1): 8.75e-3,
                      (4, 1): 4.55e-3, }
params["X1_b_sin"] = {(0, 0): 0,
                      (1, 0): 0.016,
                      (2, 0): 0.0077,
                      (3, 0): 0.008,
                      (4, 0): 0.0004,
                      (0, 1): -2.17e-4,
                      (1, 1): 7.96e-4,
                      (2, 1): 8.93e-4,
                      (3, 1): -1.12e-4,
                      (4, 1): -6.31e-5, }
params["X2_b_sin"] = {(0, 0): 0,
                      (1, 0): -3.08,
                      (2, 0): -4.497e-2,
                      (3, 0): 0.13,
                      (4, 0): 3.4e-2,
                      (0, 1): 1.92e-7,
                      (1, 1): -1.54e-1,
                      (2, 1): -2.25e-3,
                      (3, 1): -6.56e-4,
                      (4, 1): 1.72e-3, }
params["X2_b_cos"] = {(0, 0): 0.09,
                      (1, 0): 0.0117,
                      (2, 0): 0.005,
                      (3, 0): -0.00067,
                      (4, 0): 0.00285,
                      (0, 1): 4.87e-3,
                      (1, 1): 6.02e-4,
                      (2, 1): 2.67e-4,
                      (3, 1): -3.35e-5,
                      (4, 1): -1.42e-4, }

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

vis_lvl = np.linspace(0, 1 - 1e-10, 5)

ev1 = ev.sel(zeta=np.pi / 2, method="nearest")
ev2 = ev.sel(zeta=np.pi, method="nearest")

fig, ax = plt.subplots()

ax.contour(R, Z, rho_vis, vis_lvl, colors="black")
ax.contour(ev1.X1, ev1.X2, rho_vis, vis_lvl, colors="red")
ax.contour(ev2.X1, ev2.X2, rho_vis, vis_lvl, colors="green")
ax.axis("equal")
ax.set(xlabel="R", ylabel="Z")
plt.show()
