import gvec
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

from pandas.core.series import pd_array
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

params["X1_b_cos"] = {(0, 0): 9.057,
                      (1, 0): -2.2069,
                      (2, 0): 8.3827e-2,
                      (3, 0): 1.061e-1,
                      (4, 0): 1.6459e-2,
                      (0, 1): 4.77e-1,
                      (1, 1): -1.16e-1,
                      (2, 1): 4.61e-3,
                      (3, 1): 5.14e-3,
                      (4, 1): 7.95e-4, }
params["X1_b_sin"] = {(0, 0): 0,
                      (1, 0): 4.947e-2,
                      (2, 0): 4.65e-3,
                      (3, 0): 2.49894e-3,
                      (4, 0): -8.829e-4,
                      (0, 1): -0.00021,
                      (1, 1): 0.0021,
                      (2, 1): 0.00074,
                      (3, 1): -0.00036,
                      (4, 1): -0.00012, }
params["X2_b_cos"] = {(0, 0): 9.4e-2,
                      (1, 0): -3.5678e-2,
                      (2, 0): 3.35e-3,
                      (3, 0): 4.623e-3,
                      (4, 0): -1.344e-3,
                      (0, 1): 4.85e-3,
                      (1, 1): -1.86e-3,
                      (2, 1): 1.73e-4,
                      (3, 1): 2.445e-4,
                      (4, 1): 7.06e-5, }
params["X2_b_sin"] = {(0, 0): 3.42e-22,
                      (1, 0): -2.917,
                      (2, 0): -2.15e-2,
                      (3, 0): 1.36e-1,
                      (4, 0): 3.82e-2,
                      (0, 1): 1.91e-7,
                      (1, 1): -1.54e-1,
                      (2, 1): -1.13e-3,
                      (3, 1): 7.17e-3,
                      (4, 1): 2.01e-3, }

params["init_average_axis"] = True
runpath = Path("test_run")

run = gvec.run(params, runpath=runpath, keep_intermediates="all")

state = run.state

rho = np.linspace(0, 1, 20)
theta = np.linspace(0, 2*np.pi, 50)
zeta = np.linspace(0, 2*np.pi, 25)

ev = state.evaluate("X1", "X2", "LA", "p", "iota", "pos",
                    rho=rho, theta=theta, zeta=zeta)

ev_axis = state.evaluate("pos", theta=[0.0], rho=[1e-8], zeta=zeta)
