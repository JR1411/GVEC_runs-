import numpy as np
import matplotlib.pyplot as plt
import setuptools as distutils
from omfit_classes.omfit_mars import OMFITmars

import gvec
from pathlib import Path
import os
os.environ["OMP_NUM_THREADS"] = "2"

rundir = "/Users/josefreiterer/Desktop/Bachelor-Thesis/coding/workshop-material/data-demo_vmec-DEMO-MARS-DEMO_2024_NTV-OUTPUT_FLUID/DEMO/MARS/DEMO_2024_NTV/OUTPUT_FLUID/"

data = OMFITmars(rundir)

data.load()
data.get_RZ()

rho_vals = data["sim0"]["s"].values
R0 = np.asarray(data["sim0"]["R0EXP"])

R = data["sim0"]["R"]
Z = data["sim0"]["Z"]

outer_flux_surf = 0.85
rho_max = rho_vals < outer_flux_surf

R = R[rho_max] * R0
Z = Z[rho_max] * R0

R_flat = R[-1, :]
Z_flat = Z[-1, :]

data.get_Xcyl()
data.get_UnitVec()

dRds = data["UnitVector"]["dRds"]
dRdchi = data["UnitVector"]["dRdchi"]
dZdchi = data["UnitVector"]["dZdchi"]
dZds = data["UnitVector"]["dZds"]

Xr = data["XPLASMA"]["X1"].values[0:len(dRds.values[rho_max])] * dRds.values[rho_max] + \
    data['XPLASMA']['X2'].values[0: len(dRds.values[rho_max]), :] * \
    dRdchi.values[rho_max]
Xr[0:2, :] = Xr[2, :]
Xz = data['XPLASMA']['X1'].values[0:len(dZds[rho_max]), :] * dZds.values[rho_max] + \
    data['XPLASMA']['X2'].values[0:len(
        dZds[rho_max]), :] * dZdchi.values[rho_max]
Xz[0:2, :] = Xz[2, :]


Xphi = data['XPLASMA']['X3'].values * R0
Xphi[0:2, :] = Xphi[2, :]

Xr_flat = Xr[-1, :]
Xz_flat = Xz[-1, :]
phi = np.linspace(0, 2 * np.pi, 20, endpoint=False)
one_array = np.ones(len(phi))

R_array = np.outer(R_flat, one_array) + \
    np.real(np.outer(Xr_flat, np.exp(-1j * phi)))
Z_array = np.outer(Z_flat, one_array) + \
    np.real(np.outer(Xz_flat, np.exp(-1j * phi)))


def fourier_coefs_half(f, theta, phi, m, n):
    dtheta = np.diff(theta)[0]
    dphi = np.diff(phi)[0]

    exp_m_theta = np.exp(1.0j * m * dtheta)
    exp_n_phi = np.exp(-1.0j * n * dphi)

    exp_m_array = np.power(exp_m_theta, np.arange(len(theta)))
    exp_n_array = np.power(exp_n_phi, np.arange(len(phi)))

    exp_mn = np.outer(exp_m_array, exp_n_array)

    c_mn = np.sum(f * exp_mn) * dtheta * dphi / (2 * np.pi)**2

    cos_mn = np.real(c_mn)
    sin_mn = np.imag(c_mn)

    if m > 0:
        cos_mn *= 2
        sin_mn *= 2

    return cos_mn, sin_mn


m_max = 50
n_max = 2
R_cos_mn = np.zeros((m_max, n_max))
R_sin_mn = np.zeros((m_max, n_max))
Z_cos_mn = np.zeros((m_max, n_max))
Z_sin_mn = np.zeros((m_max, n_max))
theta = np.linspace(0, 2*np.pi, 200, endpoint=False)
phi = np.linspace(0, 2*np.pi, 20, endpoint=False)
for m in range(m_max):
    for n in range(n_max):
        Rc, Rs = fourier_coefs_half(R_array, theta, phi, m, n)
        Zc, Zs = fourier_coefs_half(Z_array, theta, phi, m, n)
        R_cos_mn[m, n] = Rc
        R_sin_mn[m, n] = Rs
        Z_cos_mn[m, n] = Zc
        Z_sin_mn[m, n] = Zs


print(R_cos_mn[0:2])
exit()
params = {}

params["ProjectName"] = "run_gvec"
params["nfp"] = 1
params["PhiEdge"] = 1
params["which_hmap"] = 1
params["X1_mn_max"] = [20, 1]
params["X2_mn_max"] = [20, 1]
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


params["X1_sin_cos"] = "_sincos_"
params["X2_sin_cos"] = "_sincos_"


params["init_avg_axis"] = True

runpath = Path("run_gvec")
run = gvec.run(params, runpath=runpath)

state = run.state

rho = np.linspace(0, 1, 20)
theta = np.linspace(0, 2*np.pi, 50)
zeta = np.linspace(0, 2*np.pi, 25)

ev = state.evaluate("X1", "X2", "LA", "p", "iota", "pos",
                    rho=rho, theta=theta, zeta=zeta)

ev_axis = state.evaluate("pos", theta=[0.0], rho=[1e-8], zeta=zeta)

R = ev.X1[:, :, 0]
Z = ev.X2[:, :, 0]

plt.plot(R.T, Z.T)
plt.axis("equal")
plt.show()
