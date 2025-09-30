from pathlib import Path
import os
import gvec
import numpy as np
import matplotlib.pyplot as plt
import setuptools as distutils
from omfit_classes.omfit_mars import OMFITmars

rundir = "/Users/josefreiterer/Desktop/Bachelor-Thesis/coding/workshop-material/KINETIC_KNTV=20_newOMEGAW/p0/"

data = OMFITmars(rundir)

data.load()
data.get_RZ()

rho_vals = data["sim0"]["s"].values
R0 = np.array(data["sim0"]["R0EXP"])

R = data["sim0"]["R"].values
Z = data["sim0"]["Z"].values

R = R*R0
Z = Z*R0

last_flux_surf = 0.8
rho_idx = rho_vals <= last_flux_surf

R = R[rho_idx]
Z = Z[rho_idx]

R_flat = R[-1, :]
Z_flat = Z[-1, :]


data.get_Xcyl()
data.get_UnitVec()


dRds = data["UnitVector"]["dRds"]
dRdchi = data["UnitVector"]["dRdchi"]
dZdchi = data["UnitVector"]["dZdchi"]
dZds = data["UnitVector"]["dZds"]


Xr = data["XPLASMA"]["X1"].values[0:len(dRds[rho_idx]), :] * dRds.values[rho_idx] + \
    data['XPLASMA']['X2'].values[0:len(dRdchi[rho_idx]), :] * \
    dRdchi.values[rho_idx]
Xr[0:2, :] = Xr[2, :]
Xz = data['XPLASMA']['X1'].values[0:len(dZds[rho_idx]), :] * dZds.values[rho_idx] + \
    data['XPLASMA']['X2'].values[0:len(
        dZdchi[rho_idx]), :] * dZdchi.values[rho_idx]
Xz[0:2, :] = Xz[2, :]


Xphi = data['XPLASMA']['X3'].values * R0
Xphi[0:2, :] = Xphi[2, :]

phi = np.linspace(0, 2 * np.pi, 20)

one_array = np.ones(len(phi))

Xr_flat = Xr[-1, :]
Xz_flat = Xz[-1, :]

R_array = np.outer(R_flat, one_array) + \
    np.real(np.outer(Xr_flat, np.exp(-1j*phi)))
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


def array_to_dict(arr):
    nx, ny = arr.shape
    return {(i, j): arr[i, j] for i in range(nx) for j in range(ny)}


R_cos_dict = array_to_dict(R_cos_mn)
R_sin_dict = array_to_dict(R_sin_mn)
Z_cos_dict = array_to_dict(Z_cos_mn)
Z_sin_dict = array_to_dict(Z_sin_mn)

os.environ["OMP_NUM_THREADS"] = "2"

params = {}

params["ProjectName"] = "final_run"
params["PhiEdge"] = -276
params["which_hmap"] = 1
params["nfp"] = 1
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

params["X1_b_cos"] = R_cos_dict
params["X1_b_sin"] = R_sin_dict
params["X2_b_cos"] = Z_cos_dict
params["X2_b_sin"] = Z_sin_dict

params["init_average_axis"] = True

runpath = Path("final_run")
run = gvec.run(params, runpath=runpath, keep_intermediates="all")

state = run.state

rho = np.linspace(0, 1, 10)
theta = np.linspace(0, 2*np.pi, 100)
zeta = np.linspace(0, 2*np.pi, 25)

ev = state.evaluate("X1", "X2", "rho", "pos", rho=rho, theta=theta, zeta=zeta)


fig, ax = plt.subplots()
ax.plot(ev.X1[:, :, 0].T, ev.X2[:, :, 0].T, color="black", linestyle="dashed")
ax.axis("equal")
ax.set(xlabel="R [m]", ylabel="Z [m]")
plt.show()
