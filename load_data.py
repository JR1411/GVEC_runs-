from scipy import fft
import setuptools as distutils
from omfit_classes.omfit_mars import OMFITmars
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.fft import fft2, fft, ifft


rundir = "/Users/josefreiterer/Desktop/Bachelor-Thesis/coding/workshop-material/KINETIC_KNTV=20_newOMEGAW/"

mars = OMFITmars(rundir)

mars.load()
mars.get_RZ()


rho_values = mars["sim0"]["s"].values

R0 = np.asarray(mars["sim0"]["R0EXP"])

R = mars["sim0"]["R"]
Z = mars["sim0"]["Z"]

R = R0 * R
Z = R0 * Z

chi = mars["sim0"]["chi"]

R_flat = R[rho_values <= 0.8][-1, :]
Z_flat = Z[rho_values <= 0.8][-1, :]


mars.get_Xcyl()
mars.get_UnitVec()


dRds = mars["UnitVector"]["dRds"]
dRdchi = mars["UnitVector"]["dRdchi"]
dZdchi = mars["UnitVector"]["dZdchi"]
dZds = mars["UnitVector"]["dZds"]


Xr = mars["XPLASMA"]["X1"].values[0:128, :] * dRds.values[rho_values <= 0.8] + \
    mars['XPLASMA']['X2'].values[0:128, :] * \
    dRdchi.values[rho_values <= 0.8]
Xr[0:2, :] = Xr[2, :]
Xz = mars['XPLASMA']['X1'].values[0:128, :] * dZds.values[rho_values <= 0.8] + \
    mars['XPLASMA']['X2'].values[0:128, :] * dZdchi.values[rho_values <= 0.8]
Xz[0:2, :] = Xz[2, :]


Xphi = mars['XPLASMA']['X3'].values * R0
Xphi[0:2, :] = Xphi[2, :]

phi = np.linspace(0, 2 * np.pi, 20, endpoint=False)


Xr_flat = Xr[-1, :]
Xz_flat = Xz[-1, :]

one_array = np.ones(len(phi))

R_array = np.outer(R_flat, one_array) + \
    np.real(np.outer(Xr_flat, np.exp(-1j * phi)))

Z_array = np.outer(Z_flat, one_array) + np.outer(Xz_flat, np.exp(-1j * phi))


def reconstruct(data, type: str):
    theta = np.linspace(0, 2 * np.pi, 50)
    # a = (data[0] / 2) + np.zeros(len(data))
    a = np.zeros(len(data))
    if type == "cos":
        for m in range(len(data)):
            a += data[m] * np.cos(m * theta)

    if type == "sin":
        for m in range(len(data)):
            a += data[m] * np.sin(m * theta)

    return a
