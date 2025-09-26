import numpy as np
import matplotlib.pyplot as plt
import setuptools as distutils
from omfit_classes.omfit_mars import OMFITmars


rundir = "/Users/josefreiterer/Desktop/Bachelor-Thesis/coding/workshop-material/data_albert/data-demo_vmec-DEMO-MARS-DEMO_2024_NTV-OUTPUT_FLUID/DEMO/MARS/DEMO_2024_NTV/OUTPUT_FLUID/"

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

plt.plot(R_array, Z_array)
plt.axis("equal")
plt.show()
