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

R_flat = R[rho_values <= 1.0][-1, :]
Z_flat = Z[rho_values <= 1.0][-1, :]


def Test_fourier(data, theta):
    cos_coeff = np.real(data)
    sin_coeff = np.imag(data)
    N = max(len(cos_coeff), len(sin_coeff))
    result = np.zeros_like(data)
    for i in range(0, N):
        result += cos_coeff[i] * \
            np.cos(i * theta) + sin_coeff[i] * np.sin(i * theta)
    return result


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

phi = np.linspace(0, 2 * np.pi, 200)

R_total = R[rho_values <= 0.8] + (np.real(Xr))
Z_total = Z[rho_values <= 0.8] + (np.real(Xz))


def plot_Xwarp_1(self, rsurf, phi=np.linspace(0, 2 * np.pi, 200),   fig=None, with_arrows=False):
    """
    Visualize displacement perturbation on specific surface
    :param fig: specify target figure :param rsurf: normalized surface radius for plotting, default is plasma boundary """
    from matplotlib import pyplot

    if fig is None:
        fig = pyplot.figure()
    if 'Xr' not in self['XPLASMA']:
        self.get_Xcyl()

        # Find wall surface index
    IIsurf = np.argmin(np.abs(self[self.sim]['s'].values - rsurf))

    Rw = self[self.sim]['R'].isel(s=IIsurf)
    Zw = self[self.sim]['Z'].isel(s=IIsurf)

    try:
        #        phi = np.linspace(0, 2 * np.pi, 200)
        Xwr = self['XPLASMA']['Xr'].isel(sp=IIsurf) * np.exp(1j * phi)
        Xwz = self['XPLASMA']['Xz'].isel(sp=IIsurf) * np.exp(1j * phi)
    except IndexError:
        print('ERROR: Selected index is out of XPLASMA calculation domain')
        return
    B0 = self[self.sim]['B0EXP'].values
    R0 = self[self.sim]['R0EXP'].values
    h = R0 / self[self.sim]['Ns1'].values
    Xwr_r = np.real(Xwr)
    Xwz_r = np.real(Xwz)
    Xt = np.sqrt(Xwr_r**2 + Xwz_r**2)
    Xt[np.where(Xt == 0.0)] = 1.0
    R1 = Rw + h * Xwr_r / Xt
    Z1 = Zw + h * Xwz_r / Xt
    R2 = np.vstack((Rw, R1))
    Z2 = np.vstack((Zw, Z1))
    pyplot.plot((Rw * R0),
                (Zw * R0), 'c-')
    pyplot.plot((R2 * R0),
                (Z2 * R0), 'b-')
    pyplot.plot(R[rho_values <= rsurf][::5].T,
                Z[rho_values <= rsurf][::5].T, color="orange")
    if with_arrows:
        plt.quiver(Rw * R0, Zw * R0,
                   (R1 - Rw) * R0, (Z1 - Zw) * R0,
                   angles='xy', scale_units='xy', scale=1, color='b')
    pyplot.plot(R1 * R0, Z1*R0, color="red")
    pyplot.axis('equal')
    pyplot.xlabel('R [m]')
    pyplot.ylabel('Z [m]')

    return R1, Z1


def reconstruct(data, type: str):
    theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    # a = (data[0] / 2) + np.zeros(len(data))
    a = np.zeros(len(data))
    if type == "cos":
        for m in range(len(data)):
            a += data[m] * np.cos(m * theta)

    if type == "sin":
        for m in range(len(data)):
            a += data[m] * np.sin(m * theta)

    return a


phi_angle = np.array(np.real(np.exp(-1j * phi))) * R0
R_array = np.array(R_total[-1, :])

R_2d = np.array([[R_array], [phi_angle]]).reshape(2, 200)

Z_array = np.array(Z_total[-1, :])
Z_2d = np.array([[Z_array], [phi_angle]]).reshape(2, 200)
