import setuptools as distutils
from omfit_classes.omfit_mars import OMFITmars
import numpy as np
import matplotlib.pyplot as plt

rundir = "/Users/josefreiterer/Desktop/Bachelor-Thesis/coding/workshop-material/KINETIC_KNTV=20_newOMEGAW_nuDNESTROVSKIJ 2/"
data = OMFITmars(rundir)
data.load()

data.get_RZ()

R0 = np.array([data["sim0"]["R0EXP"].values])

R = data["sim0"]["R"].values
Z = data["sim0"]["Z"].values

R = R * R0
Z = Z*R0

rho_values = data["sim0"]["s"].values

data.get_Xcyl()
data.get_UnitVec()


def plot_Xwarp_1(self, fig=None, rsurf=1.0):
    """
    Visualize displacement perturbation on specific surface
    :param fig: specify target figure
    :param rsurf: normalized surface radius for plotting, default is plasma boundary
        """
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
        Xwr = self['XPLASMA']['Xr'].isel(sp=IIsurf)
        Xwz = self['XPLASMA']['Xz'].isel(sp=IIsurf)
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
    pyplot.plot(Rw * R0, Zw * R0, 'c-')
    pyplot.plot(R2 * R0, Z2 * R0, 'b-')
    pyplot.plot(R[rho_values <= 1.0].T, Z[rho_values <= 1.0].T)
    pyplot.axis('equal')
    pyplot.xlabel('R [m]')
    pyplot.ylabel('Z [m]')
    pyplot.xlim([7, 8])
    pyplot.ylim([-4, -6])
    return R1, Z1


plot_Xwarp_1(data)
plt.savefig("pert3.png")
plt.show()
