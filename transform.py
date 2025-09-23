import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2
from load_data import R_array, R_total, Z_array, Z_total, reconstruct
import tomli_w


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
phi = np.linspace(0, 2*np.pi, 20)
for m in range(m_max):
    for n in range(n_max):
        Rc, Rs = fourier_coefs_half(R_array, theta, phi, m, n)
        Zc, Zs = fourier_coefs_half(Z_array, theta, phi, m, n)
        R_cos_mn[m, n] = Rc
        R_sin_mn[m, n] = Rs
        Z_cos_mn[m, n] = Zc
        Z_sin_mn[m, n] = Zs


def array_to_lines(arr, n):
    """
    Convert array into TOML lines with '(m,n) = value,' format,
    keeping *all* coefficients (including zeros).
    """
    lines = []
    for m, val in enumerate(arr):
        lines.append(f"({m},{n}) : {float(val)},")
    return lines


# Build coefficient lines
R_cos_lines = array_to_lines(
    R_cos_mn.T[0, :], 0) + array_to_lines(R_cos_mn.T[1, :].T, 1)
Z_cos_lines = array_to_lines(
    Z_cos_mn.T[0, :].T, 0) + array_to_lines(Z_cos_mn.T[1, :].T, 1)
R_sin_lines = array_to_lines(
    R_sin_mn.T[0, :].T, 0) + array_to_lines(R_sin_mn.T[1, :].T, 1)
Z_sin_lines = array_to_lines(
    Z_sin_mn.T[0, :].T, 0) + array_to_lines(Z_sin_mn.T[1, :].T, 1)

# Write to geometry.toml manually
with open("geometry.toml", "w") as f:
    f.write("[X1_b_cos]\n")
    f.write("\n".join(R_cos_lines) + "\n\n")
    f.write("[X1_b_sin]\n")
    f.write("\n".join(R_sin_lines) + "\n\n")
    f.write("[X2_b_cos]\n")
    f.write("\n".join(Z_cos_lines) + "\n\n")
    f.write("[X2_b_sin]\n")
    f.write("\n".join(Z_sin_lines) + "\n")

exit()
R_rec = reconstruct(R_cos_mn, "cos") + \
    reconstruct(R_sin_mn, "sin")
Z_rec = reconstruct(Z_cos_mn, "cos") + \
    reconstruct(Z_sin_mn, "sin")


fig, ax = plt.subplots()

ax.plot(R_rec, Z_rec, "r--")
ax.axis("equal")
plt.show()
