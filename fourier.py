import numpy as np
import matplotlib.pyplot as plt
from load_data import R_total, Z_total, reconstruct
from scipy.fft import fft


def fourier_coefs(f, x, m):
    num_x = len(x)
    dx = np.diff(x)
    expfac = np.exp(-1.0j * np.outer(m, x[:-1]))
    fm = expfac.dot(f * dx) / (2 * np.pi)
    fm[1:] = fm[1:]  # Half-sided Fourier series
    sin_m = -np.imag(fm)
    cos_m = np.real(fm)
    cos_m[0] = cos_m[0]
    sin_m[0] = sin_m[0]
    return cos_m, sin_m


phi = np.linspace(0, 2*np.pi, 201)
m = np.arange(200)
R_cos, R_sin = fourier_coefs(R_total[-1, :], phi, m)
Z_cos, Z_sin = fourier_coefs(Z_total[-1, :], phi, m)


R_rec = reconstruct(R_cos, "cos") + reconstruct(R_sin, "sin")
Z_rec = reconstruct(Z_cos, "cos") + reconstruct(Z_sin, "sin")


diff = R_rec - R_total[-1, :].values

print(diff)

exit()

fig, ax = plt.subplots()

ax.plot(R_rec, Z_rec, "r--")
ax.plot(R_total[-1, :].T, Z_total[-1, :].T)
ax.axis("equal")
plt.show()

exit()
arrays = [R_cos, R_sin, Z_cos, Z_sin]
names = ["X1_b_cos", "X1_b_sin", "X2_b_cos", "X2_b_sin"]


is_cos = [True, False, True, False]


def write_toml_cos_sin(arrays, names, is_cos, filename="modes2.toml", skip_zeros=False):
    with open(filename, "w") as f:
        for name, arr, cos_flag in zip(names, arrays, is_cos):
            f.write(f"[{name}]\n")
            for m, coeff in enumerate(arr):
                if skip_zeros and coeff == 0.0:
                    continue
                key = f"({m},  0)" if cos_flag else f"({m} , 0 )"
                f.write(f'"{key}" = {coeff}\n')
            f.write("\n")  # Leerzeile nach jeder Tabelle


write_toml_cos_sin(arrays, names, is_cos)
