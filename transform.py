import numpy as np
import matplotlib.pyplot as plt
from load_data import R_2d, Z_2d
from scipy.fft import fft2


def transform(y):
    n = len(y[0])
    y_fft = fft2(y) / n
    cos_coeff = np.real(y_fft[0])
    sin_coeff = -np.imag(y_fft[0])
    cos_coeff[1:-1] = (3.14 / 1.4666) * cos_coeff[1:-1]
    sin_coeff[1:-1] = (3.14 / 1.4666) * sin_coeff[1:-1]
    theta_cos = np.real(y_fft[1])
    theta_sin = -np.imag(y_fft[1])
    return cos_coeff[0:3], sin_coeff[0:3], theta_cos[0:3], theta_sin[0:3]


R_cos, R_sin, theta_cos, theta_sin = transform(R_2d)
Z_cos, Z_sin, _, _ = transform(Z_2d)

print(R_cos, R_sin)
