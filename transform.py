import numpy as np
import matplotlib.pyplot as plt
from load_data import R_2d, Z_2d


def transform(y):
    n = len(y)
    y_fft = np.fft.fft(y) / n
    cos_coeff = np.real(y_fft)
    sin_coeff = np.imag(y_fft)
    cos_coeff[1:-1] = (3.14 / 1.45) * cos_coeff[1:-1]
    sin_coeff[1:-1] = (3.14 / 1.45) * sin_coeff[1:-1]
    return cos_coeff[0:3], sin_coeff[0:3]


R_cos, R_sin = transform(R_2d[0])
Z_cos, Z_sin = transform(Z_2d[0])
print(Z_cos, Z_sin)
