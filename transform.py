import numpy as np
import matplotlib.pyplot as plt
from load_data import R_2d, Z_2d
from scipy.fft import fft2


def fft_to_cos_sin(R_fft):
    """ Zerlegt komplexe FFT-Koeffizienten in reelle Cos-/Sin-Koeffizienten """
    N = len(R_fft)
    a = np.zeros(N)
    b = np.zeros(N)

    # DC-Anteil
    a[0] = R_fft[0].real

    # für 1 <= m < N/2: Koeffizientenpaare
    for m in range(1, N//2):
        a[m] = 2 * R_fft[m].real
        b[m] = -2 * R_fft[m].imag   # Vorzeichen wegen exp(i*mθ)

    # Nyquist (falls gerade N)
    if N % 2 == 0:
        a[N//2] = R_fft[N//2].real

    return a, b
