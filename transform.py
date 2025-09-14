import numpy as np
import matplotlib.pyplot as plt
from load_data import R_2d, R_array, R_total, Z_2d, Z_array, Z_total, reconstruct
from scipy.fft import fft2, fft


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


R_fft = fft(R_2d) / len(R_2d.T)
Z_fft = fft(Z_2d) / len(Z_2d.T)

R_cos, R_sin = fft_to_cos_sin(R_fft[0])
Z_cos, Z_sin = fft_to_cos_sin(Z_fft[0])

theta_R_cos, theta_R_sin = fft_to_cos_sin(R_fft[1])
theta_Z_cos, theta_Z_sin = fft_to_cos_sin(Z_fft[1])


R_rec = reconstruct(R_cos, "cos") + reconstruct(R_sin, "sin")
Z_rec = reconstruct(Z_cos, "cos") + reconstruct(Z_sin, "sin")

print(theta_R_cos[0:3], theta_R_sin[0:3])
exit()
plt.plot(R_rec, Z_rec)
plt.axis("equal")
plt.show()
