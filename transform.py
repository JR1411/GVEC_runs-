import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2
from load_data import R_array, R_total, Z_array, Z_total, reconstruct


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


def fft2_theta_phi(f):

    Ntheta, Nphi = f.shape

    # standard FFT in both directions
    F = np.fft.fft2(f) / (Ntheta * Nphi)

    # index arrays
    m = np.fft.fftfreq(Ntheta, 1/Ntheta).astype(int)  # theta index
    n = np.fft.fftfreq(Nphi,   1/Nphi).astype(int)    # phi index

    # flip the phi axis (because basis is -n*phi instead of +n*phi)
    F = np.roll(F[:, ::-1], 1, axis=1)
    n = -n

    # cosine and sine parts
    cos_modes = np.real(F)
    sin_modes = -np.imag(F)

    return m, n, cos_modes, sin_modes


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
n_max = 1
R_cos_mn = np.zeros((m_max, n_max))
R_sin_mn = np.zeros((m_max, n_max))
Z_cos_mn = np.zeros((m_max, n_max))
Z_sin_mn = np.zeros((m_max, n_max))
theta = np.linspace(0, 2*np.pi, 200)
phi = np.linspace(0, 2*np.pi, 20)
for m in range(m_max):
    for n in range(n_max):
        Rc, Rs = fourier_coefs_half(R_array, theta, phi, m, n)
        Zc, Zs = fourier_coefs_half(Z_array, theta, phi, m, n)
        R_cos_mn[m, n] = Rc
        R_sin_mn[m, n] = Rs
        Z_cos_mn[m, n] = Zc
        Z_sin_mn[m, n] = Zs

R_rec = reconstruct(R_cos_mn, "cos") + \
    reconstruct(R_sin_mn, "sin")
Z_rec = reconstruct(Z_cos_mn, "cos") + \
    reconstruct(Z_sin_mn, "sin")


fig, ax = plt.subplots()

ax.plot(R_rec, Z_rec, "r--")
ax.axis("equal")
plt.show()
