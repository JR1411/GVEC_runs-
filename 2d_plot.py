from test import ev, ev_axis
import numpy as np
import matplotlib.pyplot as plt

R = ev.X1[:, :, 0]
Z = ev.X2[:, :, 0]

rho_vis = R*0 + ev.rho
vis_lvl = np.linspace(0, 1, 11)

ev_2 = ev.sel(zeta=np.pi / 2, method="nearest")
ev_3 = ev.sel(zeta=np.pi, method="nearest")


fig, ax = plt.subplots()

ax.contour(R, Z, rho_vis, vis_lvl, colors="black")
ax.contour(ev_2.X1, ev_2.X2, rho_vis, vis_lvl, colors="red", alpha=0.5)
ax.contour(ev_3.X1, ev_3.X2, rho_vis, vis_lvl, colors="green", alpha=0.5)
ax.axis("equal")
ax.set(xlabel="R", ylabel="Z",
       title=r"black : $\zeta$ = 0 , red : $\zeta$ = $\pi / 2 $ , green : $\zeta$ = $\pi  $ ")
plt.savefig("flux_surfaces.jpeg")
