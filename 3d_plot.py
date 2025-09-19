from test import ev, ev_axis
import numpy as np
import matplotlib.pyplot as plt


ax = plt.axes(projection="3d")

x_ax, y_ax, z_ax = np.asarray(ev_axis.pos)

ax.plot3D(x_ax, y_ax, z_ax, color="green")

x, y, z = np.asarray(ev.pos)

ax.plot_surface(x[-1, :, :], y[-1, :, :], z[-1, :, :], alpha=0.5)
ax.view_init(elev=90, azim=-90)
ax.axis("equal")
ax.set(xlabel="x", ylabel="y")
plt.plot(0, 0, marker="o")
plt.show()
