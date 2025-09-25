from test import ev
import matplotlib.pyplot as plt
from load_data import R_array, Z_array
import numpy as np


R = ev.X1[-1, :, 0]
Z = ev.X2[-1, :, 0]

fig, ax = plt.subplots()

ax.plot(R_array,  Z_array, "r--")
ax.plot(R, Z)
ax.axis("equal")
ax.set(xlabel="R [m]", ylabel="Z [m]")
plt.show()
