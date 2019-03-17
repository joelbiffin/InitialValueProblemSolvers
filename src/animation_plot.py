import math
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

body_1_motion = lambda t: np.array([
    10*math.cos(t),
    9*math.sin(t),
    10
])

t = np.linspace(0, 10, 100)
fig = plt.figure()
ax = p3.Axes3D(fig)

ax.set_xlim(-12, 12)
ax.set_ylim(-12, 12)
ax.set_zlim(-12, 12)


body_1 = np.array([body_1_motion(i) for i in t])
data = ax.scatter(body_1[:, 0], body_1[:, 1], body_1[:, 2], marker='o', animated=False)

plt.show()
