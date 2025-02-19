import numpy as np
import matplotlib.pyplot as plt

THETA = np.radians(30) # simple setting that states at what angle the camera and projector image planes lie.
CAMERA_POS = (10, -20)
A = 1/20
x_c, y_c = CAMERA_POS[0], CAMERA_POS[1]
m = np.tan(THETA)
x_i = (x_c + m * y_c) / (1 + m**2)

n = np.arange(0, 400)
x = x_c - y_c * (x_c - n * A * x_i) / (y_c - m * n * A * x_i)

# DEFINING CONSTANTS FOR PROJECTION
PIX_NUM_PROJ = 20
x_proj = np.linspace(0, x[-1], PIX_NUM_PROJ)
step = x_proj[1]

x_pix_vals = x % step # how far into each pixel each x is.


# VISUAL ADJUSTMENTS
plt.rcParams['font.size'] = 16
plt.rcParams['font.weight'] = "bold"
plt.rcParams['font.family'] = "Arial"

# plt.plot(n, x)
# plt.xlabel("nth pixel ray")
# plt.ylabel("x coordinate of intersection with projector plane")

fig, ax = plt.subplots(constrained_layout = True)
ax.hist(x_pix_vals/step, 20)
ax.set_xlabel("Distance along pixel, d / L_p", fontfamily="Arial", fontweight="bold", fontsize=18)
ax.set_ylabel("Frequency", fontfamily="Arial", fontweight="bold", fontsize=18)
ax.set_xlim(0,1)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.xaxis.set_tick_params(width=2)
ax.yaxis.set_tick_params(width=2)
# plt.scatter(n, x_pix_vals/step)
plt.show()