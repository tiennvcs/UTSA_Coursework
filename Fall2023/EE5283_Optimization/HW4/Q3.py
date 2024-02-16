import matplotlib.pyplot as plt
import numpy as np

delta = 0.01
x1 = np.arange(-3.0, 3.0, delta)
x2 = np.arange(-3.0, 3.0, delta)
X, Y = np.meshgrid(x2, x1)
f0 = -3 * X + 0 * Y

# x1 <= 2 - x2**2
z = 2 - x2 ** 2

fig, ax = plt.subplots(figsize=(16, 9))
CS = ax.contour(Y, X, f0, 20, colors='k')  # Negative contours default to dashed.
ax.plot(x2, z, label=r"$x_1 = 2 - x_2^2$")
ax.scatter(0, 2, color='blue')
ax.clabel(CS, fontsize=9, inline=True)
ax.set_xlabel(r"$x_2$")
ax.set_ylabel(r"$x_1$")
ax.set_xticks(np.arange(-3.0, 3.0, 0.2))
ax.set_yticks(np.arange(-3.0, 3.0, 0.2))
ax.legend(loc="best")
plt.xlim(-3, 3)
plt.ylim(-3, 3)

plt.show()
fig.savefig("./HW4/Q3.pdf")
