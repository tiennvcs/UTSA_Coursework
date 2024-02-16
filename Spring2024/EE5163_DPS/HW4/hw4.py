import numpy as np
import matplotlib.pyplot as plt


fs = 500
step_n = 0.1

n_values = np.arange(0, 10, step_n)

x1 = np.cos(2*np.pi*100*n_values)

x2 = np.cos(2*np.pi*250*n_values)

x3 = np.cos(2*np.pi*500*n_values)

x4 = np.cos(2*np.pi*750*n_values)


fig, ax = plt.subplots()
ax.plot(n_values, x1, label=r'$f_0 = 100Hz$')
ax.plot(n_values, x2, label=r'$f_0 = 250Hz$')
ax.plot(n_values, x3, label=r'$f_0 = 500Hz$')
ax.plot(n_values, x4, label=r'$f_0 = 750Hz$')
ax.set_xlabel(r"$n$")
ax.set_ylabel(r"$\cos{2\pi f_0 n}$")
ax.legend()
plt.show()
fig.savefig('./HW4/hw4.pdf')
fig.savefig('./HW4/hw4.png')
