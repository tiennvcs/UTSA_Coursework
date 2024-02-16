import numpy as np
from matplotlib import pyplot as plt

plt.style.use('seaborn-poster')

# Sampling rate
sr = 1000

# Sampling interval
ts = 1.0/sr
n = np.arange(0, 1, ts)

freq = 400
x = np.e**(1j*2*np.pi*freq*n)

plt.figure(figsize=(12, 9))
plt.plot(n, x, 'r', label=r'$\sin{2 \pi 100 n}$')
plt.xlabel("Time (n)")
plt.ylabel("Amplitude")
plt.show()
plt.savefig("./HW5/signal_c.png")

def DFT(x: np.ndarray):
    """
        Function to calculate the discrete Fourier Transform of a 1D real-valued signal x
    """
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j*np.pi*k*n/N)
    X = np.dot(e, x)
    return X


X = DFT(x)
# Calculate the frequencyt
N = len(X)
n = np.arange(N)
T = N/sr
freq = n/T

plt.figure(figsize=(12, 9))
plt.stem(freq, abs(X), 'b', markerfmt=" ", basefmt="-b")
plt.xlabel("Freq (Hz)")
plt.ylabel("DFT Amplitude |X(freq)|")
plt.show()
plt.savefig("./HW5/dft_c.png")