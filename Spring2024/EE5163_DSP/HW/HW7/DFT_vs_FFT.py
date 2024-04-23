import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm


# Efficient DFT implementation
def efficient_DFT(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    
    # Use only half of the frequencies due to symmetry
    for n in tqdm(range(N), total=N):
        X[:N//2] += x[n] * np.exp(-2j * np.pi * np.arange(N//2) * n / N)
    
    # Use the conjugate symmetry for the second half of the frequencies
    X[N//2:] = np.conj(X[:N//2][::-1])
    
    return X


# FFT implementation using numpy
def FFT(x):
    return np.fft.fft(x)

# Values of N
n_values = [10, 16, 20]

# Lists to store times
dft_times = []
fft_times = []

for n in n_values:
    # Generate a random signal
    x = np.random.random(2**n)

    # Compute DFT and measure time
    start_time = time.time()    
    X_dft = efficient_DFT(x)
    end_time = time.time()
    dft_time = end_time - start_time
    dft_times.append(dft_time)

    # Compute FFT and measure time
    start_time = time.time()
    X_fft = FFT(x)
    end_time = time.time()
    fft_time = end_time - start_time
    fft_times.append(fft_time)

    print(f"For N = 2**{n}, DFT took {dft_time} seconds and FFT took {fft_time} seconds.")

# Plot the times
X_ranges = range(1, len(n_values)+1)
X_ticks = ["2^{}".format(str(n)) for n in n_values]
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(X_ranges, dft_times, 'o-', label='DFT')
ax.plot(X_ranges, fft_times, 'o-', label='FFT')
ax.set_yscale('log')
ax.set_xticks(X_ranges)
ax.set_xticklabels(X_ticks)
ax.set_xlabel('N')
ax.set_ylabel('Time (seconds)')
ax.legend()
ax.grid()
plt.show()
fig.savefig('./HW7/plot.png')
fig.savefig('./HW7/plot.pdf')
