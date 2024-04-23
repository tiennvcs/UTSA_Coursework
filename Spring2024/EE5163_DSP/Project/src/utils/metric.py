import numpy as np


def mean_square_error(true_f: np.ndarray, denoising_f: np.ndarray):
    return 1/true_f.shape[-1]*np.sum((true_f-denoising_f)**2)

# def signal2noise(signal: np.ndarray):
#     signal_power = np.mean(signal)**2
#     noise_power = np.var(signal)
#     return signal_power/noise_power

def signal2noise(frequency_signal: np.ndarray):
    signal_power = np.mean(frequency_signal)**2
    noise_power = np.var(frequency_signal)
    return float(10*np.log10(signal_power/noise_power))

# def signal2noise(denoising_signal, removed_frequencies):
#     signal_power = np.mean(denoising_signal)**2
# 
#     noise_power=  np.var(removed_frequencies)
