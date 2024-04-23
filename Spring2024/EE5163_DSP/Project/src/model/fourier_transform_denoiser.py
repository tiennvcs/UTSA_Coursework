from copy import deepcopy
from . import Model
import numpy as np


class FourierTransformDenoiser(Model):
    def __init__(self, threshold_ratio: float):
        self.name = "Fourier Transform based denoiser"
        self._threshold_ratio = threshold_ratio

    def _fft(self, x_t: np.ndarray, sampling_rate):
        X_f = np.fft.fft(x_t)
        N = len(X_f)
        n = np.arange(N)
        T = N/sampling_rate
        freq = n/T
        n_oneside = N//2
        # get the one side frequency
        f_oneside = freq[:n_oneside]
        # normalize the amplitude
        X_oneside = X_f[:n_oneside]
        return X_oneside, f_oneside, X_f, freq
    
    def _inv_fft(self, X_f: np.ndarray):
        return np.fft.ifft(X_f)

    def _cutdown_lowfrequencies(self, X_f: np.ndarray):
        removal_indices = np.abs(X_f) <= np.max(X_f)*self._threshold_ratio
        removed_frequency_values = np.copy(X_f[np.where(removal_indices), ])
        clean_Xf = deepcopy(X_f)
        clean_Xf[removal_indices] = 0
        return clean_Xf, removed_frequency_values, np.where(removal_indices)

    def run(self, input_signal: np.ndarray, sampling_rate: int):
        Xf_oneside, freq_oneside, X_f, freq = self._fft(input_signal, sampling_rate)
        clean_Xf, removed_frequencies, removal_indices = self._cutdown_lowfrequencies(X_f)
        clean_Xf_oneside, _, _ = self._cutdown_lowfrequencies(Xf_oneside)
        clean_xt = self._inv_fft(clean_Xf)
        return {
            'Xf_oneside': Xf_oneside,
            'freq_oneside': freq_oneside,
            'X_f': X_f,
            'freq': freq,
            'clean_xt': clean_xt.real,
            'clean_Xf': clean_Xf,
            'clean_Xf_oneside': clean_Xf_oneside,
            'removed_frequencies': removed_frequencies,
            'removal_indices': removal_indices
        }