from . import Dataset
import numpy as np
from typing import List


class TrigonometricSignal(Dataset):
    """
        Generate the 
    """
    def __init__(self, amplitudes: np.ndarray, frequencies: np.ndarray, function_names: List[str], mean: float, var: float):
        self._amplitudes = amplitudes
        self._frequencies = frequencies
        self._function_names = function_names
        self._mean = mean
        self._var = var

    def _true_f(self, t: np.ndarray):
        true_xt = np.zeros(t.shape[-1])
        for amp, freq, f_name in zip(self._amplitudes, self._frequencies, self._function_names):
            if f_name == 'sin':
                true_xt += amp*np.sin(2*np.pi*freq*t)
            elif f_name == 'cos':
                true_xt += amp*np.cos(2*np.pi*freq*t)
        return true_xt

    def generate(self, sampling_rate: int):
        ts = 1.0/sampling_rate
        t = np.arange(0, 1, ts)
        true_xt = self._true_f(t)
        x_t = true_xt + np.random.normal(self._mean, np.sqrt(self._var), true_xt.shape[-1])
        return {
            't': t,
            'true_xt': true_xt,
            'x_t': x_t
        }
    
 