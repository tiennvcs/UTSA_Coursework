from . import Dataset
import numpy as np
from typing import List


class PolynomialSignal(Dataset):
    def __init__(self, degree: int, coefficents: np.ndarray, mean: float, var: float):
        self._degree = degree
        self._coefficents = coefficents
        self._mean = mean
        self._var = var

    def _true_f(self, t: np.ndarray):
        true_xt = np.zeros(t.shape[-1])
        degree_lst = range(0, self._degree+1, 1)
        for d, coeff in zip(degree_lst, self._coefficents):
            true_xt += coeff*(t**d)
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
    
 