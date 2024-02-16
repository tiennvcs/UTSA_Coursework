import numpy as np


def build_distance_matrix(data) -> np.ndarray:
    n = len(data)
    D = np.zeros((n,n))
    for i in range(0, n):
        for j in range(0, len(data)):
            D[i,j] = np.linalg.norm(data[i] - data[j])
    return D