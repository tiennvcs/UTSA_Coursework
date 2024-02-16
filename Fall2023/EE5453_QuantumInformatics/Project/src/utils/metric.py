import numpy as np


def mean_square_error(X: np.ndarray, y: np.ndarray, w: np.ndarray):
    y_pred = np.array([w.dot(x) for x in X.T])
    N = X.T.shape[0]
    mse = 1/N * np.sqrt(np.sum((y_pred-y)**2))
    return mse