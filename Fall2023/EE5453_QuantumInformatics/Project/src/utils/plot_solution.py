import numpy as np
from matplotlib import pyplot as plt
from typing import List


def plot_regressor(X: np.ndarray, y: np.ndarray, w_list: List[np.ndarray], output_file: str):
    fig, ax = plt.subplots(figsize=(9, 6))
    # Plot dataset from (X,y)
    ax.scatter(X[1:, :], y, label="Given samples", color='red')
    color_map = {
        0: 'blue',
        1: 'green'
    }
    name_map = {
        0: 'Classical LR',
        1: 'Quantum LR'
    }
    style_map = {
        0: 'bo-',
        1: 'gx-'
    }
    # Plot the regression lines
    for i, w in  enumerate(w_list):
        y_pred = np.array([w.dot(x) for x in X.T])
        ax.plot(X[1:, :].T, y_pred, style_map[i], label=name_map[i], )

    ax.legend(loc='best')

    plt.show()

    fig.savefig(output_file)


