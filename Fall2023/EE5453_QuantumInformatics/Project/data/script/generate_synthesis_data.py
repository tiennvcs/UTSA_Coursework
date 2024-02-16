import json
import os
import numpy as np
from matplotlib import pyplot as plt
import inspect


def f(x: np.ndarray):
    return 2*x + 3


def generate_dataset(N: int, D, f, mean: float, var: float):
    """
        Generate a synthesis dataset for regression task
        - Parameters:
            + N: number of samples would be generated.
            + D: the dimension of data sample.
            + mean: the expectation value of noise variable
            + var: variance of noise variable
        - Output: 
            + data: DxN numpy array object, each column is a data sample
    """
    data = []
    X = np.random.randint(low=0, high=20, size=N+1)
    for x in X:
        noised_f = f(x) + np.random.normal(mean, np.sqrt(var))
        new_sample = [x, noised_f]
        data.append(new_sample)
    return np.array(data)


def plot(data, output_dir):
    X = data[:, 0]
    y = data[:, -1]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(X, y, color='red')
    
    ax.set_title(r"Synthesis dataset by function $y(x) = 2x+3 +\epsilon_{\mathbb{N}(0, 4)}$", fontsize=13)
    plt.show()
    fig.savefig(os.path.join(output_dir, 'data_visualization.pdf'))


def save_data(data: np.ndarray, output_dir: str):
    data_output_file = os.path.join(output_dir, 'data.csv')
    np.savetxt(data_output_file, data, delimiter=",")


def save_configuration(config: dict, output_dir: str):
    saving_config_running_file = os.path.join(output_dir, 'running_config.json')
    with open(saving_config_running_file, 'w') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    N = 20
    D = 1
    mean = 0
    var = 4
    output_dir = '/home/tiennv/Github/EE5453QuantumInformatics/Project/data/data1'
    os.makedirs(output_dir, exist_ok=True)

    np.random.seed(2505)

    data = generate_dataset(N, D, f, mean, var)
    config = {
        'true_function': inspect.getsource(f),
        'N': N,
        'D': D,
        'mean': mean,
        'var': var,
    }    

    plot(data, output_dir)

    save_data(data, output_dir)

    save_configuration(config, output_dir)
