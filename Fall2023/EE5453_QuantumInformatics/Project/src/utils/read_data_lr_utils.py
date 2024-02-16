import os
import pandas as pd
import numpy as np


def read_data_lr_from_path(data_path: str):
    if os.path.exists(data_path):
        data = pd.read_csv(data_path).to_numpy()
        X = data[:, :-1]
        y = data[:, -1]
        
        ones = np.array([1]*X.shape[0]).reshape(-1, 1)
        extend_X = np.append(ones, X, 1).T
        return extend_X, y
    else:
        print("Error, invalid data path !")
        exit(0)


if __name__ == '__main__':
    data_path = "/home/tiennv/Github/EE5453QuantumInformatics/Project/data/data1/data.csv"
    X, y = read_data_lr_from_path(data_path=data_path)
    print(X.shape)
    print(y.shape)
    print(X)
    print(y)