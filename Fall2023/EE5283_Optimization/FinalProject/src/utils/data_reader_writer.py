import pandas as pd
import numpy as np


def load_data_from_file(data_path: str):
    data = pd.read_csv(data_path, header=None)
    return data.to_numpy()



def load_weight_matrix(data_path: str):
    data = pd.read_csv(data_path, header=None)
    print(data)
    return data.to_numpy()

def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()