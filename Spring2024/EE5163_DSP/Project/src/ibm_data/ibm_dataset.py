import numpy as np
import csv


class IBMDataset:
    """
        Generate the 
    """
    def __init__(self, data_file: str):
        self._data_file = data_file
        self._data = []

        self._load_data()

    def _load_data(self):
        with open(self._data_file, 'r') as textfile:
            for row in reversed(list(csv.reader(textfile))):
                self._data.append(row)

    def generate(self, num_samples: int=2000):
        t = list(range(0, len(self._data), int(len(self._data)/num_samples)))[: num_samples]
        x_t = []
        for i in t:
            x_t.append(float(self._data[i][1]))
        return {
            't': np.arange(0, len(t), 1),
            'x_t': np.array(x_t)
        }


if __name__ == '__main__':
    data_path = "/home/tiennv/Github/UTSA_Coursework/Spring2024/EE5163_DPS/Project/data/ibm_stock_2022_Jan2March.csv"
    dataset = IBMDataset(data_file=data_path).read_data_from_file(num_samples=1000)
    print(dataset)
 