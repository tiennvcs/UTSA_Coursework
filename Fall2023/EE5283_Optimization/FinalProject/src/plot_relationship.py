import os
import json
from matplotlib import pyplot as plt


def read_data(data_dir):
    m_list = []
    objective_list = []

    for data_file in os.listdir(data_dir):
        full_data_file = os.path.join(data_dir, data_file)
        with open(full_data_file, 'r') as f:
            data = json.load(f)
            m_list.append(data['m'])
            objective_list.append(data['objective'])
    m_list, objective_list = zip(*sorted(zip(m_list, objective_list)))

    return m_list, objective_list


def plot(X, y):
    plt.style.use('ggplot')

    fig, ax = plt.subplots(figsize=(6, 4))
    

    ax.plot(X, y, "ro-")

    # ax.set_title("The optimal objective corresponding different salesmen")
    ax.set_ylabel(r"$f(X)$")
    ax.set_xlabel(r"# salesmen")
    ax.legend(loc="best")
    fig.savefig("./output/objective_vs_salesmen.pdf")


if __name__ == '__main__':
    data_dir = "/home/tiennv/Github/EE5283EngineeringOptimization/FinalProject/output/numerical_example"

    X, y = read_data(data_dir=data_dir)

    plot(X, y)
