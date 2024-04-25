import os, time
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap, EfficientSU2
from qiskit_algorithms.optimizers import COBYLA, ADAM
from qiskit_algorithms.utils import algorithm_globals

from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC

algorithm_globals.random_seed = 2505

DATA_DIR =  "./data"
KEY2DATADIR = {
    'MNIST-2': "./MNIST-2",
    "Fashion-MNIST-2": "./Fashion-MNIST-2",
    "Fashion-MNIST-3": "./Fashion-MNIST-3",
    "Syn-Dataset-4": "./Syn-Dataset-4",
    "Syn-Dataset-16": "./Syn-Dataset-16"
}
KEY2FULLDIR = {
    'MNIST-2': os.path.join(DATA_DIR, KEY2DATADIR['MNIST-2']),
    'Fashion-MNIST-2': os.path.join(DATA_DIR, KEY2DATADIR['Fashion-MNIST-2']),
    'Fashion-MNIST-3': os.path.join(DATA_DIR, KEY2DATADIR['Fashion-MNIST-3']),
    'Syn-Dataset-4': os.path.join(DATA_DIR, KEY2DATADIR['Syn-Dataset-4']),
    'Syn-Dataset-16': os.path.join(DATA_DIR, KEY2DATADIR['Syn-Dataset-16'])
}

EPOCHS = 30

def load_dataset(data_dir: str):
    x_file = os.path.join(data_dir, './x.npy')
    y_file = os.path.join(data_dir, './y.npy')
    x_train_file = os.path.join(data_dir, './x_train.npy')
    y_train_file = os.path.join(data_dir, './y_train.npy')
    x_test_file = os.path.join(data_dir, './x_test.npy')
    y_test_file = os.path.join(data_dir, './y_test.npy')

    x = np.load(x_file)
    y = np.load(y_file)
    x_train = np.load(x_train_file)
    y_train = np.load(y_train_file)
    x_test = np.load(x_test_file)
    y_test = np.load(y_test_file)

    return x_train, x_test, y_train, y_test, x, y


def callback_graph(weights, obj_func_eval):
    # clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    # plt.title("Objective function value against iteration")
    # plt.xlabel("Iteration")
    # plt.ylabel("Objective function value")
    # plt.plot(range(len(objective_func_vals)), objective_func_vals)
    # plt.show()
    print("Iter {} - objective function value: {}".format(len(objective_func_vals), obj_func_eval))


def create_simpleVQC(num_features: int, max_iteration: int):
    feature_map = ZZFeatureMap(num_features)
    ansatz = EfficientSU2(num_qubits=num_features, reps=1)
    optimizer = COBYLA(maxiter=max_iteration)
    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        loss="cross_entropy",
        optimizer=optimizer,
        callback=callback_graph
    )
    return vqc


def create_complexVQC(num_features: int, max_iteration: int):
    feature_map = ZZFeatureMap(num_features)
    ansatz = EfficientSU2(num_qubits=num_features, reps=3)
    optimizer = COBYLA(maxiter=max_iteration)
    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        loss="cross_entropy",
        optimizer=optimizer,
        callback=callback_graph
    )
    return vqc


DATA_NAME = "Syn-Dataset-16"
print(os.path.exists(KEY2FULLDIR[DATA_NAME]))
DATA_DIR = KEY2FULLDIR[DATA_NAME]


x_train, x_test, y_train, y_test, x, y = load_dataset(DATA_DIR)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

dim = x_train.shape[-1]

simpleVQC = create_simpleVQC(num_features=dim, max_iteration=EPOCHS)

objective_func_vals = []
s_time = time.time()
simpleVQC.fit(x_train, y_train)
synthetic16_qnn_train_time = time.time() - s_time

synthetic16_qnn_train_score = simpleVQC.score(x_train, y_train)
print("QNN's training score: {}".format(synthetic16_qnn_train_score))

synthetic16_qnn_test_score = simpleVQC.score(x_test, y_test)
print("QNN's testing score: {}".format(synthetic16_qnn_test_score))

print("QNN's training time: {} (seconds)".format(synthetic16_qnn_train_time))