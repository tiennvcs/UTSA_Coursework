import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap, EfficientSU2
from qiskit_algorithms.optimizers import GradientDescent, COBYLA
from qiskit_algorithms.utils import algorithm_globals

from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor, VQR
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.circuit.library import QNNCircuit
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_provider import IBMProvider
import os, time

algorithm_globals.random_seed = 2505


DATA_DIR =  "/home/tiennv/Github/UTSA_Coursework/Spring2024/EE5423_HWML/Project/data"
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

MAX_ITERATION = 100
REPS = 1
# IBM_BACKEND_NAME = "ibmq_qasm_simulator"
# IBM_INSTANCE_NAME = "ibm-q-asu/main/utsa-panagiotis-"
# IBM_TOKEN = "aac67f019f7b273de049667f903d44384ff6f874041cf8830d05f8d6aa883861d45a4fa93afdf3bc5243a4919c6d58d0d339f67dd8a014666e74d644a340f4a7"
# IBMProvider.save_account(token=IBM_TOKEN, overwrite=True)
# IBM_PROVIDER = IBMProvider(instance=IBM_INSTANCE_NAME)
# IBM_BACKEND = IBM_PROVIDER.get_backend(IBM_BACKEND_NAME)

from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService()
print(service.backends())
backend = service.backend("ibmq_qasm_simulator")
print(backend)


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
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()

def create_vqc_model(num_features: int, max_iteration: int, reps: int, backend):
    feature_map = ZZFeatureMap(num_features)
    # ansatz = RealAmplitudes(num_features, reps=reps)
    ansatz = EfficientSU2(num_qubits=num_features, reps=reps)
    optimizer = COBYLA(maxiter=max_iteration)
    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        loss="cross_entropy",
        optimizer=optimizer,
        callback=callback_graph,
        quantum_instance=backend
    )
    return vqc


print(os.path.exists(KEY2FULLDIR['Syn-Dataset-4']))

DATA_NAME = "Syn-Dataset-4"
DATA_DIR = KEY2FULLDIR[DATA_NAME]

x_train, x_test, y_train, y_test, x, y = load_dataset(DATA_DIR)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

dim = x_train.shape[-1]

vqc_model = create_vqc_model(num_features=dim, max_iteration=MAX_ITERATION, reps=REPS, backend=backend)

objective_func_vals = []
s_time = time.time()
vqc_model.fit(x_train, y_train)
synthetic4_qnn_train_time = time.time() - s_time