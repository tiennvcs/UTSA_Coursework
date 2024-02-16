import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit import Aer


def normalize_unit(v: np.ndarray) -> np.ndarray:
    if v.T.dot(v) == 1:
        return v
    else:
        return v/np.sqrt(v.T.dot(v))
    

def reconstruct_original_vector(normalized_v: np.ndarray, original_length: float) -> np.ndarray:
    return normalized_v*original_length


def state_intialization(circuit: QuantumCircuit, 
                        input_register: QuantumRegister, 
                        initial_state: np.ndarray):
    # circuit.initialize(initial_state, input_register)
    circuit.x(input_register)


def simulator_intialization():
    try:
        simulator = Aer.get_backend('statevector_simulator')
        # simulator.set_options(device='GPU')
    except ValueError as e:
        print(e)
    return simulator


if __name__ == '__main__':
    v = np.array([1, 1]).T
    l = np.sqrt(v.T.dot(v))
    res = normalize_unit(v)
    constructed_v = reconstruct_original_vector(normalized_v=res, original_length=l)
    print("Before normalization: ", v)
    print("After normalization: ", res)
    print("Reconstruct vector: ", constructed_v)