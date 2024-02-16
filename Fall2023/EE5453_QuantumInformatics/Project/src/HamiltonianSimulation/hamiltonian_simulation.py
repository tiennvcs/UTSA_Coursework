from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
import numpy as np


def create_demo_quantum_circuit(clock_qubit_num, input_qubit_num):
    clock_register = QuantumRegister(size=clock_qubit_num, name='clock')
    input_register = QuantumRegister(size=input_qubit_num, name='input')
    measurement = ClassicalRegister(size=3, name='output')
    circuit = QuantumCircuit(clock_register, input_register, measurement)
    circuit.barrier()
    return circuit, clock_register, input_register, measurement


class HamiltonianSimulation:
    """
        This class simulates the Hamiltonian evolution for a Single qubit. 
        For a Hamiltonian given by H, the Unitary operator simulated for time t is given by
            e^{-iHt}. An eigenvalue of lambda for the Hamiltonian H corresponds to 
            the eigenvalue of e^{-i*lambda*t}.
    """
    def __init__(self, H, t, exponent=1.0):
        self._H = H
        self._t = t
        self._eigen_vals, self._eigen_vecs = np.linalg.eigh(self._H)
        self._eigen_components = []
        for lam, v in zip(self._eigen_vals, self._eigen_vecs):
            theta = -lam*t/np.pi
            proj = np.outer(v, np.conj(v))
            self._eigen_components.append((theta, proj))

    def simulate(self, circuit: QuantumCircuit, clock_register: QuantumRegister):
        circuit.unitary(self._H, clock_register)

    
if __name__ == '__main__':
    A = np.array([
        [1, -1/3],
        [-1/3, 1]
    ])

    circuit, clock_register, input_register, measurement = create_demo_quantum_circuit(
        clock_qubit_num=2, input_qubit_num=1)

    simulator = HamiltonianSimulation(H=A, t=1, exponent=1.0)

    simulator.simulate(circuit=circuit, clock_register=clock_register)

    print(circuit)
