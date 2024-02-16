from qiskit import QuantumCircuit
import numpy as np


class QFT:
    """
        Quantum Fourier Transform
        Builds the QFT circuit iteratively
    """
    def __init__(self, signal_length=16, basis_to_transform='',
                 validate_inverse_fourier=False,
                 qubits=None):
        self.signal_length = signal_length
        self.basis_to_transform = basis_to_transform

        if qubits is None:
            self.num_qubits = int(np.log2(signal_length))
            self.qubits = [QuantumCircuit(i) for i in range(self.num_qubits)]

        else:
            self.qubits = qubits
            self.num_qubits = len(self.qubits)
        
        self.qubit_index = 0
        self.input_circuit = QuantumCircuit()

        self.validate_inverse_fourier = validate_inverse_fourier
        self.circuit = QuantumCircuit()
        self.inv_circuit = QuantumCircuit()

        for k, q_s in enumerate(self.basis_to_transform):
            if int(q_s) == 1:
                # Change the qubit state from |0> to |1>
                self.input_circuit.x(self.qubits[k])

        def qft_circuit_iter(self):

            if self._qubit_index > 0