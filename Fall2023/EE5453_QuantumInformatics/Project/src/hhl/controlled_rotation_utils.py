from qiskit import QuantumCircuit, QuantumRegister
import numpy as np


def controlled_rotation(circuit: QuantumCircuit, 
                        clock_register: QuantumRegister, 
                        ancilla_register: QuantumRegister):
    circuit.cry(np.pi, clock_register[0], ancilla_register)
    circuit.cry(np.pi/3, clock_register[1], ancilla_register)