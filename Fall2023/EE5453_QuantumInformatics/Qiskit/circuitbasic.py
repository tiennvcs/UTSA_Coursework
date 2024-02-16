import numpy as np
from qiskit import QuantumCircuit
from matplotlib import pyplot as plt


# Create a Quantum Circuit acting on a quantum register of three qubits
circ = QuantumCircuit(3)

# ADd a H gate on qubit 0, putting this qubit in superposition
circ.h(0)
# Add a CX (CNOT) gate on control qubit 0 and target qubt 1, putting the qubits in a Bell state
circ.cx(0, 1)
# Add a CX (CNOT) gate on control qubit 0 and target qubit 2, putting
# the qubits in a GHZ state
circ.cx(0, 2)

circ.draw('mpl', filename='./image/circuit.png')
plt.show()
