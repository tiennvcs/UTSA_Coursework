from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
import numpy as np
from hhl.qft_utils import inverse_quantum_fourier_transform, quantum_fourier_transform


def quantum_phase_estimation(circuit: QuantumCircuit, clock: QuantumRegister, input: QuantumRegister):
    # Put the clock register into bell states, or apply H on them
    circuit.h(clock)
    circuit.barrier()

    # Perform controlled rotation part
    ##  e^{i*A*t}
    circuit.cu(np.pi/2, -np.pi/2, np.pi/2, 3*np.pi/4, control_qubit=clock[0], target_qubit=input, label='U')
    # circuit.cu(np.pi, -np.pi/2, np.pi/2, 0, control_qubit=clock[0], target_qubit=input, label='U')
    ##  e^{i*A*t*2}
    circuit.cu(np.pi, np.pi, 0, 0, control_qubit=clock[1], target_qubit=input, label='U2')
    # circuit.cu(0, 0, 0, np.pi, control_qubit=clock[1], target_qubit=input, label='U2')
    circuit.barrier()

    # Perform inverse quantum fourier tranform
    inverse_quantum_fourier_transform(circuit=circuit, clock=clock, n=2)


def inverse_quantum_phase_estimation(
        circuit: QuantumCircuit, 
        clock_register: QuantumRegister, 
        input_register: QuantumRegister):
    # Perform quantum fourier transform on register holding eigenvalues
    quantum_fourier_transform(circuit=circuit, clock=clock_register, n=2)

    circuit.barrier()

    # Perform controlled rotation part
    ## Perform e^{i*A*t*2}
    circuit.cu(np.pi, np.pi, 0, 0, clock_register[1], input_register, label='U2')
    # circuit.cu(0, 0, 0, np.pi, clock_register[1], input_register, label='U2')
    ## Perform e^{i*A*t}
    circuit.cu(np.pi/2, np.pi/2, -np.pi/2, -3*np.pi/4, clock_register[0], input_register, label='U')
    # circuit.cu(np.pi, np.pi/2, -np.pi/2, 0, clock_register[0], input_register, label='U')
    circuit.barrier()

    # Again, apply H on clock register
    circuit.h(clock_register)


def create_demo_quantum_circuit(clock_qubit_num, input_qubit_num):
    clock_register = QuantumRegister(size=clock_qubit_num, name='clock')
    input_register = QuantumRegister(size=input_qubit_num, name='input')
    measurement = ClassicalRegister(size=3, name='output')
    circuit = QuantumCircuit(clock_register, input_register, measurement)
    circuit.barrier()
    return circuit, clock_register, input_register, measurement

    
if __name__ == '__main__':
    A = np.array([
        [1, -1/3],
        [-1/3, 1]
    ])
    num1 = 2
    num2 = 1

    # Initialize circuit from configuration
    circuit, clock_register, input_register, measurement = create_demo_quantum_circuit(clock_qubit_num=num1, input_qubit_num=num2)
    print("Before performing quantum phase estimation: ", circuit)

    quantum_phase_estimation(circuit=circuit, clock=clock_register, input=input_register)
    print("After performing quantum phase estimation: ", circuit)