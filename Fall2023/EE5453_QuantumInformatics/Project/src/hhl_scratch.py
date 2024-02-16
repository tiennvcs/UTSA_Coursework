import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from hhl.qpe_utils import quantum_phase_estimation, inverse_quantum_phase_estimation
from qiskit import Aer
from qiskit.visualization import plot_distribution
from hhl.utils import normalize_unit, state_intialization, simulator_intialization
from hhl.controlled_rotation_utils import controlled_rotation
from utils.ibm_connect import load_ibm_config, establish_connect, load_running_config


def circuit_initialization():
    ancilla_register = QuantumRegister(size=1, name='anc')
    clock_register = QuantumRegister(size=2, name='clock')
    input_register = QuantumRegister(size=1, name='input')
    measurement = ClassicalRegister(2, name='output')
    circuit = QuantumCircuit(ancilla_register, clock_register, input_register, measurement)
    circuit.barrier()
    return circuit, ancilla_register, clock_register, input_register, measurement


def hhl_circuit(circuit: QuantumCircuit,
                ancilla_register: QuantumRegister, 
                clock_register: QuantumRegister, 
                input_register: QuantumRegister, 
                measurement: ClassicalRegister, b: np.ndarray):
    # Initialize state
    state_intialization(circuit=circuit, input_register=input_register, initial_state=b)
    circuit.barrier()

    # Perform phase estimation
    quantum_phase_estimation(circuit=circuit, clock=clock_register, input=input_register)
    circuit.barrier()

    # Perform controlled rotation
    controlled_rotation(circuit=circuit, 
                        clock_register=clock_register, 
                        ancilla_register=ancilla_register)
    circuit.barrier()

    # Measure the ancilla qubit
    circuit.measure(ancilla_register, measurement[0])
    circuit.barrier()

    # Perform inverse quantum fourier transform
    inverse_quantum_phase_estimation(
        circuit=circuit, 
        clock_register=clock_register, 
        input_register=input_register)

def run(b, config, visualize=True):
    # Initialize the circuit
    circuit, ancilla_register, clock_register, input_register, measurement = circuit_initialization()

    # Build up components in HHL algorithm
    hhl_circuit(circuit, ancilla_register, clock_register, input_register, measurement, b=b)
    circuit.barrier()

    # Measure the input register to get the output
    circuit.measure(input_register, measurement[1])

    if visualize:
        print(circuit)
        circuit.draw("mpl", fold=18, style="iqx", filename="./output/quantum_cuircuit.pdf")

    print("2. Load IBM configuration from {} and establising the connection ...".format(config['ibm_config_path']))    
    ibm_config = load_ibm_config(config_path=config['ibm_config_path'])
    provider = establish_connect(channel=ibm_config['channel'], 
                                token=ibm_config['token'], instance=ibm_config['instance'])
    # print("\t\t-> {}".format(provider.backends()))
    backend = provider.get_backend(config['ibm_backend'])
    # Run simulation to get the expection of output
    # simulator = simulator_intialization()
    # result = execute(circuit, simulator, shots=shots).result()
    job = backend.run(circuit.decompose(), shot=config['shot'])
    counts = job.result().get_counts(circuit)
    plot_distribution(counts, title='Counts', filename='./output/scratch_count.pdf')
    # final_state_vector = job.result().get_statevector(circuit)
    # print(final_state_vector)


if __name__ == '__main__':
    visualize = True
    b = np.array([0, 1])
    nomalized_b = normalize_unit(b)
    config_path = "./src/config.json"
    print("1. Loading the configuration from {} ...".format(config_path))
    config = load_running_config(config_path=config_path)
    print("\t\t-> {}".format(config))
    run(visualize=True, b=nomalized_b, config=config) 