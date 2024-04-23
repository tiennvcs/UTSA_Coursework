from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_machine_learning.datasets import ad_hoc_data
from qiskit.circuit.library import TwoLocal
from qiskit_machine_learning.utils import split_dataset_to_data_and_labels, map_label_to_class_name

# Use IBM's qasm_simulator
quantum_instance = QuantumInstance(Aer.get_backend('ibmq_qasm_simulator'), shots=1024)

# Load ad hoc dataset
train_data, test_data, _ = ad_hoc_data(training_size=20, test_size=5, n=2, gap=0.3, plot_data=False)
train_data = split_dataset_to_data_and_labels(train_data)
test_data = split_dataset_to_data_and_labels(test_data)

# Construct feature map, ansatz, and optimizer
feature_map = TwoLocal(2, ['ry', 'rz'], 'cz', reps=3, entanglement='linear')
ansatz = TwoLocal(2, ['ry', 'rz'], 'cz', reps=3, entanglement='linear')
optimizer = COBYLA()

# Initialize the VQC
vqc = VQC(optimizer, feature_map, ansatz, quantum_instance=quantum_instance)

# Train the VQC
vqc.fit(train_data[0], train_data[1])

# Test the VQC
score = vqc.score(test_data[0], test_data[1])

print(f'Classification score: {score}')
