import numpy as np
from linear_solvers import HHL
from qiskit.quantum_info import Statevector
from qiskit import Aer
from utils.ibm_connect import load_ibm_config, establish_connect, load_running_config


class HHLSolver:
    def __init__(self, epsilon=1e-3, backend=None) -> None:
        if backend:
            self._backend = backend
        else:
            self._backend = Aer.get_backend('aer_simulator')
        self._solver = HHL(epsilon=epsilon, quantum_instance=self._backend)
    
    def _get_solution_vector(self, solution):
        """Extracts and normalizes simulated state vector
        from LinearSolverResult."""
        output_vector = Statevector(solution.state).data
        index = int(Statevector(solution.state).dim)//2
        # print(index)
        # print("Value at {} is {}".format(index, output_vector[index: index+2]))
        solution_vector = output_vector[index: index+2].real
        # if len(output_vector)>= 64:
        #     return solution_vector
        norm = solution.euclidean_norm
        return norm * solution_vector / np.linalg.norm(solution_vector)

    def construct_circuit(self, A: np.ndarray, b: np.ndarray):
        sol = self._solver.solve(A, b)
        print(sol.state.decompose())
        x = self._get_solution_vector(solution=sol)
        return x
    
    def get_solver(self):
        return self._solver
    

def run(config_path: str, A, b):
    print("1. Loading the configuration from {} ...".format(config_path))
    config = load_running_config(config_path=config_path)
    print("\t\t-> {}".format(config))

    if config['ibm_backend']:
        print("2. Load IBM configuration from {} and establising the connection ...".format(config['ibm_config_path']))    
        ibm_config = load_ibm_config(config_path=config['ibm_config_path'])

        provider = establish_connect(channel=ibm_config['channel'], 
                                    token=ibm_config['token'], instance=ibm_config['instance'])
        print("\t\t-> {}".format(provider.backends()))
        backend = provider.get_backend(config['ibm_backend'])
    else:
        print("2. Using Aer simulator ...")
        backend = None
    print("3. Initilize the HHL Solver ...")
    hhl_solver = HHLSolver(epsilon=1e-4, backend=backend)
    print("\t\t-> {}".format(hhl_solver.get_solver()))

    print("4. Build HHL circuit to solve the specific problem ...")
    x = hhl_solver.construct_circuit(A, b)
    print("\t\t-> x = {}".format(x)) 


if __name__ == '__main__':

    A = np.array([
        [1, -1/3], 
        [-1/3, 1]
    ])
    b = np.array([0, 1])

    config_path = "./src/config.json"
    run(config_path, A, b)
    

    