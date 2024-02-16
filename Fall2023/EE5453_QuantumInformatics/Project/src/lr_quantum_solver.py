import os
import numpy as np
from utils.read_data_lr_utils import read_data_lr_from_path
from utils.plot_solution import plot_regressor
from hhl_qiskit import HHLSolver
import time


class QuantumSolver:
    def __init__(self):
        self._solver = HHLSolver(epsilon=1e-6, backend=None)

    def solve(self, X: np.ndarray, y: np.ndarray):
        """
            Solve the linear regression problem with X = [x1, ..., xN]
            , and y = [y1, ..., yN].T
            
        """
        # Compute the product XX^T
        A = X.dot(X.T)
        # Compute the product Xy
        b = X.dot(y)
        # Construct circuit
        s_time = time.time()
        print(A)
        print(b)
        input()
        w = self._solver.construct_circuit(A=A, b=b)
        run_time = time.time() - s_time
        return w, run_time
    

if __name__ == '__main__':
    data_path = "./data/data1/data.csv"
    X, y = read_data_lr_from_path(data_path=data_path)
    q_solver = QuantumSolver()
    w, run_time = q_solver.solve(X=X, y=y)
    print("Found regressor: y = {}x + {}".format(w[1], w[0])) 
    w = w*1e4/2
    print("Rescale regessor: y = {}x + {}".format(w[1], w[0]))
    output_file = os.path.join("./output", "quantum_solution.pdf")
    plot_regressor(X, y, [w], output_file)