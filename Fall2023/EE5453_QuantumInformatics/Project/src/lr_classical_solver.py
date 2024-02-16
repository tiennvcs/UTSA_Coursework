import os
import numpy as np
from utils.read_data_lr_utils import read_data_lr_from_path
from utils.plot_solution import plot_regressor
import time


class ClassicalSolver:
    def __init__(self):
        pass

    def solve(self, X: np.ndarray, y: np.ndarray):
        """
            Solve the linear regression problem with X = [x1, ..., xN]
            , and y = [y1, ..., yN].T
            
        """
        # Compute the product XX^T
        A = X.dot(X.T)
        # Compute the product Xy
        b = X.dot(y)
        # Compute the inverse of A 
        inv_A = np.linalg.inv(A)
        # Compute the solution w
        s_time = time.time()
        w = inv_A.dot(b)
        run_time = time.time() - s_time
        return w, run_time
    

if __name__ == '__main__':
    data_path = "./data/data1/data.csv"
    X, y = read_data_lr_from_path(data_path=data_path)
    c_solver = ClassicalSolver()
    w, run_time = c_solver.solve(X=X, y=y) 
    
    output_file = os.path.join("./output", "classical_solution.pdf")
    plot_regressor(X, y, [w], output_file)