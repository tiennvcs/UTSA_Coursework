import cvxpy as cp
import numpy as np


class CVXSolver:
    def __init__(self):
        self.cp = cp
        self._constraints = []

    def _prepare_data(self, distance_matrix: np.ndarray, m: int):
        self._m = m
        self._D = distance_matrix
        self._n = self._D.shape[0]
        self._X = cp.Variable(self._D.shape, boolean=True)
        self._u = cp.Variable(self._n, integer=True)
        self._ones = np.ones((self._n,1))

    def _apply_constraints(self):
        self._constraints += [self._X[0,:] @ self._ones == self._m]
        self._constraints += [self._X[:,0] @ self._ones == self._m]
        self._constraints += [self._X[1:,:] @ self._ones == 1]
        self._constraints += [self._X[:,1:].T @ self._ones == 1]
        self._constraints += [cp.diag(self._X) == 0]
        self._constraints += [self._u[1:] >= 2]
        self._constraints += [self._u[1:] <= self._n]
        self._constraints += [self._u[0] == 1]

        for i in range(1, self._n):
            for j in range(1, self._n):
                if i != j:
                    self._constraints += [ self._u[i] - self._u[j] + 1  <= (self._n - 1) * (1 - self._X[i, j]) ]
    
    def _objective_construct(self):
        self._objective = cp.Minimize(cp.sum(cp.multiply(self._D, self._X)))

    def problem_establish(self):
        self._prob = cp.Problem(self._objective, self._constraints)
        
    def solve(self, distance_matrix: np.ndarray, m: int):
        self._prepare_data(distance_matrix, m)

        self._objective_construct()

        self._apply_constraints()

        self.problem_establish()

        self._prob.solve(verbose=True)
        
    def gather_path(self):
        # Transforming the solution to paths
        self._X_sol = np.argwhere(self._X.value==1)
        self._ruta = {}
        for i in range(0, self._m):
            self._ruta['Salesman_' + str(i+1)] = [1]
            j = i
            a = 10e10
            while a != 0:
                a = self._X_sol[j,1]
                self._ruta['Salesman_' + str(i+1)].append(a+1)
                j = np.where(self._X_sol[:,0] == a)
                # print(j)
                # if j[0].shape[0] == 0:
                #     continue
                j = j[0][0]
                a = j     
        # Showing the paths
        for i in self._ruta.keys():
            print('\tThe path of ' + i + ' is: {}'.format(' => '.join(map(str, self._ruta[i]))))

    def get_data(self):
        return {
            "X": self._X.value.tolist(),
            "u": self._u.value.tolist(),
            "objective": self._objective.value.tolist(),
            "D": self._D.tolist(),
        }
    
    def get_path(self):
        return self._ruta