import numpy as np
import fractions


A = np.array([
    [1, -1/3],
    [-1/3, 1]
])

eig_vals, eig_vecs = np.linalg.eigh(A)
for eig_val, eig_vec in zip(eig_vals, eig_vecs):
    print("eigenvalue: {} correponsding eigenvector: {}".format(eig_val, eig_vec))