import numpy as np


A = np.array([
    [1, -1/3],
    [-1/3, 1]
])

b = np.array([
    [0, 1]
]).T

# Find eigenvalues, eigenvectors of A
eig_vals, eig_vecs = np.linalg.eigh(A)

# Find the linear combination of b in eigenvectors of A.
inversion_A = np.zeros(shape=2)
for eig_val, eig_vec in zip(eig_vals, eig_vecs):
    sub_matrix = (1/eig_val)*(np.outer(eig_vec, eig_vec.T))
    inversion_A = inversion_A + sub_matrix
x = inversion_A.dot(b)

print("Found solution by inversion method: {}".format(x.tolist()))
print("The ratio between |x1|^2 and |x2|^2 is {}".format((x[0].T.dot(x[0])/(x[1].T.dot(x[1])))))