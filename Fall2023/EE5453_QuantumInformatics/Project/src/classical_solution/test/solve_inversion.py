import numpy as np


A = np.array([
    [1, -1/3],
    [-1/3, 1]
])

b = np.array([
    [0, 1]
]).T

# Solve Ax = b by inversion method
inverse_A = np.linalg.inv(A)
x = inverse_A.dot(b)
print("Found solution by inversion method: {}".format(x.tolist()))
print("The ratio between |x1|^2 and |x2|^2 is {}".format((x[0].T.dot(x[0])/(x[1].T.dot(x[1])))))