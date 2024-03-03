import numpy as np


# Pre-defined transformation
Bt = np.array([
    [1, 0, -1, 0],
    [0, 1, 1, 0],
    [0, -1, 1, 0],
    [0, -1, 0, 1]
])

G = np.array([
    [1, 0],
    [0.5, 0.5],
    [0.5, -0.5],
    [0, 1]
])

At = np.array([
    [1, 1, 1, 0],
    [0, 1, -1, 0],
    [0, 1, 1, 1]
])

# Define the input feature map and kernel
d = np.array([
    [1, 2, 3, 0],
    [4, 5, 6, 0],
    [7, 8, 9, 0],
    [0, 0, 0, 0]
])

g = np.array([
    [-1, 2],
    [3, 0]
])

# The output of normal convolution 
convolution_output = np.zeros(shape=(d.shape[0]-g.shape[0]+1, d.shape[1]-g.shape[1]+1))
row_idx = 0
while row_idx <= d.shape[0] - g.shape[0]:   
    col_idx = 0
    while col_idx <= d.shape[1] - g.shape[1]:
        input_tile = d[row_idx: row_idx+g.shape[0], col_idx: col_idx+g.shape[1]]
        convolution_output[row_idx, col_idx] = np.sum(np.multiply(input_tile, g))
        col_idx += 1
    row_idx += 1
print(convolution_output)

# The output of Winograd convolution
## Transform the input feature map into Winograd domain
d_w = Bt.dot(d).dot(Bt.T)
## Transform the kernel into Winograd domain
g_w = G.dot(g).dot(G.T)
## Compute the element-wise multiplication between d_w and g_w
o_w = np.multiply(d_w, g_w)
## Compute the inverse transformation, which maps o_w back to input domain
o = At.dot(o_w).dot(At.T)
print(o)