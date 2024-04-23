import numpy as np


# Pre-defined transformation
Bt = np.array([
    [1, 0, -1],
    [0, 1, 1],
    [0, -1, 1],
])

G = np.array([
    [1, 0],
    [0.5, 0.5],
    [0.5, -0.5]
])

At = np.array([
    [1, 1, 1],
    [0, 1, -1],
])

# Define the input feature map and kernel
d = np.array([
    [2, 3, 5],
    [1, 4, 0],
    [8, 7, 6]
])

g = np.array([
    [1, 0],
    [4, 2]
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
print("Normal convolution output: ", convolution_output)

# The output of Winograd convolution
## Transform the input feature map into Winograd domain
d_w = Bt.dot(d).dot(Bt.T)
print("The transformation of input feature map in Winograd domain: \n{}".format(d_w))
## Transform the kernel into Winograd domain
g_w = G.dot(g).dot(G.T)
print("The transformation of kernel in Winograd domain: \n{}".format(g_w))
## Compute the element-wise multiplication between d_w and g_w
o_w = np.multiply(d_w, g_w)
print("The element-wise product: \n{}".format(o_w))
## Compute the inverse transformation, which maps o_w back to input domain
o = At.dot(o_w).dot(At.T)
print("The output of Winograd convolution: \n{}".format(o))
