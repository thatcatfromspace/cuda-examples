from numba import cuda
import numpy as np

@cuda.jit
def matrix_add(A, B, C):
    row, col = cuda.grid(2)

    if row < C.shape[0] and col < C.shape[1]:
        C[row][col] = A[row][col] + B[row][col]

A = np.array([[1,2],
              [3,4]], dtype=np.float32)

B = np.array([[5,6],
              [7,8]], dtype=np.float32)

C = np.zeros((2,2), dtype=np.float32)

threads_per_block = (2,2)
blocks_per_grid = (1,1)

matrix_add[blocks_per_grid, threads_per_block](A, B, C)

print(C)