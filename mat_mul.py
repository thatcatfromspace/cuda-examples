from numba import cuda
import numpy as np

@cuda.jit
def matmul(A, B, C):
    row, col = cuda.grid(2)

    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0
        for k in range(A.shape[1]):
            tmp += A[row][k] * B[k][col]

        C[row][col] = tmp


A = np.array([[1,2,3],
              [4,5,6]], dtype=np.float32)

B = np.array([[7,8,9,10],
              [11,12,13,14],
              [15,16,17,18]], dtype=np.float32)

C = np.zeros((2,4), dtype=np.float32)

threads_per_block = (16,16)

blocks_x = (C.shape[1] + 15) // 16
blocks_y = (C.shape[0] + 15) // 16

blocks_per_grid = (blocks_x, blocks_y)

matmul[blocks_per_grid, threads_per_block](A, B, C)

print(C)