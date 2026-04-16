from numba import cuda
import numpy as np

@cuda.jit
def transpose(A, B):
    row, col = cuda.grid(2)

    if row < B.shape[0] and col < B.shape[1]:
        B[row][col] = A[col][row]


A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]], dtype=np.float32)

B = np.zeros((3, 3), dtype=np.float32)

threads_per_block = (16, 16)

blocks_x = (B.shape[0] + 15) // 16
blocks_y = (B.shape[1] + 15) // 16 

blocks_per_grid = (blocks_x, blocks_y)

transpose[blocks_per_grid, threads_per_block](A, B)

print(B)