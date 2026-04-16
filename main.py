from numba import cuda
import numpy as np

@cuda.jit
def square_kernel(a):
    idx = cuda.grid(1)
    if idx < len(a):
        a[idx] *= a[idx]

arr = np.array([1,2,3,4], dtype=np.float32)

threads_per_block = 32
size = arr.size
blocks_per_grid = (size + threads_per_block - 1) // threads_per_block

square_kernel[blocks_per_grid, threads_per_block](arr)

print(arr)