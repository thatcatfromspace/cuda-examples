from numba import cuda
import numpy as np

@cuda.jit
def partial_sum(arr, partial_sums):
    idx = cuda.grid(1)
    tid = cuda.threadIdx.x

    shared = cuda.shared.array(256, dtype=np.int32)

    if idx < arr.size:
        shared[tid] = arr[idx]
    else:
        shared[tid] = 0

    cuda.syncthreads()

    stride = 1
    while stride < cuda.blockDim.x:
        if tid % (2 * stride) == 0:
            shared[tid] += shared[tid + stride]
        stride *= 2
        cuda.syncthreads()

    if tid == 0:
        partial_sums[cuda.blockIdx.x] = shared[0]

A = np.array([1,2,3,4,5,6,7,8], dtype=np.int32)

threads_per_block = 4
blocks_per_grid = 2

partial_sums = np.zeros(blocks_per_grid, dtype=np.int32)

partial_sum[blocks_per_grid, threads_per_block](A, partial_sums)

print(partial_sums)
print("Final sum:", np.sum(partial_sums))