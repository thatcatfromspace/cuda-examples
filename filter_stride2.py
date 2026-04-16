from numba import cuda
import numpy as np

@cuda.jit
def blur_stride2(img, output):
    row, col = cuda.grid(2)

    stride = 2
    kernel_size = 3

    # Only valid output threads should compute
    if row < output.shape[0] and col < output.shape[1]:

        # Map output pixel -> input starting point
        start_row = row * stride
        start_col = col * stride

        total = 0

        # Apply 3x3 blur kernel
        for i in range(kernel_size):
            for j in range(kernel_size):
                total += int(img[start_row + i][start_col + j])

        output[row][col] = total // 9

        cuda.atomic.add()


img = np.array([
    [10,20,30,40,50,60],
    [15,25,35,45,55,65],
    [20,30,40,50,60,70],
    [25,35,45,55,65,75],
    [30,40,50,60,70,80],
    [35,45,55,65,75,85]
], dtype=np.uint8)

kernel_size = 3
stride = 2

# Output size formula:
# floor((input - kernel)/stride) + 1
out_rows = ((img.shape[0] - kernel_size) // stride) + 1
out_cols = ((img.shape[1] - kernel_size) // stride) + 1

output = np.zeros((out_rows, out_cols), dtype=np.uint8)

# CUDA launch config
threads_per_block = (16,16)

blocks_x = (out_cols + 15) // 16
blocks_y = (out_rows + 15) // 16

blocks_per_grid = (blocks_x, blocks_y)

# Launch kernel
blur_stride2[blocks_per_grid, threads_per_block](img, output)

# Print results
print("Input Image:")
print(img)

print("\nOutput Image (Stride 2 Blur):")
print(output)