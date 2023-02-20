import time
import numpy as np
import pycuda.autoinit
import pycuda.driver as drv

# Choose a matrix size that suits your GPU
MATRIX_SIZE = 2048

# Create two random matrices on the GPU
a_gpu = drv.mem_alloc(MATRIX_SIZE * MATRIX_SIZE * np.dtype(np.float32).itemsize)
b_gpu = drv.mem_alloc(MATRIX_SIZE * MATRIX_SIZE * np.dtype(np.float32).itemsize)

# Generate random values for the matrices
a = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
b = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)

# Copy the matrices to the GPU
drv.memcpy_htod(a_gpu, a)
drv.memcpy_htod(b_gpu, b)

# Create an empty matrix to store the result on the GPU
c_gpu = drv.mem_alloc(MATRIX_SIZE * MATRIX_SIZE * np.dtype(np.float32).itemsize)

# Define the CUDA kernel for matrix multiplication
kernel_code = """
__global__ void matrix_multiply(float *a, float *b, float *c, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        float sum = 0.0f;

        for (int i = 0; i < size; i++) {
            sum += a[row * size + i] * b[i * size + col];
        }

        c[row * size + col] = sum;
    }
}
"""

# Compile the kernel code and create a CUDA function
mod = drv.module_from_source(kernel_code)
matrix_multiply = mod.get_function("matrix_multiply")

# Define the block and grid sizes for the kernel
block_size = (16, 16, 1)
grid_size = (int(np.ceil(MATRIX_SIZE / block_size[0])),
             int(np.ceil(MATRIX_SIZE / block_size[1])),
             1)

# Time the matrix multiplication operation on the GPU
start_time = time.time()

matrix_multiply(a_gpu, b_gpu, c_gpu, np.int32(MATRIX_SIZE),
                block=block_size, grid=grid_size)

end_time = time.time()

# Copy the result matrix back to the CPU
c = np.empty((MATRIX_SIZE, MATRIX_SIZE), dtype=np.float32)
drv.memcpy_dtoh(c, c_gpu)

# Print the time taken to perform the matrix multiplication on the GPU
print(f"Time taken: {end_time - start_time:.6f} seconds")
