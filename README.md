# Matrix Multiplication Test

This is a simple script to test matrix multiplication performance on CPU and GPU. The script generates two random matrices of size N x N, and computes their product using the dot product function for CPU and a CUDA kernel for GPU.

## Requirements

Python 3.x
NumPy
PyCUDA (optional, only needed for GPU test)

## Usage

'python matrix_multiplication.py --mode=<cpu/gpu> --size=<matrix_size>'
The --mode parameter is used to specify the mode of the test (CPU or GPU). The --size parameter is used to specify the size of the matrices (N x N).

If the --mode parameter is not specified, the script will prompt the user to choose between CPU and GPU.

If the --size parameter is not specified, the script will use a default value of 2048.

## Example usage

'python matrix_multiplication.py --mode=cpu --size=2048'
This will run the matrix multiplication test on CPU, using matrices of size 2048 x 2048.

'python matrix_multiplication.py --mode=gpu --size=4096'
This will run the matrix multiplication test on GPU, using matrices of size 4096 x 4096 (assuming PyCUDA is installed).

## Output
The script will output the time taken to compute the product of the two matrices in seconds.

## License
This project is licensed under the terms of the MIT license.
