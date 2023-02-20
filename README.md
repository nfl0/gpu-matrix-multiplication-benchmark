# GPU Matrix Multiplication Benchmark

This repository contains code for benchmarking matrix multiplication on a GPU using the CUDA programming model. The benchmark is implemented in Python and uses the PyCUDA library to interface with the GPU.

## Usage

To run the benchmark, simply execute the `matrix_multiplication_benchmark.py` script:

python matrix_multiplication_benchmark.py

The script generates two random matrices on the GPU, copies them to the device, performs matrix multiplication using a CUDA kernel, and then copies the result back to the host. The script prints the time taken to perform the operation.

By default, the benchmark uses a matrix size of 2048. This can be changed by modifying the `MATRIX_SIZE` constant in the script.

## Requirements

The benchmark requires the following dependencies:

- Python 3.x
- NumPy
- PyCUDA

## Contributing

If you find a bug or would like to suggest an improvement, please open an issue or submit a pull request.

## License

This code is released under the MIT license. See the LICENSE file for more details.
