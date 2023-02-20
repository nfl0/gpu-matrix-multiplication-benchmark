# Matrix Multiplication

This repository contains a Python script for performing matrix multiplication, using either the CPU or the GPU.

## Usage

The script accepts the following command-line arguments:

- `--mode`: Choose the mode of computation. Valid options are `cpu` and `gpu`. Default is `gpu`.
- `--size`: Choose the size of the matrices. Default is `2048`.

For example, to run the script on the CPU with a matrix size of 4096, use the following command:

python matrix_multiplication.py --mode=cpu --size=4096


## Requirements

To run the script on the GPU, you will need to have PyCUDA installed. You can install PyCUDA using pip:

pip install pycuda


## License

This repository is licensed under the MIT license.
