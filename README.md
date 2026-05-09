# CUDA ML Framework

This project is a from-scratch machine learning framework built in C++ with a focus on GPU-accelerated performance using CUDA.

Rather than relying on existing ML libraries, the goal is to understand the full pipeline by implementing the core components manually—from tensor storage and matrix multiplication to forward passes, backpropagation, gradient descent, and training a simple neural network classifier.

The machine learning model itself is intentionally simple. The real focus of this project is exploring how GPU parallelization can accelerate matrix-heavy workloads such as neural networks, and understanding the systems-level concepts behind that performance.

## Project Goals

This project is primarily a learning opportunity to deepen my understanding of:

- CUDA and GPU programming
- C++ memory management and low-level performance considerations
- Parallelizing matrix operations for machine learning workloads
- Benchmarking CPU vs GPU runtime performance
- How neural networks work internally (forward pass, backpropagation, gradient descent, optimization)

## Current Features

- Custom Tensor class built from scratch
- Matrix multiplication, addition, transpose, ReLU, and softmax
- Cross-entropy loss
- Manual backpropagation implementation
- Stochastic Gradient Descent (SGD)
- XOR classifier trained from scratch without external ML libraries

## Future Work

- CUDA acceleration for matrix operations (starting with matrix multiplication)
- CPU vs GPU performance benchmarking
- Shared memory optimization for GPU kernels
- Larger toy datasets and MNIST classification
- Improved project structure and abstraction for reusable layers

## Why This Project

The goal is not to compete with frameworks like PyTorch or TensorFlow, but to fully understand what happens underneath them.

By building everything manually first and then introducing CUDA acceleration, this project serves as both a machine learning systems project and a GPU programming deep dive.
