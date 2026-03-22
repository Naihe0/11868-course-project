#!/bin/bash
mkdir -p minitorch/cuda_kernels

# Compile base MiniTorch CUDA ops (needed by CudaKernelOps backend)
nvcc -o minitorch/cuda_kernels/combine.so --shared src/combine.cu -Xcompiler -fPIC

# Compile PagedAttention CUDA kernel
nvcc -o minitorch/cuda_kernels/paged_attention.so --shared src/paged_attention.cu -Xcompiler -fPIC
