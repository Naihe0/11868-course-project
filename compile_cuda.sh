#!/bin/bash
mkdir -p minitorch/cuda_kernels

# Compile base MiniTorch CUDA ops (needed by CudaKernelOps backend)
nvcc -o minitorch/cuda_kernels/combine.so --shared src/combine.cu -Xcompiler -fPIC

# Compile softmax kernel (hw4)
nvcc -o minitorch/cuda_kernels/softmax_kernel.so --shared src/softmax_kernel.cu -Xcompiler -fPIC

# Compile layernorm kernel (hw4)
nvcc -o minitorch/cuda_kernels/layernorm_kernel.so --shared src/layernorm_kernel.cu -Isrc -Xcompiler -fPIC

# Compile PagedAttention CUDA kernel
# Use --cudart shared so the .so uses the process-wide libcudart.so instead of a
# private static copy.  A static copy cannot see the GPU once numba.cuda (driver
# API) has already initialised a context.
nvcc -o minitorch/cuda_kernels/paged_attention.so --shared src/paged_attention.cu -Xcompiler -fPIC --cudart shared
