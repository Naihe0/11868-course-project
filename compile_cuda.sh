#!/bin/bash
mkdir -p minitorch/cuda_kernels
nvcc -o minitorch/cuda_kernels/combine.so --shared src/combine.cu -Xcompiler -fPIC
nvcc -o minitorch/cuda_kernels/paged_attention.so --shared src/paged_attention.cu -Xcompiler -fPIC
