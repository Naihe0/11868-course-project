# PagedAttention in MiniTorch

Reimplementation of [PagedAttention](https://arxiv.org/abs/2309.06180) within the MiniTorch educational deep learning framework, as proposed for the 11-868 course project.

## Overview

PagedAttention is a memory-efficient attention mechanism for LLM inference that manages the KV cache in non-contiguous memory blocks, inspired by virtual memory paging. This project integrates it into MiniTorch to provide a clear, accessible implementation that demonstrates system-level challenges in modern LLM serving.

### Key Components

- **Block Manager** (`minitorch/block_manager.py`): Block-based memory allocator for KV cache pages
- **PagedAttention Kernel** (`src/paged_attention.cu`): Custom CUDA kernel for attention over non-contiguous memory blocks
- **Transformer Integration** (`minitorch/transformer.py`): Transformer model with PagedAttention support
- **Benchmarking** (`project/run_benchmark.py`): Evaluation scripts for memory and throughput

## Project Structure

```
11868-course-project/
├── README.md
├── setup.py
├── setup.cfg
├── requirements.txt
├── compile_cuda.sh
├── pytest.ini
├── src/
│   ├── combine.cu                 # Base MiniTorch CUDA ops
│   └── paged_attention.cu         # PagedAttention CUDA kernel
├── minitorch/
│   ├── __init__.py
│   ├── autodiff.py                
│   ├── module.py                  
│   ├── modules_basic.py           
│   ├── tensor.py                  
│   ├── tensor_data.py             
│   ├── tensor_functions.py        
│   ├── tensor_ops.py              
│   ├── nn.py                     
│   ├── operators.py              
│   ├── optim.py                   
│   ├── fast_ops.py                
│   ├── cuda_ops.py                
│   ├── cuda_kernel_ops.py         
│   ├── datasets.py                
│   ├── testing.py                 
│   ├── scalar.py                  
│   ├── scalar_functions.py       
│   ├── cuda_kernels/              # Compiled .so files
│   ├── block_manager.py           # NEW: Block-based KV cache memory manager
│   ├── paged_attention.py         # NEW: PagedAttention mechanism
│   └── transformer.py             # MODIFIED: Transformer with PagedAttention
├── project/
│   ├── run_inference.py           # Inference with PagedAttention
│   └── run_benchmark.py           # Performance evaluation
├── tests/
│   ├── __init__.py
│   ├── test_block_manager.py      # Block manager unit tests
│   ├── test_paged_attention.py    # PagedAttention correctness tests
│   └── test_benchmark.py          # Performance regression tests
├── benchmarks/
│   └── README.md                  # Benchmark results and plots
└── docs/
    └── design.md                  # Design document
```

## Setup

### Prerequisites

- Python 3.10+
- CUDA Toolkit (compatible with V100 GPUs)
- PSC compute node access (V100-16GB or V100-32GB)

### Installation

```bash
# 1. Copy base MiniTorch files from hw3
#    Copy all files marked "(from hw3)" in the structure above
#    from ../llmsys_hw3/minitorch/ into ./minitorch/

# 2. Install dependencies
pip install -r requirements.txt

# 3. Compile CUDA kernels
bash compile_cuda.sh

# 4. Install the package in development mode
pip install -e .
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test suites
pytest tests/test_block_manager.py -v
pytest tests/test_paged_attention.py -v
```

### Running Benchmarks

```bash
# Run inference with PagedAttention
python project/run_inference.py

# Run performance benchmarks
python project/run_benchmark.py --batch-sizes 1 2 4 8 --seq-lengths 128 256 512 1024
```

## Evaluation Metrics

- **Memory fragmentation**: Internal and external fragmentation of KV cache
- **Maximum batch size**: Largest batch before OOM
- **Throughput**: Tokens/second for generation
- **Correctness**: Output matches standard attention exactly

## References

- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023
- vLLM: https://github.com/vllm-project/vllm
