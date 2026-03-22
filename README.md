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

# Run only BlockManager tests quietly
pytest tests/test_block_manager.py -q

# Run only PagedAttention reference / correctness tests quietly
pytest tests/test_paged_attention.py -q

# Run one specific PagedAttention test
pytest tests/test_paged_attention.py -k test_single_sequence_multi_block -v
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

## TODO

### Repository completeness

- [ ] Copy the base MiniTorch files from hw3 into `minitorch/`
- [ ] Add missing CUDA source files required by `compile_cuda.sh` (for example `src/combine.cu`)
- [ ] Verify that `pip install -e .` and `import minitorch` work in a clean environment

### Block manager

- [x] Implement `BlockManager.allocate_block()`
- [x] Implement `BlockManager.allocate_blocks_for_sequence()`
- [x] Implement `BlockManager.append_token_to_sequence()`
- [x] Implement `BlockManager.free_sequence()`
- [x] Implement `BlockManager.compute_fragmentation()`
- [x] Move KV storage to global `key_cache` / `value_cache` owned by `BlockManager`

### Attention implementation

- [x] Implement `standard_attention()` as the contiguous correctness baseline
- [x] Implement `paged_attention_ref()` for Python-side validation
- [ ] Implement `PagedAttentionKernel._load_library()`
- [ ] Implement `PagedAttentionKernel.forward()`
- [x] Initialize Q / K / V / output projections in `PagedMultiHeadAttention`
- [x] Implement `PagedMultiHeadAttention.forward_prefill()`
- [x] Implement `PagedMultiHeadAttention.forward_decode()`

### Transformer integration

- [x] Implement `PagedTransformerLayer.forward_prefill()`
- [x] Implement `PagedTransformerLayer.forward_decode()`
- [x] Implement `PagedDecoderLM.generate()`
- [x] Verify end-to-end prefill/decode behavior with sequence position tracking

### CUDA kernel

- [ ] Implement `warp_reduce_sum()`
- [ ] Implement `warp_reduce_max()`
- [ ] Implement `paged_attention_v1_kernel()`
- [ ] Validate kernel launch configuration across supported head dimensions
- [ ] Compile and test `minitorch/cuda_kernels/paged_attention.so`

### Optional advanced TODOs

- [ ] Rework KV cache memory layout to better match coalesced GPU memory access
- [ ] Add a higher-performance kernel path inspired by vLLM's paged attention design
- [ ] Introduce thread-group / warp-level work partitioning instead of a purely naive per-head kernel
- [ ] Use shared memory staging for query vectors and partial reductions
- [ ] Implement more efficient online softmax and value accumulation reductions
- [ ] Add template-specialized kernels for different `head_dim`, `block_size`, and thread-count configurations
- [ ] Benchmark multiple kernel variants and select launch parameters per configuration
- [ ] Explore multi-block-per-sequence kernel decomposition for long-context decode
- [ ] Add CUDA profiling with Nsight or equivalent tooling to analyze memory bandwidth and occupancy
- [ ] Compare the course-project kernel against a vLLM-style design in the final report
- [ ] Evaluate tradeoffs between implementation simplicity, numerical stability, and throughput
- [ ] Document which optimizations are correctness-preserving versus performance-only

### Tests

- [x] Fill in `tests/test_block_manager.py`
- [x] Fill in `tests/test_paged_attention.py`
- [x] Fill in the remaining performance cycle test in `tests/test_benchmark.py`
- [x] Add an end-to-end inference smoke test once the model path is runnable

### Benchmarks and reporting

- [ ] Implement throughput measurement in `project/run_benchmark.py`
- [ ] Implement fragmentation measurement in `project/run_benchmark.py`
- [ ] Implement max-batch-size search in `project/run_benchmark.py`
- [ ] Implement correctness comparison in `project/run_benchmark.py`
- [ ] Save benchmark outputs to `benchmarks/results/benchmark_results.csv`
- [ ] Add benchmark plots / summary discussion to `benchmarks/README.md`

## References

- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023
- vLLM: https://github.com/vllm-project/vllm
