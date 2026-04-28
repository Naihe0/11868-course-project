# PagedAttention in MiniTorch

This repository reimplements the core ideas behind PagedAttention inside the MiniTorch educational deep learning framework. It includes a block-based KV cache allocator, Python reference PagedAttention, an optional CUDA decode kernel/runtime, a decoder-only language model integration, tests, and benchmark scripts.

The best place to understand the project is the maintained guide in [docs/README.md](docs/README.md). It is written for readers who are new to PagedAttention and only lightly familiar with transformers.

## What This Project Implements

- `minitorch/block_manager.py`: fixed-size physical KV blocks, per-sequence block tables, prefix-cache lookup/publish, allocation/freeing, fragmentation, and memory accounting.
- `minitorch/paged_attention.py`: contiguous reference attention, Python paged-attention reference, ctypes wrapper for the CUDA library, and the MiniTorch multi-head attention module.
- `minitorch/transformer.py`: decoder-only LM with prefill, decode, prefix-hit grouping, generation, and runtime cleanup.
- `src/paged_attention.cu`: custom CUDA PagedAttention decode kernel plus a stateful runtime API for device-side cache synchronization.
- `project/run_inference.py`: synthetic inference driver.
- `project/run_benchmark.py`: general throughput, fragmentation, capacity, correctness, baseline, and prefix-cache benchmark driver.
- `project/run_rigorous_benchmark.py`: six-experiment benchmark suite used for report/poster figures.

## Documentation

Start here:

- [docs/README.md](docs/README.md): documentation hub and reading order.
- [docs/README.zh-CN.md](docs/README.zh-CN.md): single-file Simplified Chinese version of the guide.
- [docs/foundations/minitorch-primer.md](docs/foundations/minitorch-primer.md): MiniTorch concepts used by this project.
- [docs/foundations/transformer-and-kv-cache.md](docs/foundations/transformer-and-kv-cache.md): beginner transformer, KV cache, and PagedAttention primer.
- [docs/core/block-manager.md](docs/core/block-manager.md): allocator walkthrough.
- [docs/core/paged-attention-python.md](docs/core/paged-attention-python.md): Python/reference attention walkthrough.
- [docs/core/cuda-kernel-and-runtime.md](docs/core/cuda-kernel-and-runtime.md): CUDA kernel/runtime walkthrough.
- [docs/core/transformer-integration.md](docs/core/transformer-integration.md): model lifecycle walkthrough.
- [docs/reference/code-inventory.md](docs/reference/code-inventory.md): map of code-bearing files and artifacts.

Historical project material remains in [docs/design.md](docs/design.md), [docs/REVIEW.md](docs/REVIEW.md), [docs/reports/](docs/reports/), and [benchmarks/README.md](benchmarks/README.md).

## Repository Layout

```text
11868-course-project/
├── README.md
├── compile_cuda.sh
├── requirements.txt
├── setup.py
├── setup.cfg
├── pytest.ini
├── minitorch/
│   ├── block_manager.py
│   ├── paged_attention.py
│   ├── transformer.py
│   └── ... MiniTorch framework files ...
├── src/
│   ├── paged_attention.cu
│   ├── combine.cu
│   ├── softmax_kernel.cu
│   ├── layernorm_kernel.cu
│   └── includes/
├── project/
│   ├── run_inference.py
│   ├── run_benchmark.py
│   ├── run_rigorous_benchmark.py
│   ├── run_gpt2_paged_benchmark.py
│   ├── contiguous_kv_baseline.py
│   ├── plot.py
│   ├── plot_gpt2_benchmark.py
│   └── plot_rigorous_figures.py
├── tests/
│   ├── test_block_manager.py
│   ├── test_paged_attention.py
│   ├── test_parity.py
│   └── test_benchmark.py
├── benchmarks/
└── docs/
```

## Setup

CPU reference path:

```bash
pip install -r requirements.txt
pip install -e .
```

CUDA path:

```bash
module load cuda/12.4  # if your environment uses modules
bash compile_cuda.sh
```

Detailed setup notes, including Windows/WSL and CUDA context caveats, are in [docs/usage/setup-and-build.md](docs/usage/setup-and-build.md).

## Run Inference

CPU reference decode:

```bash
python project/run_inference.py \
  --backend cpu \
  --decode-backend ref \
  --batch-size 1 \
  --prompt-len 16 \
  --max-new-tokens 8
```

CUDA decode after compiling kernels:

```bash
python project/run_inference.py \
  --backend cuda \
  --decode-backend cuda \
  --compare-to-ref \
  --batch-size 1 \
  --prompt-len 16 \
  --max-new-tokens 8
```

More examples and flag explanations are in [docs/usage/running-inference.md](docs/usage/running-inference.md).

## Run Tests

```bash
pytest tests/ -q
```

Focused suites:

```bash
pytest tests/test_block_manager.py -q
pytest tests/test_paged_attention.py -q
pytest tests/test_parity.py -q
pytest tests/test_benchmark.py -q
```

CUDA-specific tests skip automatically when CUDA or the compiled shared library is unavailable. See [docs/usage/testing-and-benchmarking.md](docs/usage/testing-and-benchmarking.md).

## Run Benchmarks

General benchmark:

```bash
python project/run_benchmark.py \
  --batch-sizes 1 2 4 \
  --seq-lengths 32 64 128 \
  --block-sizes 8 16 \
  --max-new-tokens 16 \
  --compare-baseline \
  --compare-prefix-cache
```

Plot general benchmark results:

```bash
python project/plot.py
```

GPT-2 real-data MiniTorch GPU KV benchmark:

```bash
python project/run_gpt2_paged_benchmark.py \
  --model-name gpt2 \
  --dataset-name wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --dataset-split test \
  --batch-sizes 1 \
  --seq-lengths 1 \
  --warmup-iters 0 \
  --timed-iters 1
python project/plot_gpt2_benchmark.py
```

Rigorous report/poster benchmark:

```bash
python project/run_rigorous_benchmark.py \
  --output-dir benchmarks/results_rigorous \
  --backend auto \
  --decode-backend auto
python project/plot_rigorous_figures.py
```

Benchmark interpretation is documented in [docs/advanced/memory-and-capacity.md](docs/advanced/memory-and-capacity.md) and [docs/usage/testing-and-benchmarking.md](docs/usage/testing-and-benchmarking.md).

## References

- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023.
- vLLM: https://github.com/vllm-project/vllm
