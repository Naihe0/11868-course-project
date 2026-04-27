# Code Inventory

This inventory explains the role of each code-bearing area in the `11868-course-project` repository. The goal is not to give equal weight to every file. The PagedAttention files get deep explanations elsewhere; inherited MiniTorch and report-generation files are documented enough to help you understand how they support the project.

## Categories

| Category | Meaning |
| --- | --- |
| PagedAttention core | Project-specific implementation that must be understood to understand the system. |
| MiniTorch substrate | Inherited educational framework code used by the project. |
| CUDA/build infrastructure | CUDA kernels, headers, build scripts, and import/runtime glue. |
| Tests | Behavior contracts and regression coverage. |
| Benchmark/reporting | Scripts and data pipelines for measuring and plotting results. |
| Historical/report artifact | Proposal, report, poster, or generated presentation material. |
| Odd/peripheral | Present in the repo but not part of the main runnable path. |

## Root Files

| Path | Category | Role |
| --- | --- | --- |
| `README.md` | Documentation entry point | Short project overview, setup commands, and links into this docs guide. |
| `requirements.txt` | CUDA/build infrastructure | Python dependencies: NumPy, pytest, hypothesis, numba, PyCUDA, matplotlib, tqdm, and related helpers. |
| `setup.py` | CUDA/build infrastructure | Minimal setuptools entry point used with `pip install -e .`. |
| `setup.cfg` | CUDA/build infrastructure | Package metadata (`minitorch`, version `0.5`). |
| `pytest.ini` | Tests | Points pytest at the `tests/` directory. |
| `compile_cuda.sh` | CUDA/build infrastructure | Compiles `combine.so`, `softmax_kernel.so`, `layernorm_kernel.so`, and `paged_attention.so`. The PagedAttention library is built with `--cudart shared`. |
| `.run.sh` | CUDA/build infrastructure | Convenience wrapper that activates the workspace virtual environment, moves to the project directory, and executes a command. It is machine-specific. |
| `.check_env.py` | CUDA/build infrastructure | Quick environment probe for torch CUDA, numba, NumPy, pytest, and the imported MiniTorch path. |
| `.gitignore` | Repository metadata | Excludes generated/intermediate artifacts from version control. |

## PagedAttention Core Python

| Path | Main symbols | Role |
| --- | --- | --- |
| `minitorch/block_manager.py` | `KVBlock`, `BlockTable`, `PrefixCacheMatch`, `BlockManager` | Owns physical KV blocks, per-sequence block tables, global per-layer key/value caches, prefix-cache metadata, allocation/freeing, fragmentation, and memory accounting. Deep dive: [../core/block-manager.md](../core/block-manager.md). |
| `minitorch/paged_attention.py` | `standard_attention`, `paged_attention_ref`, `PagedAttentionKernel`, `PagedMultiHeadAttention` | Implements contiguous reference attention, Python paged attention, ctypes bindings to the CUDA library, runtime cache synchronization, prefix-aware prefill, and prefill/decode attention paths. Deep dives: [../core/paged-attention-python.md](../core/paged-attention-python.md) and [../core/cuda-kernel-and-runtime.md](../core/cuda-kernel-and-runtime.md). |
| `minitorch/transformer.py` | `FeedForward`, `PagedTransformerLayer`, `PagedDecoderLM` | Integrates paged attention into a decoder-only language model with prefill, decode, prefix-hit grouping, generation, and runtime cleanup. Deep dive: [../core/transformer-integration.md](../core/transformer-integration.md). |

## MiniTorch Substrate

These files are inherited MiniTorch framework pieces. They matter because the PagedAttention code uses their tensor, module, backend, and operation APIs.

| Path | Category | Role in this project |
| --- | --- | --- |
| `minitorch/__init__.py` | MiniTorch substrate plus CUDA runtime glue | Exports MiniTorch APIs and project modules. It initializes PyTorch CUDA first when available so CUDA runtime API calls are not poisoned by an early numba driver context. |
| `minitorch/tensor.py` | MiniTorch substrate | User-facing `Tensor`, NumPy conversion helpers, operators, views, permutations, and backend dispatch. |
| `minitorch/tensor_data.py` | MiniTorch substrate | Low-level tensor storage, shape, strides, indexing, and broadcasting mechanics. |
| `minitorch/tensor_functions.py` | MiniTorch substrate | Tensor operations exposed as differentiable functions. |
| `minitorch/tensor_ops.py` | MiniTorch substrate | Backend abstraction and tensor map/zip/reduce primitives used by CPU and CUDA backends. |
| `minitorch/fast_ops.py` | MiniTorch substrate | Faster CPU tensor operations. The default CPU examples use `TensorBackend(FastOps)`. |
| `minitorch/cuda_ops.py` | CUDA/build infrastructure | CUDA tensor backend inherited from MiniTorch, built on numba/PyCUDA support. |
| `minitorch/cuda_kernel_ops.py` | CUDA/build infrastructure | CUDA kernel-backed tensor operations and lazy PyCUDA loading. |
| `minitorch/fast_conv.py` | MiniTorch substrate | Convolution support inherited from MiniTorch; not central to PagedAttention inference. |
| `minitorch/module.py` | MiniTorch substrate | `Module` and `Parameter`, used by attention and transformer layers. |
| `minitorch/modules.py` | MiniTorch substrate | General module helpers inherited from MiniTorch. |
| `minitorch/modules_basic.py` | MiniTorch substrate | `Linear`, `Embedding`, `Dropout`, and `LayerNorm1d`, all used by `PagedDecoderLM`. |
| `minitorch/modules_transfomer.py` | MiniTorch substrate and benchmark baseline | Non-paged transformer implementation inherited from earlier homework. Used as a no-cache baseline in benchmarks. The filename has a typo (`transfomer`). |
| `minitorch/nn.py` | MiniTorch substrate | Neural-network functions such as `softmax`, `dropout`, `GELU`, and reductions. |
| `minitorch/operators.py` | MiniTorch substrate | Scalar/math primitives that tensor functions build on. |
| `minitorch/autodiff.py` | MiniTorch substrate | Autodiff graph infrastructure. The project mostly runs inference, but modules and tensors still come from this framework. |
| `minitorch/scalar.py` | MiniTorch substrate | Scalar value and scalar autodiff support. |
| `minitorch/scalar_functions.py` | MiniTorch substrate | Scalar differentiable operations. |
| `minitorch/scalar_modules.py` | MiniTorch substrate | Scalar module examples/inherited helpers. |
| `minitorch/optim.py` | MiniTorch substrate | Optimizer support, not central to the inference path. |
| `minitorch/datasets.py` | MiniTorch substrate | Toy datasets inherited from MiniTorch. |
| `minitorch/testing.py` | Tests/MiniTorch substrate | Testing helpers inherited from MiniTorch. |
| `minitorch/tmp.py` | Odd/peripheral | Small leftover helper/import stub. It is not part of the PagedAttention execution path. |
| `minitorch/cuda_kernels/.gitkeep` | CUDA/build infrastructure | Keeps the compiled-kernel output directory present before `compile_cuda.sh` runs. |

## CUDA Source And Headers

| Path | Main symbols | Role |
| --- | --- | --- |
| `src/paged_attention.cu` | `warp_reduce_sum`, `warp_reduce_max`, `paged_attention_v1_kernel`, `paged_attention_prefill_with_prefix_kernel`, `PagedAttentionRuntime`, `paged_attention_runtime_*`, `paged_attention_forward` | Project-specific CUDA implementation. Provides both stateless host-cache forward and stateful runtime cache APIs. Deep dive: [../core/cuda-kernel-and-runtime.md](../core/cuda-kernel-and-runtime.md). |
| `src/combine.cu` | `MatrixMultiplyKernel`, `mapKernel`, `zipKernel`, `reduceKernel`, `MatrixMultiply`, `tensorMap`, `tensorZip`, `tensorReduce` | Inherited MiniTorch CUDA tensor primitives. Required by `CudaKernelOps`. |
| `src/softmax_kernel.cu` | `ker_attn_softmax_lt32`, `ker_attn_softmax`, `launch_attn_softmax`, `ker_attn_softmax_bw`, `launch_attn_softmax_bw` | Inherited attention-softmax kernels. The project-specific paged kernel has its own softmax loop. |
| `src/layernorm_kernel.cu` | `ker_layer_norm`, `launch_layernorm`, `ker_ln_bw_dgamma_dbetta`, `ker_ln_bw_dinp`, `launch_layernorm_bw` | Inherited CUDA LayerNorm kernels. |
| `src/includes/block_reduce.h` | warp/block reduction macros | Shared CUDA reduction utilities inherited from prior homework. |
| `src/includes/common.h` | common CUDA constants and includes | Shared CUDA constants such as `WARP_SIZE` and `MAX_THREADS`. |
| `src/includes/cuda_util.h` | `CHECK_GPU_ERROR`, CUDA utility declarations | Error checking, diagnostics, and CUDA helper declarations. |
| `src/includes/kernels.h` | many kernel launcher declarations | Broad inherited header with declarations for LayerNorm, softmax, dropout, transform, quantization, and related kernels. |

## Project Scripts

| Path | Main symbols | Role |
| --- | --- | --- |
| `project/run_inference.py` | `parse_args`, `_sample`, `_reference_last_logits`, `main` | CLI for synthetic generation using `PagedDecoderLM` and `BlockManager`. Supports CPU/CUDA tensor backends, reference/CUDA decode backends, CUDA-reference comparison, and full recomputation checks. |
| `project/run_benchmark.py` | `benchmark_throughput`, `benchmark_baseline_throughput`, `benchmark_prefix_cache_prefill`, `benchmark_fragmentation`, `benchmark_max_batch_size`, `check_correctness`, `main` | General benchmark suite producing throughput, latency, fragmentation, max-batch, correctness, optional baseline, and optional prefix-cache CSV outputs. |
| `project/run_rigorous_benchmark.py` | `run_memory_breakdown`, `run_capacity_curve`, `run_decode_speed`, `run_prefix_prefill`, `run_parallel_sampling_memory`, `run_beam_search_memory`, `main` | Cleaner six-experiment benchmark used by poster/report figures. Includes memory breakdown, capacity, decode speed, prefix prefill, parallel sampling memory, and beam-search memory. |
| `project/run_gpt2_paged_benchmark.py` | `main`, `_make_model_from_hf`, `_benchmark_one`, `NoKVDecoder`, `GpuContiguousKVDecoder` | Loads pretrained GPT-2 weights into the MiniTorch model, tokenizes real WikiText prompts, and compares full-model no-cache recompute, contiguous GPU KV decode, and GPU-resident PagedAttention. |
| `project/contiguous_kv_baseline.py` | `ContiguousKVDecoderLM` | HuggingFace-style contiguous KV baseline that shares model weights with `PagedDecoderLM` for fair benchmark comparisons. |
| `project/plot.py` | `plot_throughput_vs_batch`, `plot_latency_vs_seqlen`, `plot_speedup_vs_seqlen`, `plot_fragmentation_vs_blocksize`, `plot_kv_memory_vs_contiguous` | Plots CSV output from `run_benchmark.py`. |
| `project/plot_rigorous_figures.py` | `plot_fig1_memory_breakdown` through `plot_fig6_beam_search`, `main` | Produces report/poster-ready figures from rigorous benchmark CSVs. |
| `project/plot_gpt2_benchmark.py` | `plot_latency`, `plot_throughput`, `plot_kv_memory`, `main` | Plots the GPT-2 benchmark comparison CSV into latency, throughput, and KV-memory figures. |

## Tests

| Path | Main coverage | Role |
| --- | --- | --- |
| `tests/test_block_manager.py` | allocation, sequence lifecycle, block tables, fragmentation, prefix-cache hash/lookup/publish/eviction | Defines core allocator behavior and edge cases such as OOM rollback and active sequence rejection. |
| `tests/test_paged_attention.py` | standard attention, paged reference, module prefill/decode, transformer layer/model behavior, prefix reuse, generation cleanup, CUDA runtime synchronization, CUDA kernel parity | The broadest correctness suite for the attention and model integration path. CUDA-specific tests are skipped when the runtime/kernel is unavailable. |
| `tests/test_parity.py` | NumPy MHA parity, decode vs full recompute, KV memory accounting | End-to-end sanity checks that the paged implementation is mathematically aligned with standard attention and the baseline model. |
| `tests/test_benchmark.py` | allocator speed and free/reallocate cycle speed | Lightweight performance regression checks for block allocation. |
| `tests/__init__.py` | package marker | Makes the test directory importable. |

## Documentation And Report Artifacts

| Path | Category | Role |
| --- | --- | --- |
| `docs/design.md` | Historical/report artifact | Original architecture snapshot and milestone plan. Useful background, but some status items may be stale. |
| `docs/REVIEW.md` | Historical/report artifact | Project audit, reproduction notes, gaps, and benchmark notes. |
| `docs/reports/proposal/example_paper.tex` plus local `.sty`/`.bst` files | Historical/report artifact | Proposal source and local conference-style support files. Generated `.aux`, `.log`, `.pdf`, `.fls`, `.fdb_latexmk`, `.synctex`, and image files are build artifacts. |
| `docs/reports/midterm_report/example_paper.tex` plus local `.sty`/`.bst` files | Historical/report artifact | Midterm report source and local style support files. Generated outputs are build artifacts. |
| `docs/reports/poster/poster.tex` | Historical/report artifact | Poster source for presentation figures. |
| `docs/reports/poster/README.md` | Historical/report artifact | Poster build/usage notes. |
| `docs/reports/poster/figure*.png`, `poster_preview*.png`, `poster.pdf`, logs, aux files | Historical/report artifact | Generated or copied presentation outputs. Not part of the implementation path. |
| `docs/poster-draft-image.png` | Historical/report artifact | Poster draft image. |

## Benchmark Data And Plots

CSV and PNG outputs under `benchmarks/` are data artifacts, not implementation code. They are useful for interpreting results, but the commands in [../usage/testing-and-benchmarking.md](../usage/testing-and-benchmarking.md) explain how to regenerate them.

Important locations:

- `benchmarks/results/benchmark_results.csv`
- `benchmarks/results/fragmentation_results.csv`
- `benchmarks/results_*/*.csv`
- `benchmarks/results_medium/*.csv`
- `benchmarks/report_figures_v2/figure*.png`
- `benchmarks/plots/*.png`
- `benchmarks/README.md`

## What To Read Before Editing

| Editing target | Read first |
| --- | --- |
| Block allocation, prefix cache, memory metrics | [../core/block-manager.md](../core/block-manager.md) |
| Python reference attention or MHA wrapper | [../core/paged-attention-python.md](../core/paged-attention-python.md) |
| CUDA kernel or ctypes runtime | [../core/cuda-kernel-and-runtime.md](../core/cuda-kernel-and-runtime.md) |
| Model prefill/decode/generate path | [../core/transformer-integration.md](../core/transformer-integration.md) |
| CLI flags, test commands, benchmark outputs | [../usage/testing-and-benchmarking.md](../usage/testing-and-benchmarking.md) |
