# Project Review: PagedAttention in MiniTorch

**Date:** 2026-04-20
**Reviewer environment:** WSL Ubuntu 24, Python 3.12.3, PyTorch 2.9.1+cu128, CUDA available (1 device), numba 0.63.1, numpy 2.3.5.

---

## 1. How to reproduce the review

```bash
source /mnt/c/Users/yangn/projects/11868/.venv/bin/activate
cd /mnt/c/Users/yangn/projects/11868/11868-course-project
python -m pytest tests/ -v
python project/run_benchmark.py --batch-sizes 1 2 4 --seq-lengths 32 64 128 \
    --block-sizes 8 16 --max-new-tokens 16 --compare-baseline
```

A convenience wrapper `.run.sh` is included that activates the venv and `cd`s into the project, e.g. `bash .run.sh python -m pytest tests/`.

## 2. Test status

```
33 passed in 26.10s
```

This **includes** the previously CUDA-gated tests:
- `TestPagedAttentionKernel::test_kernel_matches_reference` ✅
- `TestPagedAttentionKernel::test_kernel_batch` ✅

The compiled kernel `minitorch/cuda_kernels/paged_attention.so` produces output that matches the Python reference within `1e-5` for both single-sequence and batched cases.

## 3. End-to-end benchmark snapshot (CPU backend, CUDA kernel optional)

Model: `vocab=1000, n_embd=64, n_head=4, n_layers=2`, decode `max_new_tokens=16`, baseline = no-KV-cache `DecoderLM` from hw3.

| block | batch | seq | tok/s | ms/tok decode | baseline ms/tok | speedup |
|------:|------:|----:|------:|--------------:|----------------:|--------:|
| 8  | 1 | 128 | 16.4 | 51.5 | 207.2 | **4.02×** |
| 8  | 4 | 128 | 29.4 | 25.2 | 106.9 | **4.24×** |
| 16 | 2 | 128 | 28.3 | 26.6 | 146.3 | **5.49×** |
| 16 | 4 | 128 | 40.7 | 18.3 | 111.6 | **6.09×** |

Headline: paged decode beats the no-cache baseline by **2.4×–6.1×** purely from KV reuse. Speedup grows with sequence length, which is the expected scaling.

`max_batch_size` (capacity at `num_kv_blocks=512`):

| block_size | seq=32 | seq=64 | seq=128 |
|-----------:|-------:|-------:|--------:|
| 8  | 128 | 64  | 32 |
| 16 | 256 | 128 | 64 |

## 4. Component inventory

### A. Block manager — `minitorch/block_manager.py`
Implemented and tested:
- `KVBlock`, `BlockTable`, `BlockManager` with global `key_cache` / `value_cache` of shape `(num_blocks, block_size, n_head, head_dim)`
- `allocate_block`, `allocate_blocks_for_sequence`, `append_token_to_sequence`, `free_sequence`
- `write_kv_slot`, `write_token_kv`
- `get_physical_location`, `get_block_table_array` (padded int32)
- `compute_fragmentation` (internal + external)
- 16 unit tests cover allocation, sequence lifecycle, block-table layout, fragmentation flavors

### B. Python attention reference — `minitorch/paged_attention.py`
- `standard_attention()` with optional mask
- `paged_attention_ref()` gathers KV through block tables, calls `standard_attention` per batch element
- `PagedMultiHeadAttention` (Q/K/V/out projections, `forward_prefill` writes K/V into cache, `forward_decode` appends new token + paged attention)

### C. CUDA kernel — `src/paged_attention.cu` + `PagedAttentionKernel`
- 3-pass V1 kernel: Q·K → block-wide softmax (warp-reduce + shared scratch) → weighted-V accumulation
- Grid `(batch, n_head)`, block size = `head_dim` (clamped to `[32, 1024]`)
- Dynamic shared memory: `logits[max_context_len] + out_accum[head_dim] + warp_scratch[num_warps]`
- Host launcher `paged_attention_forward`: device alloc / H2D / launch / D2H / free
- Compile uses `--cudart shared` (records the CUDA runtime / driver context fix)
- **Verified** against the Python reference on this WSL+RTX setup

### D. Transformer integration — `minitorch/transformer.py`
- `FeedForward`, `PagedTransformerLayer` (pre-LN → attention → residual → pre-LN → FFN → residual)
- `PagedDecoderLM` (token + positional embeddings, N stacked layers, LN, LM head)
- `PagedDecoderLM.generate()` orchestrates prefill → sample → decode loop → `free_sequence`

### E. Inference & benchmarking
- `project/run_inference.py`: one-shot generation + timing
- `project/run_benchmark.py`: sweeps block × batch × seq_len; measures throughput, fragmentation, max batch (binary search), correctness, and an optional no-cache baseline

### F. Tests (33 total, all passing)
- `test_block_manager.py` — 16 cases
- `test_paged_attention.py` — 13 reference / module / generation / kernel cases
- `test_benchmark.py` — 2 perf-regression cases
- 2 of the above are CUDA kernel cases (now active, previously gated)

## 5. Findings — gaps and opportunities

### 5.1 Already-finished but mis-flagged
- `block_manager.py` line 253 and `paged_attention.py` line 293 still carry `# TODO:` comments above code that **is** implemented. Cosmetic.

### 5.2 Real gaps

| # | Gap | Impact |
|---|-----|--------|
| G1 | Internal fragmentation in the existing CSV is **always 0.0** because every `seq_len` chosen happens to be a multiple of `block_size`. The metric is real but the experiment doesn't show it. | Under-sells the central tradeoff. |
| G2 | Max-batch number is reported in *sequences* but never converted to **bytes saved**. The vLLM headline ("paged saves >2× KV memory vs naive contiguous reservation") requires a contiguous baseline. | Missing the strongest narrative. |
| G3 | Throughput is reported as a single mean. **Latency p50/p95** were promised in the proposal. | Easy fix, big credibility gain. |
| G4 | There is **no end-to-end logits parity** test against the hw3/hw4 `DecoderLM`. We've tested attention parity in isolation, but never that the full paged model produces the same logits. | Necessary correctness evidence. |
| G5 | `benchmarks/README.md` is a stub. No plots. | Required for the report. |
| G6 | The CUDA `paged_attention_forward` host launcher does a full `cudaMalloc + cudaMemcpy` of the **entire** key/value cache every call. This makes the GPU path slower than pure NumPy at small sizes. | Out of scope for this enhancement pass; flagged for future work. |

### 5.3 Optional advanced (still open in TODO list)
- V2 multi-block-per-(seq,head) kernel
- Online softmax single-pass kernel
- Shared-memory Q staging + head_dim templating
- Nsight profiling
- vLLM kernel-style comparison

---

## 6. Enhancements implemented in this pass

The following items address gaps G1–G5. Items in §5.3 and G6 are deliberately deferred (each is multi-day GPU work).

### E1. `compute_kv_memory()` on `BlockManager`
New helper that returns:
- `kv_bytes_paged` — actually-allocated cache bytes
- `kv_bytes_contiguous_naive` — what a max-position-sized contiguous allocation would cost
- `memory_savings_ratio`

This makes the bytes-saved comparison a one-liner from the benchmark.

### E2. Benchmark sweep over **non-multiple** sequence lengths
`project/run_benchmark.py` now also reports a small fragmentation-focused sweep with seq lengths chosen to **not** divide evenly into the block sizes (e.g. 33, 65, 100), and writes those rows to a separate CSV `benchmarks/results/fragmentation_results.csv`. Internal fragmentation is now visibly non-zero.

### E3. Latency p50/p95
`benchmark_throughput()` now repeats the decode loop multiple times (configurable, default 5) and reports `decode_p50_ms`, `decode_p95_ms` per token alongside the mean.

### E4. End-to-end logits parity test
New test `tests/test_parity.py::test_paged_vs_decoderlm_prefill_logits` that builds a `PagedDecoderLM` and a hw3 `DecoderLM` with the **same parameter values**, runs prefill on the same prompt, and asserts logits match within `1e-4`.

### E5. Plotting script `project/plot.py`
Reads `benchmarks/results/benchmark_results.csv` (and the new fragmentation CSV) and emits five PNGs into `benchmarks/plots/`:
1. `throughput_vs_batch.png` — tok/s vs batch, line per (block_size, seq_len)
2. `latency_vs_seqlen.png` — ms/token decode vs seq_len, line per block_size
3. `speedup_vs_seqlen.png` — speedup over no-cache baseline
4. `fragmentation_vs_blocksize.png` — internal fragmentation vs block_size at fixed seq_len
5. `kv_memory_vs_contiguous.png` — paged vs naive contiguous KV bytes

### E6. Cleanup
- Removed two stale `# TODO` comments
- Expanded `benchmarks/README.md` with a results-and-plots section that pulls from the regenerated CSV/PNGs

---

## 7. What is *not* changed

- The CUDA kernel itself is untouched. G6 (device-resident KV cache) and the §5.3 advanced-optimization work are scoped for a follow-up sprint.
- No dependencies added.
- No public APIs broken; new CSV columns are additive.

## 8. Suggested next steps for the report

1. Lead with the **memory-saved** plot (E1 + E5 plot 5) — that is the PagedAttention thesis statement.
2. Pair the **fragmentation-vs-block-size** plot with a short discussion of the speed/memory tradeoff.
3. Use the **p95** numbers in the latency table; means hide the long tail that paged attention is meant to mitigate.
4. The end-to-end logits-parity test can be stated as a one-line correctness guarantee.
5. Add a ½-page "limitations" section listing G6, no FP16, no copy-on-write sharing, no preemption — answers the obvious questions before they get asked.
