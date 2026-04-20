# Benchmark Results

Outputs from `project/run_benchmark.py` and `project/plot.py`.

## Reproduce

```bash
source /mnt/c/Users/yangn/projects/11868/.venv/bin/activate
cd /mnt/c/Users/yangn/projects/11868/11868-course-project
python project/run_benchmark.py \
    --batch-sizes 1 2 4 \
    --seq-lengths 32 64 128 \
    --block-sizes 8 16 \
    --max-new-tokens 16 \
    --compare-baseline
python project/plot.py
```

## Layout

```
benchmarks/
├── README.md
├── results/
│   ├── benchmark_results.csv         # Throughput / latency / fragmentation / correctness
│   └── fragmentation_results.csv     # Dedicated non-aligned seq-length sweep
└── plots/
    ├── throughput_vs_batch.png
    ├── latency_vs_seqlen.png
    ├── speedup_vs_seqlen.png
    ├── fragmentation_vs_blocksize.png
    └── kv_memory_vs_contiguous.png
```

## Headline findings (CPU backend, n_embd=64, n_head=4, n_layers=2)

### 1. Decode is 2.3×–5.6× faster than the no-KV-cache baseline

The paged decode path always wins, and the gap **grows with sequence length**:

| block | batch | seq | tok/s | ms/tok (mean) | p50  | p95  | baseline ms/tok | speedup |
|------:|------:|----:|------:|--------------:|-----:|-----:|----------------:|--------:|
| 16 | 1 | 128 | 15.8 | 55.3 | 56.0 | 71.2 | 221.8 | **4.01×** |
| 16 | 2 | 128 | 26.2 | 31.1 | 32.8 | 37.0 | 153.2 | **4.93×** |
| 16 | 4 | 128 | 36.1 | 21.1 | 20.2 | 24.7 | 118.5 | **5.63×** |
| 8  | 4 | 128 | 30.7 | 24.2 | 22.6 | 34.9 | 116.7 | **4.82×** |

See [plots/throughput_vs_batch.png](plots/throughput_vs_batch.png), [plots/latency_vs_seqlen.png](plots/latency_vs_seqlen.png), [plots/speedup_vs_seqlen.png](plots/speedup_vs_seqlen.png).

### 2. Paged KV cache uses 86–97% less memory than naive contiguous allocation

A system without paging must reserve `n_positions` worth of KV cache per active sequence. Paging only allocates blocks for tokens that actually exist:

| block | batch | seq | KV bytes (paged) | KV bytes (contig, n_pos=1024) | savings |
|------:|------:|----:|-----------------:|------------------------------:|--------:|
| 8  | 4 | 33  |  80 KB | 2,048 KB | **96.1%** |
| 16 | 4 | 33  |  96 KB | 2,048 KB | **95.3%** |
| 8  | 4 | 130 | 272 KB | 2,048 KB | **86.7%** |
| 16 | 4 | 130 | 288 KB | 2,048 KB | **85.9%** |

See [plots/kv_memory_vs_contiguous.png](plots/kv_memory_vs_contiguous.png).

### 3. Internal fragmentation tracks block size as expected

For sequences whose length doesn't divide evenly by the block size:

| block | seq | internal fragmentation |
|------:|----:|-----------------------:|
| 8  |  33 | 17.5% |
| 16 |  33 | 31.3% |
| 8  | 100 |  3.9% |
| 16 | 100 | 10.7% |
| 8  | 130 |  4.4% |
| 16 | 130 |  9.7% |

Smaller blocks reduce internal fragmentation but increase per-token bookkeeping overhead. See [plots/fragmentation_vs_blocksize.png](plots/fragmentation_vs_blocksize.png).

### 4. Max sustainable batch size (512 KV blocks budget)

| block_size | seq=32 | seq=64 | seq=128 |
|-----------:|-------:|-------:|--------:|
| 8  | 128 | 64  | 32 |
| 16 | 256 | 128 | 64 |

### 5. Correctness

- All 39 unit tests pass (`pytest tests/ -q`).
- 2 tests compare the **CUDA kernel** against the Python reference (`max_abs_error < 1e-5`).
- 6 parity tests confirm `PagedMultiHeadAttention` matches a pure-NumPy MHA reference within `2e-4` for varying `(seq_len, n_head, head_dim)` and across the prefill+decode boundary.
- Every CSV row has `correctness_pass=1` with `max_abs_error ≈ 1.4e-7`.

See [../docs/REVIEW.md](../docs/REVIEW.md) for the full project review, gaps analysis, and design notes.

