# GPT-2/WikiText GPU-Resident PagedAttention Benchmark

Date: April 27, 2026

This benchmark loads HuggingFace `gpt2`, tokenizes real WikiText-2 text, runs GPT-2 inference on CUDA, extracts real layer-11 decode-attention inputs, and then compares the current CUDA PagedAttention kernel with a contiguous no-paging CUDA baseline.

The timed comparison still satisfies the strict GPU-resident requirement: K/V, query, and output tensors live on GPU during timed iterations. The CSV records `timed_h2d_kv_bytes=0`, `timed_h2d_query_bytes=0`, and `timed_d2h_output_bytes=0` for every measured row.

## Artifacts

Primary real-model CSV:

- `benchmarks/results_gpu_resident/gpt2_wikitext_paged_vs_contiguous_gpu_resident.csv`

Primary real-model plots:

- `benchmarks/results_gpu_resident/gpt2_wikitext_paged_vs_contiguous_gpu_resident_latency.png`
- `benchmarks/results_gpu_resident/gpt2_wikitext_paged_vs_contiguous_gpu_resident_memory.png`

Supplemental synthetic-kernel CSVs are still available for controlled stress testing:

- `benchmarks/results_gpu_resident/paged_vs_contiguous_gpu_resident.csv`
- `benchmarks/results_gpu_resident/paged_vs_contiguous_gpu_resident_heavy.csv`

Implementation files:

- `project/run_gpu_resident_paged_vs_contiguous.py`
- `project/plot_gpu_resident_paged_vs_contiguous.py`
- `src/paged_attention.cu`
- `minitorch/paged_attention.py`

## How The Real-Data Path Works

For each benchmark row, the script:

1. Loads `gpt2` with HuggingFace Transformers on CUDA.
2. Loads real text from `wikitext/wikitext-2-raw-v1:test`.
3. Tokenizes enough real dataset text for the requested batch and sequence length.
4. Runs GPT-2 prefill with `use_cache=True`.
5. Captures the selected layer's next-token `c_attn` output during a decode step and uses the Q slice as the real query tensor.
6. Extracts the selected layer's prefill K/V cache as the real contiguous K/V tensor.
7. Re-packs that same real K/V tensor into paged physical blocks on GPU.
8. Runs the paged CUDA kernel and contiguous CUDA baseline through raw device pointers.

The default source is now `--source gpt2`. Synthetic mode remains available with `--source synthetic` for isolated kernel experiments.

## Primary Configuration

| Parameter | Value |
| --- | --- |
| Model | `gpt2` |
| Dataset | `wikitext/wikitext-2-raw-v1:test` |
| Extracted layer | 11 |
| Batch sizes | 1, 2 |
| Sequence lengths | 64, 128, 256, 512 |
| Heads | 12 |
| Head dimension | 64 |
| Block size | 16 |
| Static contiguous context length | 1024 |
| Warmup iterations | 10 |
| Timed iterations | 30 |

## Results

| Batch | Seq len | Paged median (ms) | Contiguous median (ms) | Paged / contiguous | Paged KV allocated | Static contiguous KV | Savings vs static |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 64 | 0.10627 | 0.05445 | 1.9517x | 0.375 MiB | 6.000 MiB | 93.75% |
| 1 | 128 | 0.08834 | 0.06577 | 1.3432x | 0.750 MiB | 6.000 MiB | 87.50% |
| 1 | 256 | 0.08326 | 0.08773 | 0.9491x | 1.500 MiB | 6.000 MiB | 75.00% |
| 1 | 512 | 0.10546 | 0.09456 | 1.1154x | 3.000 MiB | 6.000 MiB | 50.00% |
| 2 | 64 | 0.08585 | 0.10658 | 0.8055x | 0.750 MiB | 12.000 MiB | 93.75% |
| 2 | 128 | 0.11815 | 0.06702 | 1.7629x | 1.500 MiB | 12.000 MiB | 87.50% |
| 2 | 256 | 0.09635 | 0.07136 | 1.3503x | 3.000 MiB | 12.000 MiB | 75.00% |
| 2 | 512 | 0.10105 | 0.09391 | 1.0760x | 6.000 MiB | 12.000 MiB | 50.00% |

Summary over the 8 GPT-2/WikiText rows:

- Latency ratio range: `0.8055x` to `1.9517x` paged / contiguous.
- Average latency ratio: `1.2943x` paged / contiguous.
- Memory savings vs static contiguous reservation: `50.00%` to `93.75%`.
- Max output error: `0.0`.

## Interpretation

The latency plot shows that the paged kernel usually pays a block-table/indexing cost relative to compact contiguous K/V when all tensors are already GPU-resident. Most rows are around `0.95x-1.76x` contiguous latency. The `batch=1, seq=64` row is the slowest relative paged row at `1.95x`; these per-call latencies are still sub-millisecond, so launch/synchronization jitter and simple teaching-kernel effects can move individual ratios.

The key memory result is cleaner. Paged allocation tracks the actual number of context tokens, while static contiguous reservation assumes each sequence reserves the configured 1024-token context window. That gives `93.75%` savings at 64 tokens, `87.50%` at 128 tokens, `75.00%` at 256 tokens, and `50.00%` at 512 tokens. The fractional savings are the same for batch size 1 and 2 because both layouts scale linearly with batch.

Correctness is exact in this benchmark grid: paged and contiguous outputs match with `max_abs_error=0.0` for every row. This matters because both kernels consume the same real GPT-2 K/V and query tensors; the only difference is physical KV layout and block-table lookup.

## What This Benchmark Does And Does Not Claim

This benchmark does use GPT-2 inference on real WikiText data to produce the measured attention inputs. It does not time the whole GPT-2 model end to end. The timed region intentionally isolates the decode-attention kernel comparison after real GPT-2 has generated the query and K/V tensors.

That separation is useful for this project because it directly tests the PagedAttention implementation under realistic activation values while preserving a strict no timed CPU/GPU copy condition. It should be described as a real-GPT-2-activation kernel benchmark, not as a full serving throughput benchmark.

## Reproduction Command

```bash
python project/run_gpu_resident_paged_vs_contiguous.py \
  --source gpt2 \
  --model-name gpt2 \
  --dataset-name wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --dataset-split test \
  --batch-sizes 1 2 \
  --seq-lengths 64 128 256 512 \
  --block-size 16 \
  --static-context-len 1024 \
  --warmup-iters 10 \
  --timed-iters 30 \
  --output-dir benchmarks/results_gpu_resident \
  --output-csv gpt2_wikitext_paged_vs_contiguous_gpu_resident.csv
```

Plot command:

```bash
python project/plot_gpu_resident_paged_vs_contiguous.py \
  --input benchmarks/results_gpu_resident/gpt2_wikitext_paged_vs_contiguous_gpu_resident.csv \
  --output-dir benchmarks/results_gpu_resident \
  --output-prefix gpt2_wikitext_paged_vs_contiguous_gpu_resident
```

## Next Steps

The next improvements would make the benchmark closer to a serving workload:

1. Add CUDA event timing in addition to synchronized wall-clock timing.
2. Benchmark variable-length batches from real dataset prompts.
3. Extract and benchmark multiple GPT-2 layers rather than one selected layer.
4. Integrate the device-pointer runtime into full MiniTorch decode so model-level decode avoids query/output host transfers too.