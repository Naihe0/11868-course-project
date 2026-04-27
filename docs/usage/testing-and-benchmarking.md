# Testing And Benchmarking

This page explains how to validate the implementation and reproduce benchmark outputs. Run commands from the repository root.

## Unit Tests

Run all tests:

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

Verbose run for one test pattern:

```bash
pytest tests/test_paged_attention.py -k cuda -v
```

CUDA-specific tests are skipped when CUDA or the compiled shared library is unavailable.

## What The Tests Cover

| Test file | Main coverage |
| --- | --- |
| `tests/test_block_manager.py` | allocation, block tables, K/V writes, fragmentation, prefix cache, eviction |
| `tests/test_paged_attention.py` | standard attention, reference paged attention, module prefill/decode, transformer integration, CUDA parity |
| `tests/test_parity.py` | pure NumPy MHA parity, decode/full-recompute parity, memory accounting |
| `tests/test_benchmark.py` | allocator speed and free/reallocate cycle performance |

## Quick Benchmark Smoke Run

Use a tiny configuration first:

```bash
python project/run_benchmark.py \
  --batch-sizes 1 \
  --seq-lengths 16 \
  --block-sizes 8 \
  --max-new-tokens 4 \
  --skip-max-batch
```

Outputs are written to `benchmarks/results/` by default.

## GPT-2 Pretrained Benchmark Path

`project/run_gpt2_paged_benchmark.py` loads a HuggingFace GPT-2-family checkpoint, copies the pretrained weights into the local MiniTorch `PagedDecoderLM`, tokenizes real text prompts from WikiText by default, and runs the full MiniTorch model for all three paths. Timed iterations compare no-cache full-context recompute, contiguous GPU KV decode, and the project's GPU-resident PagedAttention runtime. The contiguous and paged KV caches are device-owned and updated with device-to-device copies; timed rows report zero host-to-device K/V bytes.

Install the optional dependencies first:

```bash
pip install -r requirements.gpt2.txt
```

Fast loader and shape smoke run:

```bash
python project/run_gpt2_paged_benchmark.py \
  --model-name sshleifer/tiny-gpt2 \
  --batch-sizes 1 \
  --seq-lengths 4 \
  --warmup-iters 0 \
  --timed-iters 1
```

Real GPT-2 run on WikiText. The MiniTorch CUDA backend is intentionally simple, so start small and scale sequence length only if you have time:

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
```

Output defaults to:

```text
benchmarks/results_gpt2/gpt2_gpu_kv_benchmark.csv
```

Plot the comparison:

```bash
python project/plot_gpt2_benchmark.py
```

The plots show full-model decode latency, query-token throughput, and KV memory for the three compared paths. Start with `sshleifer/tiny-gpt2` when checking code changes; use full `gpt2` for the final measurement.

## General Benchmark Run

```bash
python project/run_benchmark.py \
  --batch-sizes 1 2 4 \
  --seq-lengths 32 64 128 \
  --block-sizes 8 16 \
  --max-new-tokens 16 \
  --compare-baseline \
  --compare-prefix-cache
```

Main outputs:

```text
benchmarks/results/benchmark_results.csv
benchmarks/results/fragmentation_results.csv
```

`benchmark_results.csv` includes throughput, latency, fragmentation, KV efficiency, correctness, optional baseline columns, and optional prefix-cache columns.

`fragmentation_results.csv` comes from a dedicated non-aligned sequence-length sweep.

## Plotting General Benchmark Results

After `run_benchmark.py` writes CSV files:

```bash
python project/plot.py
```

Expected plots:

```text
benchmarks/plots/throughput_vs_batch.png
benchmarks/plots/latency_vs_seqlen.png
benchmarks/plots/speedup_vs_seqlen.png
benchmarks/plots/fragmentation_vs_blocksize.png
benchmarks/plots/kv_memory_vs_contiguous.png
```

Some plots are skipped if the required CSV or columns are missing. For example, `speedup_vs_seqlen.png` needs baseline columns from `--compare-baseline`.

## Rigorous Benchmark Suite

`project/run_rigorous_benchmark.py` runs the six experiments used for final report/poster figures.

Quick-ish run with default experiment sizes:

```bash
python project/run_rigorous_benchmark.py \
  --output-dir benchmarks/results_rigorous \
  --backend auto \
  --decode-backend auto
```

For a faster development check:

```bash
python project/run_rigorous_benchmark.py \
  --output-dir benchmarks/results_rigorous_smoke \
  --backend cpu \
  --decode-backend ref \
  --skip-decode \
  --skip-prefix
```

CSV outputs:

```text
benchmarks/results_rigorous/exp1_memory_breakdown.csv
benchmarks/results_rigorous/exp2_capacity_curve.csv
benchmarks/results_rigorous/exp3_decode_speed.csv
benchmarks/results_rigorous/exp4_prefix_prefill.csv
benchmarks/results_rigorous/exp5_parallel_sampling.csv
benchmarks/results_rigorous/exp6_beam_search.csv
```

## Plotting Rigorous Figures

After generating rigorous CSVs:

```bash
python project/plot_rigorous_figures.py
```

Expected figure outputs:

```text
benchmarks/report_figures_v2/figure1.png
benchmarks/report_figures_v2/figure2.png
benchmarks/report_figures_v2/figure3.png
benchmarks/report_figures_v2/figure4.png
benchmarks/report_figures_v2/figure5.png
benchmarks/report_figures_v2/figure6.png
```

## Interpreting Common Metrics

| Metric | Meaning |
| --- | --- |
| `tokens_per_sec` | End-to-end generated tokens per second, including prefill and decode. |
| `decode_tokens_per_sec` | Decode-only generated tokens per second. |
| `time_per_token_ms` | Mean decode latency per generated token. |
| `decode_p50_ms`, `decode_p95_ms` | Median and p95 per-token decode latency. |
| `internal_frag` | Fraction of allocated block slots wasted inside tail blocks. |
| `kv_efficiency` | Live KV bytes divided by reserved paged KV bytes. |
| `memory_savings_ratio` | Savings versus naive contiguous reservation. |
| `correctness_pass` | Whether implementation output matched reference within tolerance. |
| `prefix_prefill_speedup` | Fresh prefill time divided by cached-prefix prefill time. |

## Practical Benchmark Advice

- Start small, then scale model size and sequence lengths.
- Use `--compare-to-ref` when trying CUDA decode changes.
- Use `--skip-correctness` only after validating a configuration.
- Use `--skip-max-batch` for faster throughput-only iterations.
- Keep `n_embd` divisible by `n_head`.
- Ensure `num_kv_blocks` can hold `batch_size * ceil((seq_len + max_new_tokens) / block_size)` blocks.
