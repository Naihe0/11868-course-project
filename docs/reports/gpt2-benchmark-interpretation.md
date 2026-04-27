# GPT-2 GPU KV Benchmark Interpretation

Date: April 27, 2026

`project/run_gpt2_paged_benchmark.py` now measures a real GPT-2-family MiniTorch CUDA workload. It loads a pretrained HuggingFace checkpoint, copies the weights into the local MiniTorch `PagedDecoderLM`, tokenizes real WikiText prompts by default, and compares:

- `baseline_no_kv`: full MiniTorch model recompute over the whole context.
- `contiguous_kv`: full MiniTorch model single-token decode with a device-resident contiguous KV cache.
- `paged_attention`: full MiniTorch model single-token decode with the project's GPU-resident PagedAttention runtime.

The timed contiguous and paged paths keep the authoritative KV cache on GPU. New K/V rows are written with device-to-device copies, and timed rows report zero host-to-device K/V bytes.

## Generated Artifacts

Raw CSV:

- `benchmarks/results_gpt2/gpt2_gpu_kv_benchmark.csv`

Plots:

- `benchmarks/results_gpt2/gpt2_latency.png`
- `benchmarks/results_gpt2/gpt2_throughput.png`
- `benchmarks/results_gpt2/gpt2_kv_memory.png`

## Reproduce

Smoke test:

```bash
python project/run_gpt2_paged_benchmark.py \
  --model-name sshleifer/tiny-gpt2 \
  --batch-sizes 1 \
  --seq-lengths 4 \
  --warmup-iters 0 \
  --timed-iters 1
python project/plot_gpt2_benchmark.py
```

Full GPT-2 run:

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

## Interpretation

The latency plot compares no-cache full-model recompute against single-token full-model decode with contiguous KV and paged KV. The no-cache baseline intentionally does more work because it reruns the whole context.

The throughput plot reports decode query tokens per second from the same median timings. It should be read as a MiniTorch full-model comparison, not production serving throughput.

The memory plot compares baseline working K/V, static contiguous KV reservation, live KV, and paged allocated KV blocks across all GPT-2 layers. Paged allocation tracks the real context rounded up to `block_size`, while static contiguous KV reserves the configured maximum context length per sequence.

## Takeaways

1. The GPT-2 benchmark now uses pretrained weights and real text by default.
2. Timed contiguous and paged decode keep the authoritative K/V cache on GPU, avoiding CPU/GPU K/V transfers.
3. The plot outputs show baseline no-cache full-model recompute, contiguous GPU KV, and PagedAttention side by side.
4. This is still a MiniTorch benchmark, not a full vLLM serving benchmark with scheduling and continuous batching.
