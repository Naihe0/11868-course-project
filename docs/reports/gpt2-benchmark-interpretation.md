# GPT-2 PagedAttention Benchmark Interpretation

Date: April 27, 2026

This note interprets the real-model GPT-2-family benchmark path added in `project/run_gpt2_paged_benchmark.py`. Unlike the earlier synthetic MiniTorch runs, these measurements load HuggingFace GPT-2-family pretrained weights, tokenize natural-language prompts, and execute MiniTorch `PagedDecoderLM` prefill/decode through the block-table KV cache.

## Generated Artifacts

Raw CSV results:

- `benchmarks/results_gpt2/gpt2_paged_benchmark.csv`
- `benchmarks/results_gpt2/tiny_gpt2_paged_benchmark.csv`

Plots:

- `benchmarks/results_gpt2/gpt2_latency.png`
- `benchmarks/results_gpt2/gpt2_throughput.png`
- `benchmarks/results_gpt2/gpt2_kv_memory.png`

## Benchmark Configuration

| Model | Backend | Decode backend | Prompts | Prompt tokens | New tokens | Block size | KV blocks |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `gpt2` | CPU | reference | 3 | 8 | 2 | 16 | 2 |
| `sshleifer/tiny-gpt2` | CPU | reference | 3 | 17-20 | 4 | 16 | 3 |

The full GPT-2 run intentionally uses very short prompts because this project is an educational MiniTorch implementation. Its embedding path materializes one-hot vectors and its LM head materializes full-vocabulary logits, so full GPT-2 CPU inference is far slower than a production PyTorch or vLLM engine.

## Result Table

| Model | Prompt | Prefill (s) | Decode (s) | Total (s) | End-to-end tok/s | Decode-forward tok/s | Paged allocated KV | Contiguous estimate | Savings | Fragmentation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gpt2` | 0 | 20.607 | 5.657 | 26.264 | 0.381 | 0.177 | 1.125 MiB | 72.000 MiB | 98.44% | 43.75% |
| `gpt2` | 1 | 4.236 | 3.790 | 8.027 | 1.246 | 0.264 | 1.125 MiB | 72.000 MiB | 98.44% | 43.75% |
| `gpt2` | 2 | 4.761 | 3.674 | 8.435 | 1.185 | 0.272 | 1.125 MiB | 72.000 MiB | 98.44% | 43.75% |
| `tiny-gpt2` | 0 | 18.045 | 6.010 | 24.055 | 0.998 | 0.499 | 0.001 MiB | 0.031 MiB | 96.88% | 28.12% |
| `tiny-gpt2` | 1 | 1.136 | 4.392 | 5.528 | 4.160 | 0.683 | 0.001 MiB | 0.031 MiB | 96.88% | 31.25% |
| `tiny-gpt2` | 2 | 1.856 | 4.273 | 6.129 | 3.426 | 0.702 | 0.001 MiB | 0.031 MiB | 96.88% | 37.50% |

The first prompt for each model is much slower than the following prompts. That row includes cold-start effects from the Python/MiniTorch execution path, so the steadier rows are more useful for comparing prompt-to-prompt behavior.

## Latency Plot

`gpt2_latency.png` splits each row into prefill and decode-forward time.

For full `gpt2`, the warm rows finish in about 8.0-8.4 seconds for an 8-token prompt plus 2 generated tokens. Prefill is about 4.2-4.8 seconds, and the single measured decode forward is about 3.7-3.8 seconds. The first full-GPT2 row takes 26.3 seconds because it pays substantial first-use overhead.

For `tiny-gpt2`, the warm rows finish in about 5.5-6.1 seconds for 17-19 prompt tokens plus 4 generated tokens. Decode dominates those warm tiny runs because each row performs three decode forwards, while full `gpt2` performs only one decode forward due to the smaller `--max-new-tokens` setting.

## Throughput Plot

`gpt2_throughput.png` shows low absolute throughput because it measures the MiniTorch educational implementation, not an optimized serving engine.

Warm full-`gpt2` rows reach about 1.19-1.25 end-to-end tokens/s and about 0.26-0.27 decode-forward tokens/s. Warm `tiny-gpt2` rows reach about 3.43-4.16 end-to-end tokens/s and about 0.68-0.70 decode-forward tokens/s.

This confirms the GPT-2 path is functionally useful for real-token and real-weight validation, but it should not be presented as production throughput. The key value is that the same PagedAttention KV-cache machinery now runs under a pretrained GPT-2-family checkpoint.

## KV-Memory Plot

`gpt2_kv_memory.png` is the strongest result: paged allocation tracks the actual prompt plus generated context instead of reserving the full 1024-position context window.

For full `gpt2`, the contiguous-context estimate is 72.0 MiB per sequence at 1024 positions. The paged run allocates one 16-token block, which is 1.125 MiB, while the live KV for 9 filled token slots is about 0.633 MiB. That is a 98.44% allocated-memory reduction versus the full-context contiguous estimate.

For `tiny-gpt2`, the absolute memory is tiny because the checkpoint has only 2 hidden dimensions, but the same structure appears: the paged allocation is 1024 bytes versus a 32768-byte contiguous estimate, giving 96.88% allocated-memory reduction.

Internal fragmentation is expected because the block size is 16 and these prompts do not exactly fill the final block. Full `gpt2` uses 9 of 16 slots, so fragmentation is 43.75%. The tiny run fills 20-23 of 32 allocated slots, giving 28.12-37.50% fragmentation. This is the normal tradeoff of block-granular allocation: it avoids full-context over-reservation while wasting only the tail slots of the last block.

## Generated Text Sanity Check

The full `gpt2` continuations are short but plausible for the tiny generation budget, for example:

- `PagedAttention reduces memory waste by 50%`
- `In a language model serving system, many languages are`
- `The project compares contiguous KV cache allocation to the`

The `tiny-gpt2` checkpoint repeatedly emits `factors`, which is expected for that very small debugging checkpoint. Its main role here is fast smoke testing of the weight loader and plotting pipeline.

## Takeaways

1. Path B is now real: the project can load GPT-2-family HuggingFace weights into MiniTorch `PagedDecoderLM` and run tokenized natural-language prompts through PagedAttention.
2. The memory behavior matches the PagedAttention story: allocation is proportional to touched KV blocks, not the model's full context window.
3. The latency/throughput numbers should be treated as educational MiniTorch measurements, not vLLM-like serving numbers.
4. The first row in each benchmark is cold-start polluted. Future timing runs should add explicit warmup prompts before recording rows.
5. The next implementation step should be batched real-prompt benchmarking with padding masks or a scheduler that groups same-length prompts, followed by CUDA decode for the real GPT-2 path.