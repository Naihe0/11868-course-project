# Running Inference

`project/run_inference.py` runs synthetic generation with `PagedDecoderLM`. It does not load a tokenizer or pretrained language model weights. Token IDs are sampled randomly, and model weights are initialized by MiniTorch modules.

The script is useful for:

- exercising prefill and decode
- checking block allocation behavior
- comparing CUDA decode against Python reference decode
- checking paged decode logits against full-sequence recomputation

## Minimal CPU Run

```bash
python project/run_inference.py \
  --backend cpu \
  --decode-backend ref \
  --batch-size 1 \
  --prompt-len 16 \
  --max-new-tokens 8
```

Expected output includes:

```text
Backend: cpu
Model config: ...
Block config: ...
Generating ...
Generated ... tok/s
Block manager: ...
```

## CUDA Decode Run

After building CUDA kernels:

```bash
python project/run_inference.py \
  --backend cuda \
  --decode-backend cuda \
  --compare-to-ref \
  --batch-size 2 \
  --prompt-len 32 \
  --max-new-tokens 8
```

This uses CUDA tensor operations where available and the custom PagedAttention CUDA kernel for decode.

## Full-Recompute Correctness Check

Use `--check-correctness` to compare paged decode logits against a reference that reruns prefill on the full sequence so far:

```bash
python project/run_inference.py \
  --backend cpu \
  --decode-backend ref \
  --check-correctness \
  --batch-size 2 \
  --prompt-len 8 \
  --max-new-tokens 4
```

The script reports:

- `max_abs_error`
- `mean_abs_error`
- `argmax_match_rate`

This check is slower because it creates a fresh block manager and recomputes the full sequence at every generation step.

## Important Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--n-vocab` | `10000` | Synthetic vocabulary size. |
| `--n-embd` | `256` | Model embedding dimension. Must divide by `--n-head`. |
| `--n-head` | `8` | Number of attention heads. |
| `--n-layers` | `4` | Number of transformer layers and per-layer KV caches. |
| `--n-positions` | `1024` | Maximum position embedding length. |
| `--block-size` | `16` | Tokens per KV cache block. |
| `--num-kv-blocks` | `256` | Physical block capacity in the block manager. |
| `--batch-size` | `1` | Number of sequences generated together. |
| `--prompt-len` | `32` | Synthetic prompt length. |
| `--max-new-tokens` | `64` | Number of generated tokens to sample. |
| `--temperature` | `1.0` | Sampling temperature. `0` means argmax. |
| `--backend` | `cpu` | MiniTorch tensor backend: `cpu` or `cuda`. |
| `--decode-backend` | `ref` | Paged decode backend: Python `ref` or custom `cuda`. |
| `--compare-to-ref` | off | For CUDA decode, also run Python reference decode and assert closeness. |
| `--compare-tolerance` | `1e-4` | Numerical tolerance for CUDA/reference comparison. |
| `--check-correctness` | off | Compare paged decode logits to full-sequence recomputation. |

## Choosing `num-kv-blocks`

Each sequence needs enough blocks for the prompt and generated tokens:

```text
blocks_per_sequence = ceil((prompt_len + max_new_tokens) / block_size)
total_needed ~= batch_size * blocks_per_sequence
```

If `num_kv_blocks` is too small, the block manager raises `No free blocks available`.

## What The Script Cleans Up

At the end of a normal run, `run_inference.py` frees every `seq_id` from the block manager. The static `PagedDecoderLM.generate` helper also uses `finally` cleanup, but this CLI uses an explicit loop so it can print final block-manager state.

## What This Script Does Not Do

- It does not tokenize natural language text.
- It does not load pretrained weights.
- It does not implement a production request scheduler.
- It does not persist generated outputs to a file.

It is an implementation and correctness driver for the PagedAttention machinery.
