# Python PagedAttention Walkthrough

`minitorch/paged_attention.py` contains both the correctness-first Python implementation and the high-level attention module used by the transformer. The CUDA wrapper lives in the same file, but the CUDA details are covered separately in [cuda-kernel-and-runtime.md](cuda-kernel-and-runtime.md).

## Three Levels In One File

The file has three conceptual layers:

1. `standard_attention`: contiguous scaled dot-product attention.
2. `paged_attention_ref`: slow reference implementation that gathers K/V through block tables.
3. `PagedMultiHeadAttention`: MiniTorch module that uses standard attention in prefill and paged attention in decode.

## `standard_attention`

`standard_attention(query, key, value, mask=None)` is the baseline used for prefill and correctness.

Input shapes:

```text
query: (batch, n_head, seq_q,  head_dim)
key:   (batch, n_head, seq_kv, head_dim)
value: (batch, n_head, seq_kv, head_dim)
mask:  broadcastable to (batch, n_head, seq_q, seq_kv)
```

It computes:

```text
scores  = sum(query * key) / sqrt(head_dim)
scores += mask, if provided
weights = softmax(scores, dim=3)
output  = sum(weights * value)
```

The implementation uses explicit broadcasting with `view` so it stays inside MiniTorch tensor operations.

## `paged_attention_ref`

`paged_attention_ref` is the simplest way to understand PagedAttention. It does not try to be fast.

Inputs:

```text
query:        Tensor shaped (batch, n_head, 1, head_dim)
key_cache:    NumPy array or list of arrays shaped (num_blocks, block_size, n_head, head_dim)
value_cache:  same shape as key_cache
block_tables: Python list of per-sequence physical block IDs
context_lens: Python list of per-sequence context lengths
```

For each batch element, it reconstructs logical K/V order:

```python
for token_idx in range(context_len):
    logical_block_idx = token_idx // block_size
    slot_idx = token_idx % block_size
    block_id = block_table[logical_block_idx]
    gathered_keys.append(key_cache[layer_id][block_id, slot_idx])
    gathered_values.append(value_cache[layer_id][block_id, slot_idx])
```

After gathering, it wraps the arrays back into MiniTorch tensors and calls `standard_attention`.

This function is the ground truth for CUDA kernel tests and optional runtime comparisons.

## `PagedAttentionKernel` In Brief

The file also defines `PagedAttentionKernel`, a ctypes wrapper around `minitorch/cuda_kernels/paged_attention.so`. The high-level attention module uses it only when `decode_backend="cuda"`. See [cuda-kernel-and-runtime.md](cuda-kernel-and-runtime.md) for the full runtime story.

## `PagedMultiHeadAttention`

`PagedMultiHeadAttention` is the transformer-facing module.

Constructor fields that matter most:

- `n_embd`: model dimension.
- `n_head`: number of heads.
- `head_dim`: computed as `n_embd // n_head`.
- `block_size`: tokens per KV cache block.
- `layer_id`: which layer's `BlockManager.key_cache[layer_id]` and `value_cache[layer_id]` to use.
- `decode_backend`: `"ref"` or `"cuda"`.
- `compare_to_ref`: when using CUDA decode, also run reference decode and assert numerical closeness.

Projection modules:

- `q_proj`
- `k_proj`
- `v_proj`
- `out_proj`

## Prefill Path

`forward_prefill(x, block_manager, seq_ids)` handles a full prompt.

Input shape:

```text
x: (batch, seq_len, n_embd)
```

Steps:

1. Flatten to `(batch * seq_len, n_embd)`.
2. Project Q/K/V with `q_proj`, `k_proj`, `v_proj`.
3. Reshape to heads and permute into `(batch, n_head, seq_len, head_dim)`.
4. Build a causal mask with `-1e9` above the diagonal.
5. Run `standard_attention(q, k, v, mask)`.
6. Merge heads and run `out_proj`.
7. Convert K/V to NumPy in `(batch, seq_len, n_head, head_dim)` order.
8. Write every token's K/V into the block manager with `_write_kv_batch_to_cache`.

The prefill attention computation itself is contiguous. Paging matters because the resulting K/V are stored into blocks for later decode.

## Decode Path

`forward_decode(x, block_manager, seq_ids)` handles exactly one new token per active sequence.

Input shape:

```text
x: (batch, 1, n_embd)
```

Important ordering detail: `PagedDecoderLM.forward_decode` has already called `append_token_to_sequence` before the layer runs. That means the current token slot exists before `PagedMultiHeadAttention.forward_decode` writes layer-specific K/V.

Steps:

1. Project Q/K/V for the one new token.
2. Convert new K/V to NumPy arrays shaped `(batch, n_head, head_dim)`.
3. For each sequence, find the final token position with `context_len - 1`.
4. Use `get_physical_location` to find `(block_id, slot_idx)`.
5. Write new K/V into that slot for this layer.
6. If CUDA runtime mode is active, update the corresponding device-side slot.
7. Run either `_decode_attention_ref` or `_decode_attention_kernel`.
8. Optionally compare CUDA output against reference output.
9. Merge heads and run `out_proj`.

The decode output shape is `(batch, 1, n_embd)`.

## `_write_kv_batch_to_cache`

This helper writes a batch of K/V values into the block manager starting at logical token index `token_start`.

Expected K/V shapes:

```text
key_values:   (batch, tokens_to_write, n_head, head_dim)
value_values: (batch, tokens_to_write, n_head, head_dim)
```

For every batch item and local token index, it asks the block manager for the physical location, writes the K/V slot, and records touched block IDs. In CUDA mode it uploads touched blocks to the runtime when needed.

## `_decode_attention_ref`

This prepares the inputs for `paged_attention_ref`:

- block tables from `block_manager.block_tables[seq_id].block_ids`
- context lengths from `block_manager.get_context_len(seq_id)`
- layer-specific cache arrays from `block_manager.key_cache` and `value_cache`

Then it calls `paged_attention_ref` with this module's `layer_id`, `n_head`, `head_dim`, and `block_size`.

## Prefix-Aware Prefill

`forward_prefill_with_prefix_batch` handles prompts where a prefix is already cached.

Inputs:

- `x`: only the suffix/work tokens that still need computation.
- `prefix_token_count`: number of cached prefix tokens visible to attention.
- `cached_token_count`: number of tokens found in the prefix cache.
- `write_kv_to_cache`: whether the new suffix K/V should be written.

The reference path:

1. Compute Q/K/V for the suffix tokens.
2. Gather cached prefix K/V with `_gather_cached_prefix_kv_batch`.
3. Concatenate prefix K/V and suffix K/V.
4. Build a causal mask where suffix token `i` can see `prefix_token_count + i + 1` tokens.
5. Run `standard_attention` over the combined context.

The CUDA path calls `_prefill_attention_kernel_with_prefix_batch`, which uses the runtime's cached prefix blocks plus suffix rows.

## Runtime Synchronization Helpers

Several methods exist only for `decode_backend="cuda"`:

- `_ensure_kernel_runtime`: creates or resizes the CUDA runtime.
- `_sync_runtime_blocks_for_sequences`: finds all blocks touched by a set of sequences and token counts.
- `_upload_runtime_blocks`: uploads blocks not already known valid on the device.
- `_ensure_runtime_synced_for_sequences`: clears/rebuilds runtime validity when the block manager object changes.

These helpers prevent the CUDA path from uploading the full cache on every decode step.

## Correctness Strategy

The Python path is the correctness anchor:

- Unit tests compare `standard_attention` to a manual NumPy attention.
- Unit tests compare `paged_attention_ref` to standard attention after gathering.
- CUDA tests compare the compiled kernel against `paged_attention_ref` when available.
- `compare_to_ref=True` lets inference/benchmark code check CUDA decode against the Python reference during runtime.

## Common Shape Transformations

| Stage | Shape |
| --- | --- |
| Transformer hidden states | `(batch, seq_len, n_embd)` |
| Flattened for projections | `(batch * seq_len, n_embd)` |
| Projected Q/K/V before head permute | `(batch, seq_len, n_head, head_dim)` |
| Attention Q/K/V | `(batch, n_head, seq_len, head_dim)` |
| Cache-write K/V | `(batch, seq_len, n_head, head_dim)` |
| Decode query | `(batch, n_head, 1, head_dim)` |
| One decode K/V slot | `(batch, n_head, head_dim)` |
