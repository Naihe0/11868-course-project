# Transformer Integration Walkthrough

`minitorch/transformer.py` integrates the block manager and paged attention into a decoder-only language model.

The file defines:

- `FeedForward`
- `PagedTransformerLayer`
- `PagedDecoderLM`

## Model Structure

At a high level:

```text
token ids
  -> token embedding + position embedding
  -> PagedTransformerLayer 0
  -> PagedTransformerLayer 1
  -> ...
  -> final LayerNorm
  -> lm_head
  -> logits
```

Each transformer layer uses pre-layer normalization:

```text
x -> LayerNorm -> Attention -> residual add
  -> LayerNorm -> FeedForward -> residual add
```

## `FeedForward`

`FeedForward` is the MLP sublayer:

```text
Linear(n_embd, 4 * n_embd)
GELU
Linear(4 * n_embd, n_embd)
Dropout
```

It reshapes `(batch, seq_len, n_embd)` to `(batch * seq_len, n_embd)` for the linear layers, then reshapes back.

## `PagedTransformerLayer`

`PagedTransformerLayer` owns:

- `ln_1`
- `ln_2`
- `attention`: a `PagedMultiHeadAttention`
- `ff`: a `FeedForward`

It has two forward methods because prefill and decode use different attention behavior.

### `forward_prefill`

Input shape:

```text
x: (batch, seq_len, n_embd)
```

Steps:

1. Normalize with `ln_1`.
2. Call `attention.forward_prefill`, which uses standard attention and writes prompt K/V into the block manager.
3. Add the residual.
4. Normalize with `ln_2`.
5. Call the feed-forward network.
6. Add the residual.

### `forward_decode`

Input shape:

```text
x: (batch, 1, n_embd)
```

The structure is the same, but `attention.forward_decode` reads historical K/V through block tables.

## `PagedDecoderLM`

`PagedDecoderLM` owns the full language model:

- `token_embeddings`
- `position_embeddings`
- `layers`
- final `ln`
- `lm_head`

Constructor parameters mirror the model and cache configuration:

- `n_vocab`
- `n_embd`
- `n_head`
- `n_positions`
- `n_layers`
- `block_size`
- `decode_backend`
- `compare_to_ref`
- `compare_tolerance`

The block manager is not owned by the model. It is passed into prefill/decode calls so the caller controls cache capacity and sequence lifetime.

## Embedding With `_embed`

`_embed(idx, start_pos=0)` creates token plus positional embeddings.

For prefill:

```text
idx shape: (batch, prompt_len)
position ids: 0..prompt_len-1
```

For decode:

```text
idx shape: (batch, 1)
position id: current token position
```

The `start_pos` argument keeps decode positional embeddings aligned with the full generated sequence.

## Prefill Without Prefix Cache Hits

`forward_prefill(idx, block_manager, seq_ids)` begins by checking prefix-cache matches for every batch item. If all matches are misses, the simple path runs:

1. Allocate blocks for every sequence with `allocate_blocks_for_sequence`.
2. Embed the whole prompt.
3. Run every `PagedTransformerLayer.forward_prefill`.
4. Publish full prompt blocks with `publish_sequence_prefix_blocks` for future reuse.
5. Apply final `LayerNorm` and `lm_head`.
6. Return logits shaped `(batch, seq_len, n_vocab)`.

This path computes attention for every prompt token and fills all layer caches.

## Prefill With Prefix Cache Hits

If any sequence has cached prefix blocks, `forward_prefill` splits the batch:

- no-hit sequences run the normal full prefill path
- hit sequences are grouped by `cached_token_count`

Grouping matters because suffix work length and prefix length must match inside one batched call.

For each hit group, `_forward_prefill_group_with_prefix` runs:

1. Choose `work_start = cached_token_count`, except if the full prompt is cached, keep one token of work so logits can still be produced.
2. Slice `idx_work = idx[:, work_start:]`.
3. Embed only the work tokens with `start_pos=work_start`.
4. For each layer, call `attention.forward_prefill_with_prefix_batch`.
5. Run final norm and `lm_head` for the work tokens.
6. Return a full `(batch, seq_len, n_vocab)` array with zeros for positions that were skipped and real logits for work positions.

After each group finishes, full blocks are published back into the prefix cache.

## Decode Path

`forward_decode(idx, block_manager, seq_ids, start_pos=0)` expects one token per sequence:

```text
idx: (batch, 1)
```

Steps:

1. Embed the token at `start_pos`.
2. For each `seq_id`, allocate an empty sequence if needed.
3. Reserve one token slot with `append_token_to_sequence`.
4. Run every `PagedTransformerLayer.forward_decode`.
5. Apply final norm and `lm_head`.
6. Return logits shaped `(batch, 1, n_vocab)`.

The slot reservation happens before the layers run because every layer writes its own K/V for the same logical token position.

## Generation Loop

`PagedDecoderLM.generate` is a static helper that wraps prefill and repeated decode.

Flow:

1. Set the model to eval mode.
2. Convert the prompt tensor to a Python list for output accumulation.
3. Run `forward_prefill` on the full prompt.
4. Sample the first new token from the last prompt-position logits.
5. For every remaining generated token:
   - build `(batch, 1)` input for the previous sampled token
   - call `forward_decode`
   - sample the next token
   - append it to the generated output
6. In a `finally` block, free every active sequence and close CUDA decode runtimes.

The `finally` block is important. It releases blocks even if decode raises an error.

## Sequence IDs

`seq_ids` are caller-provided identifiers. They let the block manager store independent block tables for each active sequence.

Typical batch use:

```python
seq_ids = list(range(batch_size))
```

Rules:

- A `seq_id` cannot be prefetched twice while active.
- A sequence should be freed when generation finishes.
- Reusing a `seq_id` is safe after `free_sequence` removes the previous block table.

## Runtime Cleanup

`close_decode_runtime` walks through layers and closes CUDA runtime handles held by attention modules. This frees device memory allocated by `PagedAttentionRuntime`.

`generate` calls it automatically. Manual inference loops should call it when finished if they use `decode_backend="cuda"`.

## Full Call Flow

Prefill:

```text
PagedDecoderLM.forward_prefill
  -> BlockManager.lookup_prefix_blocks
  -> BlockManager.allocate_blocks_for_sequence or allocate_sequence_with_prefix
  -> PagedDecoderLM._embed
  -> PagedTransformerLayer.forward_prefill
       -> PagedMultiHeadAttention.forward_prefill
            -> standard_attention
            -> _write_kv_batch_to_cache
  -> BlockManager.publish_sequence_prefix_blocks
  -> final ln + lm_head
```

Decode:

```text
PagedDecoderLM.forward_decode
  -> _embed(start_pos)
  -> BlockManager.append_token_to_sequence
  -> PagedTransformerLayer.forward_decode
       -> PagedMultiHeadAttention.forward_decode
            -> write new K/V slot
            -> paged_attention_ref or CUDA runtime forward
  -> final ln + lm_head
```

## Common Failure Modes

- `Sequence X is already active`: the caller tried to prefill an existing `seq_id` without freeing it.
- `forward_decode expects a single new token`: decode input had `seq_len != 1`.
- `No free blocks available`: the block manager capacity is too small, or cached blocks cannot be evicted.
- Position mismatch in manual loops: `start_pos` should match the absolute position of the decode token.
