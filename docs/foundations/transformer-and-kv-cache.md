# Transformer And KV Cache Primer

This page explains only the transformer concepts needed to understand this project. The implementation is a decoder-only language model with a PagedAttention KV cache.

## Decoder-Only Generation

A decoder-only language model receives a prompt and predicts the next token. Generation has two phases:

1. **Prefill**: process the whole prompt at once.
2. **Decode**: generate one new token at a time.

Example with a prompt length of 4 and 3 generated tokens:

```text
Prefill input:  [t0, t1, t2, t3]
Sample:                         t4
Decode input:                   [t4]
Sample:                             t5
Decode input:                       [t5]
Sample:                                 t6
```

The prefill phase is expensive but parallel across prompt positions. The decode phase is sequential because token `t5` depends on `t4`, and token `t6` depends on both.

## Attention In One Layer

Each transformer layer turns hidden states into three vectors:

- `Q`, query: what the current token is looking for.
- `K`, key: what each previous token offers for matching.
- `V`, value: the information copied from previous tokens after matching.

Scaled dot-product attention computes:

```text
scores = Q * K^T / sqrt(head_dim)
weights = softmax(scores + mask)
output = weights * V
```

In code, `standard_attention` receives tensors with shape:

```text
query: (batch, n_head, seq_q, head_dim)
key:   (batch, n_head, seq_kv, head_dim)
value: (batch, n_head, seq_kv, head_dim)
```

For prefill, `seq_q == seq_kv == prompt_len`. For decode, `seq_q == 1` and `seq_kv == current_context_len`.

## Multi-Head Attention

The model dimension `n_embd` is split into multiple heads:

```text
n_embd = n_head * head_dim
```

Each head attends separately, then the outputs are merged back to `n_embd`. In this project, `PagedMultiHeadAttention` computes Q/K/V with `Linear(n_embd, n_embd)` projections, reshapes to heads, runs attention, then applies `out_proj`.

## Causal Masking

Decoder-only models cannot let a token attend to future tokens. During prefill, the token at position `i` can see positions `0..i`, but not `i+1..end`.

The project implements this with a mask containing `0` for allowed positions and a large negative number, `-1e9`, for blocked positions. Adding that mask before softmax makes the blocked probabilities nearly zero.

During decode, the new token is already at the end of the sequence, so it can attend to every cached token including itself.

## What Is A KV Cache?

Without a KV cache, every decode step would recompute keys and values for the entire prompt plus all generated tokens so far. That is wasteful because old tokens do not change.

A KV cache stores the keys and values once:

```text
After prefill:
  cache stores K/V for tokens 0..prompt_len-1

After decode step 1:
  append K/V for token prompt_len

After decode step 2:
  append K/V for token prompt_len+1
```

Then each decode step only computes Q/K/V for the one new token, appends K/V, and attends over the cache.

## The Contiguous KV Problem

A simple serving system can allocate a fixed contiguous cache for every request:

```text
sequence 0: [slot 0, slot 1, ..., slot 1023]
sequence 1: [slot 0, slot 1, ..., slot 1023]
```

This is easy to index but wasteful. If a sequence uses only 80 tokens and the system reserves 1024 slots, most of its KV memory is reserved but unused.

For one layer, approximate KV bytes per token are:

```text
2 * n_head * head_dim * dtype_size
```

The `2` is for K and V. For all layers:

```text
2 * n_layers * n_head * head_dim * dtype_size
```

Contiguous reservation multiplies this by the maximum reserved length per active sequence, not by the actual current sequence length.

## Paged KV Cache

PagedAttention replaces one large per-sequence slab with many fixed-size physical blocks. Each sequence stores a block table:

```text
logical tokens:  0  1  2  3 | 4  5  6  7 | 8
logical blocks:        0    |      1      | 2
block table:     [physical block 5, physical block 2, physical block 9]
```

The sequence is logically contiguous, but its memory can be physically scattered. During attention, code translates each token index into:

```text
logical_block = token_index // block_size
slot_idx      = token_index % block_size
block_id      = block_table[logical_block]
```

Then it reads:

```text
key_cache[block_id, slot_idx]
value_cache[block_id, slot_idx]
```

That is the core idea of this project.

## Prefill Versus Decode In This Repo

During prefill:

1. `PagedDecoderLM.forward_prefill` allocates blocks for each sequence.
2. Each `PagedTransformerLayer` calls `PagedMultiHeadAttention.forward_prefill`.
3. Attention is computed with normal contiguous `standard_attention`.
4. K/V are written into `BlockManager.key_cache` and `BlockManager.value_cache`.

During decode:

1. `PagedDecoderLM.forward_decode` reserves one new token slot per active sequence.
2. Each attention layer computes Q/K/V for the one new token.
3. The new K/V are written into the correct physical block and slot.
4. `paged_attention_ref` or the CUDA kernel attends over all cached K/V through the block table.

## Why Prefix Caching Fits Naturally

Many requests can share a prefix, such as a system prompt. If a complete block of tokens has already been prefetched, later requests with the same block can reuse its physical KV block instead of recomputing it.

This project hashes complete token blocks, stores hash-to-block mappings, and increments block reference counts when multiple sequences share a cached prefix block.

Tail blocks that are not full are not published into the prefix cache, because sharing partial blocks complicates mutation and copy-on-write.

## What To Remember

- Prefill handles the full prompt and fills the cache.
- Decode handles one token and reads the cache.
- A KV cache avoids recomputing keys and values for old tokens.
- Contiguous caches waste memory by reserving for worst-case length.
- Paged caches allocate fixed-size blocks only as needed.
- A block table is the bridge from logical token order to physical cache storage.
