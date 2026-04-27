# Prefix Cache And Sharing

PagedAttention becomes especially useful when multiple requests share a prompt prefix. This project implements a block-level prefix cache so later requests can reuse full KV cache blocks computed by earlier requests.

## Why Prefix Sharing Matters

Many serving workloads reuse the same beginning:

- a system prompt
- a long instruction template
- a retrieval preamble
- a shared conversation history before branching
- multiple samples from the same prompt

Without prefix caching, every request recomputes K/V for that shared prefix. With prefix caching, one request computes those full blocks once, and later requests attach the same physical blocks to their block tables.

## Unit Of Sharing

The project shares only complete blocks.

If `block_size=4` and a prompt has 10 tokens:

```text
tokens:        0 1 2 3 | 4 5 6 7 | 8 9
full blocks:   block A |  block B | partial tail
shared later:  yes     |  yes     | no
```

The partial tail is not published to the prefix cache. This keeps mutation rules simple: cached blocks are read-only full blocks, and decode appends naturally allocate a new block when the shared prefix ends at a full block boundary.

## Hash Chain

`BlockManager.compute_block_hash_chain(token_ids, extra_hash=None)` computes one hash per full logical block.

The hash includes:

1. the previous block hash
2. the current block's token IDs
3. optional `extra_hash`

That makes the hash a chain, not just a hash of one isolated token block. The same token block at a different prefix position receives a different hash if the earlier context differs.

Example:

```text
block 0 hash = H(None, tokens[0:4], extra_hash)
block 1 hash = H(block0_hash, tokens[4:8], extra_hash)
block 2 hash = H(block1_hash, tokens[8:12], extra_hash)
```

The `extra_hash` parameter lets a caller separate otherwise identical token prefixes across incompatible contexts, such as different model adapters or cache namespaces.

## Lookup

`lookup_prefix_blocks(token_ids, extra_hash=None)` walks the hash chain from the first block until it misses.

If blocks 0 and 1 exist but block 2 does not:

```text
return PrefixCacheMatch(block_ids=[id0, id1], cached_token_count=2 * block_size)
```

The match is always the longest contiguous prefix. It does not skip a missing block and resume later.

## Allocation With Prefix Reuse

`allocate_sequence_with_prefix(seq_id, num_tokens, prefix_match)` creates a normal active sequence, but its first blocks point at existing physical blocks.

For a prompt of 10 tokens with 8 cached prefix tokens:

```text
new seq block table: [cached block 5, cached block 9, newly allocated block 2]
context length:      10
```

The method increments `ref_count` on cached blocks and allocates fresh blocks for the uncached suffix.

## Publishing

`publish_sequence_prefix_blocks(seq_id, token_ids, extra_hash=None)` runs after prefill finishes.

It publishes the sequence's full blocks into:

```text
prefix_cache:       hash -> block_id
block_hash_to_ids:  hash -> set(block_id)
```

It also marks each block:

```text
block.block_hash = hash
block.is_prefix_cached = True
```

Publishing happens after all layers have written their K/V. If a block were published before layer computation finished, another request could reuse incomplete cache contents.

## Reference Counts

`ref_count` is the safety mechanism for sharing.

Possible owners:

- an active sequence's block table
- the prefix cache metadata

When a sequence is freed, shared cached blocks may remain alive. If their active reference count falls to zero but they are still prefix-cached, they move into `cached_free_lru` instead of returning directly to `free_block_ids`.

## Cached-Free Blocks And Eviction

A cached-free block is not used by an active sequence, but it still contains reusable prefix K/V.

If ordinary free blocks run out, `evict_cached_block_if_needed` can reclaim one cached-free block:

1. Pop a cached block from the LRU queue.
2. Remove its hash mappings.
3. Clear `block_hash` and `is_prefix_cached`.
4. Zero K/V storage for every layer.
5. Return the block to the ordinary free list.

This gives prefix cache reuse opportunistic value without letting old cached prefixes permanently exhaust capacity.

## How `PagedDecoderLM.forward_prefill` Uses It

Before allocating prompt blocks, `forward_prefill` checks each sequence:

```text
match = block_manager.lookup_prefix_blocks(prompt_tokens)
```

Then it splits the batch:

- sequences with no prefix hit run normal prefill
- sequences with prefix hits are grouped by `cached_token_count`

Grouping keeps tensor shapes uniform for a batched suffix computation.

For a hit group, `_forward_prefill_group_with_prefix` computes only the suffix/work tokens while attention sees the cached prefix.

## Full-Prompt Cache Hit

If a prompt is entirely cached, the implementation still keeps one token of work so it can produce logits for the final position. It uses cached prefix K/V for earlier tokens and computes the final token's contribution.

This avoids returning an empty logits tensor while still reusing nearly all cached work.

## Prefix-Aware Attention

For a suffix token at local index `i`, the visible context length is:

```text
prefix_token_count + i + 1
```

The suffix token can attend to:

- all cached prefix tokens
- suffix tokens up to itself

It cannot attend to future suffix tokens.

The Python reference path gathers prefix K/V from the block manager and concatenates suffix K/V. The CUDA path reads prefix K/V from runtime cache blocks and suffix K/V from scratch arrays.

## Parallel Sampling And Beam Search Interpretation

The benchmark scripts use block sharing to model two common serving patterns:

- **Parallel sampling**: several completions share one prompt, then branch into independent decode continuations.
- **Beam search**: beams share a trunk before diverging, so block reuse can avoid cloning the entire prefix for every beam.

The project simulates these memory effects at block granularity. The implementation is not a full beam-search engine, but the memory accounting follows the same sharing principle.

## What Is Not Implemented

This project keeps the sharing model intentionally narrow:

- It does not share partial tail blocks.
- It does not implement arbitrary copy-on-write for partially filled shared blocks.
- It does not deduplicate blocks outside the prefix path.
- It does not implement a production scheduler around cache eviction.

Those choices make the educational implementation easier to verify.

## Tests To Read

Prefix behavior is covered mainly in:

- `tests/test_block_manager.py`
- `tests/test_paged_attention.py`
- `tests/test_parity.py`

Look for tests covering hash chains, lookup misses, prefix allocation, publication, eviction, repeated prompt reuse, and prefix prefill parity.
