# Block Manager Walkthrough

`minitorch/block_manager.py` is the heart of this project. It turns a KV cache from one large contiguous buffer per sequence into a pool of fixed-size physical blocks. Each active sequence keeps a block table that maps its logical token order to those physical blocks.

## The Problem It Solves

During generation, every active sequence needs historical K/V vectors for every layer. A naive cache reserves memory for the maximum possible sequence length. That is simple, but it wastes memory when real prompts are shorter or have uneven lengths.

The block manager solves this by allocating blocks only when tokens exist:

```text
sequence tokens:      0  1  2  3 | 4  5
logical blocks:             0    |   1
block table:          [physical block 7, physical block 2]
physical cache pool:  block 0, block 1, block 2, ..., block 7, ...
```

The model sees a sequence of tokens. The allocator sees physical blocks.

## Data Structures

### `KVBlock`

`KVBlock` is metadata for one physical block. It does not store the actual K/V vectors. The actual vectors live in `BlockManager.key_cache` and `BlockManager.value_cache`.

Important fields:

- `block_id`: physical block index.
- `block_size`: number of token slots per block.
- `ref_count`: how many active or cached owners reference this block.
- `num_filled`: how many token slots are valid.
- `n_head`, `head_dim`: shape metadata for K/V vectors.
- `block_hash`, `is_prefix_cached`: prefix-cache metadata.

Useful properties:

- `is_full`: true when `num_filled >= block_size`.
- `num_empty_slots`: remaining slots in the block.

### `BlockTable`

`BlockTable` maps one sequence's logical blocks to physical block IDs.

```python
BlockTable(seq_id=10, block_ids=[7, 2, 11])
```

This means:

```text
sequence 10 logical block 0 -> physical block 7
sequence 10 logical block 1 -> physical block 2
sequence 10 logical block 2 -> physical block 11
```

`append_block` adds a physical block to the end of that sequence's logical block list.

### `PrefixCacheMatch`

`PrefixCacheMatch` is returned by `lookup_prefix_blocks`. It contains:

- `block_ids`: physical blocks that can be reused.
- `cached_token_count`: number of prompt tokens covered by those blocks.

Because only full blocks are cached, `cached_token_count` is always `len(block_ids) * block_size`.

### `BlockManager`

`BlockManager` owns the global KV cache and all active sequence metadata.

Key fields:

```text
blocks:             block_id -> KVBlock metadata
free_block_ids:     ordinary free physical blocks
block_tables:       seq_id -> BlockTable
context_lens:       seq_id -> number of tokens currently in the sequence
key_cache[layer]:   NumPy array shaped (num_blocks, block_size, n_head, head_dim)
value_cache[layer]: NumPy array shaped (num_blocks, block_size, n_head, head_dim)
```

There is one K cache and one V cache per transformer layer. That is why `BlockManager` requires an explicit `num_layers`.

## Cache Shape

For each layer:

```text
key_cache[layer].shape == (num_blocks, block_size, n_head, head_dim)
value_cache[layer].shape == (num_blocks, block_size, n_head, head_dim)
```

Indexing one token's key looks like:

```python
key_cache[layer][block_id, slot_idx, :, :]
```

This returns an array shaped `(n_head, head_dim)`.

## Allocation Workflow

### `allocate_block`

`allocate_block` reserves one physical block:

1. If the ordinary free list is empty, it tries `evict_cached_block_if_needed`.
2. If there is still no free block, it raises `RuntimeError`.
3. It pops the first free block ID.
4. It resets metadata: `ref_count = 1`, `num_filled = 0`.
5. It zeroes that block's K/V cache slices for every layer.
6. It returns the `KVBlock` metadata object.

The free list is simple and deterministic, which makes tests easier to reason about.

### `allocate_blocks_for_sequence`

This method reserves enough blocks for an initial prompt length:

```python
num_blocks = ceil(num_tokens / block_size)
```

For a sequence with `num_tokens=10` and `block_size=4`, it allocates 3 blocks with `num_filled` values `4`, `4`, and `2`.

It also creates:

- `block_tables[seq_id]`
- `context_lens[seq_id] = num_tokens`

If allocation fails partway through, it rolls back any blocks already allocated for that sequence. That keeps allocator state consistent after OOM.

### `append_token_to_sequence`

Decode reserves one new token slot per sequence before each layer writes K/V. `append_token_to_sequence` handles that reservation:

1. If the sequence has no blocks yet, allocate one.
2. Otherwise, inspect the last block in the sequence's block table.
3. If the last block is full, allocate a new block and append it.
4. Increment `num_filled` for the chosen block.
5. Increment `context_lens[seq_id]`.

The method returns the `KVBlock` that now contains the reserved slot.

### `free_sequence`

When a request finishes, `free_sequence` releases its block table:

1. Decrement `ref_count` for each referenced physical block.
2. If a block reaches zero references and is prefix-cached, move it to `cached_free_lru` instead of zeroing it.
3. If a block reaches zero references and is not cached, zero its K/V slices and append it to `free_block_ids`.
4. Remove the sequence's `block_table`, `context_len`, and prefix-cache info.

The important distinction is that cached blocks can outlive the sequence that created them.

## Logical To Physical Mapping

`get_physical_location(seq_id, token_index)` performs the key translation:

```python
logical_block = token_index // block_size
slot_idx = token_index % block_size
block_id = block_tables[seq_id].block_ids[logical_block]
return block_id, slot_idx
```

Every K/V cache read or write eventually relies on this mapping.

## Writing K/V

### `write_kv_slot`

`write_kv_slot` writes one token's K and V vectors into a known physical slot.

Expected shapes:

```text
key.shape   == (n_head, head_dim)
value.shape == (n_head, head_dim)
```

It writes:

```python
key_cache[layer][block_id, slot_idx, :, :] = key
value_cache[layer][block_id, slot_idx, :, :] = value
```

### `write_token_kv`

`write_token_kv` combines append and write:

1. Reserve a new slot with `append_token_to_sequence`.
2. Compute `slot_idx = block.num_filled - 1`.
3. Write K/V with `write_kv_slot`.
4. Return `(block_id, slot_idx)`.

The current transformer path usually reserves slots in `PagedDecoderLM.forward_decode` and then writes layer-specific K/V inside `PagedMultiHeadAttention.forward_decode`. The helper remains useful for tests and direct use.

## Kernel Metadata

CUDA needs block tables as a rectangular `int32` array. `get_block_table_array(seq_ids, pad_value=-1)` builds that array:

```text
seq 0 block ids: [3, 4]
seq 1 block ids: [7]

array:
[[ 3,  4],
 [ 7, -1]]
```

The kernel only reads positions covered by each sequence's `context_lens`, so padded entries should not be used.

## Prefix Cache

The prefix-cache methods allow future requests to reuse complete blocks from earlier requests.

### `compute_block_hash_chain`

This method hashes only full logical blocks. A tail partial block is skipped.

For each full block:

```text
hash_i = sha256(parent_hash, block_tokens, extra_hash)
```

The parent hash makes the same token block hash differently if it appears after a different prefix.

### `lookup_prefix_blocks`

This computes the hash chain for a prompt and walks it from the beginning until the first miss. It returns the longest contiguous cached prefix.

### `allocate_sequence_with_prefix`

This creates a new block table where the prefix physical blocks are reused:

1. Append each prefix block to the new sequence's table.
2. Increment each reused block's `ref_count`.
3. Allocate new suffix blocks for tokens not covered by the cached prefix.
4. Store a `PrefixCacheMatch` in `seq_prefix_cache_info`.

Because cached prefix blocks are full blocks, normal decode appends allocate a new block after them instead of mutating a shared partial block.

### `publish_sequence_prefix_blocks`

After prefill finishes writing K/V for every layer, this method publishes the sequence's full blocks into the prefix cache. Publishing too early would expose incomplete K/V to later requests, so `PagedDecoderLM.forward_prefill` publishes only after all transformer layers finish.

### `evict_cached_block_if_needed`

If ordinary free blocks run out, the allocator can evict a zero-reference cached block from `cached_free_lru`. Eviction removes hash mappings, clears cached metadata, zeroes K/V slices, and returns the block to the ordinary free list.

## Fragmentation And Memory Metrics

### `compute_fragmentation`

Internal fragmentation is:

```text
(allocated_token_slots - used_token_slots) / allocated_token_slots
```

Example: `block_size=16`, `seq_len=33` uses 3 blocks, so 48 slots are allocated and 33 are used. Internal fragmentation is `15 / 48`.

External fragmentation is a simple diagnostic in this code: among used block IDs, it counts free blocks between the lowest and highest used block and divides by the number of free blocks.

### `compute_kv_memory`

This compares bytes allocated by the paged cache against a naive contiguous baseline that reserves `max_seq_len` tokens for every active sequence.

Paged bytes:

```text
num_used_blocks * block_size * 2 * n_head * head_dim * dtype_size * num_layers
```

Contiguous bytes:

```text
num_active_sequences * max_seq_len * 2 * n_head * head_dim * dtype_size * num_layers
```

The reported memory savings ratio is:

```text
1 - paged_bytes / contiguous_bytes
```

## Invariants To Keep In Mind

- Every active `seq_id` has one `BlockTable` and one `context_lens` entry.
- `context_lens[seq_id]` is the number of logical tokens in that sequence.
- `num_filled` on each physical block tracks valid slots in that block.
- `key_cache` and `value_cache` are indexed by layer, physical block, slot, head, and head dimension.
- Prefix-cache blocks can remain allocated after the original sequence is freed.
- Full-block prefix reuse is safe because the reused blocks are logically read-only.

## Tests That Define Behavior

`tests/test_block_manager.py` is the best executable companion to this file. It covers:

- initial free/used state
- unique block allocation
- OOM errors
- sequence allocation and rollback
- appending tokens within a block and across block boundaries
- freeing sequences
- K/V writes into global cache
- block table padding
- physical location lookup
- fragmentation metrics
- prefix hash, lookup, allocation, publish, and eviction behavior
