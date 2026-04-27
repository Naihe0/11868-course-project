# API Reference

This is a project-specific API map. It focuses on symbols introduced or directly used by the PagedAttention implementation.

## `minitorch.block_manager`

### `KVBlock`

Metadata for one physical KV cache block.

Constructor fields:

```python
KVBlock(block_id, block_size, n_head, head_dim)
```

Important attributes:

- `block_id`
- `block_size`
- `ref_count`
- `num_filled`
- `n_head`
- `head_dim`
- `block_hash`
- `is_prefix_cached`

Properties:

- `is_full`
- `num_empty_slots`

### `BlockTable`

Per-sequence logical-to-physical block map.

```python
BlockTable(seq_id, block_ids=None)
append_block(block_id)
```

### `PrefixCacheMatch`

Dataclass returned by prefix lookup.

Fields:

- `block_ids`
- `cached_token_count`

### `BlockManager`

Owns all physical KV blocks and per-sequence block tables.

Constructor:

```python
BlockManager(
    num_blocks,
    block_size,
    n_head,
    head_dim,
    num_layers=1,
    cache_dtype=np.float32,
)
```

Core allocation:

- `allocate_block() -> KVBlock`
- `free_block(block_id) -> None`
- `allocate_blocks_for_sequence(seq_id, num_tokens) -> BlockTable`
- `append_token_to_sequence(seq_id) -> KVBlock`
- `free_sequence(seq_id) -> None`

K/V access:

- `write_kv_slot(layer_id, block_id, slot_idx, key, value) -> None`
- `write_token_kv(seq_id, layer_id, key, value) -> tuple[int, int]`
- `get_physical_location(seq_id, token_index) -> tuple[int, int]`
- `get_block_table_array(seq_ids, pad_value=-1) -> np.ndarray`
- `get_context_len(seq_id) -> int`

Metrics:

- `get_num_free_blocks() -> int`
- `get_num_used_blocks() -> int`
- `compute_fragmentation() -> dict`
- `compute_kv_memory(max_seq_len) -> dict`

Prefix cache:

- `compute_block_hash_chain(token_ids, extra_hash=None) -> list[str]`
- `lookup_prefix_blocks(token_ids, extra_hash=None) -> PrefixCacheMatch`
- `allocate_sequence_with_prefix(seq_id, num_tokens, prefix_match) -> BlockTable`
- `publish_sequence_prefix_blocks(seq_id, token_ids, extra_hash=None) -> None`
- `evict_cached_block_if_needed() -> bool`

## `minitorch.paged_attention`

### `standard_attention`

```python
standard_attention(query, key, value, mask=None)
```

Contiguous scaled dot-product attention.

Expected shapes:

```text
query: (batch, n_head, seq_q, head_dim)
key:   (batch, n_head, seq_kv, head_dim)
value: (batch, n_head, seq_kv, head_dim)
mask:  broadcastable to (batch, n_head, seq_q, seq_kv)
```

Returns:

```text
(batch, n_head, seq_q, head_dim)
```

### `paged_attention_ref`

```python
paged_attention_ref(
    query,
    key_cache,
    value_cache,
    block_tables,
    context_lens,
    block_size,
    layer_id=0,
    backend=None,
)
```

Reference PagedAttention implementation. Gathers K/V vectors through block tables and calls `standard_attention`.

### `PagedAttentionKernel`

ctypes wrapper around `minitorch/cuda_kernels/paged_attention.so`.

Important methods:

- `ensure_runtime(num_blocks, block_size, n_head, head_dim, max_batch, max_blocks_per_seq)`
- `upload_layer_cache(key_cache, value_cache)`
- `update_slot(block_id, slot_idx, key, value)`
- `upload_block(block_id, key_block, value_block)`
- `update_metadata(block_tables, context_lens)`
- `forward(query, key_cache=None, value_cache=None, block_tables=None, context_lens=None, block_size=None, max_context_len=None)`
- `close()`

### `PagedMultiHeadAttention`

```python
PagedMultiHeadAttention(
    n_embd,
    n_head,
    block_size,
    layer_id=0,
    p_dropout=0.0,
    backend=None,
    decode_backend="ref",
    compare_to_ref=False,
    compare_tolerance=1e-4,
)
```

Important methods:

- `forward_prefill(x, block_manager, seq_ids)`
- `forward_prefill_with_prefix_batch(x, block_manager, seq_ids, prefix_token_count, cached_token_count, write_kv_to_cache=True)`
- `forward_decode(x, block_manager, seq_ids)`
- `close_decode_runtime()`

## `minitorch.transformer`

### `FeedForward`

```python
FeedForward(n_embd, p_dropout=0.0, backend=None)
```

Transformer MLP block: linear, GELU, linear, dropout.

### `PagedTransformerLayer`

```python
PagedTransformerLayer(
    n_embd,
    n_head,
    block_size,
    layer_id,
    p_dropout=0.0,
    backend=None,
    decode_backend="ref",
    compare_to_ref=False,
    compare_tolerance=1e-4,
)
```

Important methods:

- `forward_prefill(x, block_manager, seq_ids)`
- `forward_decode(x, block_manager, seq_ids)`

### `PagedDecoderLM`

```python
PagedDecoderLM(
    n_vocab,
    n_embd,
    n_head,
    n_positions,
    n_layers,
    block_size,
    p_dropout=0.0,
    backend=None,
    decode_backend="ref",
    compare_to_ref=False,
    compare_tolerance=1e-4,
)
```

Important methods:

- `forward_prefill(idx, block_manager, seq_ids)`
- `forward_decode(idx, block_manager, seq_ids, start_pos=0)`
- `close_decode_runtime()`
- `generate(model, idx, max_new_tokens, block_manager, temperature=1.0)`

## Project CLIs

### `project/run_inference.py`

Synthetic generation driver.

Common flags:

- `--backend cpu|cuda`
- `--decode-backend ref|cuda`
- `--compare-to-ref`
- `--check-correctness`
- `--batch-size`
- `--prompt-len`
- `--max-new-tokens`
- `--block-size`
- `--num-kv-blocks`

### `project/run_benchmark.py`

General benchmark driver.

Common flags:

- `--batch-sizes ...`
- `--seq-lengths ...`
- `--block-sizes ...`
- `--max-new-tokens`
- `--output-dir`
- `--backend cpu|cuda`
- `--decode-backend ref|cuda`
- `--compare-baseline`
- `--compare-prefix-cache`
- `--skip-correctness`
- `--skip-max-batch`
- `--skip-frag-sweep`

### `project/run_rigorous_benchmark.py`

Six-experiment benchmark driver used for report figures.

Common flags:

- `--backend auto|cpu|cuda`
- `--decode-backend auto|ref|cuda`
- `--output-dir`
- `--warmup-trials`
- `--timed-trials`
- `--skip-memory`
- `--skip-capacity`
- `--skip-decode`
- `--skip-prefix`
- `--skip-sharing`

### Plot scripts

- `project/plot.py`: reads `benchmarks/results/*.csv`, writes `benchmarks/plots/*.png`.
- `project/plot_rigorous_figures.py`: reads `benchmarks/results_rigorous/*.csv`, writes `benchmarks/report_figures_v2/figure*.png`.
