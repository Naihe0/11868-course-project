# Memory And Capacity

PagedAttention is a memory management technique. Its main benefit is not that it changes the math of attention. Its benefit is that it changes how KV cache memory is reserved, used, shared, and reclaimed.

## KV Cache Bytes Per Token

For one token in one layer:

```text
bytes_per_token_per_layer = 2 * n_head * head_dim * dtype_size
```

The factor of `2` is for key and value.

For all layers:

```text
bytes_per_token = 2 * n_layers * n_head * head_dim * dtype_size
```

Since `n_embd = n_head * head_dim`, this can also be written as:

```text
bytes_per_token = 2 * n_layers * n_embd * dtype_size
```

## Paged Reservation

Paged KV memory is reserved in full blocks. If a sequence has length `L` and block size `B`, it needs:

```text
ceil(L / B) blocks
```

Reserved token slots for that sequence:

```text
ceil(L / B) * B
```

The unused slots in the last block are internal fragmentation.

Example with `block_size=16` and `seq_len=33`:

```text
ceil(33 / 16) = 3 blocks
reserved slots = 48
used slots = 33
wasted tail slots = 15
```

## Static Contiguous Reservation

A simple contiguous cache often reserves a maximum sequence length for every active sequence:

```text
reserved_slots = batch_size * max_seq_len
```

This is easy to index but can be much larger than the tokens actually present. Paged allocation instead tracks the current lengths.

## Memory Accounting In `BlockManager`

`compute_kv_memory(max_seq_len)` reports:

- `paged_bytes`: blocks currently used by active or cached state.
- `contiguous_bytes`: naive contiguous reservation for active sequences at `max_seq_len`.
- `memory_savings_ratio`: `1 - paged_bytes / contiguous_bytes`.

The paged formula is:

```text
used_blocks * block_size * 2 * n_head * head_dim * dtype_size * num_layers
```

The contiguous comparison is:

```text
active_sequences * max_seq_len * 2 * n_head * head_dim * dtype_size * num_layers
```

## Internal Fragmentation

Internal fragmentation is waste inside allocated blocks:

```text
internal = (allocated_slots - used_slots) / allocated_slots
```

It depends strongly on block size and sequence length distribution.

Tradeoff:

- Smaller blocks reduce tail waste.
- Larger blocks reduce block-table overhead and can improve locality.

The benchmark fragmentation sweep intentionally uses non-aligned sequence lengths such as `33`, `65`, `100`, and `130` so tail waste is visible.

## External Fragmentation

The project reports a simple external-fragmentation diagnostic: it looks at free block IDs between used block IDs.

This is not a full production allocator metric because blocks are fixed-size and do not require coalescing. It is still useful for observing whether used blocks are scattered throughout the physical pool.

## Capacity Under A Block Budget

With a fixed number of KV blocks, max batch size roughly follows:

```text
max_batch ~= num_blocks / ceil(seq_len / block_size)
```

For example, with `num_blocks=512`:

```text
block_size=16, seq_len=128 -> 8 blocks per sequence -> about 64 sequences
block_size=8,  seq_len=128 -> 16 blocks per sequence -> about 32 sequences
```

This is why a larger block size can support more sequences for lengths that divide neatly, even though it may waste more tail slots for non-aligned lengths.

## Runtime Cache Versus Allocator Capacity

There are two related but separate capacities:

1. Host-side `BlockManager` capacity: `num_blocks` physical blocks.
2. CUDA runtime capacity: maximum batch and maximum block-table width allocated in `PagedAttentionRuntime`.

If the Python runtime needs a larger batch or wider block table than the current CUDA runtime supports, it recreates the runtime with larger capacity.

## Prefix Sharing And Effective Capacity

Prefix sharing improves effective capacity because several sequences can point at the same physical prefix blocks.

Without sharing:

```text
N sequences * prefix_blocks
```

With sharing:

```text
1 shared prefix copy + N suffixes
```

This is the memory effect measured by the parallel-sampling and beam-search benchmark experiments.

## Contiguous KV Baseline

`project/contiguous_kv_baseline.py` implements a HuggingFace-style contiguous KV cache around the same `PagedDecoderLM` weights. It reserves arrays shaped:

```text
(max_batch_size, n_head, max_seq_len, head_dim)
```

for K and V in each layer.

This is a fairer decode-speed baseline than a no-cache model because it avoids recomputing the entire prompt every step. It still reserves static contiguous memory, so it does not get the memory benefits of paging.

## Benchmark Families

`project/run_benchmark.py` produces general benchmark CSVs:

- throughput and latency
- fragmentation sweep
- max-batch search
- correctness comparison
- optional no-cache baseline comparison
- optional prefix-cache prefill comparison

`project/run_rigorous_benchmark.py` produces six report-grade experiments:

1. Memory breakdown on a realistic non-aligned workload.
2. Capacity curve under a fixed KV block budget.
3. Decode speed against no-cache and contiguous-KV baselines.
4. Prefix prefill speedup across share ratios.
5. Parallel sampling memory.
6. Beam search memory.

The rigorous script writes CSVs under `benchmarks/results_rigorous/`, and `project/plot_rigorous_figures.py` renders poster/report figures under `benchmarks/report_figures_v2/`.

## Interpreting Results Carefully

When reading benchmark output, keep these distinctions in mind:

- CPU reference decode is useful for correctness and educational visibility, not production speed.
- CUDA decode depends on successful kernel compilation and CUDA availability.
- No-cache baselines measure a different algorithmic cost model from paged decode.
- Contiguous KV baselines are better for decode-speed fairness but still over-reserve memory.
- Prefix-cache wins depend on shared prefix length rounded down to full blocks.
- Memory savings are largest when actual sequence lengths are much shorter than the reserved maximum.
