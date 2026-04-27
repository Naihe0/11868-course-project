# CUDA Kernel And Runtime Walkthrough

The CUDA path has two halves:

1. `PagedAttentionKernel` in `minitorch/paged_attention.py`, which loads a shared library and exposes Python methods.
2. `src/paged_attention.cu`, which implements the GPU kernels and a small stateful runtime.

The CUDA implementation is optional for understanding correctness. The reference Python path explains the algorithm. The CUDA path explains how the same algorithm is made faster and how device-side cache state is synchronized.

## Build Output

`compile_cuda.sh` compiles:

```text
minitorch/cuda_kernels/combine.so
minitorch/cuda_kernels/softmax_kernel.so
minitorch/cuda_kernels/layernorm_kernel.so
minitorch/cuda_kernels/paged_attention.so
```

The PagedAttention library is compiled with:

```bash
nvcc -std=c++17 -o minitorch/cuda_kernels/paged_attention.so \
  --shared src/paged_attention.cu -Xcompiler -fPIC --cudart shared
```

The `--cudart shared` flag matters because the Python process may already contain CUDA runtime state from PyTorch and driver state from numba. Using the process-wide CUDA runtime avoids a separate statically linked runtime copy that cannot see the initialized device state.

## CUDA Context Import Order

`minitorch/__init__.py` tries to initialize PyTorch CUDA before importing modules that may touch `numba.cuda`. This avoids a runtime/driver API ordering failure where CUDA runtime calls such as `cudaSetDevice` or `cudaMalloc` later report that no CUDA-capable device exists.

The practical lesson: if CUDA behavior looks impossible, check import order and whether the shared library was built with `--cudart shared`.

## Python Wrapper: `PagedAttentionKernel`

`PagedAttentionKernel` loads `minitorch/cuda_kernels/paged_attention.so` with `ctypes.CDLL`. It registers ctypes signatures for two modes:

1. Stateless forward:
   - `paged_attention_forward`
2. Stateful runtime API:
   - `paged_attention_runtime_create`
   - `paged_attention_runtime_destroy`
   - `paged_attention_runtime_upload_layer_cache`
   - `paged_attention_runtime_update_slot`
   - `paged_attention_runtime_upload_block`
   - `paged_attention_runtime_update_metadata`
   - `paged_attention_runtime_forward`
   - `paged_attention_runtime_prefill_with_prefix_forward`

The wrapper converts `Tensor` or array inputs into contiguous NumPy arrays before passing pointers to ctypes.

## Stateless Mode

Stateless mode passes the full host-side cache and metadata into every call:

```python
kernel.forward(
    query,
    key_cache,
    value_cache,
    block_tables,
    context_lens,
    block_size,
    max_context_len,
)
```

This is straightforward but copies cache data each call inside `paged_attention_forward`. It is useful as a simple ABI and for tests.

## Stateful Runtime Mode

Stateful mode creates a `PagedAttentionRuntime` object on the C++ side. It owns device buffers:

```text
d_key_cache
d_value_cache
d_block_tables
d_context_lens
d_query
d_output
d_prefill_query
d_prefill_suffix_key
d_prefill_suffix_value
d_prefill_output
```

The Python side then updates only what changed:

- Upload an entire block with `upload_block`.
- Update one decode slot with `update_slot`.
- Upload current block tables and context lengths with `update_metadata`.
- Run `forward(query, max_context_len=...)` without passing host cache arrays.

`PagedMultiHeadAttention` tracks `_runtime_valid_block_ids` so it does not upload blocks that are already valid on the device. It clears that set when the runtime is recreated or when a different `BlockManager` object is used.

## Runtime Creation

`paged_attention_runtime_create` receives:

```text
num_blocks
block_size
n_head
head_dim
max_batch
max_blocks_per_seq
```

It allocates device buffers sized for the maximum batch and maximum block-table width. The Python wrapper keeps `_runtime_config` and reuses a runtime when the existing capacity is compatible.

## Main Decode Kernel

`paged_attention_v1_kernel` computes attention for decode queries.

Launch shape:

```text
grid  = (batch_size, n_head)
block = max(WARP_SIZE, min(head_dim, 1024)) threads
```

Each CUDA thread block handles one `(sequence, head)` pair.

Inputs:

```text
output:       (batch, n_head, head_dim)
query:        (batch, n_head, head_dim)
key_cache:    (num_blocks, block_size, n_head, head_dim)
value_cache:  (num_blocks, block_size, n_head, head_dim)
block_tables: (batch, max_blocks_per_seq)
context_lens: (batch)
```

## Token Lookup In The Kernel

For each context token, the kernel performs the same translation as the Python reference:

```cuda
int logical_block = token_idx / block_size;
int slot_in_block = token_idx % block_size;
int physical_block = seq_block_table[logical_block];
```

Then it computes the pointer into K or V:

```cuda
cache + physical_block * (block_size * n_head * head_dim)
      + slot_in_block * (n_head * head_dim)
      + head_idx * head_dim
```

This pointer math is the CUDA version of the block table lookup.

## Three-Pass Attention

The decode kernel uses dynamic shared memory split into:

```text
logits[max_context_len]
out_accum[head_dim]
q_shared[head_dim]
warp_scratch[num_warps]
```

The algorithm:

1. **Score pass**: compute `Q dot K` for each context token and store logits.
2. **Softmax pass**: subtract the global max for numerical stability, exponentiate, reduce the sum, and normalize logits into weights.
3. **Value pass**: accumulate `sum(weight[token] * V[token])` for each output dimension.

`warp_reduce_max` and `warp_reduce_sum` use `__shfl_xor_sync` to reduce values within a warp. Warp leaders write into shared memory so the block can reduce across warps.

## Prefix-Prefill Kernel

`paged_attention_prefill_with_prefix_kernel` handles a batch of suffix rows whose prefix K/V is already cached.

Launch shape:

```text
grid = (batch_size * work_len, n_head)
```

Each row corresponds to one suffix token position in one batch item. Its context is:

```text
cached prefix tokens + suffix tokens up to this local row
```

For `token_idx < prefix_token_count`, it reads from the paged runtime cache. For suffix positions, it reads from the in-flight suffix K/V arrays passed for this call.

This lets prefix-aware prefill avoid recomputing attention over cached prompt blocks while still applying a causal mask over the full prefix-plus-suffix context.

## Host-Callable Functions

The `extern "C"` section exposes stable C symbols for ctypes.

Important functions:

| Function | Purpose |
| --- | --- |
| `paged_attention_runtime_create` | Allocates the runtime and device buffers. |
| `paged_attention_runtime_destroy` | Frees runtime buffers. |
| `paged_attention_runtime_upload_layer_cache` | Uploads the entire layer cache. Useful but heavy. |
| `paged_attention_runtime_update_slot` | Copies one `(n_head, head_dim)` K/V slot to device cache. |
| `paged_attention_runtime_upload_block` | Copies one whole physical block to device cache. |
| `paged_attention_runtime_update_metadata` | Copies block tables and context lengths to device. |
| `paged_attention_runtime_forward` | Runs decode attention using runtime-resident cache. |
| `paged_attention_runtime_prefill_with_prefix_forward` | Runs prefix-aware prefill over cached prefix plus suffix rows. |
| `paged_attention_forward` | Stateless API that allocates/copies/runs/frees inside one call. |

## How Python Keeps Runtime State Correct

`PagedMultiHeadAttention.forward_decode` does this in CUDA mode:

1. Sync runtime blocks for the prior context length.
2. Write the new K/V slot on the host `BlockManager`.
3. Call `update_slot` so the device cache gets the new token.
4. Update metadata with current block tables and context lengths.
5. Call runtime forward.

The ordering matters. The new token's K/V must be present before attention reads the full context including that token.

## Validation

The CUDA path is validated by tests in `tests/test_paged_attention.py` when CUDA and the compiled shared library are available:

- kernel output against `paged_attention_ref`
- batched kernel output
- runtime block reuse within one manager
- runtime reupload after manager switch
- runtime resync after reference prefill

For command-line runs, `--compare-to-ref` enables a runtime comparison between CUDA decode and Python reference decode.

## Limitations And Reading Notes

- The kernel is intentionally readable and educational. It is not a full vLLM production kernel.
- The V1 kernel uses one thread block per `(sequence, head)` pair, which is simple but not ideal for very long contexts.
- Dynamic shared memory scales with `max_context_len`, so very large contexts need care.
- CUDA availability depends on the local environment, compiled `.so` files, and import/runtime order.
