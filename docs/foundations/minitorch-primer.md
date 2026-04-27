# MiniTorch Primer For This Project

MiniTorch is an educational deep learning framework. This project does not need you to understand every MiniTorch file before reading the PagedAttention code, but it does rely on a few core ideas: tensors, backends, modules, parameters, and NumPy conversion.

## Why MiniTorch Matters Here

The PagedAttention code is not written in PyTorch. It uses MiniTorch's `Tensor` and `Module` APIs so it can plug into the course framework. That means the project has two kinds of arrays:

1. MiniTorch `Tensor` values, used for model computation.
2. NumPy arrays, used for the KV cache in `BlockManager` and as the bridge to ctypes/CUDA.

The common flow is:

```python
idx_np = np.array([[1, 2, 3]], dtype=np.float32)
idx = minitorch.tensor_from_numpy(idx_np, backend=backend)
logits = model.forward_prefill(idx, block_manager, seq_ids=[0])
logits_np = logits.to_numpy()
```

## Tensor

`minitorch/tensor.py` defines the user-facing `Tensor` object. A `Tensor` carries:

- Shape, such as `(batch, seq_len, n_embd)`.
- Backend, such as `TensorBackend(FastOps)` or `TensorBackend(CudaKernelOps)`.
- Methods such as `view`, `permute`, `contiguous`, `sum`, and `to_numpy`.
- Autodiff history, even though this project mostly runs inference.

The attention code depends heavily on reshaping and permuting tensors. For example, `PagedMultiHeadAttention.forward_prefill` starts with `(batch, seq_len, n_embd)` and reshapes projected Q/K/V into `(batch, seq_len, n_head, head_dim)`, then permutes to `(batch, n_head, seq_len, head_dim)` for attention.

## TensorData

`minitorch/tensor_data.py` handles lower-level storage details:

- `shape` describes logical dimensions.
- `strides` describe how to step through storage.
- `storage` is the flat backing array.
- Index helpers translate multidimensional indices into positions.

Most project code does not manipulate `TensorData` directly, but you see the effects whenever code calls `view`, `permute`, or `contiguous`. If a tensor is not laid out contiguously after a permutation, code often calls `.contiguous()` before another `view`.

## Backends

MiniTorch separates tensor semantics from implementation with `TensorBackend` in `minitorch/tensor_ops.py`.

The two relevant backends are:

| Backend | How it is created | Purpose |
| --- | --- | --- |
| CPU fast path | `minitorch.TensorBackend(minitorch.FastOps)` | Default for most tests and examples. |
| CUDA tensor backend | `minitorch.TensorBackend(minitorch.CudaKernelOps)` | Uses compiled MiniTorch CUDA primitives for tensor operations. |

Do not confuse the MiniTorch tensor backend with the PagedAttention decode backend:

- `--backend cpu|cuda` controls MiniTorch tensor operations.
- `--decode-backend ref|cuda` controls whether decode attention uses Python `paged_attention_ref` or the custom `src/paged_attention.cu` kernel.

Those choices can be combined, though practical CUDA use requires compiled shared libraries.

## Module And Parameter

`minitorch/module.py` defines `Module` and `Parameter`. This is MiniTorch's equivalent of the PyTorch `nn.Module` pattern.

Project-specific modules inherit from `Module`:

- `PagedMultiHeadAttention` in `minitorch/paged_attention.py`
- `FeedForward`, `PagedTransformerLayer`, and `PagedDecoderLM` in `minitorch/transformer.py`

The inherited layers in `minitorch/modules_basic.py` also use `Module`:

- `Linear`
- `Embedding`
- `Dropout`
- `LayerNorm1d`

These are the building blocks of the transformer.

## Linear

`Linear` applies an affine projection. In this project it is used for:

- `q_proj`: input embedding to query vector.
- `k_proj`: input embedding to key vector.
- `v_proj`: input embedding to value vector.
- `out_proj`: attention output back to model dimension.
- `lm_head`: final hidden state to vocabulary logits.
- Feed-forward network input and output projections.

The attention projections are all shape-preserving at the model dimension: `n_embd -> n_embd`. The code then splits that dimension into `n_head * head_dim`.

## Embedding

`Embedding` maps token IDs and position IDs into vectors.

`PagedDecoderLM._embed` does:

1. `token_embeddings(idx)` for token identity.
2. `position_embeddings(pos_ids)` for token position.
3. Adds them and applies dropout.

During prefill, `start_pos=0`. During decode, `start_pos` is the position of the one new token.

## LayerNorm, Dropout, GELU, Softmax

The transformer stack uses:

- `LayerNorm1d` before attention and before the feed-forward network.
- `Dropout`, usually set to `0.0` in benchmarks for deterministic behavior.
- `GELU` in the feed-forward network.
- `softmax` in `standard_attention` to turn attention scores into attention weights.

The key implementation point: `standard_attention` is written using MiniTorch tensor operations, while `BlockManager` cache storage is NumPy.

## NumPy Conversion Boundary

The project crosses the Tensor/NumPy boundary in several important places:

- `BlockManager.key_cache` and `BlockManager.value_cache` are lists of NumPy arrays.
- `PagedMultiHeadAttention._write_kv_batch_to_cache` converts K/V tensors to NumPy and writes them into those arrays.
- `paged_attention_ref` gathers NumPy cache blocks and wraps them back into MiniTorch tensors for reference attention.
- `PagedAttentionKernel` converts MiniTorch tensors and arrays into contiguous NumPy arrays for ctypes calls.

This boundary is one reason the code is readable for education: the memory manager is plain NumPy, while the model computation remains MiniTorch.

## Autodiff In This Project

MiniTorch supports automatic differentiation through `autodiff.py`, `tensor_functions.py`, and scalar/tensor history objects. PagedAttention inference does not rely on training-time gradients. You mainly need to know that tensors still carry enough framework metadata to behave like normal MiniTorch tensors.

## Practical Checklist For Reading Code

When reading a tensor expression in this repo, ask:

1. Is this a MiniTorch `Tensor` or a NumPy array?
2. What is the shape at this point?
3. Did a `permute` make the tensor non-contiguous before a `view`?
4. Which backend is responsible for this computation?
5. Is this path used during prefill, decode, or both?
