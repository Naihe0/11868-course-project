# Glossary

This glossary is ordered roughly by the reading path in [../README.md](../README.md).

## MiniTorch And Tensor Terms

| Term | Meaning |
| --- | --- |
| Tensor | MiniTorch multidimensional array object used for model computation. |
| TensorData | Low-level storage, shape, and stride representation behind a `Tensor`. |
| Shape | Tuple of dimension sizes, for example `(batch, seq_len, n_embd)`. |
| Stride | Number of storage positions to jump when an index changes along a dimension. |
| Contiguous | A layout where tensor elements are stored in the expected row-major order for a `view`. |
| Backend | MiniTorch execution implementation such as `FastOps` or `CudaKernelOps`. |
| Module | MiniTorch class for layers with parameters and training/evaluation state. |
| Parameter | A trainable value owned by a `Module`. |
| `tensor_from_numpy` | Helper that wraps a NumPy array as a MiniTorch tensor. |
| `to_numpy` | Helper that copies/converts a MiniTorch tensor into a NumPy array. |

## Transformer Terms

| Term | Meaning |
| --- | --- |
| Decoder-only model | Language model that predicts next tokens using only previous tokens as context. |
| Token embedding | Learned vector for a token ID. |
| Positional embedding | Learned vector for a token position. |
| Hidden state | Per-token vector carried through the transformer layers. |
| Query (`Q`) | Vector representing what the current token asks for. |
| Key (`K`) | Vector representing what each token can match against. |
| Value (`V`) | Vector containing information copied according to attention weights. |
| Attention score | Dot product between query and key, scaled by `sqrt(head_dim)`. |
| Attention weight | Softmax-normalized score. |
| Multi-head attention | Attention split across several heads, each with smaller `head_dim`. |
| Causal mask | Mask that prevents a token from attending to future positions. |
| Logits | Unnormalized scores over vocabulary tokens. |
| Temperature | Sampling control that sharpens or flattens the probability distribution. |

## Inference And KV Cache Terms

| Term | Meaning |
| --- | --- |
| Prefill | First inference phase that processes the whole prompt and fills the KV cache. |
| Decode | Autoregressive phase that processes one new token at a time. |
| KV cache | Stored key/value vectors for previous tokens. |
| Context length | Number of tokens currently present for a sequence. |
| Contiguous KV cache | Cache that reserves one contiguous slab per sequence. |
| Paged KV cache | Cache that stores sequence KV in fixed-size physical blocks. |
| Internal fragmentation | Wasted token slots inside allocated blocks, usually in the tail block. |
| External fragmentation | Free blocks located between used blocks. In this project it is a simple diagnostic, not a full allocator coalescing model. |

## Block Manager Terms

| Term | Meaning |
| --- | --- |
| Physical block | A real storage block in `BlockManager.key_cache` and `value_cache`. |
| Logical block | A sequence-local block index computed from a token position. |
| Block table | Per-sequence mapping from logical block indices to physical block IDs. |
| `seq_id` | Identifier for an active sequence/request. |
| `block_id` | Identifier for a physical block in the global block pool. |
| Slot | Token position inside one physical block. |
| `num_filled` | Number of token slots currently occupied in a physical block. |
| `ref_count` | Number of active/cached owners of a physical block. Used for sharing. |
| Free list | List of ordinary unallocated physical block IDs. |
| Prefix cache | Mapping from full-block token hashes to physical blocks that can be reused. |
| Full block | A block containing exactly `block_size` tokens. Only full blocks are published to the prefix cache. |
| Hash chain | Prefix-cache hash where each block's hash depends on the previous block hash and current block tokens. |
| Copy-on-write | Sharing policy where a block is copied before mutation if more than one sequence references it. |
| LRU eviction | Least-recently-used eviction. This project uses a simple cached-free block LRU for reusable prefix blocks. |

## PagedAttention Terms

| Term | Meaning |
| --- | --- |
| Standard attention | Attention over contiguous key and value tensors. Used for prefill and correctness reference. |
| Paged attention | Attention that reads key/value vectors through block tables. |
| Gather | The process of reading scattered K/V blocks into logical token order. |
| Reference backend | Python/NumPy/MiniTorch implementation used for correctness. In flags this is `--decode-backend ref`. |
| CUDA backend | Custom compiled PagedAttention kernel used for decode. In flags this is `--decode-backend cuda`. |
| Stateless kernel call | CUDA call where host cache arrays and metadata are passed every time. |
| Stateful runtime | CUDA runtime object that keeps device-side cache buffers and updates only changed blocks/metadata. |

## CUDA Terms

| Term | Meaning |
| --- | --- |
| Kernel | Function running on the GPU. |
| Grid | Collection of CUDA thread blocks for one kernel launch. |
| Thread block | Group of threads that can synchronize and share memory. |
| Warp | Group of 32 NVIDIA GPU threads executing together. |
| Shared memory | Fast on-chip memory visible to threads in one block. |
| Warp reduction | Operation that combines values across a warp, such as sum or max. |
| Dynamic shared memory | Shared memory size chosen at kernel launch time. |
| CUDA runtime API | CUDA API layer used by `cudaMalloc`, `cudaMemcpy`, and kernel launches. |
| CUDA driver API | Lower-level CUDA API used by libraries such as numba. Import order can matter when both runtime and driver APIs are used. |

## Benchmark Terms

| Term | Meaning |
| --- | --- |
| Throughput | Tokens generated per second. |
| Latency | Time per generated token. |
| p50 | Median latency. Half of observations are below this value. |
| p95 | Tail latency. 95 percent of observations are below this value. |
| Speedup | Baseline time divided by optimized time. Larger is better. |
| Max batch size | Largest number of concurrent sequences that fit under a fixed KV block budget. |
| Memory savings ratio | `1 - paged_bytes / contiguous_bytes`. |
| Correctness tolerance | Maximum allowed numerical difference between implementation and reference outputs. |
