# Design Document: PagedAttention in MiniTorch

## 1. Overview

This document describes the design of PagedAttention integrated into MiniTorch.
The implementation consists of three main components:

1. **Block Manager** — manages KV cache memory in fixed-size blocks
2. **PagedAttention Kernel** — CUDA kernel for attention over non-contiguous blocks
3. **Transformer Integration** — modified transformer supporting prefill + decode

## 2. Architecture

```
┌─────────────────────────────────────────────────┐
│                PagedDecoderLM                    │
│  ┌───────────────────────────────────────────┐  │
│  │         PagedTransformerLayer (×N)         │  │
│  │  ┌─────────────────────────────────────┐  │  │
│  │  │     PagedMultiHeadAttention          │  │  │
│  │  │  ┌──────────┐  ┌─────────────────┐  │  │  │
│  │  │  │ Standard  │  │ Paged Attention │  │  │  │
│  │  │  │ Attention │  │ (CUDA kernel)   │  │  │  │
│  │  │  │ (prefill) │  │ (decode)        │  │  │  │
│  │  │  └──────────┘  └────────┬────────┘  │  │  │
│  │  └─────────────────────────┼───────────┘  │  │
│  └────────────────────────────┼──────────────┘  │
│                               │                  │
│  ┌────────────────────────────▼──────────────┐  │
│  │             Block Manager                  │  │
│  │  ┌────────┐ ┌────────┐ ┌────────┐        │  │
│  │  │Block 0 │ │Block 1 │ │Block 2 │ ...    │  │
│  │  │(K,V)   │ │(K,V)   │ │(K,V)   │        │  │
│  │  └────────┘ └────────┘ └────────┘        │  │
│  │         Physical Block Pool               │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

## 3. Block Manager Design

### 3.1 Block Structure
- Each `KVBlock` stores `block_size` token slots for K and V vectors
- Shape: `(block_size, n_head, head_dim)` for both K and V
- Tracks `ref_count` for potential sharing (e.g., beam search)

### 3.2 Block Table
- Each sequence has a `BlockTable` mapping logical → physical block ids
- Analogous to a page table in virtual memory
- Passed to CUDA kernel as a flat int array

### 3.3 Allocation Strategy
- Free list: simple FIFO queue of available block ids
- `allocate_blocks_for_sequence()`: allocates ceil(tokens / block_size) blocks
- `append_token_to_sequence()`: checks if last block has space; if not, allocates new
- `free_sequence()`: decrements ref counts, returns zero-ref blocks to free list

## 4. PagedAttention Kernel

### 4.1 Kernel Design (V1)
- One thread block per (batch_item, attention_head) pair
- Threads within a block cooperatively iterate over KV blocks
- Uses online softmax for numerical stability
- Shared memory for partial sums and max tracking

### 4.2 Memory Layout
```
key_cache:   [num_physical_blocks, block_size, n_head, head_dim]
value_cache: [num_physical_blocks, block_size, n_head, head_dim]
block_table: [batch_size, max_blocks_per_seq]
```

### 4.3 Potential Optimizations (Future)
- V2 kernel: partition across multiple thread blocks per sequence
- Block size tuning for different GPU architectures
- Shared memory optimization for different head dimensions

## 5. Inference Pipeline

### 5.1 Prefill Phase
1. Full prompt processed with standard causal attention
2. K, V written into allocated blocks via block manager
3. Returns logits for the last token (for sampling)

### 5.2 Decode Phase (per step)
1. Single new token embedded and projected to Q, K_new, V_new
2. K_new, V_new appended to block manager
3. PagedAttention kernel computes attention over all cached blocks
4. Output projected and returned

### 5.3 Generation Loop
```
prefill(prompt) → sample token → decode(token) → sample → ... → free()
```

## 6. Evaluation Plan

| Metric | Description | How |
|--------|-------------|-----|
| Internal fragmentation | Wasted space in partially-filled blocks | `block_manager.compute_fragmentation()` |
| External fragmentation | Unusable free blocks between used ones | After alloc/free cycles |
| Max batch size | Largest batch before OOM | Binary search with try/except |
| Throughput | Tokens/second during decode | Wall-clock timing |
| Correctness | Paged == Standard output | Element-wise comparison |

## 7. Implementation Milestones

- [ ] Week 2-3: Block manager (allocate, free, fragmentation metrics)
- [ ] Week 4-5: CUDA kernel (V1) + Python reference implementation
- [ ] Week 6: Transformer integration (prefill + decode paths)
- [ ] Week 7: Benchmarking + report
