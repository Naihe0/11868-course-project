/*
 * PagedAttention CUDA Kernel for MiniTorch
 *
 * Computes scaled dot-product attention over non-contiguous KV cache blocks.
 * Each sequence's KV cache is stored in fixed-size blocks that are mapped
 * via a block table (analogous to a page table in virtual memory).
 *
 * Kernel interface:
 *   paged_attention_forward(
 *     output,       // [batch, n_head, head_dim]
 *     query,        // [batch, n_head, head_dim]
 *     key_cache,    // [num_blocks, block_size, n_head, head_dim]
 *     value_cache,  // [num_blocks, block_size, n_head, head_dim]
 *     block_tables, // [batch, max_blocks_per_seq]
 *     context_lens, // [batch]
 *     block_size,
 *     max_context_len,
 *     n_head,
 *     head_dim
 *   )
 *
 * References:
 *   - Kwon et al., "Efficient Memory Management for Large Language Model
 *     Serving with PagedAttention", SOSP 2023
 *   - vLLM: https://github.com/vllm-project/vllm
 */

#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <assert.h>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

#define MAX_BLOCK_SIZE 64
#define WARP_SIZE 32

// ---------------------------------------------------------------------------
// Utility device functions
// ---------------------------------------------------------------------------

__device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}


// ---------------------------------------------------------------------------
// PagedAttention Kernel (V1 — one thread-block per head per sequence)
// ---------------------------------------------------------------------------

__global__ void paged_attention_v1_kernel(
    float* __restrict__ output,          // [batch, n_head, head_dim]
    const float* __restrict__ query,     // [batch, n_head, head_dim]
    const float* __restrict__ key_cache, // [num_blocks, block_size, n_head, head_dim]
    const float* __restrict__ value_cache,// [num_blocks, block_size, n_head, head_dim]
    const int* __restrict__ block_tables,// [batch, max_blocks_per_seq]
    const int* __restrict__ context_lens,// [batch]
    const int block_size,
    const int max_blocks_per_seq,
    const int n_head,
    const int head_dim,
    const float scale
) {
    // Thread block handles one (sequence, head) pair.
    // Grid: (batch_size, n_head)
    const int seq_idx  = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid      = threadIdx.x;
    const int num_threads = blockDim.x;

    const int context_len = context_lens[seq_idx];
    if (context_len == 0) return;

    // Pointer to this sequence's query vector for this head.
    // query layout: [batch, n_head, head_dim]
    const float* q = query + (seq_idx * n_head + head_idx) * head_dim;

    // block_tables layout: [batch, max_blocks_per_seq]
    const int* seq_block_table = block_tables + seq_idx * max_blocks_per_seq;

    // -----------------------------------------------------------------------
    // Pass 1: Compute Q·K scores for every context token, track global max.
    // -----------------------------------------------------------------------
    // Shared memory for storing per-token attention scores.
    // Maximum tokens = MAX_BLOCK_SIZE * max_blocks_per_seq, but we don't know
    // max_blocks_per_seq at compile time. Use dynamic shared memory or iterate
    // in two passes. For simplicity, we use a two-pass approach with a fixed
    // shared memory buffer for scores (up to max_context_len tokens).
    //
    // For V1 we use a simple approach: each thread loops over dimensions it
    // owns to compute partial dot products, then warp-reduce to get the full
    // score per token.

    extern __shared__ float shared_mem[];
    // shared_mem[0 .. context_len-1] = attention logits
    // shared_mem[context_len .. context_len + head_dim - 1] = output accumulator
    float* logits = shared_mem;
    float* out_accum = shared_mem + context_len;

    // Initialize output accumulator to zero.
    for (int d = tid; d < head_dim; d += num_threads) {
        out_accum[d] = 0.0f;
    }

    // Compute Q·K for each context token.
    float thread_max = -FLT_MAX;

    for (int token_idx = tid; token_idx < context_len; token_idx += num_threads) {
        // Map token_idx to physical cache location.
        int logical_block = token_idx / block_size;
        int slot_in_block = token_idx % block_size;
        int physical_block = seq_block_table[logical_block];

        // key_cache layout: [num_blocks, block_size, n_head, head_dim]
        const float* k = key_cache
            + physical_block * (block_size * n_head * head_dim)
            + slot_in_block * (n_head * head_dim)
            + head_idx * head_dim;

        // Full dot product (each thread does the full dot for its token).
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += q[d] * k[d];
        }
        dot *= scale;
        logits[token_idx] = dot;
        thread_max = fmaxf(thread_max, dot);
    }

    __syncthreads();

    // -----------------------------------------------------------------------
    // Reduce global max across all threads.
    // -----------------------------------------------------------------------
    // Warp-level reduce first.
    float warp_max = warp_reduce_max(thread_max);
    // Use first element of each warp to do block-level reduce via shared mem.
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int num_warps = (num_threads + WARP_SIZE - 1) / WARP_SIZE;

    // Reuse a small region after out_accum for inter-warp communication.
    // out_accum has head_dim floats; we need num_warps floats for reduce.
    // Place warp reduce scratch after out_accum.
    float* warp_scratch = shared_mem + context_len + head_dim;

    if (lane_id == 0) {
        warp_scratch[warp_id] = warp_max;
    }
    __syncthreads();

    float global_max = -FLT_MAX;
    if (tid < num_warps) {
        global_max = warp_scratch[tid];
    }
    global_max = warp_reduce_max(global_max);
    // Broadcast: after warp_reduce every lane in warp 0 has the value.
    // Store to shared for other warps.
    if (tid == 0) {
        warp_scratch[0] = global_max;
    }
    __syncthreads();
    global_max = warp_scratch[0];

    // -----------------------------------------------------------------------
    // Pass 2: Compute exp(logit - max) and sum, then normalize.
    // -----------------------------------------------------------------------
    float thread_sum = 0.0f;
    for (int token_idx = tid; token_idx < context_len; token_idx += num_threads) {
        float val = expf(logits[token_idx] - global_max);
        logits[token_idx] = val;
        thread_sum += val;
    }

    // Reduce sum across threads.
    float warp_sum = warp_reduce_sum(thread_sum);
    if (lane_id == 0) {
        warp_scratch[warp_id] = warp_sum;
    }
    __syncthreads();

    float global_sum = 0.0f;
    if (tid < num_warps) {
        global_sum = warp_scratch[tid];
    }
    global_sum = warp_reduce_sum(global_sum);
    if (tid == 0) {
        warp_scratch[0] = global_sum;
    }
    __syncthreads();
    global_sum = warp_scratch[0];

    // Normalize logits to get attention weights.
    for (int token_idx = tid; token_idx < context_len; token_idx += num_threads) {
        logits[token_idx] /= global_sum;
    }
    __syncthreads();

    // -----------------------------------------------------------------------
    // Pass 3: Compute weighted sum of values.
    // -----------------------------------------------------------------------
    // Each thread accumulates over dimensions it owns.
    for (int d = tid; d < head_dim; d += num_threads) {
        float acc = 0.0f;
        for (int token_idx = 0; token_idx < context_len; token_idx++) {
            int logical_block = token_idx / block_size;
            int slot_in_block = token_idx % block_size;
            int physical_block = seq_block_table[logical_block];

            // value_cache layout: [num_blocks, block_size, n_head, head_dim]
            const float* v = value_cache
                + physical_block * (block_size * n_head * head_dim)
                + slot_in_block * (n_head * head_dim)
                + head_idx * head_dim;

            acc += logits[token_idx] * v[d];
        }
        out_accum[d] = acc;
    }

    __syncthreads();

    // -----------------------------------------------------------------------
    // Write output: [batch, n_head, head_dim]
    // -----------------------------------------------------------------------
    float* out = output + (seq_idx * n_head + head_idx) * head_dim;
    for (int d = tid; d < head_dim; d += num_threads) {
        out[d] = out_accum[d];
    }
}

__global__ void paged_attention_prefill_with_prefix_kernel(
    float* __restrict__ output,               // [batch, work_len, n_head, head_dim]
    const float* __restrict__ query,          // [batch, work_len, n_head, head_dim]
    const float* __restrict__ suffix_keys,    // [batch, work_len, n_head, head_dim]
    const float* __restrict__ suffix_values,  // [batch, work_len, n_head, head_dim]
    const float* __restrict__ key_cache,      // [num_blocks, block_size, n_head, head_dim]
    const float* __restrict__ value_cache,    // [num_blocks, block_size, n_head, head_dim]
    const int* __restrict__ block_tables,     // [batch, max_blocks_per_seq]
    const int block_size,
    const int max_blocks_per_seq,
    const int work_len,
    const int prefix_token_count,
    const int n_head,
    const int head_dim,
    const float scale
) {
    const int row_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    const int batch_idx = row_idx / work_len;
    const int local_idx = row_idx % work_len;
    const int context_len = prefix_token_count + local_idx + 1;

    const float* q = query + ((row_idx * n_head + head_idx) * head_dim);
    const int* seq_block_table = block_tables + batch_idx * max_blocks_per_seq;

    extern __shared__ float shared_mem[];
    float* logits = shared_mem;
    float* out_accum = shared_mem + context_len;

    for (int d = tid; d < head_dim; d += num_threads) {
        out_accum[d] = 0.0f;
    }

    float thread_max = -FLT_MAX;
    for (int token_idx = tid; token_idx < context_len; token_idx += num_threads) {
        const float* k = nullptr;
        if (token_idx < prefix_token_count) {
            int logical_block = token_idx / block_size;
            int slot_in_block = token_idx % block_size;
            int physical_block = seq_block_table[logical_block];
            k = key_cache
                + physical_block * (block_size * n_head * head_dim)
                + slot_in_block * (n_head * head_dim)
                + head_idx * head_dim;
        } else {
            int suffix_idx = token_idx - prefix_token_count;
            k = suffix_keys
                + (((batch_idx * work_len + suffix_idx) * n_head + head_idx) * head_dim);
        }

        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += q[d] * k[d];
        }
        dot *= scale;
        logits[token_idx] = dot;
        thread_max = fmaxf(thread_max, dot);
    }

    __syncthreads();

    float warp_max = warp_reduce_max(thread_max);
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int num_warps = (num_threads + WARP_SIZE - 1) / WARP_SIZE;
    float* warp_scratch = shared_mem + context_len + head_dim;

    if (lane_id == 0) {
        warp_scratch[warp_id] = warp_max;
    }
    __syncthreads();

    float global_max = -FLT_MAX;
    if (tid < num_warps) {
        global_max = warp_scratch[tid];
    }
    global_max = warp_reduce_max(global_max);
    if (tid == 0) {
        warp_scratch[0] = global_max;
    }
    __syncthreads();
    global_max = warp_scratch[0];

    float thread_sum = 0.0f;
    for (int token_idx = tid; token_idx < context_len; token_idx += num_threads) {
        float val = expf(logits[token_idx] - global_max);
        logits[token_idx] = val;
        thread_sum += val;
    }

    float warp_sum = warp_reduce_sum(thread_sum);
    if (lane_id == 0) {
        warp_scratch[warp_id] = warp_sum;
    }
    __syncthreads();

    float global_sum = 0.0f;
    if (tid < num_warps) {
        global_sum = warp_scratch[tid];
    }
    global_sum = warp_reduce_sum(global_sum);
    if (tid == 0) {
        warp_scratch[0] = global_sum;
    }
    __syncthreads();
    global_sum = warp_scratch[0];

    for (int token_idx = tid; token_idx < context_len; token_idx += num_threads) {
        logits[token_idx] /= global_sum;
    }
    __syncthreads();

    for (int d = tid; d < head_dim; d += num_threads) {
        float acc = 0.0f;
        for (int token_idx = 0; token_idx < context_len; ++token_idx) {
            const float* v = nullptr;
            if (token_idx < prefix_token_count) {
                int logical_block = token_idx / block_size;
                int slot_in_block = token_idx % block_size;
                int physical_block = seq_block_table[logical_block];
                v = value_cache
                    + physical_block * (block_size * n_head * head_dim)
                    + slot_in_block * (n_head * head_dim)
                    + head_idx * head_dim;
            } else {
                int suffix_idx = token_idx - prefix_token_count;
                v = suffix_values
                    + (((batch_idx * work_len + suffix_idx) * n_head + head_idx) * head_dim);
            }
            acc += logits[token_idx] * v[d];
        }
        out_accum[d] = acc;
    }

    __syncthreads();

    float* out = output + ((row_idx * n_head + head_idx) * head_dim);
    for (int d = tid; d < head_dim; d += num_threads) {
        out[d] = out_accum[d];
    }
}

// ---------------------------------------------------------------------------
// Host-callable launcher
// ---------------------------------------------------------------------------

extern "C" {

#include <stdio.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
} while(0)

struct PagedAttentionRuntime {
    float* d_key_cache;
    float* d_value_cache;
    int* d_block_tables;
    int* d_context_lens;
    float* d_query;
    float* d_output;
    float* d_prefill_query;
    float* d_prefill_suffix_key;
    float* d_prefill_suffix_value;
    float* d_prefill_output;

    int num_blocks;
    int block_size;
    int n_head;
    int head_dim;
    int max_batch;
    int max_blocks_per_seq;
    int active_max_blocks_per_seq;
    int prefill_row_capacity;
};

static inline int paged_attention_threads(int head_dim) {
    int threads = head_dim;
    if (threads < WARP_SIZE) threads = WARP_SIZE;
    if (threads > 1024) threads = 1024;
    return threads;
}

static inline size_t cache_numel(const PagedAttentionRuntime* runtime) {
    return (size_t)runtime->num_blocks * runtime->block_size * runtime->n_head * runtime->head_dim;
}

static inline size_t query_numel(const PagedAttentionRuntime* runtime) {
    return (size_t)runtime->max_batch * runtime->n_head * runtime->head_dim;
}

static inline size_t block_table_numel(const PagedAttentionRuntime* runtime) {
    return (size_t)runtime->max_batch * runtime->max_blocks_per_seq;
}

static inline size_t prefill_row_numel(
    const PagedAttentionRuntime* runtime,
    int row_capacity
) {
    return (size_t)row_capacity * runtime->n_head * runtime->head_dim;
}

static void ensure_prefill_scratch_capacity(
    PagedAttentionRuntime* runtime,
    int required_rows
) {
    if (required_rows <= runtime->prefill_row_capacity) {
        return;
    }

    if (runtime->d_prefill_query != nullptr) {
        CUDA_CHECK(cudaFree(runtime->d_prefill_query));
        runtime->d_prefill_query = nullptr;
    }
    if (runtime->d_prefill_suffix_key != nullptr) {
        CUDA_CHECK(cudaFree(runtime->d_prefill_suffix_key));
        runtime->d_prefill_suffix_key = nullptr;
    }
    if (runtime->d_prefill_suffix_value != nullptr) {
        CUDA_CHECK(cudaFree(runtime->d_prefill_suffix_value));
        runtime->d_prefill_suffix_value = nullptr;
    }
    if (runtime->d_prefill_output != nullptr) {
        CUDA_CHECK(cudaFree(runtime->d_prefill_output));
        runtime->d_prefill_output = nullptr;
    }

    size_t row_bytes = prefill_row_numel(runtime, required_rows) * sizeof(float);
    CUDA_CHECK(cudaMalloc(&runtime->d_prefill_query, row_bytes));
    CUDA_CHECK(cudaMalloc(&runtime->d_prefill_suffix_key, row_bytes));
    CUDA_CHECK(cudaMalloc(&runtime->d_prefill_suffix_value, row_bytes));
    CUDA_CHECK(cudaMalloc(&runtime->d_prefill_output, row_bytes));
    runtime->prefill_row_capacity = required_rows;
}

void* paged_attention_runtime_create(
    int num_blocks,
    int block_size,
    int n_head,
    int head_dim,
    int max_batch,
    int max_blocks_per_seq
) {
    CUDA_CHECK(cudaSetDevice(0));

    PagedAttentionRuntime* runtime = new PagedAttentionRuntime();
    runtime->d_key_cache = nullptr;
    runtime->d_value_cache = nullptr;
    runtime->d_block_tables = nullptr;
    runtime->d_context_lens = nullptr;
    runtime->d_query = nullptr;
    runtime->d_output = nullptr;
    runtime->d_prefill_query = nullptr;
    runtime->d_prefill_suffix_key = nullptr;
    runtime->d_prefill_suffix_value = nullptr;
    runtime->d_prefill_output = nullptr;

    runtime->num_blocks = num_blocks;
    runtime->block_size = block_size;
    runtime->n_head = n_head;
    runtime->head_dim = head_dim;
    runtime->max_batch = max_batch;
    runtime->max_blocks_per_seq = max_blocks_per_seq;
    runtime->active_max_blocks_per_seq = max_blocks_per_seq;
    runtime->prefill_row_capacity = 0;

    size_t cache_bytes = cache_numel(runtime) * sizeof(float);
    size_t query_bytes = query_numel(runtime) * sizeof(float);
    size_t bt_bytes = block_table_numel(runtime) * sizeof(int);
    size_t cl_bytes = (size_t)runtime->max_batch * sizeof(int);

    CUDA_CHECK(cudaMalloc(&runtime->d_key_cache, cache_bytes));
    CUDA_CHECK(cudaMalloc(&runtime->d_value_cache, cache_bytes));
    CUDA_CHECK(cudaMalloc(&runtime->d_block_tables, bt_bytes));
    CUDA_CHECK(cudaMalloc(&runtime->d_context_lens, cl_bytes));
    CUDA_CHECK(cudaMalloc(&runtime->d_query, query_bytes));
    CUDA_CHECK(cudaMalloc(&runtime->d_output, query_bytes));

    return runtime;
}

void paged_attention_runtime_destroy(void* handle) {
    if (handle == nullptr) return;

    CUDA_CHECK(cudaSetDevice(0));

    PagedAttentionRuntime* runtime = static_cast<PagedAttentionRuntime*>(handle);
    cudaFree(runtime->d_key_cache);
    cudaFree(runtime->d_value_cache);
    cudaFree(runtime->d_block_tables);
    cudaFree(runtime->d_context_lens);
    cudaFree(runtime->d_query);
    cudaFree(runtime->d_output);
    cudaFree(runtime->d_prefill_query);
    cudaFree(runtime->d_prefill_suffix_key);
    cudaFree(runtime->d_prefill_suffix_value);
    cudaFree(runtime->d_prefill_output);
    delete runtime;
}

void paged_attention_runtime_upload_layer_cache(
    void* handle,
    const float* key_cache_host,
    const float* value_cache_host
) {
    if (handle == nullptr) return;
    CUDA_CHECK(cudaSetDevice(0));

    PagedAttentionRuntime* runtime = static_cast<PagedAttentionRuntime*>(handle);
    size_t cache_bytes = cache_numel(runtime) * sizeof(float);

    CUDA_CHECK(cudaMemcpy(
        runtime->d_key_cache,
        key_cache_host,
        cache_bytes,
        cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaMemcpy(
        runtime->d_value_cache,
        value_cache_host,
        cache_bytes,
        cudaMemcpyHostToDevice
    ));
}

void paged_attention_runtime_update_slot(
    void* handle,
    int block_id,
    int slot_idx,
    const float* key_host,
    const float* value_host
) {
    if (handle == nullptr) return;
    CUDA_CHECK(cudaSetDevice(0));

    PagedAttentionRuntime* runtime = static_cast<PagedAttentionRuntime*>(handle);
    size_t slot_elems = (size_t)runtime->n_head * runtime->head_dim;
    size_t slot_offset = ((size_t)block_id * runtime->block_size + slot_idx) * slot_elems;
    size_t slot_bytes = slot_elems * sizeof(float);

    CUDA_CHECK(cudaMemcpy(
        runtime->d_key_cache + slot_offset,
        key_host,
        slot_bytes,
        cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaMemcpy(
        runtime->d_value_cache + slot_offset,
        value_host,
        slot_bytes,
        cudaMemcpyHostToDevice
    ));
}

void paged_attention_runtime_upload_block(
    void* handle,
    int block_id,
    const float* key_block_host,
    const float* value_block_host
) {
    if (handle == nullptr) return;
    CUDA_CHECK(cudaSetDevice(0));

    PagedAttentionRuntime* runtime = static_cast<PagedAttentionRuntime*>(handle);
    if (block_id < 0 || block_id >= runtime->num_blocks) {
        fprintf(stderr, "paged_attention_runtime_upload_block: block_id=%d out of range [0, %d)\n",
                block_id, runtime->num_blocks);
        return;
    }

    size_t block_elems = (size_t)runtime->block_size * runtime->n_head * runtime->head_dim;
    size_t block_offset = (size_t)block_id * block_elems;
    size_t block_bytes = block_elems * sizeof(float);

    CUDA_CHECK(cudaMemcpy(
        runtime->d_key_cache + block_offset,
        key_block_host,
        block_bytes,
        cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaMemcpy(
        runtime->d_value_cache + block_offset,
        value_block_host,
        block_bytes,
        cudaMemcpyHostToDevice
    ));
}

void paged_attention_runtime_update_metadata(
    void* handle,
    const int* block_tables_host,
    const int* context_lens_host,
    int batch_size,
    int max_blocks_per_seq
) {
    if (handle == nullptr) return;
    CUDA_CHECK(cudaSetDevice(0));

    PagedAttentionRuntime* runtime = static_cast<PagedAttentionRuntime*>(handle);
    if (batch_size > runtime->max_batch) {
        fprintf(stderr, "paged_attention_runtime_update_metadata: batch_size=%d exceeds max_batch=%d\n",
                batch_size, runtime->max_batch);
        return;
    }
    if (max_blocks_per_seq > runtime->max_blocks_per_seq) {
        fprintf(stderr, "paged_attention_runtime_update_metadata: max_blocks_per_seq=%d exceeds runtime capacity=%d\n",
                max_blocks_per_seq, runtime->max_blocks_per_seq);
        return;
    }
    runtime->active_max_blocks_per_seq = max_blocks_per_seq;

    size_t bt_bytes = (size_t)batch_size * max_blocks_per_seq * sizeof(int);
    size_t cl_bytes = (size_t)batch_size * sizeof(int);

    CUDA_CHECK(cudaMemcpy(
        runtime->d_block_tables,
        block_tables_host,
        bt_bytes,
        cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaMemcpy(
        runtime->d_context_lens,
        context_lens_host,
        cl_bytes,
        cudaMemcpyHostToDevice
    ));
}

void paged_attention_runtime_forward(
    void* handle,
    const float* query_host,
    float* output_host,
    int batch_size,
    int max_context_len
) {
    if (handle == nullptr) return;
    CUDA_CHECK(cudaSetDevice(0));

    PagedAttentionRuntime* runtime = static_cast<PagedAttentionRuntime*>(handle);
    if (batch_size > runtime->max_batch) {
        fprintf(stderr, "paged_attention_runtime_forward: batch_size=%d exceeds max_batch=%d\n",
                batch_size, runtime->max_batch);
        return;
    }

    float scale = 1.0f / sqrtf((float)runtime->head_dim);
    dim3 grid(batch_size, runtime->n_head);
    int threads = paged_attention_threads(runtime->head_dim);
    dim3 block(threads);
    int num_warps = (threads + WARP_SIZE - 1) / WARP_SIZE;
    size_t smem_bytes = (max_context_len + runtime->head_dim + num_warps) * sizeof(float);
    size_t query_bytes = (size_t)batch_size * runtime->n_head * runtime->head_dim * sizeof(float);

    CUDA_CHECK(cudaMemcpy(
        runtime->d_query,
        query_host,
        query_bytes,
        cudaMemcpyHostToDevice
    ));

    paged_attention_v1_kernel<<<grid, block, smem_bytes>>>(
        runtime->d_output,
        runtime->d_query,
        runtime->d_key_cache,
        runtime->d_value_cache,
        runtime->d_block_tables,
        runtime->d_context_lens,
        runtime->block_size,
        runtime->active_max_blocks_per_seq,
        runtime->n_head,
        runtime->head_dim,
        scale
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        output_host,
        runtime->d_output,
        query_bytes,
        cudaMemcpyDeviceToHost
    ));
}

void paged_attention_runtime_prefill_with_prefix_forward(
    void* handle,
    const float* query_host,
    const float* suffix_key_host,
    const float* suffix_value_host,
    float* output_host,
    int batch_size,
    int work_len,
    int prefix_token_count
) {
    if (handle == nullptr) return;
    CUDA_CHECK(cudaSetDevice(0));

    PagedAttentionRuntime* runtime = static_cast<PagedAttentionRuntime*>(handle);
    if (batch_size > runtime->max_batch) {
        fprintf(stderr, "paged_attention_runtime_prefill_with_prefix_forward: batch_size=%d exceeds max_batch=%d\n",
                batch_size, runtime->max_batch);
        return;
    }

    const int total_rows = batch_size * work_len;
    const size_t row_elems = (size_t)total_rows * runtime->n_head * runtime->head_dim;
    const size_t row_bytes = row_elems * sizeof(float);

    ensure_prefill_scratch_capacity(runtime, total_rows);

    CUDA_CHECK(cudaMemcpy(
        runtime->d_prefill_query,
        query_host,
        row_bytes,
        cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaMemcpy(
        runtime->d_prefill_suffix_key,
        suffix_key_host,
        row_bytes,
        cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaMemcpy(
        runtime->d_prefill_suffix_value,
        suffix_value_host,
        row_bytes,
        cudaMemcpyHostToDevice
    ));

    float scale = 1.0f / sqrtf((float)runtime->head_dim);
    dim3 grid(total_rows, runtime->n_head);
    int threads = paged_attention_threads(runtime->head_dim);
    dim3 block(threads);
    int num_warps = (threads + WARP_SIZE - 1) / WARP_SIZE;
    size_t smem_bytes = (prefix_token_count + work_len + runtime->head_dim + num_warps) * sizeof(float);

    paged_attention_prefill_with_prefix_kernel<<<grid, block, smem_bytes>>>(
        runtime->d_prefill_output,
        runtime->d_prefill_query,
        runtime->d_prefill_suffix_key,
        runtime->d_prefill_suffix_value,
        runtime->d_key_cache,
        runtime->d_value_cache,
        runtime->d_block_tables,
        runtime->block_size,
        runtime->active_max_blocks_per_seq,
        work_len,
        prefix_token_count,
        runtime->n_head,
        runtime->head_dim,
        scale
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        output_host,
        runtime->d_prefill_output,
        row_bytes,
        cudaMemcpyDeviceToHost
    ));
}

void paged_attention_forward(
    float* output,
    const float* query,
    const float* key_cache,
    const float* value_cache,
    const int* block_tables,
    const int* context_lens,
    int batch_size,
    int n_head,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    int max_context_len
) {
    CUDA_CHECK(cudaSetDevice(0));

    float scale = 1.0f / sqrtf((float)head_dim);

    // Grid: one block per (batch, head) pair
    dim3 grid(batch_size, n_head);
    // Block: at least WARP_SIZE threads (needed for warp reductions),
    // up to head_dim if larger, capped at 1024.
    int threads = paged_attention_threads(head_dim);
    dim3 block(threads);

    // Dynamic shared memory: logits[max_context_len] + out_accum[head_dim]
    //                        + warp_scratch[ceil(threads/WARP_SIZE)]
    int num_warps = (threads + WARP_SIZE - 1) / WARP_SIZE;
    size_t smem_bytes = (max_context_len + head_dim + num_warps) * sizeof(float);

    // --- Compute buffer sizes ---
    int num_blocks_cache = 0;
    for (int i = 0; i < batch_size * max_blocks_per_seq; i++) {
        if (block_tables[i] + 1 > num_blocks_cache)
            num_blocks_cache = block_tables[i] + 1;
    }

    size_t query_bytes     = (size_t)batch_size * n_head * head_dim * sizeof(float);
    size_t cache_bytes     = (size_t)num_blocks_cache * block_size * n_head * head_dim * sizeof(float);
    size_t bt_bytes        = (size_t)batch_size * max_blocks_per_seq * sizeof(int);
    size_t cl_bytes        = (size_t)batch_size * sizeof(int);
    size_t output_bytes    = query_bytes;

    // --- Allocate device memory ---
    float *d_output, *d_query, *d_key_cache, *d_value_cache;
    int   *d_block_tables, *d_context_lens;
    CUDA_CHECK(cudaMalloc(&d_output,       output_bytes));
    CUDA_CHECK(cudaMalloc(&d_query,        query_bytes));
    CUDA_CHECK(cudaMalloc(&d_key_cache,    cache_bytes));
    CUDA_CHECK(cudaMalloc(&d_value_cache,  cache_bytes));
    CUDA_CHECK(cudaMalloc(&d_block_tables, bt_bytes));
    CUDA_CHECK(cudaMalloc(&d_context_lens, cl_bytes));

    // --- Copy inputs host → device ---
    CUDA_CHECK(cudaMemcpy(d_query,        query,        query_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_key_cache,    key_cache,    cache_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_value_cache,  value_cache,  cache_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_block_tables, block_tables, bt_bytes,    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_context_lens, context_lens, cl_bytes,    cudaMemcpyHostToDevice));

    // --- Launch kernel ---
    paged_attention_v1_kernel<<<grid, block, smem_bytes>>>(
        d_output, d_query, d_key_cache, d_value_cache,
        d_block_tables, d_context_lens,
        block_size, max_blocks_per_seq,
        n_head, head_dim, scale
    );
    CUDA_CHECK(cudaGetLastError());

    // --- Copy output device → host ---
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(output, d_output, output_bytes, cudaMemcpyDeviceToHost));

    // --- Free device memory ---
    cudaFree(d_output);
    cudaFree(d_query);
    cudaFree(d_key_cache);
    cudaFree(d_value_cache);
    cudaFree(d_block_tables);
    cudaFree(d_context_lens);
}

}  // extern "C"
