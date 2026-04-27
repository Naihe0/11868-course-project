"""
Benchmark PagedAttention vs standard attention in MiniTorch.

Measures:
  - Memory fragmentation (internal / external)
  - Maximum batch size before OOM
  - Generation throughput (tokens/second)
  - Correctness (output match)

Usage:
    python project/run_benchmark.py \
        --batch-sizes 1 2 4 8 \
        --seq-lengths 128 256 512 1024 \
        --block-sizes 8 16 32
"""

import argparse
import csv
import math
import os
import sys
import time
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, ".")

import minitorch
from minitorch.transformer import PagedDecoderLM
from minitorch.block_manager import BlockManager, CACHE_DTYPE
from minitorch.paged_attention import paged_attention_ref
from minitorch.tensor import tensor_from_numpy
from minitorch.modules_transfomer import DecoderLM

datatype = np.float32


def parse_args():
    parser = argparse.ArgumentParser(description="PagedAttention Benchmark")
    parser.add_argument("--n-vocab", type=int, default=1000)
    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-positions", type=int, default=1024)
    parser.add_argument("--num-kv-blocks", type=int, default=512)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--seq-lengths", type=int, nargs="+",
                        default=[32, 64, 128])
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--warmup-tokens", type=int, default=4,
                        help="Warmup generation tokens (not timed)")
    parser.add_argument("--output-dir", type=str, default="benchmarks/results")
    parser.add_argument("--backend", type=str, default="cpu",
                        choices=["cpu", "cuda"])
    parser.add_argument("--decode-backend", type=str, default="ref",
                        choices=["ref", "cuda"])
    parser.add_argument("--compare-to-ref", action="store_true",
                        help="When using CUDA decode, also run the reference "
                             "paged attention and assert the outputs match.")
    parser.add_argument("--compare-tolerance", type=float, default=1e-4)
    parser.add_argument("--skip-correctness", action="store_true",
                        help="Skip correctness check to speed up benchmarking")
    parser.add_argument("--skip-max-batch", action="store_true",
                        help="Skip max batch size search")
    parser.add_argument("--compare-baseline", action="store_true",
                        help="Also benchmark the non-paged DecoderLM (hw3/hw4) "
                             "for throughput comparison")
    parser.add_argument("--frag-seq-lengths", type=int, nargs="+",
                        default=[33, 65, 100, 130],
                        help="Sequence lengths used for the dedicated "
                             "fragmentation sweep (chosen to NOT divide "
                             "evenly into the block sizes).")
    parser.add_argument("--skip-frag-sweep", action="store_true",
                        help="Skip the dedicated non-aligned fragmentation sweep")
    parser.add_argument("--compare-prefix-cache", action="store_true",
                        help="Benchmark second-request prefill when the prompt "
                             "shares a prefix with a previously cached request")
    parser.add_argument("--prefix-shared-ratio", type=float, default=0.5,
                        help="Fraction of each prompt to reuse in the prefix-cache "
                             "benchmark. Rounded down to full blocks.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_backend(backend_name: str):
    if backend_name == "cuda":
        return minitorch.TensorBackend(minitorch.CudaKernelOps)
    return minitorch.TensorBackend(minitorch.FastOps)


def _create_model(n_vocab, n_embd, n_head, n_positions, n_layers,
                   block_size, backend, decode_backend,
                   compare_to_ref, compare_tolerance):
    return PagedDecoderLM(
        n_vocab=n_vocab,
        n_embd=n_embd,
        n_head=n_head,
        n_positions=n_positions,
        n_layers=n_layers,
        block_size=block_size,
        p_dropout=0.0,
        backend=backend,
        decode_backend=decode_backend,
        compare_to_ref=compare_to_ref,
        compare_tolerance=compare_tolerance,
    )


def _create_block_manager(num_kv_blocks, block_size, n_head, head_dim, n_layers):
    return BlockManager(
        num_blocks=num_kv_blocks,
        block_size=block_size,
        n_head=n_head,
        head_dim=head_dim,
        num_layers=n_layers,
    )


def _make_prompt(batch_size, seq_len, n_vocab, backend):
    prompt_np = np.random.randint(0, n_vocab, size=(batch_size, seq_len)).astype(datatype)
    return tensor_from_numpy(prompt_np, backend=backend)


def _create_baseline_model(n_vocab, n_embd, n_head, n_positions, n_layers, backend):
    """Create a non-paged DecoderLM baseline.

    The reference DecoderLM in this repo is fixed to 4 transformer layers.
    We only benchmark it when the requested paged model is also 4 layers so the
    comparison remains structurally fair.
    """
    if n_layers != 4:
        raise ValueError(
            "Baseline DecoderLM is only available for n_layers=4 in this repo"
        )
    return DecoderLM(
        n_vocab=n_vocab,
        n_embd=n_embd,
        n_head=n_head,
        n_positions=n_positions,
        p_dropout=0.0,
        backend=backend,
    )


def _kv_cache_metrics(block_manager: BlockManager) -> Dict[str, float]:
    """Summarize reserved/used KV cache capacity for the current manager state."""
    dtype_size = np.dtype(block_manager.cache_dtype).itemsize
    reserved_token_slots = block_manager.num_blocks * block_manager.block_size
    used_token_slots = sum(block.num_filled for block in block_manager.blocks.values())
    kv_reserved_bytes = (
        block_manager.num_blocks
        * block_manager.block_size
        * block_manager.n_head
        * block_manager.head_dim
        * block_manager.num_layers
        * 2
        * dtype_size
    )
    kv_used_bytes = (
        used_token_slots
        * block_manager.n_head
        * block_manager.head_dim
        * block_manager.num_layers
        * 2
        * dtype_size
    )
    kv_efficiency = (
        kv_used_bytes / kv_reserved_bytes if kv_reserved_bytes > 0 else 0.0
    )
    return {
        "reserved_token_slots": int(reserved_token_slots),
        "used_token_slots": int(used_token_slots),
        "kv_reserved_bytes": int(kv_reserved_bytes),
        "kv_used_bytes": int(kv_used_bytes),
        "kv_efficiency": round(kv_efficiency, 4),
    }


# ---------------------------------------------------------------------------
# Benchmark routines
# ---------------------------------------------------------------------------

def benchmark_throughput(
    model, block_manager, prompt, seq_ids, max_new_tokens, backend,
) -> Dict[str, float]:
    """Measure generation throughput and latency.

    Returns dict with: tokens_per_sec, total_time_s, time_per_token_ms,
                       prefill_time_s, decode_time_s, decode_p50_ms, decode_p95_ms
    """
    batch_size = prompt.shape[0]

    # --- Prefill timing ---
    prefill_start = time.perf_counter()
    logits = model.forward_prefill(prompt, block_manager, seq_ids)
    prefill_end = time.perf_counter()
    prefill_time = prefill_end - prefill_start

    # Sample first token from last prompt position
    last_logits_np = logits.to_numpy()[:, -1, :]
    next_tokens = np.argmax(last_logits_np, axis=-1)

    # --- Decode timing (per-step latency captured for percentile stats) ---
    prompt_len = prompt.shape[1]
    per_step_ms: List[float] = []
    decode_start = time.perf_counter()
    for step in range(1, max_new_tokens):
        token_input = tensor_from_numpy(
            next_tokens.reshape(batch_size, 1).astype(datatype),
            backend=model.backend,
        )
        start_pos = prompt_len + step - 1
        step_t0 = time.perf_counter()
        logits = model.forward_decode(
            token_input, block_manager, seq_ids, start_pos=start_pos,
        )
        logits_np = logits.to_numpy()[:, 0, :]
        next_tokens = np.argmax(logits_np, axis=-1)
        step_t1 = time.perf_counter()
        # Per-token latency (ms / token), normalized over the batch
        per_step_ms.append((step_t1 - step_t0) * 1000.0 / batch_size)
    decode_end = time.perf_counter()
    decode_time = decode_end - decode_start

    total_time = prefill_time + decode_time
    total_generated = batch_size * max_new_tokens
    decode_tokens = batch_size * (max_new_tokens - 1) if max_new_tokens > 1 else 0

    kv_metrics = _kv_cache_metrics(block_manager)

    # Free sequences for reuse
    for seq_id in seq_ids:
        block_manager.free_sequence(seq_id)

    if per_step_ms:
        p50 = float(np.percentile(per_step_ms, 50))
        p95 = float(np.percentile(per_step_ms, 95))
    else:
        p50 = p95 = 0.0

    metrics = {
        "total_time_s": round(total_time, 4),
        "prefill_time_s": round(prefill_time, 4),
        "decode_time_s": round(decode_time, 4),
        "tokens_per_sec": round(total_generated / total_time, 2) if total_time > 0 else 0,
        "decode_tokens_per_sec": round(decode_tokens / decode_time, 2) if decode_time > 0 else 0,
        "time_per_token_ms": round((decode_time / decode_tokens) * 1000, 2) if decode_tokens > 0 else 0,
        "decode_p50_ms": round(p50, 2),
        "decode_p95_ms": round(p95, 2),
    }
    metrics.update(kv_metrics)
    return metrics


def benchmark_baseline_throughput(
    model, prompt, max_new_tokens, backend,
) -> Dict[str, float]:
    """Measure prompt+generation throughput for the non-paged baseline DecoderLM.

    The baseline has no KV cache, so autoregressive generation feeds the
    *entire* accumulated sequence through forward() at each step.

    Returns dict with:
      - baseline_prefill_time_s
      - baseline_decode_time_s
      - baseline_total_time_s
      - baseline_tokens_per_sec
      - baseline_decode_tokens_per_sec
      - baseline_time_per_token_ms
    """
    batch_size, seq_len = prompt.shape
    generated_np = prompt.to_numpy().astype(datatype)

    model.eval()
    prefill_start = time.perf_counter()
    prompt_logits = model.forward(prompt)
    prefill_end = time.perf_counter()
    prefill_time = prefill_end - prefill_start

    if max_new_tokens <= 0:
        return {
            "baseline_prefill_time_s": round(prefill_time, 4),
            "baseline_decode_time_s": 0.0,
            "baseline_total_time_s": round(prefill_time, 4),
            "baseline_tokens_per_sec": 0.0,
            "baseline_decode_tokens_per_sec": 0.0,
            "baseline_time_per_token_ms": 0.0,
        }

    last_logits = prompt_logits.to_numpy()[:, -1, :]
    next_tokens = np.argmax(last_logits, axis=-1).astype(datatype)
    generated_np = np.concatenate(
        [generated_np, next_tokens.reshape(batch_size, 1)],
        axis=1,
    )

    decode_start = time.perf_counter()
    for _ in range(1, max_new_tokens):
        input_t = tensor_from_numpy(generated_np, backend=backend)
        logits = model.forward(input_t)
        last_logits = logits.to_numpy()[:, -1, :]
        next_tokens = np.argmax(last_logits, axis=-1).astype(datatype)
        generated_np = np.concatenate(
            [generated_np, next_tokens.reshape(batch_size, 1)],
            axis=1,
        )
    decode_end = time.perf_counter()

    decode_time = decode_end - decode_start
    total_time = prefill_time + decode_time
    total_generated = batch_size * max_new_tokens
    decode_tokens = batch_size * max(max_new_tokens - 1, 0)

    return {
        "baseline_prefill_time_s": round(prefill_time, 4),
        "baseline_decode_time_s": round(decode_time, 4),
        "baseline_total_time_s": round(total_time, 4),
        "baseline_tokens_per_sec": round(total_generated / total_time, 2) if total_time > 0 else 0,
        "baseline_decode_tokens_per_sec": round(decode_tokens / decode_time, 2) if decode_time > 0 else 0,
        "baseline_time_per_token_ms": round((decode_time / decode_tokens) * 1000, 2) if decode_tokens > 0 else 0,
        "baseline_kv_reserved_bytes": 0,
        "baseline_kv_used_bytes": 0,
    }


def benchmark_prefix_cache_prefill(
    n_vocab,
    n_embd,
    n_head,
    n_layers,
    n_positions,
    num_kv_blocks,
    block_size,
    seq_len,
    backend,
    decode_backend,
    compare_to_ref,
    compare_tolerance,
    shared_ratio: float,
) -> Dict[str, float]:
    """Measure prefill savings for a second request with a shared prefix."""
    if seq_len < block_size:
        return {
            "prefix_cached_tokens": 0,
            "prefix_cached_blocks": 0,
            "prefix_hit_rate": 0.0,
            "prefix_cached_prefill_time_s": 0.0,
            "prefix_fresh_prefill_time_s": 0.0,
            "prefix_prefill_speedup": 0.0,
        }

    head_dim = n_embd // n_head
    shared_blocks = int((seq_len * shared_ratio) // block_size)
    shared_tokens = shared_blocks * block_size
    if shared_tokens <= 0:
        return {
            "prefix_cached_tokens": 0,
            "prefix_cached_blocks": 0,
            "prefix_hit_rate": 0.0,
            "prefix_cached_prefill_time_s": 0.0,
            "prefix_fresh_prefill_time_s": 0.0,
            "prefix_prefill_speedup": 0.0,
        }

    model = _create_model(
        n_vocab,
        n_embd,
        n_head,
        n_positions,
        n_layers,
        block_size,
        backend,
        decode_backend,
        compare_to_ref,
        compare_tolerance,
    )
    model.eval()

    cached_manager = _create_block_manager(
        num_kv_blocks,
        block_size,
        n_head,
        head_dim,
        n_layers,
    )
    fresh_manager = _create_block_manager(
        num_kv_blocks,
        block_size,
        n_head,
        head_dim,
        n_layers,
    )

    base_prompt = np.random.randint(0, n_vocab, size=(1, seq_len)).astype(datatype)
    second_prompt = base_prompt.copy()
    if shared_tokens < seq_len:
        second_prompt[:, shared_tokens:] = np.random.randint(
            0, n_vocab, size=(1, seq_len - shared_tokens)
        ).astype(datatype)

    model.forward_prefill(
        tensor_from_numpy(base_prompt, backend=backend),
        cached_manager,
        [9000],
    )
    cached_manager.free_sequence(9000)

    second_prompt_t = tensor_from_numpy(second_prompt, backend=backend)

    cached_start = time.perf_counter()
    cached_logits = model.forward_prefill(second_prompt_t, cached_manager, [9001])
    cached_end = time.perf_counter()
    cached_prefill_time = cached_end - cached_start
    cached_hit_tokens = cached_manager.seq_prefix_cache_info[9001].cached_token_count

    fresh_start = time.perf_counter()
    fresh_logits = model.forward_prefill(second_prompt_t, fresh_manager, [9101])
    fresh_end = time.perf_counter()
    fresh_prefill_time = fresh_end - fresh_start

    if cached_hit_tokens < seq_len:
        np.testing.assert_allclose(
            cached_logits.to_numpy()[:, cached_hit_tokens:, :],
            fresh_logits.to_numpy()[:, cached_hit_tokens:, :],
            atol=1e-5,
            rtol=1e-5,
        )
    else:
        np.testing.assert_allclose(
            cached_logits.to_numpy()[:, -1:, :],
            fresh_logits.to_numpy()[:, -1:, :],
            atol=1e-5,
            rtol=1e-5,
        )

    cached_manager.free_sequence(9001)
    fresh_manager.free_sequence(9101)
    model.close_decode_runtime()

    return {
        "prefix_cached_tokens": int(cached_hit_tokens),
        "prefix_cached_blocks": int(cached_hit_tokens // block_size),
        "prefix_hit_rate": round(cached_hit_tokens / seq_len, 4),
        "prefix_cached_prefill_time_s": round(cached_prefill_time, 4),
        "prefix_fresh_prefill_time_s": round(fresh_prefill_time, 4),
        "prefix_prefill_speedup": round(
            (fresh_prefill_time / cached_prefill_time) if cached_prefill_time > 0 else 0.0,
            2,
        ),
    }


def benchmark_fragmentation(block_manager, batch_size, seq_len,
                             block_size, contiguous_max_seq_len: Optional[int] = None) -> Dict[str, float]:
    """Measure internal/external fragmentation and KV memory vs contiguous.

    Allocates blocks for `batch_size` sequences of `seq_len` tokens,
    measures fragmentation and KV bytes (paged vs naive contiguous),
    then frees them.

    Args:
        contiguous_max_seq_len: Max seq length the naive contiguous baseline
            would reserve per sequence. Defaults to ``seq_len`` (an optimistic
            baseline that already knows the exact length); pass a larger value
            (e.g. ``n_positions``) to model the realistic worst case.
    """
    seq_ids = list(range(1000, 1000 + batch_size))
    for seq_id in seq_ids:
        block_manager.allocate_blocks_for_sequence(seq_id, seq_len)

    frag = block_manager.compute_fragmentation()
    if contiguous_max_seq_len is None:
        contiguous_max_seq_len = seq_len
    mem = block_manager.compute_kv_memory(max_seq_len=contiguous_max_seq_len)

    utilization = 0.0
    used = block_manager.num_used_blocks
    total = block_manager.num_blocks
    if total > 0:
        utilization = used / total

    for seq_id in seq_ids:
        block_manager.free_sequence(seq_id)

    return {
        "internal_frag": round(frag["internal"], 4),
        "external_frag": round(frag["external"], 4),
        "blocks_used": used,
        "blocks_total": total,
        "utilization": round(utilization, 4),
        "kv_bytes_paged": int(mem["kv_bytes_paged"]),
        "kv_bytes_contiguous_naive": int(mem["kv_bytes_contiguous_naive"]),
        "memory_savings_ratio": round(mem["memory_savings_ratio"], 4),
    }


def benchmark_max_batch_size(
    n_vocab, n_embd, n_head, n_layers, n_positions,
    num_kv_blocks, block_size, seq_len, backend_name,
) -> int:
    """Find the maximum batch size that fits in the KV cache blocks.

    Uses binary search: tries to allocate blocks for batch_size sequences
    of seq_len tokens. Returns the largest batch_size that fits.
    """
    head_dim = n_embd // n_head
    import math
    blocks_per_seq = math.ceil(seq_len / block_size)

    # Analytical max: total blocks / blocks_per_seq
    analytical_max = num_kv_blocks // blocks_per_seq if blocks_per_seq > 0 else 0

    # Verify by actually allocating
    lo, hi = 1, analytical_max
    best = 0

    while lo <= hi:
        mid = (lo + hi) // 2
        bm = BlockManager(
            num_blocks=num_kv_blocks,
            block_size=block_size,
            n_head=n_head,
            head_dim=head_dim,
            num_layers=1,
        )
        try:
            for i in range(mid):
                bm.allocate_blocks_for_sequence(seq_id=i, num_tokens=seq_len)
            best = mid
            lo = mid + 1
        except RuntimeError:
            hi = mid - 1

    return best


def _manual_attention(query, key, value, mask=None):
    """Pure NumPy attention for correctness reference."""
    scores = np.matmul(query, np.swapaxes(key, -1, -2)) / np.sqrt(query.shape[-1])
    if mask is not None:
        scores = scores + mask
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / np.sum(weights, axis=-1, keepdims=True)
    return np.matmul(weights, value)


def check_correctness(
    n_vocab,
    n_embd,
    n_head,
    n_layers,
    n_positions,
    block_size,
    num_kv_blocks,
    backend,
    decode_backend,
    compare_to_ref,
    compare_tolerance,
) -> Dict[str, float]:
    """Verify the selected decode path against full-sequence recomputation."""

    def _reference_last_logits(model, full_tokens_np, seq_ids):
        ref_bm = _create_block_manager(
            num_kv_blocks,
            block_size,
            n_head,
            n_embd // n_head,
            n_layers,
        )
        ref_input = tensor_from_numpy(full_tokens_np.astype(datatype), backend=backend)
        ref_logits = model.forward_prefill(ref_input, ref_bm, seq_ids)
        return ref_logits.to_numpy()[:, -1, :]

    batch_size = 2
    prompt_len = min(7, n_positions - 2)
    max_new_tokens = 3
    np.random.seed(42)

    model = _create_model(
        n_vocab,
        n_embd,
        n_head,
        n_positions,
        n_layers,
        block_size,
        backend,
        decode_backend,
        compare_to_ref,
        compare_tolerance,
    )
    model.eval()
    block_manager = _create_block_manager(
        num_kv_blocks,
        block_size,
        n_head,
        n_embd // n_head,
        n_layers,
    )
    seq_ids = list(range(batch_size))
    prompt_np = np.random.randint(
        0,
        n_vocab,
        size=(batch_size, prompt_len),
    ).astype(datatype)
    prompt = tensor_from_numpy(prompt_np, backend=backend)

    generated = prompt_np.copy()
    max_abs_err = 0.0
    mean_abs_errs = []

    logits = model.forward_prefill(prompt, block_manager, seq_ids)
    logits_np = logits.to_numpy()[:, -1, :]
    ref_logits_np = _reference_last_logits(model, generated, seq_ids)
    diff = logits_np - ref_logits_np
    max_abs_err = max(max_abs_err, float(np.max(np.abs(diff))))
    mean_abs_errs.append(float(np.mean(np.abs(diff))))

    next_tokens = np.argmax(logits_np, axis=-1).astype(datatype)
    generated = np.concatenate([generated, next_tokens.reshape(batch_size, 1)], axis=1)

    for step in range(1, max_new_tokens):
        token_input = tensor_from_numpy(
            next_tokens.reshape(batch_size, 1).astype(datatype),
            backend=backend,
        )
        start_pos = prompt_len + step - 1
        logits = model.forward_decode(
            token_input,
            block_manager,
            seq_ids,
            start_pos=start_pos,
        )
        logits_np = logits.to_numpy()[:, 0, :]
        ref_logits_np = _reference_last_logits(model, generated, seq_ids)
        diff = logits_np - ref_logits_np
        max_abs_err = max(max_abs_err, float(np.max(np.abs(diff))))
        mean_abs_errs.append(float(np.mean(np.abs(diff))))
        next_tokens = np.argmax(logits_np, axis=-1).astype(datatype)
        generated = np.concatenate([generated, next_tokens.reshape(batch_size, 1)], axis=1)

    for seq_id in seq_ids:
        if seq_id in block_manager.block_tables:
            block_manager.free_sequence(seq_id)

    mean_abs_err = float(np.mean(mean_abs_errs)) if mean_abs_errs else 0.0
    matches = max_abs_err < compare_tolerance

    return {
        "correctness_pass": 1 if matches else 0,
        "max_abs_error": round(max_abs_err, 8),
        "mean_abs_error": round(mean_abs_err, 8),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results_file = os.path.join(args.output_dir, "benchmark_results.csv")

    print("=" * 70)
    print("PagedAttention Benchmark")
    print("=" * 70)
    print(f"Model:       vocab={args.n_vocab}, embd={args.n_embd}, "
          f"head={args.n_head}, layers={args.n_layers}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Seq lengths: {args.seq_lengths}")
    print(f"Block sizes: {args.block_sizes}")
    print(f"New tokens:  {args.max_new_tokens}")
    print(f"Backend:     {args.backend}")
    print(f"Baseline:    {'yes' if args.compare_baseline else 'no'}")
    print(f"Output:      {results_file}")
    print("=" * 70)
    print()

    backend = _create_backend(args.backend)
    head_dim = args.n_embd // args.n_head

    all_results: List[Dict] = []

    for block_size in args.block_sizes:
        # --- Correctness check (once per block size) ---
        if not args.skip_correctness:
            print(f"[block_size={block_size}] Correctness check ... ", end="", flush=True)
            corr = check_correctness(
                args.n_vocab,
                args.n_embd,
                args.n_head,
                args.n_layers,
                args.n_positions,
                block_size,
                args.num_kv_blocks,
                backend,
                args.decode_backend,
                args.compare_to_ref,
                args.compare_tolerance,
            )
            status = "PASS" if corr["correctness_pass"] else "FAIL"
            print(f"{status} (max_err={corr['max_abs_error']:.2e})")

        for batch_size in args.batch_sizes:
            for seq_len in args.seq_lengths:
                config = {
                    "block_size": block_size,
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "max_new_tokens": args.max_new_tokens,
                    "n_embd": args.n_embd,
                    "n_head": args.n_head,
                    "n_layers": args.n_layers,
                }
                label = (f"bs={block_size}, batch={batch_size}, "
                         f"seq={seq_len}")
                print(f"  [{label}] ", end="", flush=True)

                # Check if we have enough blocks
                import math
                blocks_needed = batch_size * math.ceil(
                    (seq_len + args.max_new_tokens) / block_size
                )
                if blocks_needed > args.num_kv_blocks:
                    print(f"SKIP (need {blocks_needed} blocks, "
                          f"have {args.num_kv_blocks})")
                    continue

                try:
                    # Create fresh model and block manager
                    model = _create_model(
                        args.n_vocab, args.n_embd, args.n_head,
                        args.n_positions, args.n_layers,
                        block_size, backend,
                        args.decode_backend,
                        args.compare_to_ref,
                        args.compare_tolerance,
                    )
                    bm = _create_block_manager(
                        args.num_kv_blocks, block_size,
                        args.n_head, head_dim,
                        args.n_layers,
                    )
                    prompt = _make_prompt(batch_size, seq_len, args.n_vocab, backend)
                    seq_ids = list(range(batch_size))

                    # Throughput
                    throughput = benchmark_throughput(
                        model, bm, prompt, seq_ids,
                        args.max_new_tokens, backend,
                    )

                    # Fragmentation (fresh block manager)
                    bm_frag = _create_block_manager(
                        args.num_kv_blocks, block_size,
                        args.n_head, head_dim,
                        args.n_layers,
                    )
                    frag = benchmark_fragmentation(
                        bm_frag, batch_size, seq_len, block_size,
                        contiguous_max_seq_len=args.n_positions,
                    )

                    result = {**config, **throughput, **frag}
                    if not args.skip_correctness:
                        result.update(corr)

                    # Baseline comparison (non-paged DecoderLM)
                    baseline_info = {}
                    if args.compare_baseline:
                        try:
                            baseline_model = _create_baseline_model(
                                args.n_vocab, args.n_embd, args.n_head,
                                args.n_positions, args.n_layers, backend,
                            )
                            baseline_info = benchmark_baseline_throughput(
                                baseline_model,
                                prompt,
                                args.max_new_tokens,
                                backend,
                            )
                            total_speedup = (
                                baseline_info["baseline_total_time_s"] / throughput["total_time_s"]
                                if throughput["total_time_s"] > 0
                                else 0
                            )
                            decode_speedup = (
                                baseline_info["baseline_time_per_token_ms"] / throughput["time_per_token_ms"]
                                if throughput["time_per_token_ms"] > 0
                                else 0
                            )
                            baseline_info["total_speedup_vs_baseline"] = round(total_speedup, 2)
                            baseline_info["decode_speedup_vs_baseline"] = round(decode_speedup, 2)
                        except Exception as e:
                            print(f"\n    Baseline error: {e}")

                    prefix_info = {}
                    if args.compare_prefix_cache:
                        try:
                            prefix_info = benchmark_prefix_cache_prefill(
                                args.n_vocab,
                                args.n_embd,
                                args.n_head,
                                args.n_layers,
                                args.n_positions,
                                args.num_kv_blocks,
                                block_size,
                                seq_len,
                                backend,
                                args.decode_backend,
                                args.compare_to_ref,
                                args.compare_tolerance,
                                args.prefix_shared_ratio,
                            )
                        except Exception as e:
                            print(f"\n    Prefix-cache error: {e}")
                    result.update(prefix_info)
                    result.update(baseline_info)
                    all_results.append(result)

                    msg = (f"OK  {throughput['tokens_per_sec']:.1f} tok/s, "
                           f"int_frag={frag['internal_frag']:.3f}, "
                           f"decode={throughput['time_per_token_ms']:.1f}ms/tok, "
                           f"kv_eff={throughput['kv_efficiency']:.3f}")
                    if baseline_info:
                        msg += (
                            f" | baseline total={baseline_info['baseline_total_time_s']:.3f}s"
                            f" decode={baseline_info['baseline_time_per_token_ms']:.1f}ms/tok"
                            f" total_spd={baseline_info.get('total_speedup_vs_baseline', 0):.2f}x"
                            f" decode_spd={baseline_info.get('decode_speedup_vs_baseline', 0):.2f}x"
                        )
                    if prefix_info:
                        msg += (
                            f" | prefix_hit={prefix_info['prefix_cached_tokens']}"
                            f" prefill_spd={prefix_info['prefix_prefill_speedup']:.2f}x"
                        )
                    print(msg)

                except Exception as e:
                    print(f"ERROR: {e}")
                    traceback.print_exc()

        # --- Max batch size search (once per block size per seq_len) ---
        if not args.skip_max_batch:
            for seq_len in args.seq_lengths:
                max_bs = benchmark_max_batch_size(
                    args.n_vocab, args.n_embd, args.n_head,
                    args.n_layers, args.n_positions,
                    args.num_kv_blocks, block_size, seq_len,
                    args.backend,
                )
                print(f"  [block_size={block_size}, seq={seq_len}] "
                      f"Max batch size: {max_bs}")

    # ---------------- Dedicated fragmentation sweep ----------------
    # Use seq lengths that DO NOT divide evenly by the block sizes so
    # internal fragmentation is non-zero and visible in the report.
    frag_results: List[Dict] = []
    if not args.skip_frag_sweep:
        print("\n" + "=" * 70)
        print("FRAGMENTATION SWEEP (non-aligned seq lengths)")
        print("=" * 70)
        for block_size in args.block_sizes:
            for seq_len in args.frag_seq_lengths:
                for batch_size in args.batch_sizes:
                    bm = _create_block_manager(
                        args.num_kv_blocks, block_size,
                        args.n_head, head_dim,
                        args.n_layers,
                    )
                    blocks_needed = batch_size * (
                        (seq_len + block_size - 1) // block_size
                    )
                    if blocks_needed > args.num_kv_blocks:
                        continue
                    frag_row = benchmark_fragmentation(
                        bm, batch_size, seq_len, block_size,
                        contiguous_max_seq_len=args.n_positions,
                    )
                    row = {
                        "block_size": block_size,
                        "batch_size": batch_size,
                        "seq_len": seq_len,
                        "n_positions": args.n_positions,
                        **frag_row,
                    }
                    frag_results.append(row)
                    print(f"  [bs={block_size}, batch={batch_size}, seq={seq_len}] "
                          f"int_frag={frag_row['internal_frag']:.4f} "
                          f"savings_vs_contig={frag_row['memory_savings_ratio']:.4f} "
                          f"({frag_row['kv_bytes_paged']/1024:.1f}KB paged "
                          f"vs {frag_row['kv_bytes_contiguous_naive']/1024:.1f}KB contig)")

        if frag_results:
            frag_file = os.path.join(args.output_dir, "fragmentation_results.csv")
            with open(frag_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(frag_results[0].keys()))
                writer.writeheader()
                writer.writerows(frag_results)
            print(f"\nFragmentation results written to {frag_file}")

    # Write main CSV
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(results_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults written to {results_file}")
    else:
        print("\nNo results collected.")

    # Print summary table
    if all_results:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        has_baseline = any("baseline_tokens_per_sec" in r for r in all_results)
        has_prefix = any("prefix_prefill_speedup" in r for r in all_results)
        header = (f"{'BS':>3} {'Batch':>5} {'SeqLen':>6} "
                  f"{'tok/s':>8} {'ms/tok':>8} {'p50':>6} {'p95':>6} "
                  f"{'IntFrag':>8} {'Util':>6} {'KVeff':>6} {'MemSav':>7}")
        if has_baseline:
            header += (
                f"  {'BL tok/s':>8} {'BL ms/tok':>9}"
                f" {'TotSpd':>7} {'DecSpd':>7}"
            )
        if has_prefix:
            header += f" {'PHit':>5} {'PSpd':>6}"
        print(header)
        print("-" * (len(header) + 4))
        for r in all_results:
            row = (f"{r['block_size']:>3} {r['batch_size']:>5} "
                   f"{r['seq_len']:>6} "
                   f"{r['tokens_per_sec']:>8.1f} "
                   f"{r['time_per_token_ms']:>8.1f} "
                   f"{r.get('decode_p50_ms', 0):>6.1f} "
                   f"{r.get('decode_p95_ms', 0):>6.1f} "
                   f"{r['internal_frag']:>8.4f} "
                   f"{r['utilization']:>6.3f} "
                   f"{r['kv_efficiency']:>6.3f} "
                   f"{r.get('memory_savings_ratio', 0):>7.3f}")
            if has_baseline and "baseline_time_per_token_ms" in r:
                row += (
                    f"  {r['baseline_tokens_per_sec']:>8.1f}"
                    f" {r['baseline_time_per_token_ms']:>9.1f}"
                    f" {r.get('total_speedup_vs_baseline', 0):>6.2f}x"
                    f" {r.get('decode_speedup_vs_baseline', 0):>6.2f}x"
                )
            if has_prefix and "prefix_prefill_speedup" in r:
                row += (
                    f" {r['prefix_cached_tokens']:>5}"
                    f" {r['prefix_prefill_speedup']:>5.2f}x"
                )
            print(row)


if __name__ == "__main__":
    main()
