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
import os
import sys
import time
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, ".")

import minitorch
from minitorch.transformer import PagedDecoderLM
from minitorch.block_manager import BlockManager
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
    parser.add_argument("--skip-correctness", action="store_true",
                        help="Skip correctness check to speed up benchmarking")
    parser.add_argument("--skip-max-batch", action="store_true",
                        help="Skip max batch size search")
    parser.add_argument("--compare-baseline", action="store_true",
                        help="Also benchmark the non-paged DecoderLM (hw3/hw4) "
                             "for throughput comparison")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_backend(backend_name: str):
    if backend_name == "cuda":
        return minitorch.TensorBackend(minitorch.CudaKernelOps)
    return minitorch.TensorBackend(minitorch.FastOps)


def _create_model(n_vocab, n_embd, n_head, n_positions, n_layers,
                   block_size, backend):
    return PagedDecoderLM(
        n_vocab=n_vocab,
        n_embd=n_embd,
        n_head=n_head,
        n_positions=n_positions,
        n_layers=n_layers,
        block_size=block_size,
        p_dropout=0.0,
        backend=backend,
    )


def _create_block_manager(num_kv_blocks, block_size, n_head, head_dim):
    return BlockManager(
        num_blocks=num_kv_blocks,
        block_size=block_size,
        n_head=n_head,
        head_dim=head_dim,
    )


def _make_prompt(batch_size, seq_len, n_vocab, backend):
    prompt_np = np.random.randint(0, n_vocab, size=(batch_size, seq_len)).astype(datatype)
    return tensor_from_numpy(prompt_np, backend=backend)


def _create_baseline_model(n_vocab, n_embd, n_head, n_positions, backend):
    """Create a non-paged DecoderLM (hw3/hw4 baseline, always 4 layers)."""
    return DecoderLM(
        n_vocab=n_vocab,
        n_embd=n_embd,
        n_head=n_head,
        n_positions=n_positions,
        p_dropout=0.0,
        backend=backend,
    )


# ---------------------------------------------------------------------------
# Benchmark routines
# ---------------------------------------------------------------------------

def benchmark_throughput(
    model, block_manager, prompt, seq_ids, max_new_tokens, backend,
) -> Dict[str, float]:
    """Measure generation throughput and latency.

    Returns dict with: tokens_per_sec, total_time_s, time_per_token_ms,
                       prefill_time_s, decode_time_s
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

    # --- Decode timing ---
    prompt_len = prompt.shape[1]
    decode_start = time.perf_counter()
    for step in range(1, max_new_tokens):
        token_input = tensor_from_numpy(
            next_tokens.reshape(batch_size, 1).astype(datatype),
            backend=model.backend,
        )
        start_pos = prompt_len + step - 1
        logits = model.forward_decode(
            token_input, block_manager, seq_ids, start_pos=start_pos,
        )
        logits_np = logits.to_numpy()[:, 0, :]
        next_tokens = np.argmax(logits_np, axis=-1)
    decode_end = time.perf_counter()
    decode_time = decode_end - decode_start

    total_time = prefill_time + decode_time
    total_generated = batch_size * max_new_tokens
    decode_tokens = batch_size * (max_new_tokens - 1) if max_new_tokens > 1 else 0

    # Free sequences for reuse
    for seq_id in seq_ids:
        block_manager.free_sequence(seq_id)

    return {
        "total_time_s": round(total_time, 4),
        "prefill_time_s": round(prefill_time, 4),
        "decode_time_s": round(decode_time, 4),
        "tokens_per_sec": round(total_generated / total_time, 2) if total_time > 0 else 0,
        "decode_tokens_per_sec": round(decode_tokens / decode_time, 2) if decode_time > 0 else 0,
        "time_per_token_ms": round((decode_time / decode_tokens) * 1000, 2) if decode_tokens > 0 else 0,
    }


def benchmark_baseline_throughput(
    model, prompt, max_new_tokens, n_vocab, backend,
) -> Dict[str, float]:
    """Measure generation throughput for the non-paged baseline DecoderLM.

    The baseline has no KV cache, so autoregressive generation feeds the
    *entire* accumulated sequence through forward() at each step.

    Returns dict with: baseline_total_time_s, baseline_tokens_per_sec,
                       baseline_time_per_token_ms
    """
    batch_size, seq_len = prompt.shape
    # Start with prompt tokens as lists
    generated_np = prompt.to_numpy().astype(datatype)

    model.eval()
    start = time.perf_counter()

    for step in range(max_new_tokens):
        # Feed entire sequence so far
        input_t = tensor_from_numpy(generated_np, backend=backend)
        logits = model.forward(input_t)
        # Greedy: take argmax at last position
        last_logits = logits.to_numpy()[:, -1, :]
        next_tokens = np.argmax(last_logits, axis=-1).astype(datatype)
        generated_np = np.concatenate(
            [generated_np, next_tokens.reshape(batch_size, 1)], axis=1,
        )

    end = time.perf_counter()
    total_time = end - start
    total_generated = batch_size * max_new_tokens

    return {
        "baseline_total_time_s": round(total_time, 4),
        "baseline_tokens_per_sec": round(total_generated / total_time, 2) if total_time > 0 else 0,
        "baseline_time_per_token_ms": round((total_time / total_generated) * 1000, 2) if total_generated > 0 else 0,
    }


def benchmark_fragmentation(block_manager, batch_size, seq_len,
                             block_size) -> Dict[str, float]:
    """Measure internal and external memory fragmentation after allocation.

    Allocates blocks for `batch_size` sequences of `seq_len` tokens,
    measures fragmentation, then frees them.
    """
    seq_ids = list(range(1000, 1000 + batch_size))
    for seq_id in seq_ids:
        block_manager.allocate_blocks_for_sequence(seq_id, seq_len)

    frag = block_manager.compute_fragmentation()

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
    n_head, head_dim, block_size, num_kv_blocks, backend,
) -> Dict[str, float]:
    """Verify that paged attention matches standard attention.

    Creates random Q/K/V, populates a paged cache, and compares outputs
    between numpy manual_attention and paged_attention_ref.
    """
    batch_size = 2
    seq_len = 17  # Odd number to test partial blocks

    np.random.seed(42)
    q_np = np.random.randn(batch_size, n_head, 1, head_dim).astype(datatype)
    keys_by_seq = [
        np.random.randn(seq_len, n_head, head_dim).astype(datatype)
        for _ in range(batch_size)
    ]
    values_by_seq = [
        np.random.randn(seq_len, n_head, head_dim).astype(datatype)
        for _ in range(batch_size)
    ]

    # Build paged cache (same approach as test suite)
    total_blocks_needed = batch_size * ((seq_len + block_size - 1) // block_size)
    key_cache = np.zeros((max(num_kv_blocks, total_blocks_needed),
                          block_size, n_head, head_dim), dtype=datatype)
    value_cache = np.zeros_like(key_cache)
    block_tables = []
    next_block_id = 0

    for b in range(batch_size):
        seq_block_ids = []
        for start in range(0, seq_len, block_size):
            end = min(start + block_size, seq_len)
            key_cache[next_block_id, :end - start] = keys_by_seq[b][start:end]
            value_cache[next_block_id, :end - start] = values_by_seq[b][start:end]
            seq_block_ids.append(next_block_id)
            next_block_id += 1
        block_tables.append(seq_block_ids)

    context_lens = [seq_len] * batch_size

    # Paged attention output
    q_t = tensor_from_numpy(q_np, backend=backend)
    paged_out = paged_attention_ref(
        q_t, key_cache, value_cache,
        block_tables, context_lens,
        block_size=block_size, n_head=n_head, head_dim=head_dim,
    ).to_numpy()

    # NumPy reference (per-batch, matches test approach)
    expected_parts = []
    for b in range(batch_size):
        k_ref = keys_by_seq[b][None].transpose(0, 2, 1, 3)
        v_ref = values_by_seq[b][None].transpose(0, 2, 1, 3)
        expected_parts.append(
            _manual_attention(q_np[b:b+1], k_ref, v_ref)
        )
    expected = np.concatenate(expected_parts, axis=0)

    max_abs_err = float(np.max(np.abs(paged_out - expected)))
    mean_abs_err = float(np.mean(np.abs(paged_out - expected)))
    matches = max_abs_err < 1e-4

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
                args.n_head, head_dim, block_size, args.num_kv_blocks, backend,
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
                    )
                    bm = _create_block_manager(
                        args.num_kv_blocks, block_size,
                        args.n_head, head_dim,
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
                    )
                    frag = benchmark_fragmentation(
                        bm_frag, batch_size, seq_len, block_size,
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
                                args.n_positions, backend,
                            )
                            baseline_prompt = _make_prompt(
                                batch_size, seq_len, args.n_vocab, backend,
                            )
                            baseline_info = benchmark_baseline_throughput(
                                baseline_model, baseline_prompt,
                                args.max_new_tokens, args.n_vocab, backend,
                            )
                            speedup = (baseline_info["baseline_time_per_token_ms"]
                                       / throughput["time_per_token_ms"]
                                       if throughput["time_per_token_ms"] > 0
                                       else 0)
                            baseline_info["speedup_vs_baseline"] = round(speedup, 2)
                        except Exception as e:
                            print(f"\n    Baseline error: {e}")
                    result.update(baseline_info)
                    all_results.append(result)

                    msg = (f"OK  {throughput['tokens_per_sec']:.1f} tok/s, "
                           f"int_frag={frag['internal_frag']:.3f}, "
                           f"decode={throughput['time_per_token_ms']:.1f}ms/tok")
                    if baseline_info:
                        msg += (f" | baseline={baseline_info['baseline_time_per_token_ms']:.1f}ms/tok"
                                f" speedup={baseline_info.get('speedup_vs_baseline', 0):.2f}x")
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

    # Write CSV
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
        header = (f"{'BS':>3} {'Batch':>5} {'SeqLen':>6} "
                  f"{'tok/s':>8} {'ms/tok':>8} "
                  f"{'IntFrag':>8} {'Util':>6}")
        if has_baseline:
            header += f"  {'BL ms/tok':>9} {'Speedup':>7}"
        print(header)
        print("-" * (70 if not has_baseline else 90))
        for r in all_results:
            row = (f"{r['block_size']:>3} {r['batch_size']:>5} "
                   f"{r['seq_len']:>6} "
                   f"{r['tokens_per_sec']:>8.1f} "
                   f"{r['time_per_token_ms']:>8.1f} "
                   f"{r['internal_frag']:>8.4f} "
                   f"{r['utilization']:>6.3f}")
            if has_baseline and "baseline_time_per_token_ms" in r:
                row += (f"  {r['baseline_time_per_token_ms']:>9.1f} "
                        f"{r.get('speedup_vs_baseline', 0):>6.2f}x")
            print(row)


if __name__ == "__main__":
    main()
