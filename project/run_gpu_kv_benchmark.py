"""Benchmark PagedAttention decode with GPU-resident KV cache.

This script isolates the custom CUDA PagedAttention runtime from the full
MiniTorch transformer.  It uploads a paged KV cache into the CUDA runtime once,
then times repeated decode-attention calls that read K/V from device memory.

For comparison, it can also run the older stateless wrapper path that copies
the full host KV cache into CUDA memory for every attention call.

Example:
    python project/run_gpu_kv_benchmark.py \
        --batch-sizes 1 2 4 \
        --seq-lengths 32 64 128 \
        --block-size 16 \
        --n-head 8 \
        --head-dim 64 \
        --timed-iters 20
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, ".")

import minitorch
from minitorch.paged_attention import PagedAttentionKernel
from minitorch.tensor import tensor_from_numpy

datatype = np.float32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark CUDA PagedAttention with GPU-resident KV cache"
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--seq-lengths", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--timed-iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260426)
    parser.add_argument("--output-dir", type=str, default="benchmarks/results_gpu_kv")
    parser.add_argument(
        "--skip-stateless",
        action="store_true",
        help="Skip the full-KV-copy stateless comparison.",
    )
    return parser.parse_args()


def _cuda_available() -> bool:
    try:
        import numba.cuda

        return bool(numba.cuda.is_available())
    except Exception:
        return False


def _make_block_tables(
    batch_size: int,
    seq_len: int,
    block_size: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    blocks_per_seq = math.ceil(seq_len / block_size)
    block_tables = np.zeros((batch_size, blocks_per_seq), dtype=np.int32)
    for batch_idx in range(batch_size):
        start = batch_idx * blocks_per_seq
        block_tables[batch_idx] = np.arange(
            start,
            start + blocks_per_seq,
            dtype=np.int32,
        )
    context_lens = np.full((batch_size,), seq_len, dtype=np.int32)
    return block_tables, context_lens, batch_size * blocks_per_seq


def _time_call(fn, warmup_iters: int, timed_iters: int) -> Dict[str, float]:
    for _ in range(warmup_iters):
        fn()

    samples = []
    for _ in range(timed_iters):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)

    arr = np.array(samples, dtype=np.float64)
    return {
        "median_s": float(np.median(arr)),
        "min_s": float(np.min(arr)),
        "max_s": float(np.max(arr)),
        "mean_s": float(np.mean(arr)),
    }


def _benchmark_one(
    batch_size: int,
    seq_len: int,
    block_size: int,
    n_head: int,
    head_dim: int,
    warmup_iters: int,
    timed_iters: int,
    rng: np.random.Generator,
    run_stateless: bool,
) -> Dict[str, object]:
    backend = minitorch.TensorBackend(minitorch.FastOps)
    block_tables, context_lens, num_blocks = _make_block_tables(
        batch_size,
        seq_len,
        block_size,
    )

    cache_shape = (num_blocks, block_size, n_head, head_dim)
    key_cache = rng.standard_normal(cache_shape, dtype=datatype)
    value_cache = rng.standard_normal(cache_shape, dtype=datatype)
    query_np = rng.standard_normal((batch_size, n_head, 1, head_dim), dtype=datatype)
    query = tensor_from_numpy(query_np, backend=backend)

    runtime_kernel = PagedAttentionKernel()
    runtime_kernel.ensure_runtime(
        num_blocks=num_blocks,
        block_size=block_size,
        n_head=n_head,
        head_dim=head_dim,
        max_batch=batch_size,
        max_blocks_per_seq=block_tables.shape[1],
    )
    runtime_kernel.upload_layer_cache(key_cache, value_cache)
    runtime_kernel.update_metadata(block_tables, context_lens)

    runtime_stats = _time_call(
        lambda: runtime_kernel.forward(query, max_context_len=seq_len),
        warmup_iters,
        timed_iters,
    )
    runtime_kernel.close()

    stateless_stats = None
    if run_stateless:
        stateless_kernel = PagedAttentionKernel()
        stateless_stats = _time_call(
            lambda: stateless_kernel.forward(
                query,
                key_cache=key_cache,
                value_cache=value_cache,
                block_tables=block_tables,
                context_lens=context_lens,
                block_size=block_size,
                max_context_len=seq_len,
            ),
            warmup_iters,
            timed_iters,
        )
        stateless_kernel.close()

    live_tokens = batch_size * seq_len
    reserved_tokens = num_blocks * block_size
    kv_bytes = int(num_blocks * block_size * n_head * head_dim * 2 * np.dtype(datatype).itemsize)
    runtime_ms = runtime_stats["median_s"] * 1000.0
    runtime_us_per_query_token = runtime_stats["median_s"] * 1e6 / max(batch_size, 1)

    row: Dict[str, object] = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "block_size": block_size,
        "n_head": n_head,
        "head_dim": head_dim,
        "num_blocks": num_blocks,
        "live_tokens": live_tokens,
        "reserved_tokens": reserved_tokens,
        "kv_bytes": kv_bytes,
        "gpu_resident_median_ms": round(runtime_ms, 4),
        "gpu_resident_min_ms": round(runtime_stats["min_s"] * 1000.0, 4),
        "gpu_resident_max_ms": round(runtime_stats["max_s"] * 1000.0, 4),
        "gpu_resident_mean_ms": round(runtime_stats["mean_s"] * 1000.0, 4),
        "gpu_resident_us_per_query_token": round(runtime_us_per_query_token, 4),
    }

    if stateless_stats is not None:
        stateless_ms = stateless_stats["median_s"] * 1000.0
        row.update(
            {
                "stateless_full_copy_median_ms": round(stateless_ms, 4),
                "stateless_full_copy_min_ms": round(stateless_stats["min_s"] * 1000.0, 4),
                "stateless_full_copy_max_ms": round(stateless_stats["max_s"] * 1000.0, 4),
                "speedup_vs_stateless_full_copy": round(
                    stateless_stats["median_s"] / runtime_stats["median_s"],
                    3,
                )
                if runtime_stats["median_s"] > 0
                else None,
            }
        )

    return row


def _write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not _cuda_available():
        raise RuntimeError("CUDA is not available; cannot run GPU KV benchmark")

    rng = np.random.default_rng(args.seed)
    rows = []
    for batch_size in args.batch_sizes:
        for seq_len in args.seq_lengths:
            row = _benchmark_one(
                batch_size=batch_size,
                seq_len=seq_len,
                block_size=args.block_size,
                n_head=args.n_head,
                head_dim=args.head_dim,
                warmup_iters=args.warmup_iters,
                timed_iters=args.timed_iters,
                rng=rng,
                run_stateless=not args.skip_stateless,
            )
            rows.append(row)
            msg = (
                f"batch={batch_size}, seq={seq_len}: "
                f"gpu_resident={row['gpu_resident_median_ms']} ms"
            )
            if "stateless_full_copy_median_ms" in row:
                msg += (
                    f", stateless_full_copy={row['stateless_full_copy_median_ms']} ms, "
                    f"speedup={row['speedup_vs_stateless_full_copy']}x"
                )
            print(msg)

    out_path = os.path.join(args.output_dir, "gpu_kv_decode_attention.csv")
    _write_csv(out_path, rows)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()