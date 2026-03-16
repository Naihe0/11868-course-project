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
from typing import Dict, List

import numpy as np

sys.path.insert(0, ".")


def parse_args():
    parser = argparse.ArgumentParser(description="PagedAttention Benchmark")
    parser.add_argument("--n-vocab", type=int, default=10000)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-positions", type=int, default=1024)
    parser.add_argument("--num-kv-blocks", type=int, default=512)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--seq-lengths", type=int, nargs="+",
                        default=[128, 256, 512, 1024])
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="benchmarks/results")
    parser.add_argument("--backend", type=str, default="cpu",
                        choices=["cpu", "cuda"])
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Benchmark routines
# ---------------------------------------------------------------------------

def benchmark_throughput(
    model, block_manager, prompt, seq_ids, max_new_tokens, backend,
) -> Dict[str, float]:
    """Measure generation throughput and latency."""
    # TODO: Implement throughput benchmark
    # Returns dict with: tokens_per_sec, total_time_s, time_per_token_ms
    raise NotImplementedError


def benchmark_fragmentation(block_manager) -> Dict[str, float]:
    """Measure internal and external memory fragmentation."""
    # TODO: Implement fragmentation measurement
    raise NotImplementedError


def benchmark_max_batch_size(
    model_cls, model_kwargs, block_manager_kwargs, seq_len, backend,
) -> int:
    """Find the maximum batch size before OOM."""
    # TODO: Implement binary search for max batch size
    raise NotImplementedError


def check_correctness(
    model_paged, model_standard, prompt, backend,
) -> bool:
    """Verify that paged attention output matches standard attention."""
    # TODO: Implement correctness check
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results_file = os.path.join(args.output_dir, "benchmark_results.csv")

    print("=" * 60)
    print("PagedAttention Benchmark")
    print("=" * 60)
    print(f"Model: vocab={args.n_vocab}, embd={args.n_embd}, "
          f"head={args.n_head}, layers={args.n_layers}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Seq lengths: {args.seq_lengths}")
    print(f"Block sizes: {args.block_sizes}")
    print(f"Output: {results_file}")
    print()

    all_results: List[Dict] = []

    for block_size in args.block_sizes:
        for batch_size in args.batch_sizes:
            for seq_len in args.seq_lengths:
                config = {
                    "block_size": block_size,
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                }
                print(f"Running: {config} ... ", end="", flush=True)

                # TODO: Create model, block manager, run benchmarks
                # result = {**config, **throughput, **fragmentation}
                # all_results.append(result)

                print("SKIPPED (not yet implemented)")

    # Write CSV
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(results_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults written to {results_file}")
    else:
        print("\nNo results collected (benchmarks not yet implemented).")


if __name__ == "__main__":
    main()
