"""Generate plots from the benchmark CSV outputs.

Reads:
  - benchmarks/results/benchmark_results.csv
  - benchmarks/results/fragmentation_results.csv (optional)

Writes PNGs into benchmarks/plots/.

Usage:
    python project/plot.py
"""

from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS_DIR = "benchmarks/results"
PLOTS_DIR = "benchmarks/plots"

MAIN_CSV = os.path.join(RESULTS_DIR, "benchmark_results.csv")
FRAG_CSV = os.path.join(RESULTS_DIR, "fragmentation_results.csv")


def _read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _f(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    v = row.get(key)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _i(row: Dict[str, str], key: str, default: int = 0) -> int:
    return int(_f(row, key, default))


def plot_throughput_vs_batch(rows, out_path):
    """tok/s vs batch_size, one line per (block_size, seq_len)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    series = defaultdict(list)  # (bs, seq) -> [(batch, tok/s)]
    for r in rows:
        key = (_i(r, "block_size"), _i(r, "seq_len"))
        series[key].append((_i(r, "batch_size"), _f(r, "tokens_per_sec")))
    for (bs, seq), pts in sorted(series.items()):
        pts.sort()
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", label=f"block={bs}, seq={seq}")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Throughput (tokens / sec)")
    ax.set_title("Generation throughput vs batch size")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_latency_vs_seqlen(rows, out_path):
    """ms/token decode vs seq_len, one line per (block_size, batch_size).

    Uses p95 if available, otherwise mean ms/tok.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    series = defaultdict(list)
    for r in rows:
        key = (_i(r, "block_size"), _i(r, "batch_size"))
        latency = _f(r, "decode_p95_ms") or _f(r, "time_per_token_ms")
        series[key].append((_i(r, "seq_len"), latency))
    for (bs, batch), pts in sorted(series.items()):
        pts.sort()
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="s", label=f"block={bs}, batch={batch}")
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Decode latency (ms / token, p95)")
    ax.set_title("Decode latency vs sequence length")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_speedup_vs_seqlen(rows, out_path):
    """Speedup over the no-cache baseline vs seq_len."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    series = defaultdict(list)
    for r in rows:
        speedup = _f(r, "speedup_vs_baseline")
        if speedup <= 0:
            continue
        key = (_i(r, "block_size"), _i(r, "batch_size"))
        series[key].append((_i(r, "seq_len"), speedup))
    if not series:
        plt.close(fig)
        return False
    for (bs, batch), pts in sorted(series.items()):
        pts.sort()
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="^", label=f"block={bs}, batch={batch}")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Speedup vs no-KV-cache baseline (×)")
    ax.set_title("Decode speedup vs no-cache baseline")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def plot_fragmentation_vs_blocksize(frag_rows, out_path):
    """Internal fragmentation vs block_size, one bar group per seq_len."""
    if not frag_rows:
        return False
    fig, ax = plt.subplots(figsize=(7, 4.5))
    # Group by seq_len, take batch_size==1 row per (block, seq) for clarity
    grouped = defaultdict(dict)  # seq -> {block_size: internal_frag}
    for r in frag_rows:
        if _i(r, "batch_size") != 1:
            continue
        grouped[_i(r, "seq_len")][_i(r, "block_size")] = _f(r, "internal_frag")
    seqs = sorted(grouped.keys())
    block_sizes = sorted({bs for s in grouped.values() for bs in s.keys()})
    width = 0.8 / max(len(seqs), 1)
    x = list(range(len(block_sizes)))
    for i, seq in enumerate(seqs):
        ys = [grouped[seq].get(bs, 0.0) for bs in block_sizes]
        offsets = [xi + (i - (len(seqs) - 1) / 2) * width for xi in x]
        ax.bar(offsets, ys, width=width, label=f"seq={seq}")
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in block_sizes])
    ax.set_xlabel("Block size")
    ax.set_ylabel("Internal fragmentation (fraction of allocated slots wasted)")
    ax.set_title("Internal fragmentation vs block size (non-aligned seq lengths)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def plot_kv_memory_vs_contiguous(frag_rows, out_path):
    """Paged KV bytes vs naive contiguous KV bytes, per (block_size, batch)."""
    if not frag_rows:
        return False
    fig, ax = plt.subplots(figsize=(7, 4.5))
    # Pick block_size == max available, plot vs batch_size for one seq_len
    if not frag_rows:
        return False
    seqs = sorted({_i(r, "seq_len") for r in frag_rows})
    target_seq = seqs[len(seqs) // 2]  # median
    blocks = sorted({_i(r, "block_size") for r in frag_rows})
    target_block = blocks[-1]

    rows_to_plot = sorted(
        (r for r in frag_rows
         if _i(r, "seq_len") == target_seq and _i(r, "block_size") == target_block),
        key=lambda r: _i(r, "batch_size"),
    )
    if not rows_to_plot:
        return False

    batches = [_i(r, "batch_size") for r in rows_to_plot]
    paged = [_f(r, "kv_bytes_paged") / 1024 for r in rows_to_plot]
    contig = [_f(r, "kv_bytes_contiguous_naive") / 1024 for r in rows_to_plot]

    x = list(range(len(batches)))
    width = 0.35
    ax.bar([xi - width / 2 for xi in x], paged, width=width,
           label="Paged (actual)", color="#2a7ae2")
    ax.bar([xi + width / 2 for xi in x], contig, width=width,
           label=f"Naive contiguous (n_positions={_i(rows_to_plot[0], 'n_positions')})",
           color="#e2a72a")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in batches])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("KV cache bytes (KB, log scale)")
    ax.set_title(
        f"KV memory: paged vs naive contiguous "
        f"(block_size={target_block}, seq_len={target_seq})"
    )
    ax.grid(True, axis="y", alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    rows = _read_csv(MAIN_CSV)
    frag_rows = _read_csv(FRAG_CSV)

    if not rows and not frag_rows:
        print(f"ERROR: no CSVs found. Run project/run_benchmark.py first.")
        sys.exit(1)

    written = []

    if rows:
        p1 = os.path.join(PLOTS_DIR, "throughput_vs_batch.png")
        plot_throughput_vs_batch(rows, p1); written.append(p1)
        p2 = os.path.join(PLOTS_DIR, "latency_vs_seqlen.png")
        plot_latency_vs_seqlen(rows, p2); written.append(p2)
        p3 = os.path.join(PLOTS_DIR, "speedup_vs_seqlen.png")
        if plot_speedup_vs_seqlen(rows, p3):
            written.append(p3)

    if frag_rows:
        p4 = os.path.join(PLOTS_DIR, "fragmentation_vs_blocksize.png")
        if plot_fragmentation_vs_blocksize(frag_rows, p4):
            written.append(p4)
        p5 = os.path.join(PLOTS_DIR, "kv_memory_vs_contiguous.png")
        if plot_kv_memory_vs_contiguous(frag_rows, p5):
            written.append(p5)

    print("Wrote:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
