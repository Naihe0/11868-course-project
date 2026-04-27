"""Plot GPU-resident KV cache benchmark results.

Reads:
    benchmarks/results_gpu_kv/gpu_kv_decode_attention.csv

Writes:
    benchmarks/results_gpu_kv/gpu_kv_decode_attention.png
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PALETTE = {
    "gpu_resident": "#1C7293",
    "stateless": "#E89B26",
    "speedup": "#065A82",
    "grid": "#A9A9A9",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 170,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot GPU KV benchmark results")
    parser.add_argument(
        "--input",
        default="benchmarks/results_gpu_kv/gpu_kv_decode_attention.csv",
        help="Input CSV produced by project/run_gpu_kv_benchmark.py",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/results_gpu_kv/gpu_kv_decode_attention.png",
        help="Output PNG path",
    )
    return parser.parse_args()


def _read_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(value: str) -> float:
    return float(value) if value not in {"", "None", None} else np.nan


def plot_gpu_kv(rows: List[Dict[str, str]], out_path: str) -> None:
    if not rows:
        raise ValueError("No rows to plot")

    labels = [f"bs={row['batch_size']}\nseq={row['seq_len']}" for row in rows]
    gpu_ms = np.array([_to_float(row["gpu_resident_median_ms"]) for row in rows])
    stateless_ms = np.array([_to_float(row["stateless_full_copy_median_ms"]) for row in rows])
    speedup = np.array([_to_float(row["speedup_vs_stateless_full_copy"]) for row in rows])
    us_per_query_token = np.array([
        _to_float(row["gpu_resident_us_per_query_token"]) for row in rows
    ])

    x = np.arange(len(rows))
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.0), gridspec_kw={"width_ratios": [1.45, 1.05]})
    latency_ax, speedup_ax = axes

    latency_ax.bar(
        x - width / 2,
        gpu_ms,
        width,
        color=PALETTE["gpu_resident"],
        label="GPU-resident KV runtime",
    )
    latency_ax.bar(
        x + width / 2,
        stateless_ms,
        width,
        color=PALETTE["stateless"],
        label="Stateless full-KV copy",
    )
    for i, value in enumerate(gpu_ms):
        latency_ax.text(
            i - width / 2,
            value + 0.04,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            rotation=90,
        )

    latency_ax.set_xticks(x)
    latency_ax.set_xticklabels(labels, rotation=0)
    latency_ax.set_ylabel("Median decode-attention call time (ms)")
    latency_ax.set_title("GPU-resident KV avoids repeated full-cache transfers")
    latency_ax.grid(axis="y", linestyle="--", color=PALETTE["grid"], alpha=0.35)
    latency_ax.legend(loc="upper left")
    latency_ax.set_ylim(0, max(stateless_ms) * 1.2)

    speedup_ax.plot(
        x,
        speedup,
        marker="o",
        linewidth=2.4,
        color=PALETTE["speedup"],
        label="Speedup vs stateless",
    )
    speedup_ax.axhspan(2.0, 4.0, color=PALETTE["gpu_resident"], alpha=0.10)
    speedup_ax.text(
        0.15,
        3.82,
        "2-4x range",
        color=PALETTE["speedup"],
        fontsize=9,
        fontweight="bold",
    )
    for i, value in enumerate(speedup):
        speedup_ax.annotate(
            f"{value:.2f}x",
            xy=(i, value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=PALETTE["speedup"],
            fontweight="bold",
        )

    twin_ax = speedup_ax.twinx()
    twin_ax.plot(
        x,
        us_per_query_token,
        marker="s",
        linewidth=1.8,
        linestyle="--",
        color="#777777",
        label="GPU runtime us/query token",
    )
    twin_ax.set_ylabel("GPU-resident latency per query token (us)", color="#555555")
    twin_ax.tick_params(axis="y", labelcolor="#555555")
    twin_ax.spines["top"].set_visible(False)

    speedup_ax.set_xticks(x)
    speedup_ax.set_xticklabels(labels, rotation=28, ha="right")
    speedup_ax.set_ylabel("Speedup")
    speedup_ax.set_title("GPU-resident KV gives 2.66-3.98x speedup here")
    speedup_ax.set_ylim(0, max(4.2, np.nanmax(speedup) * 1.15))
    speedup_ax.grid(axis="y", linestyle="--", color=PALETTE["grid"], alpha=0.35)

    handles, names = speedup_ax.get_legend_handles_labels()
    twin_handles, twin_names = twin_ax.get_legend_handles_labels()
    speedup_ax.legend(handles + twin_handles, names + twin_names, loc="lower right")

    fig.suptitle(
        "GPU-only KV-cache microbenchmark: K/V uploaded once, decode reads device-resident cache",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        -0.02,
        "Measured through the ctypes runtime wrapper; query H2D and output D2H copies are still included in each call.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    args = parse_args()
    rows = _read_rows(args.input)
    plot_gpu_kv(rows, args.output)


if __name__ == "__main__":
    main()