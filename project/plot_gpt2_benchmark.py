"""Plot MiniTorch GPT-2 PagedAttention benchmark results.

Reads one or more CSVs produced by ``project/run_gpt2_paged_benchmark.py`` and
writes three figures:

    benchmarks/results_gpt2/gpt2_latency.png
    benchmarks/results_gpt2/gpt2_throughput.png
    benchmarks/results_gpt2/gpt2_kv_memory.png
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PALETTE = {
    "prefill": "#1C7293",
    "decode": "#E89B26",
    "throughput": "#065A82",
    "decode_throughput": "#A64942",
    "paged_allocated": "#1C7293",
    "paged_live": "#BBD6E4",
    "contiguous": "#E89B26",
    "savings": "#065A82",
    "fragmentation": "#A64942",
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
    parser = argparse.ArgumentParser(description="Plot GPT-2 PagedAttention benchmark results")
    parser.add_argument(
        "--input",
        nargs="+",
        default=["benchmarks/results_gpt2/*.csv"],
        help="Input CSV path(s) or glob(s) produced by run_gpt2_paged_benchmark.py",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks/results_gpt2",
        help="Directory for generated PNGs",
    )
    return parser.parse_args()


def _expand_inputs(patterns: List[str]) -> List[str]:
    paths: List[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(matches)
        elif os.path.exists(pattern):
            paths.append(pattern)
    return sorted(dict.fromkeys(paths))


def _read_rows(paths: List[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in paths:
        with open(path, "r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["source_csv"] = path
                rows.append(row)
    return rows


def _to_float(value: str, default=np.nan) -> float:
    if value in {None, "", "None"}:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _short_model_name(name: str) -> str:
    return name.split("/")[-1]


def _row_label(row: Dict[str, str]) -> str:
    model = _short_model_name(row.get("model_name", "model"))
    return (
        f"{model}\n"
        f"p{int(_to_float(row.get('prompt_tokens', '0'), 0))}/"
        f"g{int(_to_float(row.get('generated_tokens', '0'), 0))}\n"
        f"b{int(_to_float(row.get('block_size', '0'), 0))}"
    )


def _sort_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: (
            row.get("model_name", ""),
            _to_float(row.get("block_size", "0"), 0),
            _to_float(row.get("prompt_tokens", "0"), 0),
            _to_float(row.get("prompt_id", "0"), 0),
        ),
    )


def plot_latency(rows: List[Dict[str, str]], out_path: Path) -> None:
    labels = [_row_label(row) for row in rows]
    prefill_s = np.array([_to_float(row["prefill_s"]) for row in rows])
    decode_s = np.array([_to_float(row["decode_s"]) for row in rows])
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(max(8.5, 1.05 * len(rows)), 4.8))
    ax.bar(x, prefill_s, color=PALETTE["prefill"], label="Prefill")
    ax.bar(x, decode_s, bottom=prefill_s, color=PALETTE["decode"], label="Decode forwards")
    for i, total in enumerate(prefill_s + decode_s):
        ax.text(i, total + max(total * 0.03, 0.03), f"{total:.2f}s", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("MiniTorch GPT-2 PagedAttention latency by prompt")
    ax.grid(axis="y", linestyle="--", color=PALETTE["grid"], alpha=0.35)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_throughput(rows: List[Dict[str, str]], out_path: Path) -> None:
    labels = [_row_label(row) for row in rows]
    end_to_end = np.array([_to_float(row["end_to_end_tokens_per_s"]) for row in rows])
    decode = np.array([_to_float(row["decode_forward_tokens_per_s"]) for row in rows])
    x = np.arange(len(rows))
    width = 0.36

    fig, ax = plt.subplots(figsize=(max(8.5, 1.05 * len(rows)), 4.8))
    ax.bar(x - width / 2, end_to_end, width, color=PALETTE["throughput"], label="End-to-end tokens/s")
    ax.bar(x + width / 2, decode, width, color=PALETTE["decode_throughput"], label="Decode-forward tokens/s")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Tokens per second")
    ax.set_title("MiniTorch GPT-2 PagedAttention throughput")
    ax.grid(axis="y", linestyle="--", color=PALETTE["grid"], alpha=0.35)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_kv_memory(rows: List[Dict[str, str]], out_path: Path) -> None:
    labels = [_row_label(row) for row in rows]
    paged_live = np.array([_to_float(row["paged_live_kv_bytes"]) / (1024 ** 2) for row in rows])
    paged_allocated = np.array([_to_float(row["paged_allocated_kv_bytes"]) / (1024 ** 2) for row in rows])
    contiguous = np.array([_to_float(row["contiguous_kv_bytes_estimate"]) / (1024 ** 2) for row in rows])
    savings = np.array([100.0 * _to_float(row["paged_vs_contiguous_allocated_savings"]) for row in rows])
    fragmentation = np.array([100.0 * _to_float(row["paged_internal_fragmentation"]) for row in rows])
    x = np.arange(len(rows))
    width = 0.28

    fig, ax = plt.subplots(figsize=(max(9.5, 1.15 * len(rows)), 5.1))
    ax.bar(x - width, contiguous, width, color=PALETTE["contiguous"], label="Contiguous estimate")
    ax.bar(x, paged_allocated, width, color=PALETTE["paged_allocated"], label="Paged allocated")
    ax.bar(x + width, paged_live, width, color=PALETTE["paged_live"], edgecolor="#1C7293", label="Paged live")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    positive_values = np.concatenate([contiguous, paged_allocated, paged_live])
    positive_values = positive_values[positive_values > 0]
    if positive_values.size and np.nanmax(positive_values) / np.nanmin(positive_values) > 100:
        ax.set_yscale("log")
        ax.set_ylabel("KV cache memory (MiB, log scale)")
    else:
        ax.set_ylabel("KV cache memory (MiB)")
    ax.set_title("Paged KV allocation versus contiguous-context estimate")
    ax.grid(axis="y", linestyle="--", color=PALETTE["grid"], alpha=0.35)

    twin = ax.twinx()
    twin.plot(x, savings, marker="o", linewidth=2.2, color=PALETTE["savings"], label="Allocated savings")
    twin.plot(x, fragmentation, marker="s", linewidth=1.8, linestyle="--", color=PALETTE["fragmentation"], label="Internal fragmentation")
    twin.set_ylabel("Percent")
    twin.set_ylim(0, max(105, np.nanmax(np.concatenate([savings, fragmentation])) * 1.15))
    handles, names = ax.get_legend_handles_labels()
    twin_handles, twin_names = twin.get_legend_handles_labels()
    twin.legend(handles + twin_handles, names + twin_names, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    args = parse_args()
    input_paths = _expand_inputs(args.input)
    if not input_paths:
        raise SystemExit("No GPT-2 benchmark CSV files found")
    rows = _sort_rows(_read_rows(input_paths))
    if not rows:
        raise SystemExit("GPT-2 benchmark CSV files did not contain any rows")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_latency(rows, output_dir / "gpt2_latency.png")
    plot_throughput(rows, output_dir / "gpt2_throughput.png")
    plot_kv_memory(rows, output_dir / "gpt2_kv_memory.png")


if __name__ == "__main__":
    main()