"""Plot GPU-resident paged-vs-contiguous attention benchmark results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PALETTE = {
    "paged": "#1C7293",
    "contiguous": "#E89B26",
    "ratio": "#065A82",
    "static": "#A64942",
    "live": "#BBD6E4",
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
    parser = argparse.ArgumentParser(description="Plot GPU-resident paged-vs-contiguous attention results")
    parser.add_argument(
        "--input",
        default="benchmarks/results_gpu_resident/paged_vs_contiguous_gpu_resident.csv",
    )
    parser.add_argument("--output-dir", default="benchmarks/results_gpu_resident")
    parser.add_argument("--output-prefix", default="gpu_resident_paged_vs_contiguous")
    return parser.parse_args()


def _read_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _to_float(value: str) -> float:
    return float(value) if value not in {None, "", "None"} else np.nan


def _labels(rows: List[Dict[str, str]]) -> List[str]:
    return [f"bs={row['batch_size']}\nseq={row['seq_len']}" for row in rows]


def _source_title(rows: List[Dict[str, str]]) -> str:
    first = rows[0]
    if first.get("source") == "gpt2":
        model = first.get("model_name", "gpt2")
        dataset = first.get("dataset_name", "dataset")
        config = first.get("dataset_config", "")
        split = first.get("dataset_split", "")
        dataset_label = "/".join(part for part in [dataset, config, split] if part)
        return f"{model} activations from {dataset_label}"
    return "synthetic GPU-resident Q/K/V"


def plot_latency(rows: List[Dict[str, str]], output_path: Path) -> None:
    labels = _labels(rows)
    x = np.arange(len(rows))
    width = 0.36
    paged = np.array([_to_float(row["paged_median_ms"]) for row in rows])
    contiguous = np.array([_to_float(row["contiguous_median_ms"]) for row in rows])
    ratio = np.array([_to_float(row["paged_latency_over_contiguous"]) for row in rows])

    fig, ax = plt.subplots(figsize=(max(11.0, 0.72 * len(rows)), 5.2))
    ax.bar(x - width / 2, paged, width, color=PALETTE["paged"], label="PagedAttention GPU-resident")
    ax.bar(x + width / 2, contiguous, width, color=PALETTE["contiguous"], label="Contiguous no-paging GPU-resident")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Median decode-attention latency (ms)")
    ax.set_title(
        "GPU-resident attention latency: paged block table vs contiguous KV\n"
        f"{_source_title(rows)}"
    )
    ax.grid(axis="y", linestyle="--", color=PALETTE["grid"], alpha=0.35)
    ax.legend(loc="upper left")

    twin = ax.twinx()
    twin.plot(x, ratio, marker="o", linewidth=2.1, color=PALETTE["ratio"], label="Paged / contiguous latency")
    twin.axhline(1.0, linestyle=":", color="#555555", linewidth=1.2)
    twin.set_ylabel("Latency ratio")
    handles, names = ax.get_legend_handles_labels()
    twin_handles, twin_names = twin.get_legend_handles_labels()
    twin.legend(handles + twin_handles, names + twin_names, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")


def plot_memory(rows: List[Dict[str, str]], output_path: Path) -> None:
    labels = _labels(rows)
    x = np.arange(len(rows))
    width = 0.28
    live = np.array([_to_float(row["live_kv_bytes"]) / (1024 ** 2) for row in rows])
    paged = np.array([_to_float(row["paged_allocated_kv_bytes"]) / (1024 ** 2) for row in rows])
    static = np.array([_to_float(row["static_contiguous_kv_bytes"]) / (1024 ** 2) for row in rows])
    savings = np.array([100.0 * _to_float(row["paged_savings_vs_static_contiguous"]) for row in rows])
    fragmentation = np.array([100.0 * _to_float(row["paged_internal_fragmentation"]) for row in rows])

    fig, ax = plt.subplots(figsize=(max(11.0, 0.72 * len(rows)), 5.2))
    ax.bar(x - width, static, width, color=PALETTE["static"], label="Static contiguous reservation")
    ax.bar(x, paged, width, color=PALETTE["paged"], label="Paged allocated blocks")
    ax.bar(x + width, live, width, color=PALETTE["live"], edgecolor=PALETTE["paged"], label="Live KV")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("KV memory (MiB, log scale)")
    ax.set_title(
        "KV memory: paged allocation avoids static full-context reservation\n"
        f"{_source_title(rows)}"
    )
    ax.grid(axis="y", linestyle="--", color=PALETTE["grid"], alpha=0.35)

    twin = ax.twinx()
    twin.plot(x, savings, marker="o", linewidth=2.1, color=PALETTE["ratio"], label="Savings vs static")
    twin.plot(x, fragmentation, marker="s", linewidth=1.7, linestyle="--", color=PALETTE["contiguous"], label="Internal fragmentation")
    twin.set_ylim(0, 105)
    twin.set_ylabel("Percent")
    handles, names = ax.get_legend_handles_labels()
    twin_handles, twin_names = twin.get_legend_handles_labels()
    twin.legend(handles + twin_handles, names + twin_names, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")


def main() -> None:
    args = parse_args()
    rows = _read_rows(args.input)
    if not rows:
        raise SystemExit("No rows found in benchmark CSV")
    rows = sorted(rows, key=lambda row: (int(row["batch_size"]), int(row["seq_len"])))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_latency(rows, output_dir / f"{args.output_prefix}_latency.png")
    plot_memory(rows, output_dir / f"{args.output_prefix}_memory.png")


if __name__ == "__main__":
    main()