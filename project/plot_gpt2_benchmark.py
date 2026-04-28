"""Plot GPT-2 MiniTorch GPU-resident KV benchmark results.

Reads CSVs produced by ``project/run_gpt2_paged_benchmark.py`` and writes:

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
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


IMPLEMENTATION_ORDER = ["baseline_no_kv", "contiguous_kv", "paged_attention"]
IMPLEMENTATION_LABELS = {
    "baseline_no_kv": "Baseline: no KV cache",
    "contiguous_kv": "Contiguous GPU KV",
    "paged_attention": "PagedAttention GPU KV",
}
PALETTE = {
    "baseline_no_kv": "#A64942",
    "contiguous_kv": "#E89B26",
    "paged_attention": "#1C7293",
    "reserved": "#BBD6E4",
    "live": "#BBD6E4",
    "savings": "#065A82",
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
    parser = argparse.ArgumentParser(description="Plot GPT-2 MiniTorch GPU KV benchmark results.")
    parser.add_argument(
        "--input",
        nargs="+",
        default=["benchmarks/results_gpt2/gpt2_gpu_kv_benchmark.csv"],
        help="Input CSV path(s) or glob(s) produced by run_gpt2_paged_benchmark.py.",
    )
    parser.add_argument("--output-dir", default="benchmarks/results_gpt2")
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


def _config_key(row: Dict[str, str]) -> Tuple[int, int, int]:
    return (
        int(_to_float(row.get("batch_size", "0"), 0)),
        int(_to_float(row.get("prompt_tokens", "0"), 0)),
        int(_to_float(row.get("block_size", "0"), 0)),
    )


def _config_label(key: Tuple[int, int, int]) -> str:
    batch_size, prompt_tokens, block_size = key
    return f"bs={batch_size}\nprompt={prompt_tokens}\nblock={block_size}"


def _group_rows(rows: List[Dict[str, str]]) -> Dict[Tuple[int, int, int], Dict[str, Dict[str, str]]]:
    grouped: Dict[Tuple[int, int, int], Dict[str, Dict[str, str]]] = {}
    for row in rows:
        implementation = row.get("implementation", "")
        if implementation not in IMPLEMENTATION_ORDER:
            continue
        grouped.setdefault(_config_key(row), {})[implementation] = row
    return dict(sorted(grouped.items()))


def _source_title(rows: List[Dict[str, str]]) -> str:
    first = rows[0]
    model = first.get("model_name", "gpt2")
    dataset = first.get("dataset_name", "dataset")
    config = first.get("dataset_config", "")
    split = first.get("dataset_split", "")
    layer_id = first.get("layer_id", "")
    dataset_label = "/".join(part for part in [dataset, config, split] if part)
    n_layers = first.get("n_layers", "")
    if layer_id == "all":
        layer_note = f"all {n_layers} MiniTorch layers" if n_layers else "all MiniTorch layers"
    else:
        layer_note = f"layer {layer_id}"
    if n_layers and layer_id != "all":
        layer_note += f" of {n_layers}"
    return f"{model} {layer_note}, real tokens from {dataset_label}"


def plot_latency(rows: List[Dict[str, str]], out_path: Path) -> None:
    grouped = _group_rows(rows)
    keys = list(grouped.keys())
    labels = [_config_label(key) for key in keys]
    x = np.arange(len(keys))
    width = 0.24

    fig, ax = plt.subplots(figsize=(max(10.5, 1.25 * len(keys)), 5.2))
    all_values = []
    for offset, implementation in enumerate(IMPLEMENTATION_ORDER):
        values = np.array([
            _to_float(grouped[key].get(implementation, {}).get("median_ms", ""))
            for key in keys
        ])
        all_values.extend([value for value in values if np.isfinite(value) and value > 0])
        ax.bar(
            x + (offset - 1) * width,
            values,
            width,
            color=PALETTE[implementation],
            label=IMPLEMENTATION_LABELS[implementation],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    if all_values and max(all_values) / min(all_values) > 50:
        ax.set_yscale("log")
        ax.set_ylabel("Median full-model decode latency (ms, log scale)")
    else:
        ax.set_ylabel("Median full-model decode latency (ms)")
    ax.set_title("MiniTorch GPT-2 decode: no KV vs contiguous KV vs PagedAttention\n" + _source_title(rows))
    ax.grid(axis="y", linestyle="--", color=PALETTE["grid"], alpha=0.35)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_throughput(rows: List[Dict[str, str]], out_path: Path) -> None:
    grouped = _group_rows(rows)
    keys = list(grouped.keys())
    labels = [_config_label(key) for key in keys]
    x = np.arange(len(keys))
    width = 0.24

    fig, ax = plt.subplots(figsize=(max(10.5, 1.25 * len(keys)), 5.2))
    for offset, implementation in enumerate(IMPLEMENTATION_ORDER):
        values = np.array([
            _to_float(grouped[key].get(implementation, {}).get("query_tokens_per_s", ""))
            for key in keys
        ])
        ax.bar(
            x + (offset - 1) * width,
            values,
            width,
            color=PALETTE[implementation],
            label=IMPLEMENTATION_LABELS[implementation],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Full-model decode query tokens per second")
    ax.set_title("MiniTorch GPT-2 decode throughput\n" + _source_title(rows))
    ax.grid(axis="y", linestyle="--", color=PALETTE["grid"], alpha=0.35)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_kv_memory(rows: List[Dict[str, str]], out_path: Path) -> None:
    grouped = _group_rows(rows)
    keys = list(grouped.keys())
    width = 0.74
    bar_stride = 1.28
    group_stride = len(IMPLEMENTATION_ORDER) * bar_stride + 1.15
    bar_records = []
    max_reserved = 0.0

    for group_idx, key in enumerate(keys):
        rows_for_key = grouped[key]
        for offset, implementation in enumerate(IMPLEMENTATION_ORDER):
            row = rows_for_key.get(implementation)
            if not row:
                continue
            live_mib = _to_float(row.get("live_kv_bytes", "")) / (1024 ** 2)
            if implementation == "baseline_no_kv":
                reserved_mib = _to_float(row.get("working_kv_bytes", "")) / (1024 ** 2)
            else:
                reserved_mib = _to_float(row.get("allocated_kv_bytes", "")) / (1024 ** 2)
            if not np.isfinite(reserved_mib) or reserved_mib < live_mib:
                reserved_mib = live_mib
            max_reserved = max(max_reserved, reserved_mib)
            bar_records.append(
                {
                    "x": group_idx * group_stride + offset * bar_stride,
                    "key": key,
                    "implementation": implementation,
                    "row": row,
                    "live_mib": live_mib,
                    "reserved_mib": reserved_mib,
                }
            )

    fig, ax = plt.subplots(figsize=(max(14.0, 3.7 * len(keys)), 6.1))
    for record in bar_records:
        xpos = record["x"]
        implementation = record["implementation"]
        live_mib = record["live_mib"]
        reserved_mib = record["reserved_mib"]
        row = record["row"]

        ax.bar(
            xpos,
            reserved_mib,
            width,
            color=PALETTE["reserved"],
            edgecolor="#303030",
            linewidth=0.9,
            zorder=1,
        )
        ax.bar(
            xpos,
            live_mib,
            width,
            color=PALETTE[implementation],
            edgecolor="#303030",
            linewidth=0.9,
            hatch="//" if implementation == "baseline_no_kv" else None,
            zorder=2,
        )

        annotation_y = reserved_mib + max(max_reserved * 0.035, 0.08)
        if implementation == "baseline_no_kv":
            annotation = "no persistent\nKV cache"
        else:
            efficiency = 100.0 * live_mib / reserved_mib if reserved_mib > 0 else 0.0
            unused = max(0.0, 100.0 - efficiency)
            suffix = "frag." if implementation == "paged_attention" else "unused"
            annotation = f"{efficiency:.1f}% live\n{unused:.1f}% {suffix}"
            if implementation == "paged_attention":
                savings = 100.0 * _to_float(
                    row.get("paged_savings_vs_static_contiguous", ""), 0.0
                )
                annotation += f"\n{savings:.1f}% less vs contig."
        ax.text(
            xpos,
            annotation_y,
            annotation,
            ha="center",
            va="bottom",
            fontsize=8.5,
            linespacing=1.05,
        )

    xticks = [record["x"] for record in bar_records]
    xticklabels = []
    short_labels = {
        "baseline_no_kv": "Baseline\n(no KV)",
        "contiguous_kv": "Contiguous\nstatic",
        "paged_attention": "Paged\nblocks",
    }
    for record in bar_records:
        xticklabels.append(short_labels[record["implementation"]])
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    for group_idx, key in enumerate(keys):
        center = group_idx * group_stride + bar_stride
        ax.text(
            center,
            -0.18,
            _config_label(key),
            ha="center",
            va="top",
            transform=ax.get_xaxis_transform(),
            fontsize=9,
        )
        if group_idx:
            ax.axvline(
                group_idx * group_stride - 0.72,
                color=PALETTE["grid"],
                linewidth=0.8,
                alpha=0.35,
            )

    ax.set_ylabel("KV memory allocation (MiB)")
    ax.set_title(
        "MiniTorch GPT-2 KV memory breakdown: reserved vs live KV\n"
        + _source_title(rows)
    )
    ax.grid(axis="y", linestyle="--", color=PALETTE["grid"], alpha=0.35)
    ax.set_ylim(0, max_reserved * 1.32 if max_reserved > 0 else 1.0)
    ax.legend(
        handles=[
            Patch(facecolor=PALETTE["reserved"], edgecolor="#303030", label="Reserved allocation"),
            Patch(facecolor=PALETTE["baseline_no_kv"], edgecolor="#303030", hatch="//", label="Baseline temporary live KV"),
            Patch(facecolor=PALETTE["contiguous_kv"], edgecolor="#303030", label="Contiguous live KV"),
            Patch(facecolor=PALETTE["paged_attention"], edgecolor="#303030", label="Paged live KV"),
        ],
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
    )

    fig.tight_layout(rect=(0.0, 0.0, 0.86, 1.0))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    args = parse_args()
    input_paths = _expand_inputs(args.input)
    if not input_paths:
        raise SystemExit("No GPT-2 benchmark CSV files found.")
    rows = _read_rows(input_paths)
    if not rows:
        raise SystemExit("GPT-2 benchmark CSV files did not contain rows.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_latency(rows, output_dir / "gpt2_latency.png")
    plot_throughput(rows, output_dir / "gpt2_throughput.png")
    plot_kv_memory(rows, output_dir / "gpt2_kv_memory.png")


if __name__ == "__main__":
    main()
