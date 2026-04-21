"""
Generate the six figures for the final poster from the rigorous benchmark
CSVs produced by ``run_rigorous_benchmark.py``.

Reads:
    benchmarks/results_rigorous/exp1_memory_breakdown.csv
    benchmarks/results_rigorous/exp2_capacity_curve.csv
    benchmarks/results_rigorous/exp3_decode_speed.csv
    benchmarks/results_rigorous/exp4_prefix_prefill.csv
    benchmarks/results_rigorous/exp5_parallel_sampling.csv
    benchmarks/results_rigorous/exp6_beam_search.csv

Writes:
    benchmarks/report_figures_v2/figure1.png   (memory breakdown)
    benchmarks/report_figures_v2/figure2.png   (capacity curve)
    benchmarks/report_figures_v2/figure3.png   (decode speed, 3 methods)
    benchmarks/report_figures_v2/figure4.png   (prefix prefill speedup)
    benchmarks/report_figures_v2/figure5.png   (parallel sampling memory)
    benchmarks/report_figures_v2/figure6.png   (beam search memory)

Style goals: clean MLSys-style figures with error bars where timing is
involved, honest axis labels that name what is actually measured, and
legends that distinguish paged vs the two contiguous baselines.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

PALETTE = {
    "paged": "#1C7293",
    "paged_dark": "#065A82",
    "contiguous_kv": "#E89B26",
    "no_cache": "#A64942",
    "static_worst": "#A64942",
    "static_realistic": "#E89B26",
    "fork": "#1C7293",
    "clone": "#A64942",
    "reserved": "#BBD6E4",
    "used": "#1C7293",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 160,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        print(f"  [skip] {path} does not exist")
        return []
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def _to_float(x, default=np.nan):
    if x is None or x == "" or x == "None":
        return default
    try:
        return float(x)
    except Exception:
        return default


def _to_int(x, default=0):
    if x is None or x == "" or x == "None":
        return default
    try:
        return int(float(x))
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Figure 1 — Memory breakdown
# ---------------------------------------------------------------------------

def plot_fig1_memory_breakdown(rows: List[Dict], out_path: str) -> None:
    if not rows:
        return
    methods = []
    reserved_mb = []
    used_mb = []
    efficiencies = []
    descriptions = []
    colors = []
    display_name = {
        "static_worst_case": "Static\n(worst case)",
        "static_realistic": "Static\n(realistic cap)",
        "paged": "Paged\n(block-granular)",
    }
    color_key = {
        "static_worst_case": "static_worst",
        "static_realistic": "static_realistic",
        "paged": "paged",
    }
    for r in rows:
        m = r["method"]
        methods.append(display_name.get(m, m))
        reserved_mb.append(_to_float(r["reserved_bytes"]) / 1e6)
        used_mb.append(_to_float(r["used_bytes"]) / 1e6)
        efficiencies.append(_to_float(r["efficiency"]))
        descriptions.append(r.get("description", ""))
        colors.append(PALETTE[color_key.get(m, "paged")])

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(methods))
    width = 0.55
    bars_reserved = ax.bar(
        x, reserved_mb, width,
        color=[PALETTE["reserved"]] * len(methods),
        edgecolor="#444", label="Reserved",
    )
    bars_used = ax.bar(
        x, used_mb, width,
        color=colors, edgecolor="#222",
        label="Used (live KV)",
    )
    for i, (res, use, eff) in enumerate(zip(reserved_mb, used_mb, efficiencies)):
        ax.text(i, res + max(reserved_mb) * 0.02,
                f"{eff*100:.1f}%\nefficient",
                ha="center", va="bottom", fontsize=9.5)

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("KV cache (MB)")
    ax.set_title(
        "KV memory breakdown on a realistic non-aligned workload"
    )
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(reserved_mb) * 1.22)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 2 — Capacity curve
# ---------------------------------------------------------------------------

def plot_fig2_capacity_curve(rows: List[Dict], out_path: str) -> None:
    if not rows:
        return
    seq_lens = [int(r["seq_len"]) for r in rows]
    paged = [_to_int(r["paged_max_batch"]) for r in rows]
    static_worst = [_to_int(r["static_worst_case_max_batch"]) for r in rows]
    static_realistic = [_to_int(r["static_realistic_max_batch"]) for r in rows]
    # decode_margin is the same for every row; grab from first.
    decode_margin = _to_int(rows[0].get("decode_margin", "0")) if rows else 0

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.plot(seq_lens, paged, marker="o", linewidth=2.4, color=PALETTE["paged"],
            label="Paged (empirical)", zorder=5)
    ax.plot(seq_lens, static_realistic, marker="s", linewidth=2.0,
            linestyle="--",
            color=PALETTE["static_realistic"],
            label=f"Static, realistic cap = seq_len + {decode_margin}")
    ax.plot(seq_lens, static_worst, marker="^", linewidth=2.0,
            color=PALETTE["static_worst"],
            label="Static, worst-case cap = n_positions")

    # Annotate paged-vs-realistic gain at each point.
    for i, sl in enumerate(seq_lens):
        if static_realistic[i] > 0:
            gain = paged[i] / static_realistic[i]
            if gain > 1.05:
                ax.annotate(
                    f"{gain:.2f}x",
                    xy=(sl, paged[i]),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center", fontsize=9,
                    color=PALETTE["paged_dark"], fontweight="bold",
                )

    ax.set_xlabel("Prompt length (tokens)")
    ax.set_ylabel("Max concurrent sequences under budget")
    ax.set_title(
        "Capacity curve — sequences that fit at a fixed KV budget\n"
        f"(static-realistic over-provisions decode_margin = {decode_margin} tokens)"
    )
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 3 — Decode speed with error bars
# ---------------------------------------------------------------------------

def plot_fig3_decode_speed(rows: List[Dict], out_path: str) -> None:
    if not rows:
        return
    # Group by (batch, seq_len) → tuple → stats for each method.
    configs: List[str] = []
    paged_med, paged_lo, paged_hi = [], [], []
    contig_med, contig_lo, contig_hi = [], [], []
    nocache_med, nocache_lo, nocache_hi = [], [], []

    for r in rows:
        configs.append(f"bs={r['batch_size']}\nseq={r['seq_len']}")
        pm = _to_float(r["paged_median_s"]) * 1000
        paged_med.append(pm)
        paged_lo.append(pm - _to_float(r["paged_min_s"]) * 1000)
        paged_hi.append(_to_float(r["paged_max_s"]) * 1000 - pm)

        cm = _to_float(r["contiguous_kv_median_s"]) * 1000
        contig_med.append(cm)
        contig_lo.append(cm - _to_float(r["contiguous_kv_min_s"]) * 1000)
        contig_hi.append(_to_float(r["contiguous_kv_max_s"]) * 1000 - cm)

        nm = _to_float(r["no_cache_median_s"]) * 1000
        nocache_med.append(nm)
        nocache_lo.append(nm - _to_float(r["no_cache_min_s"]) * 1000)
        nocache_hi.append(_to_float(r["no_cache_max_s"]) * 1000 - nm)

    x = np.arange(len(configs))
    width = 0.26
    fig, ax = plt.subplots(figsize=(max(7.5, len(configs) * 1.05), 4.6))

    ax.bar(x - width, paged_med, width, yerr=[paged_lo, paged_hi],
           capsize=3, color=PALETTE["paged"], label="Paged (ours)")
    ax.bar(x, contig_med, width, yerr=[contig_lo, contig_hi],
           capsize=3, color=PALETTE["contiguous_kv"],
           label="Contiguous KV (HF-style)")
    ax.bar(x + width, nocache_med, width, yerr=[nocache_lo, nocache_hi],
           capsize=3, color=PALETTE["no_cache"],
           label="No cache (re-prefill)")

    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylabel("End-to-end wall time (ms)")
    ax.set_title(
        "Figure 3: Decode speed — paged vs contiguous KV vs no-cache baseline\n"
        "(lower is better; error bars span min–max across trials)"
    )
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    # Annotate paged speedup over the contiguous baseline
    for i, r in enumerate(rows):
        spd = _to_float(r.get("paged_speedup_vs_contiguous", ""))
        if np.isfinite(spd):
            ax.text(
                x[i] - width, paged_med[i] + paged_hi[i] + max(nocache_med) * 0.03,
                f"{spd:.2f}x",
                ha="center", va="bottom", fontsize=9, color=PALETTE["paged_dark"],
            )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 4 — Prefix prefill speedup
# ---------------------------------------------------------------------------

def plot_fig4_prefix_prefill(rows: List[Dict], out_path: str) -> None:
    if not rows:
        return
    ratios = [_to_float(r["share_ratio_effective"]) for r in rows]
    fresh_med = [_to_float(r["fresh_prefill_median_s"]) * 1000 for r in rows]
    fresh_min = [_to_float(r["fresh_prefill_min_s"]) * 1000 for r in rows]
    fresh_max = [_to_float(r["fresh_prefill_max_s"]) * 1000 for r in rows]
    cached_med = [_to_float(r["cached_prefill_median_s"]) * 1000 for r in rows]
    cached_min = [_to_float(r["cached_prefill_min_s"]) * 1000 for r in rows]
    cached_max = [_to_float(r["cached_prefill_max_s"]) * 1000 for r in rows]
    speedup = [_to_float(r["prefix_prefill_speedup"]) for r in rows]

    x = np.arange(len(ratios))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7.4, 4.6))

    fresh_err = [
        [fm - fl for fm, fl in zip(fresh_med, fresh_min)],
        [fh - fm for fh, fm in zip(fresh_max, fresh_med)],
    ]
    cached_err = [
        [cm - cl for cm, cl in zip(cached_med, cached_min)],
        [ch - cm for ch, cm in zip(cached_max, cached_med)],
    ]
    ax.bar(x - width / 2, fresh_med, width, yerr=fresh_err, capsize=3,
           color=PALETTE["no_cache"], label="Fresh (cache miss)")
    ax.bar(x + width / 2, cached_med, width, yerr=cached_err, capsize=3,
           color=PALETTE["paged"], label="Prefix-cached (hit)")

    for i, spd in enumerate(speedup):
        if np.isfinite(spd) and spd > 0:
            ax.text(
                x[i], max(fresh_max) * 1.02,
                f"{spd:.2f}x",
                ha="center", va="bottom", fontsize=10,
                color=PALETTE["paged_dark"], fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r*100:.0f}%" for r in ratios])
    ax.set_xlabel("Shared-prefix fraction of prompt")
    ax.set_ylabel("Prefill time (ms)s")
    ax.set_title("Prefix-cache prefill speedup at varying shared fractions")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.set_ylim(0, max(fresh_max) * 1.18)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 5 — Parallel sampling memory
# ---------------------------------------------------------------------------

def plot_fig5_parallel_sampling(rows: List[Dict], out_path: str) -> None:
    if not rows:
        return
    grouped: Dict[int, List[Dict]] = defaultdict(list)
    for r in rows:
        grouped[int(r["prompt_len"])].append(r)

    prompt_lens = sorted(grouped.keys())
    n_panels = len(prompt_lens)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.0 * n_panels + 0.3, 4.3),
                             sharey=False)
    if n_panels == 1:
        axes = [axes]

    for ax, prompt_len in zip(axes, prompt_lens):
        sub = sorted(grouped[prompt_len], key=lambda r: int(r["n_outputs"]))
        n_outputs = [int(r["n_outputs"]) for r in sub]
        fork = [_to_int(r["fork_blocks_total"]) for r in sub]
        clone = [_to_int(r["clone_blocks_total"]) for r in sub]
        saving = [_to_float(r["saving_fraction"]) for r in sub]
        x = np.arange(len(n_outputs))
        width = 0.38
        ax.bar(x - width / 2, clone, width,
               color=PALETTE["clone"], label="Naive clone")
        ax.bar(x + width / 2, fork, width,
               color=PALETTE["fork"], label="Paged fork (shared)")
        for i, s in enumerate(saving):
            ax.text(x[i], max(clone) * 1.02,
                    f"−{s*100:.0f}%",
                    ha="center", va="bottom", fontsize=9,
                    color=PALETTE["paged_dark"], fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in n_outputs])
        ax.set_xlabel("Number of sampled outputs")
        ax.set_title(f"prompt_len = {prompt_len}"
                     + (" (non-aligned)" if prompt_len % 16 != 0 else ""))
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        if ax is axes[0]:
            ax.set_ylabel("KV blocks used")
        ax.set_ylim(0, max(clone) * 1.22)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), ncol=2, frameon=False)
    fig.suptitle("Parallel-sampling KV memory", y=0.95)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 6 — Beam search memory
# ---------------------------------------------------------------------------

def plot_fig6_beam_search(rows: List[Dict], out_path: str) -> None:
    if not rows:
        return
    grouped: Dict[int, List[Dict]] = defaultdict(list)
    for r in rows:
        grouped[int(r["prompt_len"])].append(r)

    # Pick up decode length so the caption tells the reader what workload
    # this figure describes (typically longer than Exp 5 so the trunk-
    # sharing advantage is visible).
    decode_tokens_any = _to_int(rows[0].get("decode_tokens", "0")) if rows else 0

    prompt_lens = sorted(grouped.keys())
    n_panels = len(prompt_lens)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.0 * n_panels + 0.3, 4.3),
                             sharey=False)
    if n_panels == 1:
        axes = [axes]

    for ax, prompt_len in zip(axes, prompt_lens):
        sub = sorted(grouped[prompt_len], key=lambda r: int(r["beam_width"]))
        beams = [int(r["beam_width"]) for r in sub]
        fork_peak = [_to_int(r["fork_blocks_total"]) for r in sub]
        fork_post = [
            _to_int(r.get("fork_blocks_post_prune", r["fork_blocks_total"]))
            for r in sub
        ]
        clone = [_to_int(r["clone_blocks_total"]) for r in sub]
        saving = [_to_float(r["saving_fraction"]) for r in sub]
        x = np.arange(len(beams))
        width = 0.38
        ax.bar(x - width / 2, clone, width,
               color=PALETTE["clone"], label="Naive clone")
        ax.bar(x + width / 2, fork_peak, width,
               color=PALETTE["fork"],
               label="Paged fork (peak, trunk+tips)")
        # Overlay post-prune steady-state as a hatched bar inside fork_peak
        # so readers can see how much the trunk dies once beams take over.
        ax.bar(x + width / 2, fork_post, width,
               color=PALETTE["fork"], alpha=0.55,
               edgecolor="white", hatch="//",
               label="Paged fork (post-prune)")
        for i, s in enumerate(saving):
            ax.text(x[i], max(clone) * 1.02,
                    f"−{s*100:.0f}%",
                    ha="center", va="bottom", fontsize=9,
                    color=PALETTE["paged_dark"], fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([str(b) for b in beams])
        ax.set_xlabel("Beam width")
        ax.set_title(f"prompt_len = {prompt_len}"
                     + (" (non-aligned)" if prompt_len % 16 != 0 else ""))
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        if ax is axes[0]:
            ax.set_ylabel("KV blocks used")
        ax.set_ylim(0, max(clone) * 1.28)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), ncol=3, frameon=False,
               fontsize=9.5)
    fig.suptitle(
        "Beam-search KV memory",
        y=0.95
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=str, default="benchmarks/results_rigorous")
    ap.add_argument("--output-dir", type=str, default="benchmarks/report_figures_v2")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    plot_fig1_memory_breakdown(
        _read_csv(os.path.join(args.input_dir, "exp1_memory_breakdown.csv")),
        os.path.join(args.output_dir, "figure1.png"),
    )
    plot_fig2_capacity_curve(
        _read_csv(os.path.join(args.input_dir, "exp2_capacity_curve.csv")),
        os.path.join(args.output_dir, "figure2.png"),
    )
    plot_fig3_decode_speed(
        _read_csv(os.path.join(args.input_dir, "exp3_decode_speed.csv")),
        os.path.join(args.output_dir, "figure3.png"),
    )
    plot_fig4_prefix_prefill(
        _read_csv(os.path.join(args.input_dir, "exp4_prefix_prefill.csv")),
        os.path.join(args.output_dir, "figure4.png"),
    )
    plot_fig5_parallel_sampling(
        _read_csv(os.path.join(args.input_dir, "exp5_parallel_sampling.csv")),
        os.path.join(args.output_dir, "figure5.png"),
    )
    plot_fig6_beam_search(
        _read_csv(os.path.join(args.input_dir, "exp6_beam_search.csv")),
        os.path.join(args.output_dir, "figure6.png"),
    )


if __name__ == "__main__":
    main()
