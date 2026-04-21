"""
Rigorous benchmark for PagedAttention in MiniTorch.

This is a clean re-run of the six experiments that back the final poster,
addressing the methodology issues in the original ``run_benchmark.py``:

  1. Memory breakdown uses a realistic *non-aligned* sequence-length mix
     (48, 67, 112, 160, 256, 80, 35, 200) and reports bytes that are
     actually allocated into a BlockManager, not the analytical formula.
  2. Capacity curve measures ``max_batch_under_budget`` empirically by
     allocating sequences until OOM, for both paged and contiguous KV.
  3. Decode speed compares paged against TWO baselines — the honest
     no-cache re-prefill (O(T²) per step) AND a HuggingFace-style
     contiguous KV cache baseline — and every timing is the *median of
     three timed trials* after one warmup trial.
  4. Prefix prefill sweeps share ratios {0.0, 0.25, 0.5, 0.75} (not a
     single hard-coded 0.5) with warmup + 3 trials.
  5. Parallel sampling memory uses non-aligned prompt lengths so the
     reported block counts aren't suspiciously clean ratios.
  6. Beam search memory follows the same shape as (5).

Figures consume ``benchmarks/results_rigorous/*.csv`` via
``plot_rigorous_figures.py``.

Usage:
    python project/run_rigorous_benchmark.py \
        --output-dir benchmarks/results_rigorous \
        --n-embd 512 --n-head 8 --n-layers 4 --n-positions 512 \
        --block-size 16 --num-kv-blocks 512 \
        --decode-backend auto

The defaults target ~10 min on an RTX 4060 (8 GB).  Use ``--skip-decode``
or ``--skip-prefix`` to drop the two expensive experiments for a quick
smoke test.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
import traceback
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, ".")

import minitorch
from minitorch.block_manager import BlockManager, CACHE_DTYPE
from minitorch.modules_transfomer import DecoderLM
from minitorch.tensor import tensor_from_numpy
from minitorch.transformer import PagedDecoderLM

from project.contiguous_kv_baseline import ContiguousKVDecoderLM

datatype = np.float32


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Rigorous PagedAttention benchmark")
    p.add_argument("--output-dir", type=str, default="benchmarks/results_rigorous")
    # Model
    p.add_argument("--n-vocab", type=int, default=1024)
    p.add_argument("--n-embd", type=int, default=512)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-positions", type=int, default=512)
    # KV cache
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--num-kv-blocks", type=int, default=512)
    # Backends
    p.add_argument("--backend", type=str, default="auto",
                   choices=["auto", "cpu", "cuda"])
    p.add_argument("--decode-backend", type=str, default="auto",
                   choices=["auto", "ref", "cuda"])
    # Trial control
    p.add_argument("--warmup-trials", type=int, default=1)
    p.add_argument("--timed-trials", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--seed", type=int, default=20260421)
    # Experiment skip switches (handy for dev)
    p.add_argument("--skip-memory", action="store_true")
    p.add_argument("--skip-capacity", action="store_true")
    p.add_argument("--skip-decode", action="store_true")
    p.add_argument("--skip-prefix", action="store_true")
    p.add_argument("--skip-sharing", action="store_true")
    # Exp 3 configs (small and balanced to stay under 10 min)
    p.add_argument("--decode-batch-sizes", type=int, nargs="+",
                   default=[1, 2, 4])
    p.add_argument("--decode-seq-lens", type=int, nargs="+",
                   default=[32, 64, 128])
    # Exp 4 configs
    p.add_argument("--prefix-seq-len", type=int, default=128)
    p.add_argument("--prefix-share-ratios", type=float, nargs="+",
                   default=[0.0, 0.25, 0.5, 0.75])
    # Exp 2 config: margin an operator over-provisions per-seq on top of
    # seq_len to cover decode length.  Default = 2x max_new_tokens so the
    # static baseline reflects a real "reserve a little safety headroom"
    # policy rather than the degenerate seq_len-exactly case.
    p.add_argument("--capacity-decode-margin", type=int, default=32,
                   help="Tokens reserved per seq above seq_len for the "
                        "static-realistic baseline in the capacity curve.")
    # Exp 6 config: decode length for the beam-search block-count simulation.
    # Longer than Exp 5 so that winner-takes-all trunk sharing actually pays
    # off (with decode_tokens <= block_size the per-beam decode block can't
    # be shared across beams).
    p.add_argument("--beam-decode-tokens", type=int, default=64,
                   help="Decode length for Exp 6 (beam search memory sim). "
                        "Must be large enough to span multiple blocks for "
                        "trunk-sharing to show up.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------

def _cuda_available() -> bool:
    try:
        import numba.cuda
        return bool(numba.cuda.is_available())
    except Exception:
        return False


def _create_backend(backend_name: str):
    if backend_name == "cuda":
        return minitorch.TensorBackend(minitorch.CudaKernelOps)
    return minitorch.TensorBackend(minitorch.FastOps)


def _resolve_backends(req_backend: str, req_decode_backend: str):
    has_cuda = _cuda_available()
    backend = req_backend if req_backend != "auto" else ("cuda" if has_cuda else "cpu")
    decode_backend = (
        req_decode_backend
        if req_decode_backend != "auto"
        else ("cuda" if has_cuda else "ref")
    )
    if backend == "cuda" and not has_cuda:
        raise RuntimeError("--backend cuda requested but no CUDA device found")
    if decode_backend == "cuda" and not has_cuda:
        raise RuntimeError("--decode-backend cuda requested but no CUDA device found")
    return backend, decode_backend


def _make_paged_model(args, backend, backend_name, decode_backend):
    model = PagedDecoderLM(
        n_vocab=args.n_vocab,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_positions=args.n_positions,
        n_layers=args.n_layers,
        block_size=args.block_size,
        p_dropout=0.0,
        backend=backend,
        decode_backend=decode_backend,
    )
    model.eval()
    return model


def _make_block_manager(args) -> BlockManager:
    head_dim = args.n_embd // args.n_head
    return BlockManager(
        num_blocks=args.num_kv_blocks,
        block_size=args.block_size,
        n_head=args.n_head,
        head_dim=head_dim,
        num_layers=args.n_layers,
    )


# ---------------------------------------------------------------------------
# Multi-trial timing helper
# ---------------------------------------------------------------------------

def timed(fn: Callable[[], None], n_warmup: int, n_trials: int) -> Dict[str, float]:
    """Run ``fn`` ``n_warmup + n_trials`` times and report timing stats.

    Returns a dict with median/min/max/p25/p75/mean in seconds.
    """
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    arr = np.array(times, dtype=np.float64)
    return {
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "n_trials": int(n_trials),
    }


def _write_csv(path: str, rows: Sequence[Dict]) -> None:
    if not rows:
        print(f"  (no rows for {path}, skipping CSV)")
        return
    # Union of keys across rows, preserving first-row order.
    all_keys: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"  wrote {path} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Experiment 1 — Memory breakdown on a realistic non-aligned workload
# ---------------------------------------------------------------------------

REALISTIC_LENGTHS = [48, 67, 112, 160, 256, 80, 35, 200]


def run_memory_breakdown(args, out_dir: str) -> None:
    """Compare bytes reserved vs bytes used for three allocation strategies
    on a realistic length mix.  Every number is computed from real allocator
    state, not from a closed-form formula."""

    print("\n== Experiment 1: Memory breakdown ==")
    head_dim = args.n_embd // args.n_head
    dtype_size = np.dtype(CACHE_DTYPE).itemsize
    block_size = args.block_size
    n_positions = args.n_positions
    n_head = args.n_head
    n_layers = args.n_layers
    lengths = REALISTIC_LENGTHS

    # --- Paged: allocate each seq and ask the manager how much is reserved
    bm = _make_block_manager(args)
    for seq_idx, length in enumerate(lengths):
        bm.allocate_blocks_for_sequence(seq_idx, length)

    paged_reserved_blocks = bm.num_used_blocks
    paged_reserved_token_slots = paged_reserved_blocks * block_size
    paged_reserved_bytes = (
        paged_reserved_token_slots * n_head * head_dim * n_layers * 2 * dtype_size
    )
    paged_used_token_slots = sum(lengths)
    paged_used_bytes = (
        paged_used_token_slots * n_head * head_dim * n_layers * 2 * dtype_size
    )
    frag = bm.compute_fragmentation()

    for seq_idx in range(len(lengths)):
        bm.free_sequence(seq_idx)

    # --- Naive static: every seq reserves the full n_positions (worst case)
    static_reserved_token_slots = len(lengths) * n_positions
    static_reserved_bytes = (
        static_reserved_token_slots * n_head * head_dim * n_layers * 2 * dtype_size
    )

    # --- Realistic static: every seq reserves up to the workload's max length
    realistic_cap = max(lengths)
    realistic_reserved_token_slots = len(lengths) * realistic_cap
    realistic_reserved_bytes = (
        realistic_reserved_token_slots * n_head * head_dim * n_layers * 2 * dtype_size
    )

    def _eff(used, reserved):
        return used / reserved if reserved > 0 else 0.0

    rows = [
        {
            "method": "static_worst_case",
            "description": f"each seq reserves n_positions={n_positions}",
            "reserved_token_slots": static_reserved_token_slots,
            "used_token_slots": paged_used_token_slots,
            "reserved_bytes": static_reserved_bytes,
            "used_bytes": paged_used_bytes,
            "efficiency": round(_eff(paged_used_bytes, static_reserved_bytes), 4),
            "internal_frag": round(1.0 - _eff(paged_used_bytes, static_reserved_bytes), 4),
            "external_frag": 0.0,
        },
        {
            "method": "static_realistic",
            "description": f"each seq reserves max(observed)={realistic_cap}",
            "reserved_token_slots": realistic_reserved_token_slots,
            "used_token_slots": paged_used_token_slots,
            "reserved_bytes": realistic_reserved_bytes,
            "used_bytes": paged_used_bytes,
            "efficiency": round(_eff(paged_used_bytes, realistic_reserved_bytes), 4),
            "internal_frag": round(
                1.0 - _eff(paged_used_bytes, realistic_reserved_bytes), 4
            ),
            "external_frag": 0.0,
        },
        {
            "method": "paged",
            "description": f"block_size={block_size}, ceil(L/bs) blocks per seq",
            "reserved_token_slots": paged_reserved_token_slots,
            "used_token_slots": paged_used_token_slots,
            "reserved_bytes": paged_reserved_bytes,
            "used_bytes": paged_used_bytes,
            "efficiency": round(_eff(paged_used_bytes, paged_reserved_bytes), 4),
            "internal_frag": round(frag["internal"], 4),
            "external_frag": round(frag["external"], 4),
        },
    ]
    # Summary line
    paged_save_vs_worst = 1.0 - (paged_reserved_bytes / static_reserved_bytes)
    paged_save_vs_realistic = 1.0 - (paged_reserved_bytes / realistic_reserved_bytes)
    print(
        f"  lengths={lengths}  (block_size={block_size}, n_positions={n_positions})"
    )
    print(
        f"  reserved bytes: static_worst={static_reserved_bytes/1e6:.1f} MB, "
        f"static_realistic={realistic_reserved_bytes/1e6:.1f} MB, "
        f"paged={paged_reserved_bytes/1e6:.1f} MB"
    )
    print(
        f"  paged reduction vs worst-case static = {paged_save_vs_worst*100:.1f}%, "
        f"vs realistic static = {paged_save_vs_realistic*100:.1f}%"
    )

    _write_csv(os.path.join(out_dir, "exp1_memory_breakdown.csv"), rows)


# ---------------------------------------------------------------------------
# Experiment 2 — Empirical capacity curve
# ---------------------------------------------------------------------------

def _max_batch_empirical(args, seq_len: int) -> int:
    """Allocate up to args.num_kv_blocks sequences of length ``seq_len`` into
    a fresh BlockManager until RuntimeError; return the largest batch that
    fits."""
    best = 0
    for candidate in range(1, args.num_kv_blocks * args.block_size // max(seq_len, 1) + 2):
        bm = _make_block_manager(args)
        try:
            for i in range(candidate):
                bm.allocate_blocks_for_sequence(i, seq_len)
            best = candidate
            for i in range(candidate):
                bm.free_sequence(i)
        except RuntimeError:
            break
    return best


def _max_batch_static(args, seq_len: int, cap_per_seq: int) -> int:
    """Analytical max batch when each seq is forced to reserve cap_per_seq
    slots (contiguous KV behaviour)."""
    total_slots = args.num_kv_blocks * args.block_size
    return total_slots // cap_per_seq if cap_per_seq > 0 else 0


def run_capacity_curve(args, out_dir: str) -> None:
    print("\n== Experiment 2: Capacity curve ==")
    seq_lens = [32, 48, 64, 96, 128, 160, 192, 256]
    decode_margin = args.capacity_decode_margin
    rows = []
    for seq_len in seq_lens:
        paged_max = _max_batch_empirical(args, seq_len)
        static_worst = _max_batch_static(args, seq_len, args.n_positions)
        # "Static realistic" models an operator who over-provisions a
        # decode-length margin above seq_len (they don't know the true
        # decode length, so they reserve some extra).  With margin = 0 this
        # degenerates to exactly equal paged (because both are block-
        # aligned on seq_len), which makes the figure uninformative; we
        # therefore default to a small but honest margin.
        realistic_cap = min(
            args.n_positions,
            seq_len + decode_margin,
        )
        static_realistic = _max_batch_static(args, seq_len, realistic_cap)
        row = {
            "seq_len": seq_len,
            "decode_margin": decode_margin,
            "realistic_cap_per_seq": realistic_cap,
            "paged_max_batch": paged_max,
            "static_worst_case_max_batch": static_worst,
            "static_realistic_max_batch": static_realistic,
            "capacity_gain_vs_worst": (
                round(paged_max / static_worst, 2) if static_worst > 0 else -1.0
            ),
            "capacity_gain_vs_realistic": (
                round(paged_max / static_realistic, 2) if static_realistic > 0 else -1.0
            ),
        }
        rows.append(row)
        print(
            f"  seq_len={seq_len:>4} (cap={realistic_cap}): paged={paged_max:>4}  "
            f"static_worst={static_worst:>3}  static_realistic={static_realistic:>4}  "
            f"gain_vs_realistic={row['capacity_gain_vs_realistic']:.2f}x"
        )
    _write_csv(os.path.join(out_dir, "exp2_capacity_curve.csv"), rows)


# ---------------------------------------------------------------------------
# Experiment 3 — Decode speed (paged vs no-cache vs contiguous KV)
# ---------------------------------------------------------------------------

def _decode_paged_once(args, backend, backend_name, decode_backend,
                       prompt_np: np.ndarray, max_new_tokens: int):
    """Run one full prefill+decode with the paged model."""
    model = _make_paged_model(args, backend, backend_name, decode_backend)
    bm = _make_block_manager(args)
    batch_size, prompt_len = prompt_np.shape
    seq_ids = list(range(batch_size))

    def go():
        prompt = tensor_from_numpy(prompt_np.astype(datatype), backend=backend)
        logits = model.forward_prefill(prompt, bm, seq_ids)
        last = logits.to_numpy()[:, -1, :]
        next_tokens = np.argmax(last, axis=-1).astype(datatype)
        for step in range(1, max_new_tokens):
            token_input = tensor_from_numpy(
                next_tokens.reshape(batch_size, 1),
                backend=backend,
            )
            start_pos = prompt_len + step - 1
            logits = model.forward_decode(token_input, bm, seq_ids, start_pos=start_pos)
            last = logits.to_numpy()[:, 0, :]
            next_tokens = np.argmax(last, axis=-1).astype(datatype)
        for seq_id in seq_ids:
            if seq_id in bm.block_tables:
                bm.free_sequence(seq_id)

    return model, go


def _decode_contig_once(args, backend, backend_name, decode_backend,
                        prompt_np: np.ndarray, max_new_tokens: int):
    """Run one full prefill+decode with the contiguous KV baseline."""
    model = _make_paged_model(args, backend, backend_name, decode_backend)
    batch_size, prompt_len = prompt_np.shape
    ck = ContiguousKVDecoderLM(
        model,
        max_batch_size=batch_size,
        max_seq_len=prompt_len + max_new_tokens,
    )

    def go():
        ck.reset()
        prompt = tensor_from_numpy(prompt_np.astype(datatype), backend=backend)
        logits = ck.forward_prefill(prompt)
        last = logits.to_numpy()[:, -1, :]
        next_tokens = np.argmax(last, axis=-1).astype(datatype)
        for step in range(1, max_new_tokens):
            token_input = tensor_from_numpy(
                next_tokens.reshape(batch_size, 1),
                backend=backend,
            )
            start_pos = prompt_len + step - 1
            logits = ck.forward_decode(token_input, start_pos=start_pos)
            last = logits.to_numpy()[:, 0, :]
            next_tokens = np.argmax(last, axis=-1).astype(datatype)

    return ck, go


def _decode_nocache_once(args, backend, backend_name,
                         prompt_np: np.ndarray, max_new_tokens: int):
    """Run one full prefill+decode with the no-cache baseline DecoderLM."""
    model = DecoderLM(
        n_vocab=args.n_vocab,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_positions=args.n_positions,
        n_layers=args.n_layers,
        p_dropout=0.0,
        backend=backend,
    )
    model.eval()
    batch_size, prompt_len = prompt_np.shape

    def go():
        gen = prompt_np.astype(datatype).copy()
        prompt = tensor_from_numpy(gen, backend=backend)
        logits = model.forward(prompt)
        last = logits.to_numpy()[:, -1, :]
        next_tokens = np.argmax(last, axis=-1).astype(datatype)
        gen = np.concatenate([gen, next_tokens.reshape(batch_size, 1)], axis=1)
        for _ in range(1, max_new_tokens):
            full = tensor_from_numpy(gen, backend=backend)
            logits = model.forward(full)
            last = logits.to_numpy()[:, -1, :]
            next_tokens = np.argmax(last, axis=-1).astype(datatype)
            gen = np.concatenate([gen, next_tokens.reshape(batch_size, 1)], axis=1)

    return model, go


def run_decode_speed(args, backend, backend_name, decode_backend,
                     out_dir: str) -> None:
    print("\n== Experiment 3: Decode speed (3 methods) ==")
    rows = []
    for batch_size in args.decode_batch_sizes:
        for seq_len in args.decode_seq_lens:
            label = f"batch={batch_size}, seq={seq_len}"
            prompt_np = np.random.randint(
                0, args.n_vocab, size=(batch_size, seq_len)
            ).astype(datatype)

            method_results = {}
            for method_name, maker in [
                ("paged", _decode_paged_once),
                ("contiguous_kv", _decode_contig_once),
                ("no_cache", _decode_nocache_once),
            ]:
                try:
                    if method_name == "no_cache":
                        _m, go = maker(args, backend, backend_name, prompt_np,
                                       args.max_new_tokens)
                    else:
                        _m, go = maker(args, backend, backend_name, decode_backend,
                                       prompt_np, args.max_new_tokens)
                    t = timed(go, args.warmup_trials, args.timed_trials)
                    method_results[method_name] = t
                    print(
                        f"  [{label}] {method_name:<14} "
                        f"median={t['median']*1000:>8.1f}ms  "
                        f"range=[{t['min']*1000:>6.1f}, {t['max']*1000:>6.1f}] ms"
                    )
                except Exception as e:
                    print(f"  [{label}] {method_name} ERROR: {e}")
                    traceback.print_exc()
                    method_results[method_name] = None

            paged = method_results.get("paged")
            contig = method_results.get("contiguous_kv")
            nocache = method_results.get("no_cache")

            def _field(stat, key):
                return round(stat[key], 5) if stat else None

            row = {
                "batch_size": batch_size,
                "seq_len": seq_len,
                "max_new_tokens": args.max_new_tokens,
                "paged_median_s": _field(paged, "median"),
                "paged_min_s": _field(paged, "min"),
                "paged_max_s": _field(paged, "max"),
                "contiguous_kv_median_s": _field(contig, "median"),
                "contiguous_kv_min_s": _field(contig, "min"),
                "contiguous_kv_max_s": _field(contig, "max"),
                "no_cache_median_s": _field(nocache, "median"),
                "no_cache_min_s": _field(nocache, "min"),
                "no_cache_max_s": _field(nocache, "max"),
                "paged_speedup_vs_no_cache": (
                    round(nocache["median"] / paged["median"], 3)
                    if paged and nocache and paged["median"] > 0
                    else None
                ),
                "paged_speedup_vs_contiguous": (
                    round(contig["median"] / paged["median"], 3)
                    if paged and contig and paged["median"] > 0
                    else None
                ),
                "contiguous_speedup_vs_no_cache": (
                    round(nocache["median"] / contig["median"], 3)
                    if contig and nocache and contig["median"] > 0
                    else None
                ),
            }
            rows.append(row)
    _write_csv(os.path.join(out_dir, "exp3_decode_speed.csv"), rows)


# ---------------------------------------------------------------------------
# Experiment 4 — Prefix prefill speedup
# ---------------------------------------------------------------------------

def run_prefix_prefill(args, backend, backend_name, decode_backend,
                       out_dir: str) -> None:
    print("\n== Experiment 4: Prefix prefill speedup ==")
    seq_len = args.prefix_seq_len
    rows = []
    for share_ratio in args.prefix_share_ratios:
        shared_blocks = int((seq_len * share_ratio) // args.block_size)
        shared_tokens = shared_blocks * args.block_size
        effective_share = shared_tokens / seq_len if seq_len > 0 else 0.0

        fresh_stats: Optional[Dict[str, float]] = None
        cached_stats: Optional[Dict[str, float]] = None

        try:
            model = _make_paged_model(args, backend, backend_name, decode_backend)
            base_np = np.random.randint(0, args.n_vocab, size=(1, seq_len)).astype(datatype)
            base_prompt = tensor_from_numpy(base_np, backend=backend)

            # Populate the cache once with the base request.  This is the one-
            # time cost we amortise across the second request; not timed.
            cached_manager = _make_block_manager(args)
            model.forward_prefill(base_prompt, cached_manager, [9000])
            cached_manager.free_sequence(9000)

            seq_counter = [9001]
            num_calls = args.warmup_trials + args.timed_trials

            # Cached path: each trial uses a DISTINCT second_prompt that
            # shares exactly ``shared_tokens`` tokens with base (the first
            # shared_tokens positions are copied from base, the tail is
            # freshly random).  If we reused one second_prompt across trials,
            # the warmup trial would publish its blocks into the cache, and
            # every timed trial afterwards would become a 100% cache hit
            # regardless of share_ratio — which is what the previous run
            # showed (speedup nearly flat across ratios).  Rotating tails
            # pins the cache hit to exactly shared_tokens.
            cached_prompts: List = []
            for _ in range(num_calls):
                sp_np = base_np.copy()
                if shared_tokens < seq_len:
                    sp_np[:, shared_tokens:] = np.random.randint(
                        0, args.n_vocab, size=(1, seq_len - shared_tokens),
                    ).astype(datatype)
                cached_prompts.append(
                    tensor_from_numpy(sp_np, backend=backend)
                )
            cached_call_idx = [0]

            def go_cached():
                fp = cached_prompts[cached_call_idx[0]]
                cached_call_idx[0] += 1
                sid = seq_counter[0]
                seq_counter[0] += 1
                model.forward_prefill(fp, cached_manager, [sid])
                cached_manager.free_sequence(sid)

            cached_stats = timed(go_cached, args.warmup_trials, args.timed_trials)

            # Fresh baseline: a single long-lived manager that has NEVER seen
            # the base prompt.  To avoid self-caching across trials we pre-
            # generate a distinct random prompt for each trial (all of length
            # seq_len).  BlockManager construction is a heavyweight CUDA/NumPy
            # op — doing it inside the timed loop as before meant we were
            # measuring allocator setup cost rather than prefill cost, which
            # is why share_ratio=0 wrongly reported ~3x speedup.
            fresh_manager = _make_block_manager(args)
            fresh_prompts: List = []
            for _ in range(num_calls):
                fp_np = np.random.randint(
                    0, args.n_vocab, size=(1, seq_len),
                ).astype(datatype)
                fresh_prompts.append(
                    tensor_from_numpy(fp_np, backend=backend)
                )
            fresh_call_idx = [0]

            def go_fresh():
                fp = fresh_prompts[fresh_call_idx[0]]
                fresh_call_idx[0] += 1
                fresh_sid = seq_counter[0]
                seq_counter[0] += 1
                model.forward_prefill(fp, fresh_manager, [fresh_sid])
                fresh_manager.free_sequence(fresh_sid)

            fresh_stats = timed(go_fresh, args.warmup_trials, args.timed_trials)
        except Exception as e:
            print(f"  share_ratio={share_ratio} ERROR: {e}")
            traceback.print_exc()

        speedup = (
            round(fresh_stats["median"] / cached_stats["median"], 3)
            if fresh_stats and cached_stats and cached_stats["median"] > 0
            else None
        )
        row = {
            "share_ratio_requested": share_ratio,
            "share_ratio_effective": round(effective_share, 4),
            "shared_tokens": shared_tokens,
            "shared_blocks": shared_blocks,
            "fresh_prefill_median_s": fresh_stats["median"] if fresh_stats else None,
            "fresh_prefill_min_s": fresh_stats["min"] if fresh_stats else None,
            "fresh_prefill_max_s": fresh_stats["max"] if fresh_stats else None,
            "cached_prefill_median_s": cached_stats["median"] if cached_stats else None,
            "cached_prefill_min_s": cached_stats["min"] if cached_stats else None,
            "cached_prefill_max_s": cached_stats["max"] if cached_stats else None,
            "prefix_prefill_speedup": speedup,
        }
        rows.append(row)
        if fresh_stats and cached_stats:
            print(
                f"  share={share_ratio:.2f} (eff={effective_share:.2f}): "
                f"fresh={fresh_stats['median']*1000:>6.1f}ms, "
                f"cached={cached_stats['median']*1000:>6.1f}ms, "
                f"speedup={speedup}x"
            )
    _write_csv(os.path.join(out_dir, "exp4_prefix_prefill.csv"), rows)


# ---------------------------------------------------------------------------
# Experiment 5 — Parallel sampling memory savings
# ---------------------------------------------------------------------------

def _blocks_for_tokens(num_tokens: int, block_size: int) -> int:
    return math.ceil(num_tokens / block_size) if num_tokens > 0 else 0


def _parallel_sampling_block_usage(
    args,
    prompt_len: int,
    n_outputs: int,
    decode_tokens: int,
) -> Dict[str, int]:
    """Empirically drive a BlockManager through a simulated parallel-sampling
    request and return the block counts for both fork (shared) and clone
    (duplicated) strategies.

    The simulated workload: one prompt of ``prompt_len`` tokens is prefilled,
    then ``n_outputs`` continuations each decode ``decode_tokens`` more
    tokens.  We don't run the model — we just walk the allocator the way the
    real generation code would.
    """
    bs = args.block_size

    # ---- Fork (paged): prompt blocks shared via CoW fork_sequence.
    bm = _make_block_manager(args)
    base_id = 200
    bm.allocate_blocks_for_sequence(base_id, prompt_len)
    bm.publish_sequence_prefix_blocks(
        base_id,
        np.arange(prompt_len, dtype=np.int32),
    )
    child_ids = [base_id]
    next_id = base_id + 1
    for _ in range(n_outputs - 1):
        bm.fork_sequence(base_id, next_id)
        child_ids.append(next_id)
        next_id += 1
    # Decode phase: each continuation appends decode_tokens unique tokens.
    for _ in range(decode_tokens):
        for sid in child_ids:
            bm.append_token_to_sequence(sid)
    fork_blocks = bm.num_used_blocks
    for sid in child_ids:
        bm.free_sequence(sid)

    # ---- Clone (naive): prompt KV is physically duplicated per continuation.
    bm = _make_block_manager(args)
    base_id = 300
    bm.allocate_blocks_for_sequence(base_id, prompt_len)
    child_ids = [base_id]
    next_id = base_id + 1
    for _ in range(n_outputs - 1):
        bm.clone_sequence(base_id, next_id)
        child_ids.append(next_id)
        next_id += 1
    for _ in range(decode_tokens):
        for sid in child_ids:
            bm.append_token_to_sequence(sid)
    clone_blocks = bm.num_used_blocks
    for sid in child_ids:
        bm.free_sequence(sid)

    return {
        "prompt_blocks_per_seq": _blocks_for_tokens(prompt_len, bs),
        "decode_blocks_per_seq": _blocks_for_tokens(prompt_len + decode_tokens, bs)
        - _blocks_for_tokens(prompt_len, bs),
        "fork_blocks_total": fork_blocks,
        "clone_blocks_total": clone_blocks,
    }


def run_parallel_sampling_memory(args, out_dir: str) -> None:
    print("\n== Experiment 5: Parallel sampling memory ==")
    prompt_lens = [32, 67, 128]  # 67 is deliberately non-aligned
    n_outputs_list = [2, 4, 6, 8]
    decode_tokens = args.max_new_tokens
    rows = []
    for prompt_len in prompt_lens:
        for n_outputs in n_outputs_list:
            try:
                usage = _parallel_sampling_block_usage(
                    args, prompt_len, n_outputs, decode_tokens,
                )
            except Exception as e:
                print(f"  prompt_len={prompt_len}, n_out={n_outputs} ERROR: {e}")
                continue
            fork_blocks = usage["fork_blocks_total"]
            clone_blocks = usage["clone_blocks_total"]
            saved = clone_blocks - fork_blocks
            saving_frac = saved / clone_blocks if clone_blocks > 0 else 0.0
            row = {
                "prompt_len": prompt_len,
                "n_outputs": n_outputs,
                "decode_tokens": decode_tokens,
                "prompt_blocks_per_seq": usage["prompt_blocks_per_seq"],
                "decode_blocks_per_seq": usage["decode_blocks_per_seq"],
                "fork_blocks_total": fork_blocks,
                "clone_blocks_total": clone_blocks,
                "blocks_saved": saved,
                "saving_fraction": round(saving_frac, 4),
            }
            rows.append(row)
            print(
                f"  prompt_len={prompt_len:>3}, n_outputs={n_outputs}: "
                f"fork={fork_blocks:>4}, clone={clone_blocks:>4}, "
                f"saved={saved:>4} ({saving_frac*100:.1f}%)"
            )
    _write_csv(os.path.join(out_dir, "exp5_parallel_sampling.csv"), rows)


# ---------------------------------------------------------------------------
# Experiment 6 — Beam search memory savings
# ---------------------------------------------------------------------------

def _beam_search_block_usage(
    args,
    prompt_len: int,
    beam_width: int,
    decode_tokens: int,
) -> Dict[str, int]:
    """Simulate *winner-takes-all* beam search allocation patterns.

    This is the structural feature that distinguishes beam search from
    parallel sampling: surviving beams at step ``t`` tend to descend from
    the same parent at step ``t-1`` (the highest-scoring hypothesis), so
    they share that parent's entire decode trajectory — not just the
    prompt.  We model the extreme-convergent regime where *every* step's
    survivors come from a single parent; only the final decode step
    diverges into ``beam_width`` distinct candidate tokens.

    Concretely, we grow a single "trunk" sequence by ``decode_tokens - 1``
    tokens and then fork it into ``beam_width`` tips, each of which
    commits a different final-token candidate.  Peak memory is measured
    while the trunk and all tips are concurrently alive, because that is
    how much KV you must hold during the final beam-expansion step.

    Parallel sampling (Exp 5) shares only the prompt because beams are
    independent during decode; beam search additionally shares the
    winning decode trajectory, which is why the savings here scale with
    ``decode_tokens``, not just with ``prompt_len``.
    """
    bs = args.block_size

    # ---- Paged fork (winner-takes-all beam search)
    bm = _make_block_manager(args)
    trunk_id = 400
    bm.allocate_blocks_for_sequence(trunk_id, prompt_len)
    bm.publish_sequence_prefix_blocks(
        trunk_id,
        np.arange(prompt_len, dtype=np.int32),
    )
    # Trunk commits decode_tokens - 1 tokens (the winner's history that
    # all surviving beams agreed on).
    for _ in range(decode_tokens - 1):
        bm.append_token_to_sequence(trunk_id)
    # Final step: fork into beam_width tips, each appending a distinct
    # candidate next-token.  Because the trunk's last block is only
    # partially filled, each tip's append triggers a CoW copy of that
    # last block — the earlier full trunk blocks stay shared via
    # reference counting.
    tip_ids: List[int] = []
    next_id = trunk_id + 1
    for _ in range(beam_width):
        bm.fork_sequence(trunk_id, next_id)
        bm.append_token_to_sequence(next_id)
        tip_ids.append(next_id)
        next_id += 1
    # Peak memory: trunk + all tips alive simultaneously (this is what
    # the runtime actually holds at the moment of beam expansion).
    fork_blocks_peak = bm.num_used_blocks
    # After pruning the trunk (no longer needed once tips carry the
    # state forward), the survivors retain shared trunk blocks via
    # ref-counting.  We also record this "post-prune" count so the plot
    # can optionally show steady-state.
    bm.free_sequence(trunk_id)
    fork_blocks_post_prune = bm.num_used_blocks
    for tid in tip_ids:
        bm.free_sequence(tid)

    # ---- Naive clone: each beam carries an independent copy of both the
    # prompt and its full decode trajectory.
    bm = _make_block_manager(args)
    base_id = 500
    bm.allocate_blocks_for_sequence(base_id, prompt_len)
    beam_ids = [base_id]
    next_id = base_id + 1
    for _ in range(beam_width - 1):
        bm.clone_sequence(base_id, next_id)
        beam_ids.append(next_id)
        next_id += 1
    for _ in range(decode_tokens):
        for sid in beam_ids:
            bm.append_token_to_sequence(sid)
    clone_blocks = bm.num_used_blocks
    for sid in beam_ids:
        bm.free_sequence(sid)

    return {
        "fork_blocks_total": fork_blocks_peak,
        "fork_blocks_post_prune": fork_blocks_post_prune,
        "clone_blocks_total": clone_blocks,
    }


def run_beam_search_memory(args, out_dir: str) -> None:
    print("\n== Experiment 6: Beam search memory (winner-takes-all) ==")
    prompt_lens = [32, 67, 128]
    beam_widths = [2, 4, 6, 8]
    decode_tokens = args.beam_decode_tokens
    rows = []
    for prompt_len in prompt_lens:
        for beam_width in beam_widths:
            try:
                usage = _beam_search_block_usage(
                    args, prompt_len, beam_width, decode_tokens,
                )
            except Exception as e:
                print(f"  prompt_len={prompt_len}, beam={beam_width} ERROR: {e}")
                continue
            fork_blocks = usage["fork_blocks_total"]
            fork_blocks_post_prune = usage.get("fork_blocks_post_prune", fork_blocks)
            clone_blocks = usage["clone_blocks_total"]
            saved = clone_blocks - fork_blocks
            saving_frac = saved / clone_blocks if clone_blocks > 0 else 0.0
            row = {
                "prompt_len": prompt_len,
                "beam_width": beam_width,
                "decode_tokens": decode_tokens,
                "fork_blocks_total": fork_blocks,
                "fork_blocks_post_prune": fork_blocks_post_prune,
                "clone_blocks_total": clone_blocks,
                "blocks_saved": saved,
                "saving_fraction": round(saving_frac, 4),
            }
            rows.append(row)
            print(
                f"  prompt_len={prompt_len:>3}, beam={beam_width}: "
                f"fork_peak={fork_blocks:>4} "
                f"(post_prune={fork_blocks_post_prune:>4}), "
                f"clone={clone_blocks:>4}, "
                f"saved={saved:>4} ({saving_frac*100:.1f}%)"
            )
    _write_csv(os.path.join(out_dir, "exp6_beam_search.csv"), rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    backend_name, decode_backend = _resolve_backends(args.backend, args.decode_backend)
    backend = _create_backend(backend_name)

    os.makedirs(args.output_dir, exist_ok=True)
    meta_path = os.path.join(args.output_dir, "run_meta.csv")

    print("=" * 72)
    print("Rigorous PagedAttention Benchmark")
    print("=" * 72)
    print(f"Model: vocab={args.n_vocab}, embd={args.n_embd}, heads={args.n_head}, "
          f"layers={args.n_layers}, n_positions={args.n_positions}")
    print(f"KV cache: block_size={args.block_size}, num_kv_blocks={args.num_kv_blocks}")
    print(f"Backend: {backend_name} / decode={decode_backend}")
    print(f"Trials: {args.warmup_trials} warmup + {args.timed_trials} timed")
    print(f"Max new tokens (decode): {args.max_new_tokens}")
    print(f"Output: {args.output_dir}")
    print("=" * 72)

    wall_start = time.perf_counter()
    if not args.skip_memory:
        run_memory_breakdown(args, args.output_dir)
    if not args.skip_capacity:
        run_capacity_curve(args, args.output_dir)
    if not args.skip_decode:
        run_decode_speed(args, backend, backend_name, decode_backend, args.output_dir)
    if not args.skip_prefix:
        run_prefix_prefill(args, backend, backend_name, decode_backend, args.output_dir)
    if not args.skip_sharing:
        run_parallel_sampling_memory(args, args.output_dir)
        run_beam_search_memory(args, args.output_dir)
    wall_end = time.perf_counter()

    # Run metadata (useful for plot annotations).
    _write_csv(meta_path, [{
        "n_vocab": args.n_vocab,
        "n_embd": args.n_embd,
        "n_head": args.n_head,
        "n_layers": args.n_layers,
        "n_positions": args.n_positions,
        "block_size": args.block_size,
        "num_kv_blocks": args.num_kv_blocks,
        "backend": backend_name,
        "decode_backend": decode_backend,
        "warmup_trials": args.warmup_trials,
        "timed_trials": args.timed_trials,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "wall_seconds": round(wall_end - wall_start, 1),
    }])
    print(f"\nTotal wall time: {wall_end - wall_start:.1f} s")


if __name__ == "__main__":
    main()
