"""GPU-resident PagedAttention vs contiguous attention benchmark.

By default, this benchmark loads HuggingFace GPT-2, tokenizes real WikiText
examples, runs GPT-2 inference on CUDA, extracts a real layer K/V cache and the
next-token attention query, and then compares the current paged CUDA kernel with
a contiguous no-paging CUDA baseline.

The timed comparison keeps query, output, and K/V tensors on GPU. The paged path
copies a GPU-created paged KV tensor into the persistent CUDA runtime with a
device-to-device copy during setup, then timed iterations call the runtime with
CUDA tensor data pointers. No timed iteration performs host-to-device K/V,
query, or output copies.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

sys.path.insert(0, ".")

from minitorch.paged_attention import PagedAttentionKernel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark GPU-resident paged attention against contiguous attention"
    )
    parser.add_argument("--source", choices=["gpt2", "synthetic"], default="gpt2")
    parser.add_argument("--model-name", type=str, default="gpt2")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dataset-name", type=str, default="wikitext")
    parser.add_argument("--dataset-config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--dataset-split", type=str, default="test")
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--prompts-file", type=str, default=None)
    parser.add_argument("--max-dataset-rows", type=int, default=2000)
    parser.add_argument(
        "--layer-id",
        type=int,
        default=-1,
        help="GPT-2 layer to extract K/V and query from. Negative values count from the end.",
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--seq-lengths", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--static-context-len", type=int, default=2048)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--timed-iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument("--output-dir", type=str, default="benchmarks/results_gpu_resident")
    parser.add_argument("--output-csv", type=str, default="paged_vs_contiguous_gpu_resident.csv")
    return parser.parse_args()


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("This benchmark requires PyTorch with CUDA support") from exc
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available to PyTorch; cannot run GPU-resident benchmark")
    return torch


def _import_hf_dependencies():
    try:
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "GPT-2 source requires optional dependencies: pip install -r requirements.gpt2.txt"
        ) from exc
    return load_dataset, AutoModelForCausalLM, AutoTokenizer


def _make_block_tables(batch_size: int, seq_len: int, block_size: int) -> Tuple[np.ndarray, np.ndarray, int, int]:
    blocks_per_seq = math.ceil(seq_len / block_size)
    block_tables = np.zeros((batch_size, blocks_per_seq), dtype=np.int32)
    for batch_idx in range(batch_size):
        start = batch_idx * blocks_per_seq
        block_tables[batch_idx] = np.arange(start, start + blocks_per_seq, dtype=np.int32)
    context_lens = np.full((batch_size,), seq_len, dtype=np.int32)
    return block_tables, context_lens, blocks_per_seq, batch_size * blocks_per_seq


def _build_paged_cache_from_contiguous(
    torch,
    contiguous_key,
    contiguous_value,
    batch_size: int,
    seq_len: int,
    block_size: int,
    n_head: int,
    head_dim: int,
    num_blocks: int,
):
    paged_key = torch.zeros((num_blocks, block_size, n_head, head_dim), device="cuda", dtype=torch.float32)
    paged_value = torch.zeros_like(paged_key)
    blocks_per_seq = math.ceil(seq_len / block_size)
    for batch_idx in range(batch_size):
        for logical_block in range(blocks_per_seq):
            token_start = logical_block * block_size
            token_end = min(token_start + block_size, seq_len)
            valid_tokens = token_end - token_start
            physical_block = batch_idx * blocks_per_seq + logical_block
            paged_key[physical_block, :valid_tokens].copy_(
                contiguous_key[batch_idx, token_start:token_end]
            )
            paged_value[physical_block, :valid_tokens].copy_(
                contiguous_value[batch_idx, token_start:token_end]
            )
    torch.cuda.synchronize()
    return paged_key.contiguous(), paged_value.contiguous()


def _read_prompt_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _load_real_texts(args: argparse.Namespace, load_dataset) -> List[str]:
    if args.prompts_file:
        return _read_prompt_file(args.prompts_file)

    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.dataset_split,
    )
    texts: List[str] = []
    for row in dataset:
        text = str(row.get(args.text_column, "")).strip()
        if text:
            texts.append(text)
        if len(texts) >= args.max_dataset_rows:
            break
    if not texts:
        raise ValueError("No non-empty real text rows found for GPT-2 benchmark")
    return texts


def _token_stream_from_texts(tokenizer, texts: Iterable[str], min_tokens: int) -> List[int]:
    token_ids: List[int] = []
    eos_token_id = tokenizer.eos_token_id
    for text in texts:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if encoded:
            token_ids.extend(encoded)
            if eos_token_id is not None:
                token_ids.append(int(eos_token_id))
        if len(token_ids) >= min_tokens:
            break
    if len(token_ids) < min_tokens:
        raise ValueError(
            f"Real dataset did not provide enough tokens: need {min_tokens}, got {len(token_ids)}"
        )
    return token_ids


def _input_ids_from_stream(torch, token_ids: List[int], batch_size: int, seq_len: int, offset: int):
    needed = batch_size * seq_len
    if offset + needed > len(token_ids):
        offset = 0
    values = token_ids[offset : offset + needed]
    if len(values) < needed:
        raise ValueError(f"Not enough real tokens for batch={batch_size}, seq_len={seq_len}")
    return torch.tensor(values, device="cuda", dtype=torch.long).view(batch_size, seq_len)


def _normalize_layer_id(layer_id: int, n_layers: int) -> int:
    if layer_id < 0:
        layer_id = n_layers + layer_id
    if layer_id < 0 or layer_id >= n_layers:
        raise ValueError(f"layer_id must be in [0, {n_layers}), got {layer_id}")
    return layer_id


def _layer_past_to_contiguous_kv(layer_past, batch_size: int, seq_len: int):
    key, value = layer_past[0], layer_past[1]
    if key.shape[0] != batch_size or value.shape[0] != batch_size:
        raise ValueError("Unexpected GPT-2 past batch dimension")

    if key.ndim != 4 or value.ndim != 4:
        raise ValueError(f"Expected rank-4 GPT-2 K/V tensors, got {key.shape} and {value.shape}")

    if key.shape[2] == seq_len:
        key = key.permute(0, 2, 1, 3)
    elif key.shape[3] == seq_len:
        key = key.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"Could not locate sequence dimension in GPT-2 key shape {tuple(key.shape)}")

    if value.shape[2] == seq_len:
        value = value.permute(0, 2, 1, 3)
    elif value.shape[3] == seq_len:
        value = value.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"Could not locate sequence dimension in GPT-2 value shape {tuple(value.shape)}")

    return key.contiguous().float(), value.contiguous().float()


def _get_layer_past(past_key_values, layer_id: int):
    if hasattr(past_key_values, "to_legacy_cache"):
        past_key_values = past_key_values.to_legacy_cache()
    if hasattr(past_key_values, "layers"):
        layer = past_key_values.layers[layer_id]
        if hasattr(layer, "keys") and hasattr(layer, "values"):
            return layer.keys, layer.values
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return past_key_values.key_cache[layer_id], past_key_values.value_cache[layer_id]
    return past_key_values[layer_id]


def _extract_gpt2_workload(torch, model, input_ids, layer_id: int):
    batch_size, seq_len = input_ids.shape
    n_embd = int(model.config.n_embd)
    n_head = int(model.config.n_head)
    head_dim = n_embd // n_head
    captured = []

    def capture_c_attn(_module, _inputs, output):
        captured.append(output.detach())

    with torch.inference_mode():
        prefill = model(input_ids=input_ids, use_cache=True)
        selected_layer_past = _get_layer_past(prefill.past_key_values, layer_id)
        selected_layer_past = (
            selected_layer_past[0].detach().clone(),
            selected_layer_past[1].detach().clone(),
        )
        next_token = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        handle = model.transformer.h[layer_id].attn.c_attn.register_forward_hook(capture_c_attn)
        try:
            model(input_ids=next_token, past_key_values=prefill.past_key_values, use_cache=True)
        finally:
            handle.remove()

    if not captured:
        raise RuntimeError("Failed to capture GPT-2 decode c_attn output")

    c_attn_output = captured[-1]
    query = c_attn_output[..., :n_embd]
    query = query.view(batch_size, 1, n_head, head_dim).permute(0, 2, 1, 3)[:, :, 0, :]
    query = query.contiguous().float()
    contiguous_key, contiguous_value = _layer_past_to_contiguous_kv(selected_layer_past, batch_size, seq_len)
    return query, contiguous_key, contiguous_value, next_token.detach()


def _load_gpt2_source(args: argparse.Namespace, torch):
    load_dataset, AutoModelForCausalLM, AutoTokenizer = _import_hf_dependencies()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        revision=args.revision,
        local_files_only=args.local_files_only,
    ).to("cuda")
    model.eval()
    layer_id = _normalize_layer_id(args.layer_id, int(model.config.n_layer))
    max_seq_len = max(args.seq_lengths)
    if max_seq_len >= int(model.config.n_positions):
        raise ValueError(
            f"seq_len={max_seq_len} leaves no room for one decode token in GPT-2 "
            f"position capacity {model.config.n_positions}"
        )
    max_tokens_needed = sum(
        batch_size * seq_len
        for batch_size in args.batch_sizes
        for seq_len in args.seq_lengths
    )
    texts = _load_real_texts(args, load_dataset)
    token_ids = _token_stream_from_texts(tokenizer, texts, max_tokens_needed)
    print(
        f"Loaded {args.model_name} on CUDA and {len(token_ids)} real tokens from "
        f"{args.dataset_name}/{args.dataset_config}:{args.dataset_split}; layer={layer_id}"
    )
    return model, token_ids, layer_id


def _time_call(fn, torch, warmup_iters: int, timed_iters: int) -> Dict[str, float]:
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(timed_iters):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)

    arr = np.array(samples, dtype=np.float64)
    return {
        "median_ms": float(np.median(arr) * 1000.0),
        "mean_ms": float(np.mean(arr) * 1000.0),
        "min_ms": float(np.min(arr) * 1000.0),
        "p95_ms": float(np.percentile(arr, 95) * 1000.0),
    }


def _benchmark_one(
    torch,
    batch_size: int,
    seq_len: int,
    block_size: int,
    n_head: int,
    head_dim: int,
    static_context_len: int,
    warmup_iters: int,
    timed_iters: int,
    query=None,
    contiguous_key=None,
    contiguous_value=None,
    metadata: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    block_tables, context_lens_host, blocks_per_seq, num_blocks = _make_block_tables(
        batch_size, seq_len, block_size
    )
    if query is None or contiguous_key is None or contiguous_value is None:
        query = torch.randn((batch_size, n_head, head_dim), device="cuda", dtype=torch.float32)
        contiguous_key = torch.randn((batch_size, seq_len, n_head, head_dim), device="cuda", dtype=torch.float32)
        contiguous_value = torch.randn_like(contiguous_key)
    else:
        query = query.contiguous().float()
        contiguous_key = contiguous_key.contiguous().float()
        contiguous_value = contiguous_value.contiguous().float()

    if query.shape != (batch_size, n_head, head_dim):
        raise ValueError(f"query shape {tuple(query.shape)} does not match benchmark config")
    if contiguous_key.shape != (batch_size, seq_len, n_head, head_dim):
        raise ValueError(f"key shape {tuple(contiguous_key.shape)} does not match benchmark config")
    if contiguous_value.shape != (batch_size, seq_len, n_head, head_dim):
        raise ValueError(f"value shape {tuple(contiguous_value.shape)} does not match benchmark config")

    context_lens_device = torch.full((batch_size,), seq_len, device="cuda", dtype=torch.int32)
    paged_key, paged_value = _build_paged_cache_from_contiguous(
        torch,
        contiguous_key,
        contiguous_value,
        batch_size,
        seq_len,
        block_size,
        n_head,
        head_dim,
        num_blocks,
    )
    paged_output = torch.empty((batch_size, n_head, head_dim), device="cuda", dtype=torch.float32)
    contiguous_output = torch.empty_like(paged_output)

    kernel = PagedAttentionKernel()
    kernel.ensure_runtime(
        num_blocks=num_blocks,
        block_size=block_size,
        n_head=n_head,
        head_dim=head_dim,
        max_batch=batch_size,
        max_blocks_per_seq=blocks_per_seq,
    )
    kernel.copy_layer_cache_from_device(paged_key.data_ptr(), paged_value.data_ptr())
    kernel.update_metadata(block_tables, context_lens_host)

    def run_paged() -> None:
        kernel.forward_device(query.data_ptr(), paged_output.data_ptr(), batch_size, seq_len)

    def run_contiguous() -> None:
        kernel.contiguous_forward_device(
            contiguous_output.data_ptr(),
            query.data_ptr(),
            contiguous_key.data_ptr(),
            contiguous_value.data_ptr(),
            context_lens_device.data_ptr(),
            batch_size,
            n_head,
            head_dim,
            seq_len,
        )

    run_paged()
    run_contiguous()
    torch.cuda.synchronize()
    max_abs_error = float((paged_output - contiguous_output).abs().max().item())
    mean_abs_error = float((paged_output - contiguous_output).abs().mean().item())

    paged_stats = _time_call(run_paged, torch, warmup_iters, timed_iters)
    contiguous_stats = _time_call(run_contiguous, torch, warmup_iters, timed_iters)
    kernel.close()

    dtype_size = 4
    bytes_per_token = n_head * head_dim * 2 * dtype_size
    live_tokens = batch_size * seq_len
    paged_reserved_tokens = num_blocks * block_size
    live_kv_bytes = live_tokens * bytes_per_token
    paged_allocated_kv_bytes = paged_reserved_tokens * bytes_per_token
    compact_contiguous_kv_bytes = live_kv_bytes
    static_contiguous_kv_bytes = batch_size * static_context_len * bytes_per_token
    paged_latency = paged_stats["median_ms"]
    contiguous_latency = contiguous_stats["median_ms"]

    row = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "block_size": block_size,
        "n_head": n_head,
        "head_dim": head_dim,
        "blocks_per_seq": blocks_per_seq,
        "num_blocks": num_blocks,
        "timed_iters": timed_iters,
        "kv_setup_copy": "device_to_device",
        "timed_h2d_kv_bytes": 0,
        "timed_h2d_query_bytes": 0,
        "timed_d2h_output_bytes": 0,
        "paged_median_ms": round(paged_latency, 5),
        "paged_mean_ms": round(paged_stats["mean_ms"], 5),
        "paged_min_ms": round(paged_stats["min_ms"], 5),
        "paged_p95_ms": round(paged_stats["p95_ms"], 5),
        "contiguous_median_ms": round(contiguous_latency, 5),
        "contiguous_mean_ms": round(contiguous_stats["mean_ms"], 5),
        "contiguous_min_ms": round(contiguous_stats["min_ms"], 5),
        "contiguous_p95_ms": round(contiguous_stats["p95_ms"], 5),
        "paged_latency_over_contiguous": round(paged_latency / contiguous_latency, 4),
        "contiguous_speedup_over_paged": round(paged_latency / contiguous_latency, 4),
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "live_tokens": live_tokens,
        "paged_reserved_tokens": paged_reserved_tokens,
        "live_kv_bytes": live_kv_bytes,
        "paged_allocated_kv_bytes": paged_allocated_kv_bytes,
        "compact_contiguous_kv_bytes": compact_contiguous_kv_bytes,
        "static_contiguous_kv_bytes": static_contiguous_kv_bytes,
        "paged_internal_fragmentation": round(1.0 - live_kv_bytes / paged_allocated_kv_bytes, 6),
        "paged_over_compact_memory": round(paged_allocated_kv_bytes / compact_contiguous_kv_bytes, 6),
        "paged_savings_vs_static_contiguous": round(
            1.0 - paged_allocated_kv_bytes / static_contiguous_kv_bytes,
            6,
        ),
    }
    if metadata:
        row.update(metadata)
    return row


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    torch = _import_torch()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    gpt2_source = None
    if args.source == "gpt2":
        gpt2_source = _load_gpt2_source(args, torch)

    rows = []
    token_offset = 0
    for batch_size in args.batch_sizes:
        for seq_len in args.seq_lengths:
            if args.source == "gpt2":
                model, token_ids, layer_id = gpt2_source
                if token_offset + batch_size * seq_len > len(token_ids):
                    raise ValueError(
                        "Not enough real GPT-2 dataset tokens for the requested grid. "
                        "Increase --max-dataset-rows or reduce --batch-sizes/--seq-lengths."
                    )
                input_ids = _input_ids_from_stream(torch, token_ids, batch_size, seq_len, token_offset)
                token_offset += batch_size * seq_len
                query, contiguous_key, contiguous_value, next_token = _extract_gpt2_workload(
                    torch,
                    model,
                    input_ids,
                    layer_id,
                )
                n_head = int(model.config.n_head)
                head_dim = int(model.config.n_embd) // n_head
                metadata = {
                    "source": "gpt2",
                    "model_name": args.model_name,
                    "dataset_name": args.dataset_name,
                    "dataset_config": args.dataset_config,
                    "dataset_split": args.dataset_split,
                    "layer_id": layer_id,
                    "first_token_id": int(input_ids[0, 0].item()),
                    "next_token_id": int(next_token[0, 0].item()),
                }
            else:
                query = None
                contiguous_key = None
                contiguous_value = None
                n_head = args.n_head
                head_dim = args.head_dim
                metadata = {
                    "source": "synthetic",
                    "model_name": "",
                    "dataset_name": "",
                    "dataset_config": "",
                    "dataset_split": "",
                    "layer_id": "",
                    "first_token_id": "",
                    "next_token_id": "",
                }

            row = _benchmark_one(
                torch=torch,
                batch_size=batch_size,
                seq_len=seq_len,
                block_size=args.block_size,
                n_head=n_head,
                head_dim=head_dim,
                static_context_len=max(args.static_context_len, seq_len),
                warmup_iters=args.warmup_iters,
                timed_iters=args.timed_iters,
                query=query,
                contiguous_key=contiguous_key,
                contiguous_value=contiguous_value,
                metadata=metadata,
            )
            rows.append(row)
            print(
                f"bs={batch_size} seq={seq_len}: "
                f"paged={row['paged_median_ms']} ms, "
                f"contiguous={row['contiguous_median_ms']} ms, "
                f"latency_ratio={row['paged_latency_over_contiguous']}x, "
                f"static_mem_savings={100 * row['paged_savings_vs_static_contiguous']:.2f}%"
            )

    out_path = Path(args.output_dir) / args.output_csv
    _write_csv(out_path, rows)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()