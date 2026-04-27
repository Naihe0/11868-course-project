"""Benchmark pretrained GPT-2 with MiniTorch GPU-resident KV caches.

This script uses Hugging Face only to load pretrained weights, tokenize a real
dataset, and hand the weights to the local MiniTorch model.  The timed baseline,
contiguous-KV, and paged-attention paths run the full MiniTorch model.  The
paged and contiguous KV caches are device-owned and updated with device-to-device
copies; no timed KV cache path stages K/V through CPU memory.

Smoke run:

    python project/run_gpt2_paged_benchmark.py \
        --model-name gpt2 --batch-sizes 1 --seq-lengths 8 \
        --warmup-iters 1 --timed-iters 3
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numba
import numpy as np
from numba import cuda

sys.path.insert(0, ".")

import minitorch
from minitorch.block_manager import BlockManager
from minitorch.module import Parameter
from minitorch.paged_attention import PagedAttentionKernel, standard_attention
from minitorch.tensor import Tensor, tensor_from_numpy
from minitorch.transformer import PagedDecoderLM


DTYPE = np.float32
DTYPE_SIZE = np.dtype(DTYPE).itemsize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full-model GPT-2 decode benchmarks on MiniTorch."
    )
    parser.add_argument("--model-name", type=str, default="gpt2")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dataset-name", type=str, default="wikitext")
    parser.add_argument("--dataset-config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--dataset-split", type=str, default="test")
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--prompts-file", type=str, default=None)
    parser.add_argument("--max-dataset-rows", type=int, default=2000)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1])
    parser.add_argument("--seq-lengths", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--static-context-len", type=int, default=0)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--timed-iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument("--output-dir", type=str, default="benchmarks/results_gpt2")
    parser.add_argument("--output-csv", type=str, default="gpt2_gpu_kv_benchmark.csv")
    return parser.parse_args()


def _import_hf():
    try:
        import torch
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing GPT-2 benchmark dependencies. Install `requirements.gpt2.txt`."
        ) from exc
    return torch, load_dataset, AutoModelForCausalLM, AutoTokenizer


def _cuda_available() -> bool:
    try:
        return bool(cuda.is_available())
    except Exception:
        return False


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
        raise ValueError("No non-empty text rows found for the GPT-2 benchmark.")
    return texts


def _token_stream_from_texts(tokenizer, texts: Iterable[str], min_tokens: int) -> List[int]:
    token_ids: List[int] = []
    eos_token_id = tokenizer.eos_token_id
    for text in texts:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if encoded:
            token_ids.extend(int(token_id) for token_id in encoded)
            if eos_token_id is not None:
                token_ids.append(int(eos_token_id))
        if len(token_ids) >= min_tokens:
            break
    if len(token_ids) < min_tokens:
        raise ValueError(f"Need {min_tokens} real tokens, got {len(token_ids)}.")
    return token_ids


def _make_backend():
    return minitorch.TensorBackend(minitorch.CudaOps)


def _as_minitorch(array: np.ndarray, backend) -> Tensor:
    return tensor_from_numpy(np.ascontiguousarray(array.astype(DTYPE)), backend=backend)


def _device_ptr(obj) -> int:
    if isinstance(obj, Tensor):
        storage = obj._tensor._storage
    else:
        storage = obj
    if not numba.cuda.is_cuda_array(storage):
        raise RuntimeError("Expected CUDA device-backed storage")
    return int(storage.__cuda_array_interface__["data"][0])


def _empty_device_tensor(shape: Sequence[int], backend) -> Tensor:
    size = int(np.prod(shape))
    storage = cuda.device_array(size, dtype=DTYPE)
    return Tensor.make(storage, tuple(int(dim) for dim in shape), backend=backend)


def _set_parameter(param: Parameter, array: np.ndarray, backend) -> None:
    param.update(_as_minitorch(array, backend))
    param.value.history = None


def _copy_hf_weights_into_minitorch(model: PagedDecoderLM, hf_model, backend) -> None:
    transformer = hf_model.transformer
    _set_parameter(model.token_embeddings.weights, transformer.wte.weight.detach().cpu().numpy(), backend)
    _set_parameter(model.position_embeddings.weights, transformer.wpe.weight.detach().cpu().numpy(), backend)

    for layer_idx, layer in enumerate(model.layers):
        hf_block = transformer.h[layer_idx]
        _set_parameter(layer.ln_1.weights, hf_block.ln_1.weight.detach().cpu().numpy(), backend)
        _set_parameter(layer.ln_1.bias, hf_block.ln_1.bias.detach().cpu().numpy(), backend)
        _set_parameter(layer.ln_2.weights, hf_block.ln_2.weight.detach().cpu().numpy(), backend)
        _set_parameter(layer.ln_2.bias, hf_block.ln_2.bias.detach().cpu().numpy(), backend)

        c_attn_w = hf_block.attn.c_attn.weight.detach().cpu().numpy()
        c_attn_b = hf_block.attn.c_attn.bias.detach().cpu().numpy()
        q_w, k_w, v_w = np.split(c_attn_w, 3, axis=1)
        q_b, k_b, v_b = np.split(c_attn_b, 3, axis=0)
        _set_parameter(layer.attention.q_proj.weights, q_w, backend)
        _set_parameter(layer.attention.q_proj.bias, q_b, backend)
        _set_parameter(layer.attention.k_proj.weights, k_w, backend)
        _set_parameter(layer.attention.k_proj.bias, k_b, backend)
        _set_parameter(layer.attention.v_proj.weights, v_w, backend)
        _set_parameter(layer.attention.v_proj.bias, v_b, backend)

        _set_parameter(layer.attention.out_proj.weights, hf_block.attn.c_proj.weight.detach().cpu().numpy(), backend)
        _set_parameter(layer.attention.out_proj.bias, hf_block.attn.c_proj.bias.detach().cpu().numpy(), backend)
        _set_parameter(layer.ff.linear_in.weights, hf_block.mlp.c_fc.weight.detach().cpu().numpy(), backend)
        _set_parameter(layer.ff.linear_in.bias, hf_block.mlp.c_fc.bias.detach().cpu().numpy(), backend)
        _set_parameter(layer.ff.linear_out.weights, hf_block.mlp.c_proj.weight.detach().cpu().numpy(), backend)
        _set_parameter(layer.ff.linear_out.bias, hf_block.mlp.c_proj.bias.detach().cpu().numpy(), backend)

    _set_parameter(model.ln.weights, transformer.ln_f.weight.detach().cpu().numpy(), backend)
    _set_parameter(model.ln.bias, transformer.ln_f.bias.detach().cpu().numpy(), backend)
    _set_parameter(model.lm_head.weights, transformer.wte.weight.detach().cpu().numpy().T, backend)


def _make_model_from_hf(hf_model, backend, block_size: int) -> PagedDecoderLM:
    config = hf_model.config
    model = PagedDecoderLM(
        n_vocab=int(config.vocab_size),
        n_embd=int(config.n_embd),
        n_head=int(config.n_head),
        n_positions=int(getattr(config, "n_positions", getattr(config, "n_ctx", 1024))),
        n_layers=int(config.n_layer),
        block_size=block_size,
        p_dropout=0.0,
        backend=backend,
        decode_backend="cuda",
        gpu_resident_kv=True,
    )
    model.eval()
    _copy_hf_weights_into_minitorch(model, hf_model, backend)
    return model


class NoKVDecoder:
    """Full MiniTorch model path that recomputes the whole context."""

    def __init__(self, model: PagedDecoderLM) -> None:
        self.model = model
        self.backend = model.backend
        self.n_embd = model.n_embd
        self.n_vocab = model.n_vocab
        self.n_head = model.layers[0].attention.n_head
        self.head_dim = model.layers[0].attention.head_dim

    def _embed(self, idx):
        batch_size, seq_len = idx.shape
        tok_emb = self.model.token_embeddings(idx)
        pos_ids = _as_minitorch(
            np.arange(seq_len, dtype=DTYPE).reshape(1, seq_len),
            self.backend,
        )
        return self.model.dropout(tok_emb + self.model.position_embeddings(pos_ids))

    def _project_qkv(self, attn, x, seq_len):
        batch_size = x.shape[0]
        flat_x = x.contiguous().view(batch_size * seq_len, self.n_embd)
        q = attn.q_proj(flat_x).view(batch_size, seq_len, self.n_head, self.head_dim)
        k = attn.k_proj(flat_x).view(batch_size, seq_len, self.n_head, self.head_dim)
        v = attn.v_proj(flat_x).view(batch_size, seq_len, self.n_head, self.head_dim)
        return q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)

    def _forward_hidden(self, idx):
        batch_size, seq_len = idx.shape
        x = self._embed(idx)
        for layer in self.model.layers:
            attn = layer.attention
            norm_x = layer.ln_1(x.view(batch_size * seq_len, self.n_embd)).view(
                batch_size, seq_len, self.n_embd
            )
            q, k, v = self._project_qkv(attn, norm_x, seq_len)
            mask = _as_minitorch(
                np.triu(np.full((seq_len, seq_len), -1e9, dtype=DTYPE), k=1).reshape(
                    1, 1, seq_len, seq_len
                ),
                self.backend,
            )
            attn_out = standard_attention(q, k, v, mask)
            attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(
                batch_size * seq_len, self.n_embd
            )
            x = x + attn.out_proj(attn_out).view(batch_size, seq_len, self.n_embd)
            norm_x = layer.ln_2(x.view(batch_size * seq_len, self.n_embd)).view(
                batch_size, seq_len, self.n_embd
            )
            x = x + layer.ff(norm_x)
        x = self.model.ln(x.view(batch_size * seq_len, self.n_embd)).view(
            batch_size, seq_len, self.n_embd
        )
        return x

    def forward(self, idx):
        batch_size, seq_len = idx.shape
        x = self._forward_hidden(idx)
        return self.model.lm_head(x.view(batch_size * seq_len, self.n_embd)).view(
            batch_size, seq_len, self.n_vocab
        )

    def forward_last(self, idx):
        batch_size, seq_len = idx.shape
        x = self._forward_hidden(idx)
        last_mask = np.zeros((1, seq_len, 1), dtype=DTYPE)
        last_mask[:, seq_len - 1, :] = 1.0
        last_x = (x * _as_minitorch(last_mask, self.backend)).sum(1).view(
            batch_size, self.n_embd
        )
        return self.model.lm_head(last_x).view(batch_size, 1, self.n_vocab)


class GpuContiguousKVDecoder(NoKVDecoder):
    """Full MiniTorch model with a device-resident contiguous KV cache."""

    def __init__(self, model: PagedDecoderLM, max_batch_size: int, max_seq_len: int) -> None:
        super().__init__(model)
        self.max_batch_size = int(max_batch_size)
        self.max_seq_len = int(max_seq_len)
        self.n_layers = model.n_layers
        shape = (self.max_batch_size, self.max_seq_len, self.n_head, self.head_dim)
        self.key_cache = [_empty_device_tensor(shape, self.backend) for _ in range(self.n_layers)]
        self.value_cache = [_empty_device_tensor(shape, self.backend) for _ in range(self.n_layers)]
        self.kernel = PagedAttentionKernel()
        self.context_len = 0
        self.batch_size = 0

    def reset(self) -> None:
        self.context_len = 0
        self.batch_size = 0

    def reserved_bytes(self) -> int:
        return (
            2
            * self.n_layers
            * self.max_batch_size
            * self.max_seq_len
            * self.n_head
            * self.head_dim
            * DTYPE_SIZE
        )

    def used_bytes(self) -> int:
        return (
            2
            * self.n_layers
            * self.batch_size
            * self.context_len
            * self.n_head
            * self.head_dim
            * DTYPE_SIZE
        )

    def _write_slots(self, layer_idx: int, key_values: Tensor, value_values: Tensor, token_start: int) -> None:
        batch_size, token_count, _, _ = key_values.shape
        row_elems = self.n_head * self.head_dim
        itemsize = DTYPE_SIZE
        key_ptr = _device_ptr(key_values)
        value_ptr = _device_ptr(value_values)
        key_cache_ptr = _device_ptr(self.key_cache[layer_idx])
        value_cache_ptr = _device_ptr(self.value_cache[layer_idx])
        for batch_idx in range(batch_size):
            for local_idx in range(token_count):
                row_offset = (batch_idx * token_count + local_idx) * row_elems * itemsize
                self.kernel.contiguous_update_slot_from_device(
                    key_cache_ptr,
                    value_cache_ptr,
                    batch_idx,
                    token_start + local_idx,
                    key_ptr + row_offset,
                    value_ptr + row_offset,
                    self.max_seq_len,
                    self.n_head,
                    self.head_dim,
                )

    def _run_layer_prefill(self, layer, x, seq_len: int):
        batch_size = x.shape[0]
        attn = layer.attention
        norm_x = layer.ln_1(x.view(batch_size * seq_len, self.n_embd)).view(
            batch_size, seq_len, self.n_embd
        )
        q, k, v = self._project_qkv(attn, norm_x, seq_len)
        self._write_slots(
            attn.layer_id,
            k.permute(0, 2, 1, 3).contiguous(),
            v.permute(0, 2, 1, 3).contiguous(),
            0,
        )
        mask = _as_minitorch(
            np.triu(np.full((seq_len, seq_len), -1e9, dtype=DTYPE), k=1).reshape(
                1, 1, seq_len, seq_len
            ),
            self.backend,
        )
        attn_out = standard_attention(q, k, v, mask)
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(
            batch_size * seq_len, self.n_embd
        )
        x = x + attn.out_proj(attn_out).view(batch_size, seq_len, self.n_embd)
        norm_x = layer.ln_2(x.view(batch_size * seq_len, self.n_embd)).view(
            batch_size, seq_len, self.n_embd
        )
        return x + layer.ff(norm_x)

    def forward_prefill(self, idx):
        batch_size, seq_len = idx.shape
        self.batch_size = batch_size
        self.context_len = seq_len
        x = self._embed(idx)
        for layer in self.model.layers:
            x = self._run_layer_prefill(layer, x, seq_len)
        x = self.model.ln(x.view(batch_size * seq_len, self.n_embd)).view(
            batch_size, seq_len, self.n_embd
        )
        return self.model.lm_head(x.view(batch_size * seq_len, self.n_embd)).view(
            batch_size, seq_len, self.n_vocab
        )

    def _run_layer_decode(self, layer, x):
        batch_size = x.shape[0]
        attn = layer.attention
        norm_x = layer.ln_1(x.view(batch_size, self.n_embd)).view(
            batch_size, 1, self.n_embd
        )
        q, k, v = self._project_qkv(attn, norm_x, 1)
        layer_idx = attn.layer_id
        self._write_slots(
            layer_idx,
            k.permute(0, 2, 1, 3).contiguous(),
            v.permute(0, 2, 1, 3).contiguous(),
            self.context_len,
        )

        out = _empty_device_tensor((batch_size, self.n_head, self.head_dim), self.backend)
        context_lens = cuda.to_device(
            np.full((batch_size,), self.context_len + 1, dtype=np.int32)
        )
        self.kernel.contiguous_forward_device(
            _device_ptr(out),
            _device_ptr(q.contiguous().view(batch_size, self.n_head, self.head_dim)),
            _device_ptr(self.key_cache[layer_idx]),
            _device_ptr(self.value_cache[layer_idx]),
            _device_ptr(context_lens),
            batch_size,
            self.n_head,
            self.head_dim,
            self.max_seq_len,
        )
        attn_out = out.view(batch_size, self.n_head, 1, self.head_dim)
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(batch_size, self.n_embd)
        x = x + attn.out_proj(attn_out).view(batch_size, 1, self.n_embd)
        norm_x = layer.ln_2(x.view(batch_size, self.n_embd)).view(
            batch_size, 1, self.n_embd
        )
        return x + layer.ff(norm_x)

    def forward_decode(self, idx, start_pos: int):
        batch_size, seq_len = idx.shape
        if seq_len != 1:
            raise ValueError("forward_decode expects one token")
        x = self.model._embed(idx, start_pos=start_pos)
        for layer in self.model.layers:
            x = self._run_layer_decode(layer, x)
        self.context_len += 1
        x = self.model.ln(x.view(batch_size, self.n_embd)).view(batch_size, 1, self.n_embd)
        return self.model.lm_head(x.view(batch_size, self.n_embd)).view(
            batch_size, 1, self.n_vocab
        )


def _make_block_manager(batch_size: int, context_len: int, model: PagedDecoderLM, block_size: int) -> BlockManager:
    blocks_per_seq = math.ceil(context_len / block_size)
    return BlockManager(
        num_blocks=batch_size * blocks_per_seq,
        block_size=block_size,
        n_head=model.layers[0].attention.n_head,
        head_dim=model.layers[0].attention.head_dim,
        num_layers=model.n_layers,
        allocate_host_cache=False,
    )


def _rewind_one_paged_decode(block_manager: BlockManager, seq_ids: Sequence[int]) -> None:
    for seq_id in seq_ids:
        token_index = block_manager.context_lens[seq_id] - 1
        block_id, _ = block_manager.get_physical_location(seq_id, token_index)
        block = block_manager.blocks[block_id]
        block.num_filled -= 1
        block_manager.context_lens[seq_id] -= 1
        if block.num_filled == 0 and block_manager.block_tables[seq_id].block_ids[-1] == block_id:
            block.ref_count = 0
            block_manager.block_tables[seq_id].block_ids.pop()
            if block_id not in block_manager.free_block_ids:
                block_manager.free_block_ids.append(block_id)
                block_manager.free_block_ids.sort()


def _time_cuda(fn, warmup_iters: int, timed_iters: int) -> Dict[str, float]:
    for _ in range(warmup_iters):
        fn()
    cuda.synchronize()
    samples = []
    for _ in range(timed_iters):
        start = time.perf_counter()
        fn()
        cuda.synchronize()
        samples.append(time.perf_counter() - start)
    arr = np.array(samples, dtype=np.float64)
    return {
        "median_ms": float(np.median(arr) * 1000.0),
        "mean_ms": float(np.mean(arr) * 1000.0),
        "min_ms": float(np.min(arr) * 1000.0),
        "p95_ms": float(np.percentile(arr, 95) * 1000.0),
    }


def _stats_row(common, implementation, scope, stats, allocated_kv_bytes, live_kv_bytes, working_kv_bytes, kv_residency, output_error):
    median_s = stats["median_ms"] / 1000.0
    batch_size = int(common["batch_size"])
    row = dict(common)
    row.update(
        {
            "implementation": implementation,
            "scope": scope,
            "median_ms": round(stats["median_ms"], 6),
            "mean_ms": round(stats["mean_ms"], 6),
            "min_ms": round(stats["min_ms"], 6),
            "p95_ms": round(stats["p95_ms"], 6),
            "query_tokens_per_s": round(batch_size / median_s, 3) if median_s > 0 else 0.0,
            "allocated_kv_bytes": int(allocated_kv_bytes),
            "live_kv_bytes": int(live_kv_bytes),
            "working_kv_bytes": int(working_kv_bytes),
            "kv_residency": kv_residency,
            "setup_kv_copy": "none" if implementation != "paged_attention" else "device_to_device",
            "timed_h2d_kv_bytes": 0,
            "timed_h2d_query_bytes": 0,
            "timed_d2h_output_bytes": 0,
            "output_max_abs_error": round(float(output_error), 8),
            "output_mean_abs_error": round(float(output_error), 8),
        }
    )
    return row


def _benchmark_one(
    model: PagedDecoderLM,
    no_kv: NoKVDecoder,
    contiguous: GpuContiguousKVDecoder,
    prompt: Tensor,
    decode_token: Tensor,
    full_input: Tensor,
    args: argparse.Namespace,
    metadata: Dict[str, object],
) -> List[Dict[str, object]]:
    batch_size, prompt_len = prompt.shape
    context_len = prompt_len + 1
    seq_ids = list(range(batch_size))

    contiguous.reset()
    contiguous.forward_prefill(prompt)
    paged_manager = _make_block_manager(batch_size, context_len, model, args.block_size)
    model.forward_prefill(prompt, paged_manager, seq_ids)

    def baseline_once():
        return no_kv.forward_last(full_input)

    def contiguous_once():
        contiguous.context_len = prompt_len
        return contiguous.forward_decode(decode_token, start_pos=prompt_len)

    def paged_once():
        out = model.forward_decode(decode_token, paged_manager, seq_ids, start_pos=prompt_len)
        _rewind_one_paged_decode(paged_manager, seq_ids)
        return out

    baseline_logits = baseline_once()
    contiguous_logits = contiguous_once()
    contiguous.context_len = prompt_len
    paged_logits = paged_once()
    cuda.synchronize()

    baseline_last = baseline_logits.to_numpy()
    contiguous_np = contiguous_logits.to_numpy()
    paged_np = paged_logits.to_numpy()
    baseline_error = float(np.max(np.abs(baseline_last - contiguous_np)))
    paged_error = float(np.max(np.abs(paged_np - contiguous_np)))

    baseline_stats = _time_cuda(baseline_once, args.warmup_iters, args.timed_iters)
    contiguous_stats = _time_cuda(contiguous_once, args.warmup_iters, args.timed_iters)
    contiguous.context_len = prompt_len
    paged_stats = _time_cuda(paged_once, args.warmup_iters, args.timed_iters)

    n_head = model.layers[0].attention.n_head
    head_dim = model.layers[0].attention.head_dim
    bytes_per_token = model.n_layers * n_head * head_dim * 2 * DTYPE_SIZE
    live_kv_bytes = batch_size * context_len * bytes_per_token
    static_context_len = max(int(metadata["static_context_len"]), context_len)
    static_contiguous_kv_bytes = batch_size * static_context_len * bytes_per_token
    blocks_per_seq = math.ceil(context_len / args.block_size)
    paged_allocated_tokens = batch_size * blocks_per_seq * args.block_size
    paged_allocated_kv_bytes = paged_allocated_tokens * bytes_per_token

    common = {
        "source": "minitorch_gpt2_real_dataset",
        "runner": "minitorch_full_model",
        "model_name": metadata["model_name"],
        "dataset_name": metadata["dataset_name"],
        "dataset_config": metadata["dataset_config"],
        "dataset_split": metadata["dataset_split"],
        "layer_id": "all",
        "batch_size": batch_size,
        "prompt_tokens": prompt_len,
        "context_tokens": context_len,
        "block_size": args.block_size,
        "n_layers": model.n_layers,
        "n_head": n_head,
        "head_dim": head_dim,
        "blocks_per_seq": blocks_per_seq,
        "num_blocks": batch_size * blocks_per_seq,
        "static_context_len": static_context_len,
        "memory_scope": "full_model_all_layers",
        "timed_iters": args.timed_iters,
        "first_token_id": metadata["first_token_id"],
        "next_token_id": metadata["next_token_id"],
        "paged_internal_fragmentation": round(
            1.0 - live_kv_bytes / paged_allocated_kv_bytes,
            6,
        ),
        "paged_savings_vs_static_contiguous": round(
            1.0 - paged_allocated_kv_bytes / static_contiguous_kv_bytes,
            6,
        ),
    }
    return [
        _stats_row(
            common,
            "baseline_no_kv",
            "full_model_recompute_context_next_token",
            baseline_stats,
            0,
            live_kv_bytes,
            live_kv_bytes,
            "no_persistent_kv",
            baseline_error,
        ),
        _stats_row(
            common,
            "contiguous_kv",
            "full_model_single_token_decode",
            contiguous_stats,
            static_contiguous_kv_bytes,
            live_kv_bytes,
            live_kv_bytes,
            "gpu_contiguous_kv",
            0.0,
        ),
        _stats_row(
            common,
            "paged_attention",
            "full_model_single_token_decode",
            paged_stats,
            paged_allocated_kv_bytes,
            live_kv_bytes,
            live_kv_bytes,
            "gpu_paged_runtime_no_host_kv_cache",
            paged_error,
        ),
    ]


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not _cuda_available():
        raise SystemExit("CUDA is required for the MiniTorch GPU-resident KV benchmark.")
    if args.block_size <= 0 or args.block_size > 64:
        raise ValueError("--block-size must be in [1, 64] for the current kernel.")
    if args.timed_iters <= 0 or args.warmup_iters < 0:
        raise ValueError("--timed-iters must be positive and --warmup-iters non-negative.")

    np.random.seed(args.seed)
    torch, load_dataset, AutoModelForCausalLM, AutoTokenizer = _import_hf()
    torch.manual_seed(args.seed)
    backend = _make_backend()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    hf_model.eval()

    n_positions = int(getattr(hf_model.config, "n_positions", getattr(hf_model.config, "n_ctx", 1024)))
    max_context_len = max(args.seq_lengths) + 1
    if max_context_len > n_positions:
        raise ValueError(f"Requested context {max_context_len} exceeds model position limit {n_positions}.")

    tokens_needed = sum(batch * (seq_len + 1) for batch in args.batch_sizes for seq_len in args.seq_lengths)
    texts = _load_real_texts(args, load_dataset)
    token_ids = _token_stream_from_texts(tokenizer, texts, tokens_needed)

    print(
        f"Loaded {args.model_name} weights into MiniTorch CudaOps; "
        f"using {len(token_ids)} real tokens from "
        f"{args.dataset_name}/{args.dataset_config}:{args.dataset_split}."
    )
    model = _make_model_from_hf(hf_model, backend, args.block_size)
    no_kv = NoKVDecoder(model)
    static_context_len = args.static_context_len or n_positions
    contiguous = GpuContiguousKVDecoder(
        model,
        max_batch_size=max(args.batch_sizes),
        max_seq_len=static_context_len,
    )

    rows: List[Dict[str, object]] = []
    offset = 0
    for batch_size in args.batch_sizes:
        for prompt_len in args.seq_lengths:
            needed = batch_size * (prompt_len + 1)
            chunk = np.array(token_ids[offset : offset + needed], dtype=np.float32)
            offset += needed
            chunk = chunk.reshape(batch_size, prompt_len + 1)
            prompt_np = chunk[:, :prompt_len]
            decode_np = chunk[:, prompt_len : prompt_len + 1]
            full_np = chunk

            prompt = _as_minitorch(prompt_np, backend)
            decode_token = _as_minitorch(decode_np, backend)
            full_input = _as_minitorch(full_np, backend)
            metadata = {
                "model_name": args.model_name,
                "dataset_name": args.dataset_name,
                "dataset_config": args.dataset_config or "",
                "dataset_split": args.dataset_split,
                "static_context_len": static_context_len,
                "first_token_id": int(prompt_np[0, 0]),
                "next_token_id": int(decode_np[0, 0]),
            }
            new_rows = _benchmark_one(
                model,
                no_kv,
                contiguous,
                prompt,
                decode_token,
                full_input,
                args,
                metadata,
            )
            rows.extend(new_rows)
            by_impl = {row["implementation"]: row for row in new_rows}
            print(
                f"bs={batch_size} prompt={prompt_len}: "
                f"baseline={by_impl['baseline_no_kv']['median_ms']} ms, "
                f"contiguous={by_impl['contiguous_kv']['median_ms']} ms, "
                f"paged={by_impl['paged_attention']['median_ms']} ms, "
                f"paged_error={by_impl['paged_attention']['output_max_abs_error']:.3e}, "
                f"timed_h2d_kv=0 bytes"
            )

    output_path = Path(args.output_dir) / args.output_csv
    _write_csv(output_path, rows)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
