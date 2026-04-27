"""
Benchmark a HuggingFace GPT-2-family checkpoint through MiniTorch PagedAttention.

This is the first "real model" path for the project: it loads pretrained GPT-2
weights into the existing PagedDecoderLM architecture, tokenizes natural-language
prompts, runs prefill/decode with the BlockManager-backed KV cache, and writes a
CSV with latency and KV-cache accounting.

Install optional dependencies first:

    pip install -r requirements.gpt2.txt

Example smoke run:

    python project/run_gpt2_paged_benchmark.py \
        --model-name sshleifer/tiny-gpt2 \
        --max-prompts 2 \
        --max-prompt-tokens 16 \
        --max-new-tokens 4

Example GPT-2 run on WikiText:

    python project/run_gpt2_paged_benchmark.py \
        --model-name gpt2 \
        --dataset-name wikitext \
        --dataset-config wikitext-2-raw-v1 \
        --dataset-split test \
        --max-prompts 8 \
        --max-prompt-tokens 64 \
        --max-new-tokens 16
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, ".")

import minitorch
from minitorch.block_manager import BlockManager, CACHE_DTYPE, DEFAULT_BLOCK_SIZE
from minitorch.tensor import tensor_from_numpy
from minitorch.transformer import PagedDecoderLM

datatype = np.float32


BUILTIN_PROMPTS = [
    "PagedAttention reduces memory waste by storing key and value tensors in fixed-size blocks.",
    "In a language model serving system, many requests decode one token at a time after prefill.",
    "The project compares contiguous KV cache allocation with a block table based paged cache.",
]


@dataclass
class LoadedGPT2:
    model: PagedDecoderLM
    tokenizer: object
    n_vocab: int
    n_embd: int
    n_head: int
    n_layers: int
    n_positions: int

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MiniTorch PagedAttention inference with GPT-2 weights."
    )
    parser.add_argument("--model-name", type=str, default="gpt2")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--decode-backend", choices=["ref", "cuda"], default="ref")
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument(
        "--num-kv-blocks",
        type=int,
        default=0,
        help="Physical KV blocks. Use 0 to allocate just enough blocks per prompt.",
    )
    parser.add_argument("--max-prompts", type=int, default=4)
    parser.add_argument("--min-prompt-tokens", type=int, default=4)
    parser.add_argument("--max-prompt-tokens", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 uses greedy argmax; values >0 sample from softmax.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompts-file", type=str, default=None)
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--dataset-split", type=str, default="test")
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument(
        "--contiguous-context-tokens",
        type=int,
        default=0,
        help="Context length for contiguous KV memory estimate. 0 uses model n_positions.",
    )
    parser.add_argument("--output-dir", type=str, default="benchmarks/results_gpt2")
    parser.add_argument("--output-csv", type=str, default="gpt2_paged_benchmark.csv")
    parser.add_argument("--print-generations", action="store_true")
    return parser.parse_args()


def _import_hf_dependencies():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing GPT-2 benchmark dependencies. Install them with: "
            "pip install -r requirements.gpt2.txt"
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def _create_backend(backend_name: str):
    if backend_name == "cuda":
        return minitorch.TensorBackend(minitorch.CudaKernelOps)
    return minitorch.TensorBackend(minitorch.FastOps)


def _as_numpy(tensor) -> np.ndarray:
    return tensor.detach().cpu().float().numpy()


def _update_parameter(parameter, array: np.ndarray, backend) -> None:
    value = np.ascontiguousarray(array.astype(np.float32, copy=False))
    parameter.update(tensor_from_numpy(value, backend=backend))


def _conv1d_weight_to_linear(
    weight: np.ndarray,
    in_size: int,
    out_size: int,
    name: str,
) -> np.ndarray:
    if weight.shape == (in_size, out_size):
        return weight
    if weight.shape == (out_size, in_size):
        return weight.T
    raise ValueError(
        f"Unexpected {name} weight shape {weight.shape}; expected "
        f"({in_size}, {out_size}) or ({out_size}, {in_size})"
    )


def _split_qkv_weight(weight: np.ndarray, n_embd: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if weight.shape == (n_embd, 3 * n_embd):
        return tuple(np.split(weight, 3, axis=1))
    if weight.shape == (3 * n_embd, n_embd):
        return tuple(chunk.T for chunk in np.split(weight, 3, axis=0))
    raise ValueError(
        f"Unexpected GPT-2 c_attn weight shape {weight.shape}; expected "
        f"({n_embd}, {3 * n_embd}) or ({3 * n_embd}, {n_embd})"
    )


def _get_config_int(config, *names: str) -> int:
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    raise ValueError(f"Could not find any config attribute from {names}")


def load_gpt2_into_paged_model(
    model_name: str,
    backend,
    block_size: int,
    decode_backend: str,
    revision: Optional[str] = None,
    local_files_only: bool = False,
) -> LoadedGPT2:
    torch, AutoModelForCausalLM, AutoTokenizer = _import_hf_dependencies()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            local_files_only=local_files_only,
        )
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            local_files_only=local_files_only,
        )
    except OSError as exc:
        offline_hint = (
            " The checkpoint was not found in the local HuggingFace cache; "
            "rerun without --local-files-only to download it."
            if local_files_only
            else " Check network access or pre-download the checkpoint."
        )
        raise SystemExit(f"Could not load {model_name!r}.{offline_hint}") from exc
    hf_model.eval()

    config = hf_model.config
    n_vocab = _get_config_int(config, "vocab_size")
    n_embd = _get_config_int(config, "n_embd", "hidden_size")
    n_head = _get_config_int(config, "n_head", "num_attention_heads")
    n_layers = _get_config_int(config, "n_layer", "num_hidden_layers")
    n_positions = _get_config_int(config, "n_positions", "n_ctx", "max_position_embeddings")
    ln_eps = float(getattr(config, "layer_norm_epsilon", 1e-5))

    if n_embd % n_head != 0:
        raise ValueError(f"n_embd={n_embd} must be divisible by n_head={n_head}")
    if not hasattr(hf_model, "transformer") or not hasattr(hf_model.transformer, "h"):
        raise ValueError(
            "This script expects a GPT-2-style HuggingFace model with "
            "transformer.h blocks. Try --model-name gpt2 or distilgpt2."
        )

    paged_model = PagedDecoderLM(
        n_vocab=n_vocab,
        n_embd=n_embd,
        n_head=n_head,
        n_positions=n_positions,
        n_layers=n_layers,
        block_size=block_size,
        p_dropout=0.0,
        ln_eps=ln_eps,
        backend=backend,
        decode_backend=decode_backend,
    )

    transformer = hf_model.transformer
    with torch.no_grad():
        _update_parameter(paged_model.token_embeddings.weights, _as_numpy(transformer.wte.weight), backend)
        _update_parameter(paged_model.position_embeddings.weights, _as_numpy(transformer.wpe.weight), backend)

        for layer_id, hf_layer in enumerate(transformer.h):
            layer = paged_model.layers[layer_id]
            _update_parameter(layer.ln_1.weights, _as_numpy(hf_layer.ln_1.weight), backend)
            _update_parameter(layer.ln_1.bias, _as_numpy(hf_layer.ln_1.bias), backend)
            _update_parameter(layer.ln_2.weights, _as_numpy(hf_layer.ln_2.weight), backend)
            _update_parameter(layer.ln_2.bias, _as_numpy(hf_layer.ln_2.bias), backend)

            c_attn_weight = _as_numpy(hf_layer.attn.c_attn.weight)
            c_attn_bias = _as_numpy(hf_layer.attn.c_attn.bias)
            q_weight, k_weight, v_weight = _split_qkv_weight(c_attn_weight, n_embd)
            q_bias, k_bias, v_bias = np.split(c_attn_bias, 3)
            _update_parameter(layer.attention.q_proj.weights, q_weight, backend)
            _update_parameter(layer.attention.q_proj.bias, q_bias, backend)
            _update_parameter(layer.attention.k_proj.weights, k_weight, backend)
            _update_parameter(layer.attention.k_proj.bias, k_bias, backend)
            _update_parameter(layer.attention.v_proj.weights, v_weight, backend)
            _update_parameter(layer.attention.v_proj.bias, v_bias, backend)

            attn_out_weight = _conv1d_weight_to_linear(
                _as_numpy(hf_layer.attn.c_proj.weight), n_embd, n_embd, "attn.c_proj"
            )
            _update_parameter(layer.attention.out_proj.weights, attn_out_weight, backend)
            _update_parameter(layer.attention.out_proj.bias, _as_numpy(hf_layer.attn.c_proj.bias), backend)

            mlp_hidden = 4 * n_embd
            fc_weight = _conv1d_weight_to_linear(
                _as_numpy(hf_layer.mlp.c_fc.weight), n_embd, mlp_hidden, "mlp.c_fc"
            )
            proj_weight = _conv1d_weight_to_linear(
                _as_numpy(hf_layer.mlp.c_proj.weight), mlp_hidden, n_embd, "mlp.c_proj"
            )
            _update_parameter(layer.ff.linear_in.weights, fc_weight, backend)
            _update_parameter(layer.ff.linear_in.bias, _as_numpy(hf_layer.mlp.c_fc.bias), backend)
            _update_parameter(layer.ff.linear_out.weights, proj_weight, backend)
            _update_parameter(layer.ff.linear_out.bias, _as_numpy(hf_layer.mlp.c_proj.bias), backend)

        _update_parameter(paged_model.ln.weights, _as_numpy(transformer.ln_f.weight), backend)
        _update_parameter(paged_model.ln.bias, _as_numpy(transformer.ln_f.bias), backend)

        lm_head_weight = _as_numpy(hf_model.lm_head.weight)
        if lm_head_weight.shape == (n_vocab, n_embd):
            lm_head_weight = lm_head_weight.T
        elif lm_head_weight.shape != (n_embd, n_vocab):
            raise ValueError(
                f"Unexpected lm_head weight shape {lm_head_weight.shape}; expected "
                f"({n_vocab}, {n_embd}) or ({n_embd}, {n_vocab})"
            )
        _update_parameter(paged_model.lm_head.weights, lm_head_weight, backend)

    paged_model.eval()
    return LoadedGPT2(
        model=paged_model,
        tokenizer=tokenizer,
        n_vocab=n_vocab,
        n_embd=n_embd,
        n_head=n_head,
        n_layers=n_layers,
        n_positions=n_positions,
    )


def _read_prompt_file(path: str) -> List[str]:
    prompts = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                prompts.append(text)
    return prompts


def _load_dataset_prompts(args: argparse.Namespace) -> List[str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "--dataset-name requires the datasets package. Install optional "
            "dependencies with: pip install -r requirements.gpt2.txt"
        ) from exc

    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.dataset_split,
    )
    prompts = []
    for row in dataset:
        text = str(row.get(args.text_column, "")).strip()
        if text:
            prompts.append(text)
        if len(prompts) >= args.max_prompts * 8:
            break
    return prompts


def load_prompt_texts(args: argparse.Namespace) -> List[str]:
    if args.prompts_file:
        return _read_prompt_file(args.prompts_file)
    if args.dataset_name:
        return _load_dataset_prompts(args)
    return list(BUILTIN_PROMPTS)


def encode_prompts(
    tokenizer,
    texts: Iterable[str],
    max_prompts: int,
    min_prompt_tokens: int,
    max_prompt_tokens: int,
) -> List[Tuple[str, List[int]]]:
    encoded_prompts = []
    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if max_prompt_tokens > 0:
            token_ids = token_ids[:max_prompt_tokens]
        if len(token_ids) < min_prompt_tokens:
            continue
        encoded_prompts.append((text, token_ids))
        if len(encoded_prompts) >= max_prompts:
            break
    if not encoded_prompts:
        raise ValueError(
            "No usable prompts after tokenization. Lower --min-prompt-tokens "
            "or provide longer prompts."
        )
    return encoded_prompts


def _sample_next_token(logits: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    logits = logits.astype(np.float64, copy=False)
    if temperature <= 0:
        return int(np.argmax(logits))
    scaled = logits / temperature
    scaled = scaled - np.max(scaled)
    probs = np.exp(scaled)
    probs = probs / np.sum(probs)
    return int(rng.choice(len(probs), p=probs))


def _auto_num_blocks(prompt_tokens: int, max_new_tokens: int, block_size: int) -> int:
    context_tokens = max(prompt_tokens + max_new_tokens, 1)
    return max(1, math.ceil(context_tokens / block_size) + 1)


def _kv_metrics(block_manager: BlockManager) -> dict:
    dtype_size = np.dtype(block_manager.cache_dtype).itemsize
    filled_slots = sum(block.num_filled for block in block_manager.blocks.values())
    used_blocks = block_manager.num_used_blocks
    bytes_per_token = (
        block_manager.n_head
        * block_manager.head_dim
        * block_manager.num_layers
        * 2
        * dtype_size
    )
    allocated_slots = used_blocks * block_manager.block_size
    return {
        "paged_used_blocks": used_blocks,
        "paged_allocated_slots": allocated_slots,
        "paged_filled_slots": filled_slots,
        "paged_internal_fragmentation": (
            (allocated_slots - filled_slots) / allocated_slots if allocated_slots else 0.0
        ),
        "paged_live_kv_bytes": filled_slots * bytes_per_token,
        "paged_allocated_kv_bytes": allocated_slots * bytes_per_token,
        "paged_pool_kv_bytes": block_manager.num_blocks * block_manager.block_size * bytes_per_token,
    }


def _contiguous_kv_bytes(
    context_tokens: int,
    n_head: int,
    head_dim: int,
    n_layers: int,
) -> int:
    dtype_size = np.dtype(CACHE_DTYPE).itemsize
    return context_tokens * n_head * head_dim * n_layers * 2 * dtype_size


def run_one_prompt(
    loaded: LoadedGPT2,
    prompt_id: int,
    prompt_text: str,
    prompt_tokens: Sequence[int],
    args: argparse.Namespace,
    backend,
    rng: np.random.Generator,
) -> dict:
    seq_ids = [prompt_id + 1]
    token_count = len(prompt_tokens)
    num_blocks = args.num_kv_blocks or _auto_num_blocks(
        token_count, args.max_new_tokens, args.block_size
    )
    block_manager = BlockManager(
        num_blocks=num_blocks,
        block_size=args.block_size,
        n_head=loaded.n_head,
        head_dim=loaded.head_dim,
        num_layers=loaded.n_layers,
    )
    prompt_array = np.array(prompt_tokens, dtype=datatype).reshape(1, token_count)
    prompt_tensor = tensor_from_numpy(prompt_array, backend=backend)
    generated_tokens = list(prompt_tokens)
    decode_forward_tokens = 0
    decode_s = 0.0

    try:
        prefill_start = time.perf_counter()
        logits = loaded.model.forward_prefill(prompt_tensor, block_manager, seq_ids)
        prefill_s = time.perf_counter() - prefill_start
        last_logits = logits.to_numpy()[0, -1, :]

        for step in range(args.max_new_tokens):
            next_token = _sample_next_token(last_logits, args.temperature, rng)
            generated_tokens.append(next_token)
            if step == args.max_new_tokens - 1:
                break

            token_tensor = tensor_from_numpy(
                np.array([[next_token]], dtype=datatype),
                backend=backend,
            )
            decode_start = time.perf_counter()
            logits = loaded.model.forward_decode(
                token_tensor,
                block_manager,
                seq_ids,
                start_pos=token_count + step,
            )
            decode_s += time.perf_counter() - decode_start
            decode_forward_tokens += 1
            last_logits = logits.to_numpy()[0, 0, :]

        kv_metrics = _kv_metrics(block_manager)
        total_s = prefill_s + decode_s
        output_text = loaded.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        contiguous_context_tokens = args.contiguous_context_tokens or loaded.n_positions
        contiguous_bytes = _contiguous_kv_bytes(
            contiguous_context_tokens,
            loaded.n_head,
            loaded.head_dim,
            loaded.n_layers,
        )
        row = {
            "prompt_id": prompt_id,
            "model_name": args.model_name,
            "backend": args.backend,
            "decode_backend": args.decode_backend,
            "prompt_tokens": token_count,
            "max_new_tokens": args.max_new_tokens,
            "generated_tokens": max(len(generated_tokens) - token_count, 0),
            "decode_forward_tokens": decode_forward_tokens,
            "prefill_s": prefill_s,
            "decode_s": decode_s,
            "total_s": total_s,
            "end_to_end_tokens_per_s": (
                (token_count + args.max_new_tokens) / total_s if total_s > 0 else 0.0
            ),
            "decode_forward_tokens_per_s": (
                decode_forward_tokens / decode_s if decode_s > 0 else 0.0
            ),
            "block_size": args.block_size,
            "num_kv_blocks": num_blocks,
            "contiguous_context_tokens": contiguous_context_tokens,
            "contiguous_kv_bytes_estimate": contiguous_bytes,
            "paged_vs_contiguous_allocated_savings": (
                1.0 - kv_metrics["paged_allocated_kv_bytes"] / contiguous_bytes
                if contiguous_bytes > 0
                else 0.0
            ),
            "prompt_preview": prompt_text[:160].replace("\n", " "),
            "generated_text_preview": output_text[:240].replace("\n", " "),
        }
        row.update(kv_metrics)
        return row
    finally:
        for seq_id in seq_ids:
            if seq_id in block_manager.block_tables:
                block_manager.free_sequence(seq_id)
        loaded.model.close_decode_runtime()


def write_rows(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    backend = _create_backend(args.backend)

    loaded = load_gpt2_into_paged_model(
        model_name=args.model_name,
        backend=backend,
        block_size=args.block_size,
        decode_backend=args.decode_backend,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )

    prompt_texts = load_prompt_texts(args)
    max_decode_positions = max(args.max_new_tokens - 1, 0)
    max_position_safe_prompt_tokens = loaded.n_positions - max_decode_positions
    if max_position_safe_prompt_tokens <= 0:
        raise ValueError(
            f"--max-new-tokens={args.max_new_tokens} exceeds model position "
            f"capacity {loaded.n_positions}"
        )
    max_prompt_tokens = args.max_prompt_tokens
    if max_prompt_tokens <= 0:
        max_prompt_tokens = max_position_safe_prompt_tokens
    else:
        max_prompt_tokens = min(max_prompt_tokens, max_position_safe_prompt_tokens)

    encoded_prompts = encode_prompts(
        loaded.tokenizer,
        prompt_texts,
        max_prompts=args.max_prompts,
        min_prompt_tokens=args.min_prompt_tokens,
        max_prompt_tokens=max_prompt_tokens,
    )

    print(
        f"Loaded {args.model_name}: vocab={loaded.n_vocab}, embd={loaded.n_embd}, "
        f"heads={loaded.n_head}, layers={loaded.n_layers}, positions={loaded.n_positions}"
    )
    print(f"Running {len(encoded_prompts)} prompt(s) with MiniTorch PagedAttention...")

    rows = []
    for prompt_id, (prompt_text, token_ids) in enumerate(encoded_prompts):
        row = run_one_prompt(
            loaded=loaded,
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            prompt_tokens=token_ids,
            args=args,
            backend=backend,
            rng=rng,
        )
        rows.append(row)
        print(
            f"prompt={prompt_id} tokens={row['prompt_tokens']} "
            f"prefill={row['prefill_s']:.4f}s decode={row['decode_s']:.4f}s "
            f"paged_alloc={row['paged_allocated_kv_bytes'] / (1024 ** 2):.2f} MiB"
        )
        if args.print_generations:
            print(row["generated_text_preview"])

    output_path = Path(args.output_dir) / args.output_csv
    write_rows(output_path, rows)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()