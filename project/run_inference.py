"""
Run inference using the PagedAttention-enabled decoder LM.

Usage:
    python project/run_inference.py \
        --n-vocab 10000 --n-embd 256 --n-head 8 --n-layers 4 \
        --block-size 16 --max-new-tokens 64
"""

import argparse
import sys
import time
import numpy as np

sys.path.insert(0, ".")


def parse_args():
    parser = argparse.ArgumentParser(description="PagedAttention Inference")
    parser.add_argument("--n-vocab", type=int, default=10000)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-positions", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-kv-blocks", type=int, default=256,
                        help="Total number of KV cache blocks to allocate")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-len", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--backend", type=str, default="cpu",
                        choices=["cpu", "cuda"])
    parser.add_argument("--decode-backend", type=str, default="ref",
                        choices=["ref", "cuda"])
    parser.add_argument("--compare-to-ref", action="store_true",
                        help="When using CUDA decode, also run the reference "
                             "paged attention and assert the outputs match.")
    parser.add_argument("--compare-tolerance", type=float, default=1e-4)
    parser.add_argument("--check-correctness", action="store_true",
                        help="Compare paged decode logits against full-sequence "
                             "recomputation on the same model.")
    return parser.parse_args()


def _sample(logits_np: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        return np.argmax(logits_np, axis=-1)
    scaled = logits_np / temperature
    scaled = scaled - np.max(scaled, axis=-1, keepdims=True)
    probs = np.exp(scaled) / np.sum(np.exp(scaled), axis=-1, keepdims=True)
    return np.array([np.random.choice(len(p), p=p) for p in probs])


def _reference_last_logits(model, minitorch, full_tokens_np, num_kv_blocks, block_size,
                           n_head, head_dim, seq_ids):
    from minitorch.block_manager import BlockManager

    ref_block_manager = BlockManager(
        num_blocks=num_kv_blocks,
        block_size=block_size,
        n_head=n_head,
        head_dim=head_dim,
        num_layers=model.n_layers,
    )
    ref_input = minitorch.tensor_from_numpy(
        full_tokens_np.astype(np.float32),
        backend=model.backend,
    )
    ref_logits = model.forward_prefill(ref_input, ref_block_manager, seq_ids)
    return ref_logits.to_numpy()[:, -1, :]


def main():
    args = parse_args()

    import minitorch
    from minitorch.transformer import PagedDecoderLM
    from minitorch.block_manager import BlockManager

    # Select backend
    if args.backend == "cuda":
        backend = minitorch.TensorBackend(minitorch.CudaKernelOps)
    else:
        backend = minitorch.TensorBackend(minitorch.FastOps)

    print(f"Backend: {args.backend}")
    print(f"Model config: vocab={args.n_vocab}, embd={args.n_embd}, "
          f"head={args.n_head}, layers={args.n_layers}")
    print(f"Block config: block_size={args.block_size}, "
          f"num_blocks={args.num_kv_blocks}")

    # Initialize model
    model = PagedDecoderLM(
        n_vocab=args.n_vocab,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_positions=args.n_positions,
        n_layers=args.n_layers,
        block_size=args.block_size,
        backend=backend,
        decode_backend=args.decode_backend,
        compare_to_ref=args.compare_to_ref,
        compare_tolerance=args.compare_tolerance,
    )

    # Initialize block manager
    head_dim = args.n_embd // args.n_head
    block_manager = BlockManager(
        num_blocks=args.num_kv_blocks,
        block_size=args.block_size,
        n_head=args.n_head,
        head_dim=head_dim,
        num_layers=args.n_layers,
    )

    # Create synthetic prompt
    prompt = np.random.randint(
        0, args.n_vocab,
        size=(args.batch_size, args.prompt_len),
    ).astype(np.float32)
    prompt_tensor = minitorch.tensor_from_numpy(prompt, backend=backend)

    seq_ids = list(range(args.batch_size))

    # Run generation
    print(f"\nGenerating {args.max_new_tokens} tokens for {args.batch_size} "
          f"sequence(s) with prompt length {args.prompt_len}...")

    model.eval()
    generated = prompt.copy()
    correctness_max_abs = 0.0
    correctness_mean_abs = []
    token_match = []

    start = time.perf_counter()
    logits = model.forward_prefill(prompt_tensor, block_manager, seq_ids)
    last_logits_np = logits.to_numpy()[:, -1, :]

    if args.check_correctness:
        ref_logits_np = _reference_last_logits(
            model, minitorch, generated,
            args.num_kv_blocks, args.block_size,
            args.n_head, head_dim, seq_ids,
        )
        diff = last_logits_np - ref_logits_np
        correctness_max_abs = max(correctness_max_abs, float(np.max(np.abs(diff))))
        correctness_mean_abs.append(float(np.mean(np.abs(diff))))
        token_match.append(bool(np.array_equal(
            np.argmax(last_logits_np, axis=-1),
            np.argmax(ref_logits_np, axis=-1),
        )))

    next_tokens = _sample(last_logits_np, args.temperature)
    generated = np.concatenate(
        [generated, next_tokens.reshape(args.batch_size, 1).astype(np.float32)],
        axis=1,
    )

    for step in range(1, args.max_new_tokens):
        token_input = minitorch.tensor_from_numpy(
            next_tokens.reshape(args.batch_size, 1).astype(np.float32),
            backend=backend,
        )
        start_pos = args.prompt_len + step - 1
        logits = model.forward_decode(
            token_input,
            block_manager,
            seq_ids,
            start_pos=start_pos,
        )
        logits_np = logits.to_numpy()[:, 0, :]

        if args.check_correctness:
            ref_logits_np = _reference_last_logits(
                model, minitorch, generated,
                args.num_kv_blocks, args.block_size,
                args.n_head, head_dim, seq_ids,
            )
            diff = logits_np - ref_logits_np
            correctness_max_abs = max(correctness_max_abs, float(np.max(np.abs(diff))))
            correctness_mean_abs.append(float(np.mean(np.abs(diff))))
            token_match.append(bool(np.array_equal(
                np.argmax(logits_np, axis=-1),
                np.argmax(ref_logits_np, axis=-1),
            )))

        next_tokens = _sample(logits_np, args.temperature)
        generated = np.concatenate(
            [generated, next_tokens.reshape(args.batch_size, 1).astype(np.float32)],
            axis=1,
        )

    for seq_id in seq_ids:
        block_manager.free_sequence(seq_id)

    elapsed = time.perf_counter() - start

    total_tokens = args.batch_size * args.max_new_tokens
    print(f"Generated {total_tokens} tokens in {elapsed:.3f}s "
          f"({total_tokens / elapsed:.1f} tok/s)")
    print(f"Block manager: {block_manager}")

    if args.check_correctness:
        mean_abs = float(np.mean(correctness_mean_abs)) if correctness_mean_abs else 0.0
        token_match_rate = float(np.mean(token_match)) if token_match else 1.0
        print("Correctness vs full recomputation:")
        print(f"  max_abs_error={correctness_max_abs:.6f}")
        print(f"  mean_abs_error={mean_abs:.6f}")
        print(f"  argmax_match_rate={token_match_rate:.3f}")


if __name__ == "__main__":
    main()
