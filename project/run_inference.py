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
    return parser.parse_args()


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
    )

    # Initialize block manager
    head_dim = args.n_embd // args.n_head
    block_manager = BlockManager(
        num_blocks=args.num_kv_blocks,
        block_size=args.block_size,
        n_head=args.n_head,
        head_dim=head_dim,
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

    start = time.perf_counter()
    output = PagedDecoderLM.generate(
        model,
        prompt_tensor,
        max_new_tokens=args.max_new_tokens,
        block_manager=block_manager,
        seq_ids=seq_ids,
        temperature=args.temperature,
    )
    elapsed = time.perf_counter() - start

    total_tokens = args.batch_size * args.max_new_tokens
    print(f"Generated {total_tokens} tokens in {elapsed:.3f}s "
          f"({total_tokens / elapsed:.1f} tok/s)")
    print(f"Block manager: {block_manager}")


if __name__ == "__main__":
    main()
