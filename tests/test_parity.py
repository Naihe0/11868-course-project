"""End-to-end parity tests for PagedMultiHeadAttention.

Verifies that the paged attention prefill + decode path produces the same
output as a pure-NumPy multi-head attention reference when given identical
projection weights and identical input.

This is the key correctness guarantee for the paper: the paging machinery
does not change the math, only where K/V vectors live.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

sys.path.insert(0, ".")

from minitorch.block_manager import BlockManager
from minitorch.paged_attention import PagedMultiHeadAttention
from minitorch.tensor_functions import tensor_from_numpy
from minitorch.tensor_ops import TensorBackend
from minitorch.fast_ops import FastOps

_test_backend = TensorBackend(FastOps)


def _to_tensor(x: np.ndarray):
    return tensor_from_numpy(x.astype(np.float32), backend=_test_backend)


def _proj_weights(linear):
    """Pull (W, b) from a minitorch.Linear as numpy arrays."""
    w = linear.weights.value.to_numpy()
    b = linear.bias.value.to_numpy() if linear.bias is not None else None
    return w, b


def _apply_linear(x, w, b):
    """NumPy equivalent of minitorch Linear.forward.

    minitorch Linear stores weights with shape (in_size, out_size).
    """
    out = x @ w
    if b is not None:
        out = out + b
    return out


def _numpy_mha(x, paged_module, causal=True):
    """Compute multi-head attention in pure NumPy using the same Q/K/V/out
    projection weights as ``paged_module``."""
    n_head = paged_module.n_head
    head_dim = paged_module.head_dim
    n_embd = n_head * head_dim
    batch_size, seq_len, _ = x.shape

    qw, qb = _proj_weights(paged_module.q_proj)
    kw, kb = _proj_weights(paged_module.k_proj)
    vw, vb = _proj_weights(paged_module.v_proj)
    ow, ob = _proj_weights(paged_module.out_proj)

    flat = x.reshape(batch_size * seq_len, n_embd)
    q = _apply_linear(flat, qw, qb).reshape(batch_size, seq_len, n_head, head_dim)
    k = _apply_linear(flat, kw, kb).reshape(batch_size, seq_len, n_head, head_dim)
    v = _apply_linear(flat, vw, vb).reshape(batch_size, seq_len, n_head, head_dim)

    # (B, H, S, D)
    q = q.transpose(0, 2, 1, 3)
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)

    scores = np.matmul(q, np.swapaxes(k, -1, -2)) / np.sqrt(head_dim)
    if causal:
        mask = np.triu(
            np.full((1, 1, seq_len, seq_len), -1e9, dtype=np.float32), k=1,
        )
        scores = scores + mask
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / np.sum(weights, axis=-1, keepdims=True)
    attn = np.matmul(weights, v)  # (B, H, S, D)

    merged = attn.transpose(0, 2, 1, 3).reshape(batch_size * seq_len, n_embd)
    out = _apply_linear(merged, ow, ob).reshape(batch_size, seq_len, n_embd)
    return out


@pytest.mark.parametrize("seq_len,n_head,head_dim", [
    (4, 2, 8),
    (7, 4, 16),
    (16, 2, 4),
])
def test_paged_prefill_matches_numpy_reference(seq_len, n_head, head_dim):
    """PagedMultiHeadAttention.forward_prefill should equal a NumPy MHA
    that uses the same projection weights and the same input."""
    n_embd = n_head * head_dim
    batch_size = 2
    block_size = 4

    paged = PagedMultiHeadAttention(
        n_embd=n_embd, n_head=n_head, block_size=block_size,
        p_dropout=0.0, backend=_test_backend,
    )

    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((batch_size, seq_len, n_embd), dtype=np.float32)

    bm = BlockManager(
        num_blocks=64, block_size=block_size,
        n_head=n_head, head_dim=head_dim, num_layers=1,
    )
    seq_ids = list(range(batch_size))
    for seq_id in seq_ids:
        bm.allocate_blocks_for_sequence(seq_id, seq_len)
    paged_out = paged.forward_prefill(_to_tensor(x_np), bm, seq_ids).to_numpy()

    expected = _numpy_mha(x_np, paged, causal=True)

    np.testing.assert_allclose(paged_out, expected, atol=2e-4, rtol=2e-4)
    for seq_id in seq_ids:
        assert bm.context_lens[seq_id] == seq_len


def test_paged_decode_matches_full_recompute():
    """After prefill, one decode step should match the corresponding row
    of a full-sequence MHA recomputation using the same weights."""
    n_head = 2
    head_dim = 8
    n_embd = n_head * head_dim
    block_size = 4
    prompt_len = 5
    batch_size = 1

    paged = PagedMultiHeadAttention(
        n_embd=n_embd, n_head=n_head, block_size=block_size,
        p_dropout=0.0, backend=_test_backend,
    )

    rng = np.random.default_rng(1)
    prompt_np = rng.standard_normal((batch_size, prompt_len, n_embd), dtype=np.float32)
    new_tok_np = rng.standard_normal((batch_size, 1, n_embd), dtype=np.float32)
    full_np = np.concatenate([prompt_np, new_tok_np], axis=1)

    full_out_np = _numpy_mha(full_np, paged, causal=True)
    expected_last = full_out_np[:, -1:, :]

    bm = BlockManager(
        num_blocks=64, block_size=block_size,
        n_head=n_head, head_dim=head_dim, num_layers=1,
    )
    bm.allocate_blocks_for_sequence(0, prompt_len)
    paged.forward_prefill(_to_tensor(prompt_np), bm, [0])
    bm.append_token_to_sequence(0)
    decoded = paged.forward_decode(_to_tensor(new_tok_np), bm, [0]).to_numpy()

    np.testing.assert_allclose(decoded, expected_last, atol=2e-4, rtol=2e-4)
    assert bm.context_lens[0] == prompt_len + 1


def test_kv_memory_savings_vs_contiguous():
    """compute_kv_memory should report meaningful savings when sequences
    are much shorter than the contiguous baseline reservation."""
    n_head = 4
    head_dim = 16
    block_size = 16

    bm = BlockManager(
        num_blocks=128, block_size=block_size,
        n_head=n_head, head_dim=head_dim, num_layers=1,
    )
    # 4 sequences of 64 tokens each; contiguous baseline assumes 1024.
    for seq_id in range(4):
        bm.allocate_blocks_for_sequence(seq_id, 64)

    mem = bm.compute_kv_memory(max_seq_len=1024)
    assert mem["num_active_sequences"] == 4
    assert mem["kv_bytes_paged"] > 0
    assert mem["kv_bytes_contiguous_naive"] > mem["kv_bytes_paged"]
    # 64/1024 = 6.25% utilization → savings should be >90%
    assert mem["memory_savings_ratio"] > 0.9


def test_kv_memory_savings_for_aligned_sequences():
    """If contiguous baseline equals actual seq_len (i.e. perfect oracle
    sizing), savings should be zero (paged uses exactly the same bytes)."""
    n_head = 2
    head_dim = 8
    block_size = 4

    bm = BlockManager(
        num_blocks=32, block_size=block_size,
        n_head=n_head, head_dim=head_dim, num_layers=1,
    )
    # seq_len=8 fits exactly into 2 blocks of size 4 (no internal frag)
    bm.allocate_blocks_for_sequence(0, 8)

    mem = bm.compute_kv_memory(max_seq_len=8)
    # Paged stores 2 blocks * 4 tokens = 8 tokens worth. Contiguous baseline
    # also reserves 8 tokens. Savings ≈ 0.
    assert abs(mem["memory_savings_ratio"]) < 1e-6
