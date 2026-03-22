"""
Tests for PagedAttention correctness.

Verifies that PagedAttention produces the same output as standard
scaled dot-product attention for various configurations.

Tests cover:
  - Single sequence, single head
  - Multiple heads
  - Batched sequences with different context lengths
  - Edge cases (very short / very long sequences, single block)
"""

import pytest
import sys
import numpy as np

sys.path.insert(0, ".")

from minitorch.block_manager import BlockManager, DEFAULT_BLOCK_SIZE
from minitorch.paged_attention import (
    standard_attention,
    paged_attention_ref,
    PagedAttentionKernel,
    PagedMultiHeadAttention,
)
from minitorch.tensor_functions import tensor_from_numpy
from minitorch.tensor_ops import SimpleBackend


def to_tensor(x: np.ndarray):
    return tensor_from_numpy(x.astype(np.float32), backend=SimpleBackend)


def causal_mask(seq_q: int, seq_kv: int):
    mask = np.triu(
        np.full((1, 1, seq_q, seq_kv), -1e9, dtype=np.float32),
        k=1,
    )
    return to_tensor(mask)


def manual_attention(query, key, value, mask=None):
    scores = np.matmul(query, np.swapaxes(key, -1, -2)) / np.sqrt(query.shape[-1])
    if mask is not None:
        scores = scores + mask
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / np.sum(weights, axis=-1, keepdims=True)
    return np.matmul(weights, value)


def build_paged_cache(
    keys_by_seq,
    values_by_seq,
    block_size,
):
    n_head = keys_by_seq[0].shape[1]
    head_dim = keys_by_seq[0].shape[2]
    total_blocks = sum((len(seq) + block_size - 1) // block_size for seq in keys_by_seq)
    key_cache = np.zeros((total_blocks, block_size, n_head, head_dim), dtype=np.float32)
    value_cache = np.zeros_like(key_cache)
    block_tables = []
    next_block_id = 0

    for seq_keys, seq_values in zip(keys_by_seq, values_by_seq):
        seq_len = seq_keys.shape[0]
        seq_block_ids = []
        for start in range(0, seq_len, block_size):
            end = min(start + block_size, seq_len)
            key_cache[next_block_id, : end - start] = seq_keys[start:end]
            value_cache[next_block_id, : end - start] = seq_values[start:end]
            seq_block_ids.append(next_block_id)
            next_block_id += 1
        block_tables.append(seq_block_ids)

    return key_cache, value_cache, block_tables


class IdentityProjection:
    def __call__(self, x):
        return x


def merge_heads(x):
    batch, n_head, seq_len, head_dim = x.shape
    return np.transpose(x, (0, 2, 1, 3)).reshape(batch, seq_len, n_head * head_dim)


def _cuda_available() -> bool:
    try:
        import pycuda.autoinit  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def default_config():
    return dict(n_head=4, head_dim=16, block_size=4)


# ---------------------------------------------------------------------------
# Reference implementation tests
# ---------------------------------------------------------------------------

class TestStandardAttention:
    def test_single_head_identity(self, rng, default_config):
        n_head = 1
        head_dim = default_config["head_dim"]
        seq_len = 4

        base = np.eye(seq_len, dtype=np.float32).reshape(1, 1, seq_len, seq_len)
        query = to_tensor(base[:, :, :, :head_dim] if head_dim < seq_len else np.pad(base, ((0, 0), (0, 0), (0, 0), (0, head_dim - seq_len))))
        key = to_tensor(query.to_numpy())
        value_np = rng.standard_normal((1, n_head, seq_len, head_dim), dtype=np.float32)
        value = to_tensor(value_np)

        output = standard_attention(query, key, value)
        expected = manual_attention(query.to_numpy(), query.to_numpy(), value_np)

        np.testing.assert_allclose(output.to_numpy(), expected, atol=1e-5, rtol=1e-5)

    def test_causal_mask(self, rng, default_config):
        n_head = 1
        head_dim = default_config["head_dim"]
        seq_len = 4

        query_np = rng.standard_normal((1, n_head, seq_len, head_dim), dtype=np.float32)
        key_np = rng.standard_normal((1, n_head, seq_len, head_dim), dtype=np.float32)
        value_np = rng.standard_normal((1, n_head, seq_len, head_dim), dtype=np.float32)

        output = standard_attention(
            to_tensor(query_np),
            to_tensor(key_np),
            to_tensor(value_np),
            causal_mask(seq_len, seq_len),
        )
        expected = manual_attention(
            query_np,
            key_np,
            value_np,
            np.triu(np.full((1, 1, seq_len, seq_len), -1e9, dtype=np.float32), k=1),
        )

        np.testing.assert_allclose(output.to_numpy(), expected, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Paged vs standard correctness
# ---------------------------------------------------------------------------

class TestPagedVsStandard:
    def test_single_sequence_single_block(self, rng, default_config):
        """Sequence fits in one block — paged output should match standard."""
        n_head = default_config["n_head"]
        head_dim = default_config["head_dim"]
        block_size = default_config["block_size"]
        seq_len = 3

        query_np = rng.standard_normal((1, n_head, 1, head_dim), dtype=np.float32)
        keys_np = rng.standard_normal((seq_len, n_head, head_dim), dtype=np.float32)
        values_np = rng.standard_normal((seq_len, n_head, head_dim), dtype=np.float32)

        key_cache, value_cache, block_tables = build_paged_cache([keys_np], [values_np], block_size)
        paged_out = paged_attention_ref(
            to_tensor(query_np),
            key_cache,
            value_cache,
            block_tables,
            [seq_len],
            block_size=block_size,
            n_head=n_head,
            head_dim=head_dim,
        )

        expected = manual_attention(
            query_np,
            np.transpose(keys_np[None, ...], (0, 2, 1, 3)),
            np.transpose(values_np[None, ...], (0, 2, 1, 3)),
        )
        np.testing.assert_allclose(paged_out.to_numpy(), expected, atol=1e-5, rtol=1e-5)

    def test_single_sequence_multi_block(self, rng, default_config):
        """Sequence spans multiple blocks."""
        n_head = default_config["n_head"]
        head_dim = default_config["head_dim"]
        block_size = default_config["block_size"]
        seq_len = 10

        query_np = rng.standard_normal((1, n_head, 1, head_dim), dtype=np.float32)
        keys_np = rng.standard_normal((seq_len, n_head, head_dim), dtype=np.float32)
        values_np = rng.standard_normal((seq_len, n_head, head_dim), dtype=np.float32)

        key_cache, value_cache, block_tables = build_paged_cache([keys_np], [values_np], block_size)
        paged_out = paged_attention_ref(
            to_tensor(query_np),
            key_cache,
            value_cache,
            block_tables,
            [seq_len],
            block_size=block_size,
            n_head=n_head,
            head_dim=head_dim,
        )

        expected = manual_attention(
            query_np,
            np.transpose(keys_np[None, ...], (0, 2, 1, 3)),
            np.transpose(values_np[None, ...], (0, 2, 1, 3)),
        )
        np.testing.assert_allclose(paged_out.to_numpy(), expected, atol=1e-5, rtol=1e-5)

    def test_batch_different_context_lens(self, rng, default_config):
        """Batch of sequences with varying context lengths."""
        n_head = default_config["n_head"]
        head_dim = default_config["head_dim"]
        block_size = default_config["block_size"]
        seq_lens = [3, 7, 5]
        batch_size = len(seq_lens)

        query_np = rng.standard_normal((batch_size, n_head, 1, head_dim), dtype=np.float32)
        keys_by_seq = [
            rng.standard_normal((seq_len, n_head, head_dim), dtype=np.float32)
            for seq_len in seq_lens
        ]
        values_by_seq = [
            rng.standard_normal((seq_len, n_head, head_dim), dtype=np.float32)
            for seq_len in seq_lens
        ]

        key_cache, value_cache, block_tables = build_paged_cache(keys_by_seq, values_by_seq, block_size)
        paged_out = paged_attention_ref(
            to_tensor(query_np),
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            block_size=block_size,
            n_head=n_head,
            head_dim=head_dim,
        )

        expected_rows = []
        for i, seq_len in enumerate(seq_lens):
            expected_rows.append(
                manual_attention(
                    query_np[i : i + 1],
                    np.transpose(keys_by_seq[i][None, ...], (0, 2, 1, 3)),
                    np.transpose(values_by_seq[i][None, ...], (0, 2, 1, 3)),
                )
            )
        expected = np.concatenate(expected_rows, axis=0)
        np.testing.assert_allclose(paged_out.to_numpy(), expected, atol=1e-5, rtol=1e-5)

    def test_block_size_sweep(self, rng):
        """Verify correctness across different block sizes."""
        n_head = 2
        head_dim = 8
        seq_len = 11
        query_np = rng.standard_normal((1, n_head, 1, head_dim), dtype=np.float32)
        keys_np = rng.standard_normal((seq_len, n_head, head_dim), dtype=np.float32)
        values_np = rng.standard_normal((seq_len, n_head, head_dim), dtype=np.float32)

        expected = manual_attention(
            query_np,
            np.transpose(keys_np[None, ...], (0, 2, 1, 3)),
            np.transpose(values_np[None, ...], (0, 2, 1, 3)),
        )

        for block_size in [1, 4, 8, 16, 32]:
            key_cache, value_cache, block_tables = build_paged_cache(
                [keys_np],
                [values_np],
                block_size,
            )
            paged_out = paged_attention_ref(
                to_tensor(query_np),
                key_cache,
                value_cache,
                block_tables,
                [seq_len],
                block_size=block_size,
                n_head=n_head,
                head_dim=head_dim,
            )
            np.testing.assert_allclose(
                paged_out.to_numpy(),
                expected,
                atol=1e-5,
                rtol=1e-5,
            )


class TestPagedMultiHeadAttention:
    def test_forward_prefill_populates_cache(self, rng):
        n_head = 2
        head_dim = 3
        n_embd = n_head * head_dim
        block_size = 2
        seq_len = 3
        seq_id = 11

        module = PagedMultiHeadAttention(
            n_embd=n_embd,
            n_head=n_head,
            block_size=block_size,
            backend=SimpleBackend,
        )
        module.q_proj = IdentityProjection()
        module.k_proj = IdentityProjection()
        module.v_proj = IdentityProjection()
        module.out_proj = IdentityProjection()

        x_np = rng.standard_normal((1, seq_len, n_embd), dtype=np.float32)
        x = to_tensor(x_np)
        block_manager = BlockManager(
            num_blocks=4,
            block_size=block_size,
            n_head=n_head,
            head_dim=head_dim,
        )

        output = module.forward_prefill(x, block_manager, [seq_id])

        qkv_np = np.transpose(
            x_np.reshape(1, seq_len, n_head, head_dim),
            (0, 2, 1, 3),
        )
        mask = np.triu(
            np.full((1, 1, seq_len, seq_len), -1e9, dtype=np.float32),
            k=1,
        )
        expected = merge_heads(manual_attention(qkv_np, qkv_np, qkv_np, mask))

        np.testing.assert_allclose(output.to_numpy(), expected, atol=1e-5, rtol=1e-5)
        assert block_manager.context_lens[seq_id] == seq_len
        assert block_manager.block_tables[seq_id].block_ids == [0, 1]
        np.testing.assert_allclose(block_manager.key_cache[0, :2], qkv_np[0, :, :2, :].transpose(1, 0, 2))
        np.testing.assert_allclose(block_manager.key_cache[1, :1], qkv_np[0, :, 2:3, :].transpose(1, 0, 2))
        np.testing.assert_allclose(block_manager.value_cache[0, :2], qkv_np[0, :, :2, :].transpose(1, 0, 2))
        np.testing.assert_allclose(block_manager.value_cache[1, :1], qkv_np[0, :, 2:3, :].transpose(1, 0, 2))

    def test_forward_decode_appends_and_reads_cache(self, rng):
        n_head = 2
        head_dim = 3
        n_embd = n_head * head_dim
        block_size = 2
        seq_id = 23

        module = PagedMultiHeadAttention(
            n_embd=n_embd,
            n_head=n_head,
            block_size=block_size,
            backend=SimpleBackend,
        )
        module.q_proj = IdentityProjection()
        module.k_proj = IdentityProjection()
        module.v_proj = IdentityProjection()
        module.out_proj = IdentityProjection()

        prompt_np = rng.standard_normal((1, 2, n_embd), dtype=np.float32)
        decode_np = rng.standard_normal((1, 1, n_embd), dtype=np.float32)
        block_manager = BlockManager(
            num_blocks=4,
            block_size=block_size,
            n_head=n_head,
            head_dim=head_dim,
        )

        module.forward_prefill(to_tensor(prompt_np), block_manager, [seq_id])
        output = module.forward_decode(to_tensor(decode_np), block_manager, [seq_id])

        prompt_heads = np.transpose(
            prompt_np.reshape(1, 2, n_head, head_dim),
            (0, 2, 1, 3),
        )
        decode_heads = np.transpose(
            decode_np.reshape(1, 1, n_head, head_dim),
            (0, 2, 1, 3),
        )
        full_kv = np.concatenate([prompt_heads, decode_heads], axis=2)
        expected = merge_heads(manual_attention(decode_heads, full_kv, full_kv))

        np.testing.assert_allclose(output.to_numpy(), expected, atol=1e-5, rtol=1e-5)
        assert block_manager.context_lens[seq_id] == 3
        assert block_manager.block_tables[seq_id].block_ids == [0, 1]
        np.testing.assert_allclose(block_manager.key_cache[1, 0], decode_heads[0, :, 0, :])
        np.testing.assert_allclose(block_manager.value_cache[1, 0], decode_heads[0, :, 0, :])


# ---------------------------------------------------------------------------
# CUDA kernel tests (only run when GPU available)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _cuda_available(),
    reason="CUDA not available",
)
@pytest.mark.skip(reason="PagedAttention CUDA kernel wrapper not implemented yet")
class TestPagedAttentionKernel:
    def test_kernel_matches_reference(self, rng, default_config):
        """CUDA kernel output should match Python reference implementation."""
        raise NotImplementedError

    def test_kernel_batch(self, rng, default_config):
        raise NotImplementedError
