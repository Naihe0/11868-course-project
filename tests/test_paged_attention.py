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

_test_backend = SimpleBackend


def to_tensor(x: np.ndarray):
    return tensor_from_numpy(x.astype(np.float32), backend=_test_backend)


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


class ScaleModule:
    def __init__(self, scale: float, shift: float = 0.0):
        self.scale = scale
        self.shift = shift

    def __call__(self, x):
        return to_tensor(x.to_numpy() * self.scale + self.shift)


class FakePagedAttentionModule:
    def __init__(self, scale: float, n_head: int, head_dim: int, block_size: int):
        self.scale = scale
        self.n_head = n_head
        self.head_dim = head_dim
        self.block_size = block_size

    def _heads(self, x):
        x_np = x.to_numpy()
        batch_size, seq_len, _ = x_np.shape
        return x_np.reshape(batch_size, seq_len, self.n_head, self.head_dim)

    def forward_prefill(self, x, block_manager, seq_ids):
        x_heads = self._heads(x)
        batch_size, seq_len, _, _ = x_heads.shape

        for batch_idx, seq_id in enumerate(seq_ids):
            block_table = block_manager.allocate_blocks_for_sequence(seq_id, seq_len)
            for logical_block_idx, block_id in enumerate(block_table.block_ids):
                block = block_manager.blocks[block_id]
                start = logical_block_idx * self.block_size
                end = start + block.num_filled
                block_manager.key_cache[block_id, : block.num_filled] = x_heads[batch_idx, start:end]
                block_manager.value_cache[block_id, : block.num_filled] = x_heads[batch_idx, start:end]

        return to_tensor(x.to_numpy() * self.scale)

    def forward_decode(self, x, block_manager, seq_ids):
        x_heads = self._heads(x)[:, 0]

        for batch_idx, seq_id in enumerate(seq_ids):
            if seq_id not in block_manager.block_tables:
                block_manager.allocate_blocks_for_sequence(seq_id, 0)
            block_manager.write_token_kv(seq_id, x_heads[batch_idx], x_heads[batch_idx])

        return to_tensor(x.to_numpy() * self.scale)


class FakeTokenEmbedding:
    def __init__(self, n_embd: int):
        self.n_embd = n_embd

    def __call__(self, idx):
        idx_np = idx.to_numpy().astype(np.float32)
        return to_tensor(np.repeat(idx_np[..., None], self.n_embd, axis=2) / 10.0)


class FakePositionEmbedding:
    def __init__(self, n_embd: int):
        self.n_embd = n_embd

    def __call__(self, idx):
        batch_size, seq_len = idx.shape
        return to_tensor(np.zeros((batch_size, seq_len, self.n_embd), dtype=np.float32))


class FakeLMHead:
    def __init__(self, n_vocab: int):
        self.n_vocab = n_vocab

    def __call__(self, x):
        batch_size = x.shape[0]
        logits = np.zeros((batch_size, self.n_vocab), dtype=np.float32)
        logits[:, 0] = 1.0
        if self.n_vocab > 1:
            logits[:, 1] = 0.5
        return to_tensor(logits)


def merge_heads(x):
    batch, n_head, seq_len, head_dim = x.shape
    return np.transpose(x, (0, 2, 1, 3)).reshape(batch, seq_len, n_head * head_dim)


def _cuda_available() -> bool:
    try:
        import pycuda.autoinit  # noqa: F401
        return True
    except Exception:
        return False


def _kernel_available() -> bool:
    """Check if the compiled PagedAttention .so exists and a CUDA GPU is present."""
    import os
    so_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "minitorch", "cuda_kernels", "paged_attention.so",
    )
    if not os.path.exists(so_path):
        return False
    try:
        import ctypes
        ctypes.CDLL(so_path)
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def default_config():
    return dict(n_head=2, head_dim=4, block_size=4)


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
            backend=_test_backend,
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
            backend=_test_backend,
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
# PagedTransformerLayer tests
# ---------------------------------------------------------------------------

class TestPagedTransformerLayer:
    def test_prefill_output_shape(self, rng):
        """Prefill should return (batch, seq_len, n_embd)."""
        from minitorch.transformer import PagedTransformerLayer
        n_head = 2
        head_dim = 3
        n_embd = n_head * head_dim
        block_size = 2
        seq_len = 3
        batch_size = 1

        layer = PagedTransformerLayer(
            n_embd=n_embd, n_head=n_head, block_size=block_size,
            p_dropout=0.0, backend=_test_backend,
        )
        layer.ln_1 = IdentityProjection()
        layer.ln_2 = IdentityProjection()
        layer.attention = FakePagedAttentionModule(0.25, n_head, head_dim, block_size)
        layer.ff = ScaleModule(0.5)
        block_manager = BlockManager(
            num_blocks=8, block_size=block_size,
            n_head=n_head, head_dim=head_dim,
        )
        x_np = rng.standard_normal((batch_size, seq_len, n_embd), dtype=np.float32)
        x = to_tensor(x_np)

        output = layer.forward_prefill(x, block_manager, [0])
        assert output.shape == (batch_size, seq_len, n_embd)

    def test_decode_output_shape(self, rng):
        """Decode should return (batch, 1, n_embd) after prefill."""
        from minitorch.transformer import PagedTransformerLayer
        n_head = 2
        head_dim = 3
        n_embd = n_head * head_dim
        block_size = 2
        batch_size = 1

        layer = PagedTransformerLayer(
            n_embd=n_embd, n_head=n_head, block_size=block_size,
            p_dropout=0.0, backend=_test_backend,
        )
        layer.ln_1 = IdentityProjection()
        layer.ln_2 = IdentityProjection()
        layer.attention = FakePagedAttentionModule(0.25, n_head, head_dim, block_size)
        layer.ff = ScaleModule(0.5)
        block_manager = BlockManager(
            num_blocks=8, block_size=block_size,
            n_head=n_head, head_dim=head_dim,
        )
        # Prefill first
        prompt_np = rng.standard_normal((batch_size, 2, n_embd), dtype=np.float32)
        layer.forward_prefill(to_tensor(prompt_np), block_manager, [0])

        # Decode
        decode_np = rng.standard_normal((batch_size, 1, n_embd), dtype=np.float32)
        output = layer.forward_decode(to_tensor(decode_np), block_manager, [0])
        assert output.shape == (batch_size, 1, n_embd)

    def test_prefill_residual_connection(self, rng):
        """Output should differ from input (attention + ff applied) but have same shape."""
        from minitorch.transformer import PagedTransformerLayer
        n_head = 2
        head_dim = 3
        n_embd = n_head * head_dim
        block_size = 4

        layer = PagedTransformerLayer(
            n_embd=n_embd, n_head=n_head, block_size=block_size,
            p_dropout=0.0, backend=_test_backend,
        )
        layer.ln_1 = IdentityProjection()
        layer.ln_2 = IdentityProjection()
        layer.attention = FakePagedAttentionModule(0.25, n_head, head_dim, block_size)
        layer.ff = ScaleModule(0.5)
        block_manager = BlockManager(
            num_blocks=4, block_size=block_size,
            n_head=n_head, head_dim=head_dim,
        )
        x_np = rng.standard_normal((1, 3, n_embd), dtype=np.float32)
        x = to_tensor(x_np)
        output = layer.forward_prefill(x, block_manager, [0])
        # Output should not be identical to input (non-trivial transform)
        assert not np.allclose(output.to_numpy(), x_np, atol=1e-3)


class TestPagedDecoderLM:
    def test_generate_output_shape(self, rng):
        """Generate should produce (batch, prompt_len + max_new_tokens) tokens."""
        from minitorch.transformer import PagedDecoderLM
        n_vocab = 10
        n_embd = 6
        n_head = 2
        n_positions = 16
        n_layers = 1
        block_size = 4
        prompt_len = 3
        max_new_tokens = 2

        model = PagedDecoderLM(
            n_vocab=n_vocab, n_embd=n_embd, n_head=n_head,
            n_positions=n_positions, n_layers=n_layers,
            block_size=block_size, p_dropout=0.0,
            backend=_test_backend,
        )
        model.token_embeddings = FakeTokenEmbedding(n_embd)
        model.position_embeddings = FakePositionEmbedding(n_embd)
        model.layers = [FakePagedAttentionModule(0.0, n_head, n_embd // n_head, block_size)]
        model.ln = IdentityProjection()
        model.lm_head = FakeLMHead(n_vocab)
        block_manager = BlockManager(
            num_blocks=16, block_size=block_size,
            n_head=n_head, head_dim=n_embd // n_head,
        )
        idx_np = rng.integers(0, n_vocab, size=(1, prompt_len)).astype(np.float32)
        idx = to_tensor(idx_np)

        result = PagedDecoderLM.generate(
            model, idx, max_new_tokens, block_manager, [0], temperature=1.0,
        )
        assert result.shape == (1, prompt_len + max_new_tokens)

    def test_generate_frees_sequences(self, rng):
        """After generate, the block manager should have freed the sequence."""
        from minitorch.transformer import PagedDecoderLM
        n_vocab = 10
        n_embd = 6
        n_head = 2
        n_positions = 16
        n_layers = 1
        block_size = 4

        model = PagedDecoderLM(
            n_vocab=n_vocab, n_embd=n_embd, n_head=n_head,
            n_positions=n_positions, n_layers=n_layers,
            block_size=block_size, p_dropout=0.0,
            backend=_test_backend,
        )
        model.token_embeddings = FakeTokenEmbedding(n_embd)
        model.position_embeddings = FakePositionEmbedding(n_embd)
        model.layers = [FakePagedAttentionModule(0.0, n_head, n_embd // n_head, block_size)]
        model.ln = IdentityProjection()
        model.lm_head = FakeLMHead(n_vocab)
        block_manager = BlockManager(
            num_blocks=16, block_size=block_size,
            n_head=n_head, head_dim=n_embd // n_head,
        )
        idx_np = rng.integers(0, n_vocab, size=(1, 2)).astype(np.float32)
        idx = to_tensor(idx_np)

        PagedDecoderLM.generate(
            model, idx, 2, block_manager, [42], temperature=1.0,
        )
        # Sequence should be freed
        assert 42 not in block_manager.block_tables


# ---------------------------------------------------------------------------
# CUDA kernel tests (only run when GPU available)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _kernel_available(),
    reason="PagedAttention .so not available (compile with compile_cuda.sh)",
)
class TestPagedAttentionKernel:
    def test_kernel_matches_reference(self, rng, default_config):
        """CUDA kernel output should match Python reference implementation."""
        n_head = default_config["n_head"]
        head_dim = default_config["head_dim"]
        block_size = default_config["block_size"]
        seq_len = 7

        query_np = rng.standard_normal((1, n_head, 1, head_dim), dtype=np.float32)
        keys_np = rng.standard_normal((seq_len, n_head, head_dim), dtype=np.float32)
        values_np = rng.standard_normal((seq_len, n_head, head_dim), dtype=np.float32)

        key_cache, value_cache, block_tables = build_paged_cache(
            [keys_np], [values_np], block_size,
        )
        # Python reference
        ref_out = paged_attention_ref(
            to_tensor(query_np),
            key_cache, value_cache, block_tables,
            [seq_len],
            block_size=block_size, n_head=n_head, head_dim=head_dim,
        )

        # CUDA kernel
        kernel = PagedAttentionKernel()
        kernel_out = kernel.forward(
            to_tensor(query_np),
            to_tensor(key_cache),
            to_tensor(value_cache),
            to_tensor(np.array(block_tables, dtype=np.int32)),
            to_tensor(np.array([seq_len], dtype=np.int32)),
            block_size=block_size,
            max_context_len=seq_len,
        )

        np.testing.assert_allclose(
            kernel_out.to_numpy(), ref_out.to_numpy(), atol=1e-4, rtol=1e-4,
        )

    def test_kernel_batch(self, rng, default_config):
        """CUDA kernel should handle batched sequences with different lengths."""
        n_head = default_config["n_head"]
        head_dim = default_config["head_dim"]
        block_size = default_config["block_size"]
        seq_lens = [3, 7, 5]
        batch_size = len(seq_lens)

        query_np = rng.standard_normal(
            (batch_size, n_head, 1, head_dim), dtype=np.float32,
        )
        keys_by_seq = [
            rng.standard_normal((sl, n_head, head_dim), dtype=np.float32)
            for sl in seq_lens
        ]
        values_by_seq = [
            rng.standard_normal((sl, n_head, head_dim), dtype=np.float32)
            for sl in seq_lens
        ]

        key_cache, value_cache, block_tables = build_paged_cache(
            keys_by_seq, values_by_seq, block_size,
        )

        # Python reference
        ref_out = paged_attention_ref(
            to_tensor(query_np),
            key_cache, value_cache, block_tables,
            seq_lens,
            block_size=block_size, n_head=n_head, head_dim=head_dim,
        )

        # Pad block_tables to uniform width for the kernel
        max_blocks = max(len(bt) for bt in block_tables)
        padded_tables = np.zeros((batch_size, max_blocks), dtype=np.int32)
        for i, bt in enumerate(block_tables):
            padded_tables[i, :len(bt)] = bt

        kernel = PagedAttentionKernel()
        kernel_out = kernel.forward(
            to_tensor(query_np),
            to_tensor(key_cache),
            to_tensor(value_cache),
            to_tensor(padded_tables),
            to_tensor(np.array(seq_lens, dtype=np.int32)),
            block_size=block_size,
            max_context_len=max(seq_lens),
        )

        np.testing.assert_allclose(
            kernel_out.to_numpy(), ref_out.to_numpy(), atol=1e-4, rtol=1e-4,
        )
