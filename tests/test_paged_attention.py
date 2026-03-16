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
)


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
        # TODO: Q=K=V should give attention output equal to V
        pass

    def test_causal_mask(self, rng, default_config):
        # TODO: Verify future tokens do not influence past outputs
        pass


# ---------------------------------------------------------------------------
# Paged vs standard correctness
# ---------------------------------------------------------------------------

class TestPagedVsStandard:
    def test_single_sequence_single_block(self, rng, default_config):
        """Sequence fits in one block — paged output should match standard."""
        # TODO: Implement
        pass

    def test_single_sequence_multi_block(self, rng, default_config):
        """Sequence spans multiple blocks."""
        # TODO: Implement
        pass

    def test_batch_different_context_lens(self, rng, default_config):
        """Batch of sequences with varying context lengths."""
        # TODO: Implement
        pass

    def test_block_size_sweep(self, rng):
        """Verify correctness across different block sizes."""
        # TODO: Implement for block_sizes = [1, 4, 8, 16, 32]
        pass


# ---------------------------------------------------------------------------
# CUDA kernel tests (only run when GPU available)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _cuda_available(),
    reason="CUDA not available",
)
class TestPagedAttentionKernel:
    def test_kernel_matches_reference(self, rng, default_config):
        """CUDA kernel output should match Python reference implementation."""
        # TODO: Implement
        pass

    def test_kernel_batch(self, rng, default_config):
        # TODO: Test batched CUDA kernel
        pass


def _cuda_available() -> bool:
    try:
        import pycuda.autoinit  # noqa: F401
        return True
    except Exception:
        return False
