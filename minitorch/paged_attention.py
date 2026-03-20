"""
PagedAttention mechanism for MiniTorch.

Implements attention computation over non-contiguous KV cache blocks,
allowing efficient memory utilization during LLM inference.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple

from .module import Module, Parameter
from .tensor import Tensor, tensor, tensor_from_numpy
from .tensor_ops import TensorBackend
from .nn import softmax
from .block_manager import BlockManager, BlockTable, KVBlock, DEFAULT_BLOCK_SIZE

datatype = np.float32


# ---------------------------------------------------------------------------
# Reference (naive) implementation — for correctness testing
# ---------------------------------------------------------------------------

def standard_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """Standard scaled dot-product attention on contiguous tensors.

    Args:
        query: (batch, n_head, seq_q, head_dim)
        key:   (batch, n_head, seq_kv, head_dim)
        value: (batch, n_head, seq_kv, head_dim)
        mask:  Optional broadcastable mask (0 / -inf).

    Returns:
        Attention output (batch, n_head, seq_q, head_dim).
    """
    # TODO: Implement standard attention as a correctness baseline
    raise NotImplementedError


# ---------------------------------------------------------------------------
# PagedAttention — Python reference implementation
# ---------------------------------------------------------------------------

def paged_attention_ref(
    query: Tensor,
    key_cache: np.ndarray,
    value_cache: np.ndarray,
    block_tables: List[List[int]],
    context_lens: List[int],
    block_size: int = DEFAULT_BLOCK_SIZE,
    n_head: int = 8,
    head_dim: int = 64,
) -> Tensor:
    """Reference Python implementation of PagedAttention.

    Computes attention by gathering keys/values from non-contiguous
    physical blocks according to each sequence's block table.

    This is a slow but correct implementation used for validating
    the CUDA kernel.

    Args:
        query:        Query tensor (batch, n_head, 1, head_dim) for decode step.
        key_cache:    Global key cache with shape
                      (num_blocks, block_size, n_head, head_dim).
        value_cache:  Global value cache with shape
                      (num_blocks, block_size, n_head, head_dim).
        block_tables: Per-sequence mapping from logical to physical block ids.
        context_lens: Number of context tokens per sequence.
        block_size:   Tokens per block.
        n_head:       Number of attention heads.
        head_dim:     Head dimension.

    Returns:
        Attention output (batch, n_head, 1, head_dim).
    """
    # TODO: Implement reference PagedAttention
    # Steps:
    #   1. For each sequence, gather K/V from blocks using block_table
    #   2. Compute scaled dot product attention
    #   3. Return output
    raise NotImplementedError


# ---------------------------------------------------------------------------
# PagedAttention — CUDA kernel wrapper
# ---------------------------------------------------------------------------

class PagedAttentionKernel:
    """Wrapper around the compiled PagedAttention CUDA kernel.

    Loads the shared library and exposes a Python-callable interface.
    """

    def __init__(self, library_path: str = "minitorch/cuda_kernels/paged_attention.so"):
        self.library_path = library_path
        self._lib = None

    def _load_library(self):
        """Load the compiled CUDA shared library."""
        # TODO: Load .so using ctypes
        raise NotImplementedError

    def forward(
        self,
        query: Tensor,
        key_cache: Tensor,
        value_cache: Tensor,
        block_tables: Tensor,
        context_lens: Tensor,
        block_size: int,
        max_context_len: int,
    ) -> Tensor:
        """Run the PagedAttention CUDA kernel.

        Args:
            query:           (batch, n_head, head_dim)
            key_cache:       (num_blocks, block_size, n_head, head_dim)
            value_cache:     (num_blocks, block_size, n_head, head_dim)
            block_tables:    (batch, max_blocks_per_seq) int tensor
            context_lens:    (batch,) int tensor
            block_size:      Tokens per block.
            max_context_len: Maximum context length across the batch.

        Returns:
            Attention output (batch, n_head, head_dim).
        """
        # TODO: Call into CUDA kernel
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Module wrapper for use in Transformer
# ---------------------------------------------------------------------------

class PagedMultiHeadAttention(Module):
    """Multi-Head Attention that supports both standard and paged modes.

    In standard mode (e.g. prefill), uses contiguous KV tensors.
    In paged mode (decode), reads KV from the block manager's cache.

    Args:
        n_embd:    Embedding / model dimension.
        n_head:    Number of attention heads.
        block_size: Block size for paged KV cache.
        p_dropout: Dropout probability.
        bias:      Whether linear projections have bias.
        backend:   MiniTorch tensor backend.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int = DEFAULT_BLOCK_SIZE,
        p_dropout: float = 0.1,
        bias: bool = True,
        backend: TensorBackend = None,
    ):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.block_size = block_size
        self.backend = backend

        # Projections
        # TODO: Initialize Q, K, V, Output linear projections
        #       (reuse modules_basic.Linear)

        # CUDA kernel handle (lazy-loaded)
        self._kernel: Optional[PagedAttentionKernel] = None

    # ----- Forward (prefill — standard attention) --------------------------

    def forward_prefill(
        self,
        x: Tensor,
        block_manager: BlockManager,
        seq_ids: List[int],
    ) -> Tensor:
        """Prefill phase: compute attention on the full prompt and populate
        KV cache blocks via the block manager.

        Args:
            x:             Input tensor (batch, seq_len, n_embd).
            block_manager: The global block manager.
            seq_ids:       Sequence ids for each item in the batch.

        Returns:
            Output tensor (batch, seq_len, n_embd).
        """
        # TODO: Implement prefill
        # Steps:
        #   1. Project x -> Q, K, V
        #   2. Compute standard causal attention
        #   3. Write K, V into block manager's cache
        #   4. Return output
        raise NotImplementedError

    # ----- Forward (decode — paged attention) ------------------------------

    def forward_decode(
        self,
        x: Tensor,
        block_manager: BlockManager,
        seq_ids: List[int],
    ) -> Tensor:
        """Decode phase: attend to cached KV using PagedAttention.

        Args:
            x:             Input tensor (batch, 1, n_embd) — single new token.
            block_manager: The global block manager.
            seq_ids:       Sequence ids for each item in the batch.

        Returns:
            Output tensor (batch, 1, n_embd).
        """
        # TODO: Implement decode with paged attention
        # Steps:
        #   1. Project x -> Q, K_new, V_new
        #   2. Append K_new, V_new to block manager
        #   3. Gather block tables and context lengths
        #   4. Call PagedAttention kernel (or ref implementation)
        #   5. Return output
        raise NotImplementedError
