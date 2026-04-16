"""
PagedAttention mechanism for MiniTorch.

Implements attention computation over non-contiguous KV cache blocks,
allowing efficient memory utilization during LLM inference.
"""

from __future__ import annotations

import ctypes
import os
import numpy as np
from typing import Dict, List, Optional, Tuple

from .module import Module, Parameter
from .tensor import Tensor, tensor, tensor_from_numpy
from .tensor_ops import TensorBackend
from .nn import softmax
from .block_manager import BlockManager, BlockTable, KVBlock, DEFAULT_BLOCK_SIZE
from .modules_basic import Linear
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
    batch_size, n_head, seq_q, head_dim = query.shape
    _, _, seq_kv, _ = key.shape

    q_expanded = query.contiguous().view(batch_size, n_head, seq_q, 1, head_dim)
    k_expanded = key.contiguous().view(batch_size, n_head, 1, seq_kv, head_dim)
    scores = (q_expanded * k_expanded).sum(dim=4).view(
        batch_size, n_head, seq_q, seq_kv
    ) / np.sqrt(head_dim)

    if mask is not None:
        scores = scores + mask

    weights = softmax(scores, dim=3)
    weight_expanded = weights.contiguous().view(batch_size, n_head, seq_q, seq_kv, 1)
    value_expanded = value.contiguous().view(batch_size, n_head, 1, seq_kv, head_dim)
    output = (weight_expanded * value_expanded).sum(dim=3).view(
        batch_size, n_head, seq_q, head_dim
    )
    return output


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
    layer_id: int = 0,
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
    if isinstance(key_cache, np.ndarray):
        key_cache = [key_cache]
        value_cache = [value_cache]

    outputs = []
    query_np = query.to_numpy()

    for batch_idx, block_table in enumerate(block_tables):
        context_len = context_lens[batch_idx]
        gathered_keys = []
        gathered_values = []

        for token_idx in range(context_len):
            logical_block_idx = token_idx // block_size
            slot_idx = token_idx % block_size
            block_id = block_table[logical_block_idx]
            gathered_keys.append(key_cache[layer_id][block_id, slot_idx])
            gathered_values.append(value_cache[layer_id][block_id, slot_idx])

        key_np = np.stack(gathered_keys, axis=1).reshape(1, n_head, context_len, head_dim)
        value_np = np.stack(gathered_values, axis=1).reshape(1, n_head, context_len, head_dim)
        query_i = tensor_from_numpy(query_np[batch_idx : batch_idx + 1].astype(datatype), backend=query.backend)
        key_i = tensor_from_numpy(key_np.astype(datatype), backend=query.backend)
        value_i = tensor_from_numpy(value_np.astype(datatype), backend=query.backend)
        outputs.append(standard_attention(query_i, key_i, value_i).to_numpy())

    return tensor_from_numpy(np.concatenate(outputs, axis=0).astype(datatype), backend=query.backend)


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

    @staticmethod
    def _to_numpy(value, dtype):
        if isinstance(value, Tensor):
            array = value.to_numpy()
        else:
            array = np.asarray(value)
        return np.ascontiguousarray(array.astype(dtype, copy=False))

    def _load_library(self):
        """Load the compiled CUDA shared library."""
        if self._lib is not None:
            return self._lib

        library_path = self.library_path
        if not os.path.isabs(library_path):
            project_root = os.path.dirname(os.path.dirname(__file__))
            library_path = os.path.join(project_root, library_path)

        if not os.path.exists(library_path):
            raise FileNotFoundError(
                f"PagedAttention CUDA library not found: {library_path}"
            )

        lib = ctypes.CDLL(library_path)
        lib.paged_attention_forward.argtypes = [
            np.ctypeslib.ndpointer(dtype=datatype, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=datatype, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=datatype, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=datatype, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.paged_attention_forward.restype = None
        self._lib = lib
        return self._lib

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
        lib = self._load_library()

        query_np = self._to_numpy(query, datatype)
        original_query_rank = query_np.ndim
        if original_query_rank == 4:
            if query_np.shape[2] != 1:
                raise ValueError("Kernel wrapper only supports decode queries with seq_len == 1")
            query_np = query_np[:, :, 0, :]
        elif query_np.ndim != 3:
            raise ValueError("Query must have shape (batch, n_head, head_dim) or (batch, n_head, 1, head_dim)")

        key_cache_np = self._to_numpy(key_cache, datatype)
        value_cache_np = self._to_numpy(value_cache, datatype)
        block_tables_np = self._to_numpy(block_tables, np.int32)
        context_lens_np = self._to_numpy(context_lens, np.int32).reshape(-1)

        batch_size, n_head, head_dim = query_np.shape
        if key_cache_np.shape[1] != block_size:
            raise ValueError("key_cache shape does not match block_size")
        if value_cache_np.shape != key_cache_np.shape:
            raise ValueError("value_cache must have the same shape as key_cache")
        if block_tables_np.shape[0] != batch_size:
            raise ValueError("block_tables batch dimension must match query batch size")
        if context_lens_np.shape[0] != batch_size:
            raise ValueError("context_lens length must match query batch size")

        max_blocks_per_seq = block_tables_np.shape[1] if block_tables_np.ndim == 2 else 0
        output_np = np.zeros((batch_size, n_head, head_dim), dtype=datatype)

        lib.paged_attention_forward(
            output_np,
            query_np,
            key_cache_np,
            value_cache_np,
            block_tables_np,
            context_lens_np,
            batch_size,
            n_head,
            head_dim,
            block_size,
            max_blocks_per_seq,
            max_context_len,
        )

        if original_query_rank == 4:
            output_np = output_np.reshape(batch_size, n_head, 1, head_dim)
        return tensor_from_numpy(output_np.astype(datatype), backend=query.backend)


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
        layer_id: int = 0,
        decode_backend: str = "ref",
        compare_to_ref: bool = False,
        compare_tolerance: float = 1e-4,
    ):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.block_size = block_size
        self.backend = backend
        self.layer_id = layer_id
        if decode_backend not in {"ref", "cuda"}:
            raise ValueError("decode_backend must be 'ref' or 'cuda'")
        self.decode_backend = decode_backend
        self.compare_to_ref = compare_to_ref
        self.compare_tolerance = compare_tolerance
        self.last_decode_compare: Optional[Dict[str, float]] = None
        # Projections
        # TODO: Initialize Q, K, V, Output linear projections
        #       (reuse modules_basic.Linear)

        # CUDA kernel handle (lazy-loaded)
        self.q_proj = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.k_proj = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.v_proj = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.out_proj = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self._kernel: Optional[PagedAttentionKernel] = None

    def _decode_attention_ref(
        self,
        q: Tensor,
        block_manager: BlockManager,
        seq_ids: List[int],
    ) -> Tensor:
        block_tables = [
            block_manager.block_tables[seq_id].block_ids for seq_id in seq_ids
        ]
        context_lens = [
            block_manager.get_context_len(seq_id) for seq_id in seq_ids
        ]
        return paged_attention_ref(
            q,
            block_manager.key_cache,
            block_manager.value_cache,
            block_tables,
            context_lens,
            block_size=self.block_size,
            n_head=self.n_head,
            head_dim=self.head_dim,
            layer_id=self.layer_id,
        )

    def _decode_attention_kernel(
        self,
        q: Tensor,
        block_manager: BlockManager,
        seq_ids: List[int],
    ) -> Tensor:
        if self._kernel is None:
            self._kernel = PagedAttentionKernel()
        block_tables = block_manager.get_block_table_array(seq_ids)
        context_lens = np.array(
            [block_manager.get_context_len(seq_id) for seq_id in seq_ids],
            dtype=np.int32,
        )
        max_context_len = int(context_lens.max()) if len(context_lens) > 0 else 0
        return self._kernel.forward(
            q,
            block_manager.get_key_cache(self.layer_id),
            block_manager.get_value_cache(self.layer_id),
            block_tables,
            context_lens,
            block_size=self.block_size,
            max_context_len=max_context_len,
        )

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
        batch_size, seq_len, _ = x.shape
        flat_x = x.contiguous().view(batch_size * seq_len, self.n_embd)

        q = self.q_proj(flat_x).view(batch_size, seq_len, self.n_head, self.head_dim)
        k = self.k_proj(flat_x).view(batch_size, seq_len, self.n_head, self.head_dim)
        v = self.v_proj(flat_x).view(batch_size, seq_len, self.n_head, self.head_dim)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        mask_np = np.triu(
            np.full((seq_len, seq_len), -1e9, dtype=datatype),
            k=1,
        ).reshape(1, 1, seq_len, seq_len)
        mask = tensor_from_numpy(mask_np, backend=x.backend)

        output = standard_attention(q, k, v, mask)
        output = output.permute(0, 2, 1, 3).contiguous().view(
            batch_size * seq_len, self.n_embd
        )
        output = self.out_proj(output).view(batch_size, seq_len, self.n_embd)

        k_cache_values = k.permute(0, 2, 1, 3).to_numpy()
        v_cache_values = v.permute(0, 2, 1, 3).to_numpy()

        for batch_idx, seq_id in enumerate(seq_ids):
            block_table = block_manager.block_tables[seq_id]
            for logical_block_idx, block_id in enumerate(block_table.block_ids):
                block = block_manager.blocks[block_id]
                start = logical_block_idx * self.block_size
                end = start + block.num_filled

                block_manager.key_cache[self.layer_id][block_id, : block.num_filled, :, :] = (
                    k_cache_values[batch_idx, start:end, :, :]
                )
                block_manager.value_cache[self.layer_id][block_id, : block.num_filled, :, :] = (
                    v_cache_values[batch_idx, start:end, :, :]
                )

        return output

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
        batch_size, seq_len, _ = x.shape
        if seq_len != 1:
            raise ValueError("forward_decode expects a single new token per sequence")

        flat_x = x.contiguous().view(batch_size * seq_len, self.n_embd)
        q = self.q_proj(flat_x).view(batch_size, seq_len, self.n_head, self.head_dim)
        k = self.k_proj(flat_x).view(batch_size, seq_len, self.n_head, self.head_dim)
        v = self.v_proj(flat_x).view(batch_size, seq_len, self.n_head, self.head_dim)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        k_new = k.permute(0, 2, 1, 3).to_numpy()[:, 0, :, :]
        v_new = v.permute(0, 2, 1, 3).to_numpy()[:, 0, :, :]

        for batch_idx, seq_id in enumerate(seq_ids):
            block_table = block_manager.block_tables[seq_id]
            token_index = block_manager.get_context_len(seq_id) - 1
            block_id, slot_idx = block_manager.get_physical_location(seq_id, token_index)
            block_manager.write_kv_slot(
                block_id,
                slot_idx,
                k_new[batch_idx],
                v_new[batch_idx],
                layer=self.layer_id,
            )

        if self.decode_backend == "cuda":
            output = self._decode_attention_kernel(q, block_manager, seq_ids)
            if self.compare_to_ref:
                ref_output = self._decode_attention_ref(q, block_manager, seq_ids)
                output_np = output.to_numpy()
                ref_np = ref_output.to_numpy()
                max_abs_error = float(np.max(np.abs(output_np - ref_np)))
                mean_abs_error = float(np.mean(np.abs(output_np - ref_np)))
                self.last_decode_compare = {
                    "max_abs_error": max_abs_error,
                    "mean_abs_error": mean_abs_error,
                }
                if max_abs_error > self.compare_tolerance:
                    raise AssertionError(
                        "CUDA paged attention mismatch: "
                        f"max_abs_error={max_abs_error:.6f} "
                        f"> tolerance={self.compare_tolerance:.6f}"
                    )
        else:
            self.last_decode_compare = None
            output = self._decode_attention_ref(q, block_manager, seq_ids)
        output = output.permute(0, 2, 1, 3).contiguous().view(
            batch_size * seq_len, self.n_embd
        )
        output = self.out_proj(output).view(batch_size, seq_len, self.n_embd)
        return output
