"""
PagedAttention mechanism for MiniTorch.

Implements attention computation over non-contiguous KV cache blocks,
allowing efficient memory utilization during LLM inference.
"""

from __future__ import annotations

import ctypes
import os
import numpy as np
import numba
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
        self._runtime = None
        self._runtime_config: Optional[Tuple[int, int, int, int, int, int]] = None

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
        lib.paged_attention_runtime_create.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.paged_attention_runtime_create.restype = ctypes.c_void_p
        lib.paged_attention_runtime_destroy.argtypes = [ctypes.c_void_p]
        lib.paged_attention_runtime_destroy.restype = None
        lib.paged_attention_runtime_upload_layer_cache.argtypes = [
            ctypes.c_void_p,
            np.ctypeslib.ndpointer(dtype=datatype, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=datatype, flags="C_CONTIGUOUS"),
        ]
        lib.paged_attention_runtime_upload_layer_cache.restype = None
        lib.paged_attention_runtime_copy_layer_cache_from_device.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        lib.paged_attention_runtime_copy_layer_cache_from_device.restype = None
        lib.paged_attention_runtime_update_slot.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=datatype, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=datatype, flags="C_CONTIGUOUS"),
        ]
        lib.paged_attention_runtime_update_slot.restype = None
        lib.paged_attention_runtime_update_slot_from_device.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        lib.paged_attention_runtime_update_slot_from_device.restype = None
        lib.paged_attention_runtime_upload_block.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=datatype, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=datatype, flags="C_CONTIGUOUS"),
        ]
        lib.paged_attention_runtime_upload_block.restype = None
        lib.paged_attention_runtime_update_metadata.argtypes = [
            ctypes.c_void_p,
            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.paged_attention_runtime_update_metadata.restype = None
        lib.paged_attention_runtime_forward.argtypes = [
            ctypes.c_void_p,
            np.ctypeslib.ndpointer(dtype=datatype, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=datatype, flags="C_CONTIGUOUS"),
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.paged_attention_runtime_forward.restype = None
        lib.paged_attention_runtime_forward_device.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.paged_attention_runtime_forward_device.restype = None
        lib.contiguous_attention_forward_device.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.contiguous_attention_forward_device.restype = None
        lib.contiguous_kv_update_slot_from_device.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.contiguous_kv_update_slot_from_device.restype = None
        lib.paged_attention_runtime_prefill_with_prefix_forward.argtypes = [
            ctypes.c_void_p,
            np.ctypeslib.ndpointer(dtype=datatype, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=datatype, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=datatype, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=datatype, flags="C_CONTIGUOUS"),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.paged_attention_runtime_prefill_with_prefix_forward.restype = None
        self._lib = lib
        return self._lib

    def ensure_runtime(
        self,
        num_blocks: int,
        block_size: int,
        n_head: int,
        head_dim: int,
        max_batch: int,
        max_blocks_per_seq: int,
    ):
        lib = self._load_library()
        config = (
            int(num_blocks),
            int(block_size),
            int(n_head),
            int(head_dim),
            int(max_batch),
            int(max_blocks_per_seq),
        )

        if self._runtime is not None:
            existing = self._runtime_config
            if existing is None:
                self.close()
            else:
                compatible = (
                    existing[0] == config[0]
                    and existing[1] == config[1]
                    and existing[2] == config[2]
                    and existing[3] == config[3]
                    and existing[4] >= config[4]
                    and existing[5] >= config[5]
                )
                if not compatible:
                    self.close()

        if self._runtime is None:
            self._runtime = lib.paged_attention_runtime_create(*config)
            self._runtime_config = config
        return self._runtime

    def upload_layer_cache(self, key_cache, value_cache) -> None:
        lib = self._load_library()
        if self._runtime is None:
            raise ValueError("Runtime not initialized")

        key_cache_np = self._to_numpy(key_cache, datatype)
        value_cache_np = self._to_numpy(value_cache, datatype)
        if key_cache_np.shape != value_cache_np.shape:
            raise ValueError("key_cache and value_cache must have the same shape")

        lib.paged_attention_runtime_upload_layer_cache(
            self._runtime,
            key_cache_np,
            value_cache_np,
        )

    def copy_layer_cache_from_device(self, key_cache_ptr: int, value_cache_ptr: int) -> None:
        lib = self._load_library()
        if self._runtime is None:
            raise ValueError("Runtime not initialized")

        lib.paged_attention_runtime_copy_layer_cache_from_device(
            self._runtime,
            ctypes.c_void_p(int(key_cache_ptr)),
            ctypes.c_void_p(int(value_cache_ptr)),
        )

    def update_slot(self, block_id: int, slot_idx: int, key, value) -> None:
        lib = self._load_library()
        if self._runtime is None:
            raise ValueError("Runtime not initialized")

        key_np = self._to_numpy(key, datatype)
        value_np = self._to_numpy(value, datatype)
        if key_np.shape != value_np.shape:
            raise ValueError("key and value must have the same shape")

        lib.paged_attention_runtime_update_slot(
            self._runtime,
            int(block_id),
            int(slot_idx),
            key_np,
            value_np,
        )

    def update_slot_from_device(
        self,
        block_id: int,
        slot_idx: int,
        key_ptr: int,
        value_ptr: int,
    ) -> None:
        lib = self._load_library()
        if self._runtime is None:
            raise ValueError("Runtime not initialized")

        lib.paged_attention_runtime_update_slot_from_device(
            self._runtime,
            int(block_id),
            int(slot_idx),
            ctypes.c_void_p(int(key_ptr)),
            ctypes.c_void_p(int(value_ptr)),
        )

    def upload_block(self, block_id: int, key_block, value_block) -> None:
        lib = self._load_library()
        if self._runtime is None:
            raise ValueError("Runtime not initialized")

        key_block_np = self._to_numpy(key_block, datatype)
        value_block_np = self._to_numpy(value_block, datatype)
        if key_block_np.shape != value_block_np.shape:
            raise ValueError("key_block and value_block must have the same shape")

        lib.paged_attention_runtime_upload_block(
            self._runtime,
            int(block_id),
            key_block_np,
            value_block_np,
        )

    def update_metadata(self, block_tables, context_lens) -> None:
        lib = self._load_library()
        if self._runtime is None:
            raise ValueError("Runtime not initialized")

        block_tables_np = self._to_numpy(block_tables, np.int32)
        context_lens_np = self._to_numpy(context_lens, np.int32).reshape(-1)
        if block_tables_np.ndim != 2:
            raise ValueError("block_tables must be a rank-2 array")
        if context_lens_np.shape[0] != block_tables_np.shape[0]:
            raise ValueError("context_lens must match block_tables batch size")

        batch_size = int(block_tables_np.shape[0])
        max_blocks_per_seq = int(block_tables_np.shape[1]) if block_tables_np.ndim == 2 else 0
        lib.paged_attention_runtime_update_metadata(
            self._runtime,
            block_tables_np,
            context_lens_np,
            batch_size,
            max_blocks_per_seq,
        )

    def forward_device(
        self,
        query_ptr: int,
        output_ptr: int,
        batch_size: int,
        max_context_len: int,
    ) -> None:
        lib = self._load_library()
        if self._runtime is None:
            raise ValueError("Runtime not initialized")

        lib.paged_attention_runtime_forward_device(
            self._runtime,
            ctypes.c_void_p(int(query_ptr)),
            ctypes.c_void_p(int(output_ptr)),
            int(batch_size),
            int(max_context_len),
        )

    def contiguous_forward_device(
        self,
        output_ptr: int,
        query_ptr: int,
        key_cache_ptr: int,
        value_cache_ptr: int,
        context_lens_ptr: int,
        batch_size: int,
        n_head: int,
        head_dim: int,
        max_context_len: int,
    ) -> None:
        lib = self._load_library()
        lib.contiguous_attention_forward_device(
            ctypes.c_void_p(int(output_ptr)),
            ctypes.c_void_p(int(query_ptr)),
            ctypes.c_void_p(int(key_cache_ptr)),
            ctypes.c_void_p(int(value_cache_ptr)),
            ctypes.c_void_p(int(context_lens_ptr)),
            int(batch_size),
            int(n_head),
            int(head_dim),
            int(max_context_len),
        )

    def contiguous_update_slot_from_device(
        self,
        key_cache_ptr: int,
        value_cache_ptr: int,
        batch_idx: int,
        slot_idx: int,
        key_ptr: int,
        value_ptr: int,
        max_context_len: int,
        n_head: int,
        head_dim: int,
    ) -> None:
        lib = self._load_library()
        lib.contiguous_kv_update_slot_from_device(
            ctypes.c_void_p(int(key_cache_ptr)),
            ctypes.c_void_p(int(value_cache_ptr)),
            int(batch_idx),
            int(slot_idx),
            ctypes.c_void_p(int(key_ptr)),
            ctypes.c_void_p(int(value_ptr)),
            int(max_context_len),
            int(n_head),
            int(head_dim),
        )

    def close(self) -> None:
        lib = self._load_library()
        if self._runtime is not None:
            lib.paged_attention_runtime_destroy(self._runtime)
            self._runtime = None
            self._runtime_config = None

    def forward(
        self,
        query: Tensor,
        key_cache: Optional[Tensor] = None,
        value_cache: Optional[Tensor] = None,
        block_tables: Optional[Tensor] = None,
        context_lens: Optional[Tensor] = None,
        block_size: Optional[int] = None,
        max_context_len: int = 0,
    ) -> Tensor:
        """Run the PagedAttention CUDA kernel.

        In stateless mode, callers pass the full host-side cache and metadata.
        In runtime mode, callers first initialize / update the persistent runtime
        and then call `forward(query, max_context_len=...)`.
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

        batch_size, n_head, head_dim = query_np.shape
        output_np = np.zeros((batch_size, n_head, head_dim), dtype=datatype)

        use_runtime = (
            self._runtime is not None
            and key_cache is None
            and value_cache is None
            and block_tables is None
            and context_lens is None
        )

        if use_runtime:
            lib.paged_attention_runtime_forward(
                self._runtime,
                query_np,
                output_np,
                batch_size,
                int(max_context_len),
            )
        else:
            if key_cache is None or value_cache is None or block_tables is None or context_lens is None:
                raise ValueError(
                    "Stateless forward requires key_cache, value_cache, block_tables, and context_lens"
                )
            if block_size is None:
                raise ValueError("Stateless forward requires block_size")

            key_cache_np = self._to_numpy(key_cache, datatype)
            value_cache_np = self._to_numpy(value_cache, datatype)
            block_tables_np = self._to_numpy(block_tables, np.int32)
            context_lens_np = self._to_numpy(context_lens, np.int32).reshape(-1)

            if key_cache_np.shape[1] != block_size:
                raise ValueError("key_cache shape does not match block_size")
            if value_cache_np.shape != key_cache_np.shape:
                raise ValueError("value_cache must have the same shape as key_cache")
            if block_tables_np.shape[0] != batch_size:
                raise ValueError("block_tables batch dimension must match query batch size")
            if context_lens_np.shape[0] != batch_size:
                raise ValueError("context_lens length must match query batch size")

            max_blocks_per_seq = block_tables_np.shape[1] if block_tables_np.ndim == 2 else 0
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
        gpu_resident_kv: bool = False,
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
        if gpu_resident_kv and decode_backend != "cuda":
            raise ValueError("gpu_resident_kv requires decode_backend='cuda'")
        self.gpu_resident_kv = bool(gpu_resident_kv)
        self.compare_to_ref = compare_to_ref
        self.compare_tolerance = compare_tolerance
        self.last_decode_compare: Optional[Dict[str, float]] = None
        # CUDA kernel handle (lazy-loaded)
        self.q_proj = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.k_proj = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.v_proj = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.out_proj = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self._kernel: Optional[PagedAttentionKernel] = None
        self._runtime_block_manager_id: Optional[int] = None
        self._runtime_valid_block_ids: set[int] = set()

    @staticmethod
    def _device_ptr(t: Tensor) -> int:
        storage = t._tensor._storage
        if not numba.cuda.is_cuda_array(storage):
            raise RuntimeError(
                "GPU-resident KV requires MiniTorch tensors backed by CUDA device storage"
            )
        return int(storage.__cuda_array_interface__["data"][0])

    def _empty_like_attention_output(self, batch_size: int, backend: TensorBackend) -> Tensor:
        storage = numba.cuda.device_array(
            batch_size * self.n_head * self.head_dim,
            dtype=datatype,
        )
        return Tensor.make(
            storage,
            (batch_size, self.n_head, self.head_dim),
            backend=backend,
        )

    def _ensure_kernel_runtime(
        self,
        block_manager: BlockManager,
        seq_ids: List[int],
        required_runtime_batch_size: Optional[int] = None,
    ) -> None:
        if self._kernel is None:
            self._kernel = PagedAttentionKernel()

        previous_runtime = self._kernel._runtime
        self._kernel.ensure_runtime(
            num_blocks=block_manager.num_blocks,
            block_size=self.block_size,
            n_head=self.n_head,
            head_dim=self.head_dim,
            max_batch=max(required_runtime_batch_size or len(seq_ids), 1),
            max_blocks_per_seq=block_manager.num_blocks,
        )
        if self._kernel._runtime is not previous_runtime:
            self._runtime_valid_block_ids.clear()
            self._runtime_block_manager_id = None

    def _sync_runtime_blocks_for_sequences(
        self,
        block_manager: BlockManager,
        seq_ids: List[int],
        token_counts: List[int],
    ) -> None:
        if self.decode_backend != "cuda":
            return
        if self._kernel is None:
            raise ValueError("CUDA runtime must be initialized before syncing slots")
        if len(seq_ids) != len(token_counts):
            raise ValueError("token_counts must match seq_ids length")

        touched_block_ids = set()
        for seq_id, token_count in zip(seq_ids, token_counts):
            for token_idx in range(max(int(token_count), 0)):
                block_id, _ = block_manager.get_physical_location(seq_id, token_idx)
                touched_block_ids.add(block_id)

        self._upload_runtime_blocks(block_manager, list(touched_block_ids))

    def _upload_runtime_blocks(
        self,
        block_manager: BlockManager,
        block_ids: List[int],
    ) -> None:
        if self.decode_backend != "cuda":
            return
        if self.gpu_resident_kv:
            raise RuntimeError(
                "GPU-resident KV cache cannot be reconstructed from host BlockManager storage"
            )
        if self._kernel is None:
            raise ValueError("CUDA runtime must be initialized before uploading blocks")

        block_ids_to_upload = [
            block_id
            for block_id in sorted(set(int(block_id) for block_id in block_ids))
            if block_id not in self._runtime_valid_block_ids
        ]

        for block_id in block_ids_to_upload:
            self._kernel.upload_block(
                block_id,
                block_manager.key_cache[self.layer_id][block_id],
                block_manager.value_cache[self.layer_id][block_id],
            )
            self._runtime_valid_block_ids.add(block_id)

    def _ensure_runtime_synced_for_sequences(
        self,
        block_manager: BlockManager,
        seq_ids: List[int],
        token_counts: List[int],
        required_runtime_batch_size: Optional[int] = None,
    ) -> None:
        if self.decode_backend != "cuda":
            return

        self._ensure_kernel_runtime(
            block_manager,
            seq_ids,
            required_runtime_batch_size=required_runtime_batch_size,
        )
        current_manager_id = id(block_manager)
        if self._runtime_block_manager_id != current_manager_id:
            if self.gpu_resident_kv and any(int(token_count) > 0 for token_count in token_counts):
                raise RuntimeError(
                    "GPU-resident KV runtime lost its device cache; rerun prefill before decode"
                )
            self._runtime_valid_block_ids.clear()
            self._sync_runtime_blocks_for_sequences(block_manager, seq_ids, token_counts)
            self._runtime_block_manager_id = current_manager_id

    def _prefill_attention_kernel_with_prefix_batch(
        self,
        q: Tensor,
        suffix_keys: np.ndarray,
        suffix_values: np.ndarray,
        block_manager: BlockManager,
        seq_ids: List[int],
        prefix_token_count: int,
    ) -> Tensor:
        if self.decode_backend != "cuda":
            raise ValueError("Kernel prefill path requires decode_backend='cuda'")

        q_np = np.ascontiguousarray(q.to_numpy().astype(datatype, copy=False))
        if q_np.ndim != 4:
            raise ValueError("Expected q to have shape (batch, n_head, work_len, head_dim)")
        suffix_keys_np = np.ascontiguousarray(suffix_keys.astype(datatype, copy=False))
        suffix_values_np = np.ascontiguousarray(suffix_values.astype(datatype, copy=False))

        batch_size, n_head, work_len, head_dim = q_np.shape
        if batch_size != len(seq_ids):
            raise ValueError("Batch size must match seq_ids length")

        self._ensure_kernel_runtime(
            block_manager,
            seq_ids,
            required_runtime_batch_size=batch_size,
        )
        block_tables = block_manager.get_block_table_array(seq_ids)
        context_lens = np.full((batch_size,), prefix_token_count + work_len, dtype=np.int32)
        self._kernel.update_metadata(block_tables, context_lens)

        query_rows = np.ascontiguousarray(
            np.transpose(q_np, (0, 2, 1, 3)).reshape(batch_size * work_len, n_head, head_dim)
        )
        suffix_key_rows = np.ascontiguousarray(
            suffix_keys_np.reshape(batch_size * work_len, n_head, head_dim)
        )
        suffix_value_rows = np.ascontiguousarray(
            suffix_values_np.reshape(batch_size * work_len, n_head, head_dim)
        )
        output_rows = np.zeros_like(query_rows)

        lib = self._kernel._load_library()
        lib.paged_attention_runtime_prefill_with_prefix_forward(
            self._kernel._runtime,
            query_rows,
            suffix_key_rows,
            suffix_value_rows,
            output_rows,
            batch_size,
            work_len,
            prefix_token_count,
        )

        output_np = output_rows.reshape(batch_size, work_len, n_head, head_dim)
        output_np = np.ascontiguousarray(
            np.transpose(output_np, (0, 2, 1, 3)).astype(datatype)
        )
        return tensor_from_numpy(output_np, backend=q.backend)

    def _write_kv_batch_to_cache(
        self,
        block_manager: BlockManager,
        seq_ids: List[int],
        token_start: int,
        key_values: np.ndarray,
        value_values: np.ndarray,
    ) -> None:
        if key_values.shape != value_values.shape:
            raise ValueError("key_values and value_values must have matching shapes")
        if key_values.shape[0] != len(seq_ids):
            raise ValueError("Batch dimension must match seq_ids length")

        if self.decode_backend == "cuda":
            self._ensure_kernel_runtime(block_manager, seq_ids)
            current_manager_id = id(block_manager)
            if self._runtime_block_manager_id != current_manager_id:
                self._runtime_valid_block_ids.clear()

        touched_block_ids = set()
        for batch_idx, seq_id in enumerate(seq_ids):
            for local_idx in range(key_values.shape[1]):
                token_index = token_start + local_idx
                block_id, slot_idx = block_manager.get_physical_location(seq_id, token_index)
                block_manager.write_kv_slot(
                    block_id,
                    slot_idx,
                    key_values[batch_idx, local_idx],
                    value_values[batch_idx, local_idx],
                    layer=self.layer_id,
                )
                touched_block_ids.add(block_id)

        if self.decode_backend == "cuda":
            self._upload_runtime_blocks(block_manager, list(touched_block_ids))
            self._runtime_block_manager_id = id(block_manager)

    def _write_kv_batch_to_device_cache(
        self,
        block_manager: BlockManager,
        seq_ids: List[int],
        token_start: int,
        key_values: Tensor,
        value_values: Tensor,
    ) -> None:
        if not self.gpu_resident_kv:
            raise RuntimeError("Device KV writes require gpu_resident_kv=True")
        if key_values.shape != value_values.shape:
            raise ValueError("key_values and value_values must have matching shapes")
        if key_values.shape[0] != len(seq_ids):
            raise ValueError("Batch dimension must match seq_ids length")
        if key_values.shape[2:] != (self.n_head, self.head_dim):
            raise ValueError(
                f"Expected K/V shape (batch, tokens, {self.n_head}, {self.head_dim})"
            )

        self._ensure_kernel_runtime(block_manager, seq_ids)
        key_ptr = self._device_ptr(key_values)
        value_ptr = self._device_ptr(value_values)
        row_elems = self.n_head * self.head_dim
        itemsize = np.dtype(datatype).itemsize

        touched_block_ids = set()
        for batch_idx, seq_id in enumerate(seq_ids):
            for local_idx in range(key_values.shape[1]):
                token_index = token_start + local_idx
                block_id, slot_idx = block_manager.get_physical_location(seq_id, token_index)
                offset = (batch_idx * key_values.shape[1] + local_idx) * row_elems * itemsize
                self._kernel.update_slot_from_device(
                    block_id,
                    slot_idx,
                    key_ptr + offset,
                    value_ptr + offset,
                )
                touched_block_ids.add(block_id)

        self._runtime_valid_block_ids.update(touched_block_ids)
        self._runtime_block_manager_id = id(block_manager)

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
        context_lens = [
            block_manager.get_context_len(seq_id) for seq_id in seq_ids
        ]
        if self.gpu_resident_kv:
            self._ensure_kernel_runtime(block_manager, seq_ids)
            if self._runtime_block_manager_id != id(block_manager):
                raise RuntimeError(
                    "GPU-resident KV runtime has no cache for this BlockManager; run prefill first"
                )
        else:
            self._ensure_runtime_synced_for_sequences(
                block_manager,
                seq_ids,
                context_lens,
            )
        block_tables = block_manager.get_block_table_array(seq_ids)
        context_lens_np = np.array(context_lens, dtype=np.int32)
        max_context_len = int(context_lens_np.max()) if len(context_lens_np) > 0 else 0
        self._kernel.update_metadata(block_tables, context_lens_np)
        if self.gpu_resident_kv:
            batch_size = len(seq_ids)
            q_device = q.contiguous().view(batch_size, self.n_head, self.head_dim)
            output = self._empty_like_attention_output(batch_size, q.backend)
            self._kernel.forward_device(
                self._device_ptr(q_device),
                self._device_ptr(output),
                batch_size,
                max_context_len,
            )
            return output.view(batch_size, self.n_head, 1, self.head_dim)
        return self._kernel.forward(
            q,
            max_context_len=max_context_len,
        )

    def _gather_cached_prefix_kv_batch(
        self,
        block_manager: BlockManager,
        seq_ids: List[int],
        prefix_token_count: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        key_prefix = np.zeros(
            (len(seq_ids), prefix_token_count, self.n_head, self.head_dim),
            dtype=datatype,
        )
        value_prefix = np.zeros_like(key_prefix)

        for batch_idx, seq_id in enumerate(seq_ids):
            for token_idx in range(prefix_token_count):
                block_id, slot_idx = block_manager.get_physical_location(seq_id, token_idx)
                key_prefix[batch_idx, token_idx] = block_manager.key_cache[self.layer_id][block_id, slot_idx]
                value_prefix[batch_idx, token_idx] = block_manager.value_cache[self.layer_id][block_id, slot_idx]

        return key_prefix, value_prefix

    def forward_prefill_with_prefix_batch(
        self,
        x: Tensor,
        block_manager: BlockManager,
        seq_ids: List[int],
        prefix_token_count: int,
        *,
        cached_token_count: Optional[int] = None,
        write_kv_to_cache: bool = True,
    ) -> Tensor:
        """Prefix-aware prefill for a batch that shares one prefix length."""
        batch_size, work_len, _ = x.shape
        if batch_size != len(seq_ids):
            raise ValueError("Batch size must match number of seq_ids")
        if work_len <= 0:
            raise ValueError("forward_prefill_with_prefix_batch expects at least one token")
        if self.gpu_resident_kv:
            raise NotImplementedError(
                "Prefix-cache prefill is not implemented for GPU-resident KV mode"
            )

        flat_x = x.contiguous().view(batch_size * work_len, self.n_embd)
        q = self.q_proj(flat_x).view(batch_size, work_len, self.n_head, self.head_dim)
        k = self.k_proj(flat_x).view(batch_size, work_len, self.n_head, self.head_dim)
        v = self.v_proj(flat_x).view(batch_size, work_len, self.n_head, self.head_dim)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        k_work = k.permute(0, 2, 1, 3).to_numpy()
        v_work = v.permute(0, 2, 1, 3).to_numpy()

        runtime_token_counts = [cached_token_count or prefix_token_count] * len(seq_ids)
        required_runtime_batch_size = batch_size * work_len
        if self.decode_backend == "cuda" and any(token_count > 0 for token_count in runtime_token_counts):
            self._ensure_runtime_synced_for_sequences(
                block_manager,
                seq_ids,
                runtime_token_counts,
                required_runtime_batch_size=required_runtime_batch_size,
            )
        elif self.decode_backend == "cuda":
            self._ensure_kernel_runtime(
                block_manager,
                seq_ids,
                required_runtime_batch_size=required_runtime_batch_size,
            )

        if write_kv_to_cache:
            self._write_kv_batch_to_cache(
                block_manager,
                seq_ids,
                prefix_token_count,
                k_work,
                v_work,
            )

        if self.decode_backend == "cuda":
            output = self._prefill_attention_kernel_with_prefix_batch(
                q,
                k_work,
                v_work,
                block_manager,
                seq_ids,
                prefix_token_count,
            )
            output = output.permute(0, 2, 1, 3).contiguous().view(
                batch_size * work_len, self.n_embd
            )
            return self.out_proj(output).view(batch_size, work_len, self.n_embd)

        prefix_keys, prefix_values = self._gather_cached_prefix_kv_batch(
            block_manager,
            seq_ids,
            prefix_token_count,
        )
        full_keys_np = np.concatenate([prefix_keys, k_work], axis=1)
        full_values_np = np.concatenate([prefix_values, v_work], axis=1)
        total_context = full_keys_np.shape[1]

        key_tensor = tensor_from_numpy(
            np.ascontiguousarray(
                np.transpose(full_keys_np, (0, 2, 1, 3)).astype(datatype)
            ),
            backend=x.backend,
        )
        value_tensor = tensor_from_numpy(
            np.ascontiguousarray(
                np.transpose(full_values_np, (0, 2, 1, 3)).astype(datatype)
            ),
            backend=x.backend,
        )

        mask_np = np.full((work_len, total_context), -1e9, dtype=datatype)
        for local_idx in range(work_len):
            mask_np[local_idx, : prefix_token_count + local_idx + 1] = 0.0
        mask = tensor_from_numpy(
            mask_np.reshape(1, 1, work_len, total_context),
            backend=x.backend,
        )

        output = standard_attention(q, key_tensor, value_tensor, mask)
        output = output.permute(0, 2, 1, 3).contiguous().view(
            batch_size * work_len, self.n_embd
        )
        return self.out_proj(output).view(batch_size, work_len, self.n_embd)

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

        if self.gpu_resident_kv:
            k_cache_values = k.permute(0, 2, 1, 3).contiguous()
            v_cache_values = v.permute(0, 2, 1, 3).contiguous()
            self._write_kv_batch_to_device_cache(
                block_manager,
                seq_ids,
                0,
                k_cache_values,
                v_cache_values,
            )
        else:
            k_cache_values = k.permute(0, 2, 1, 3).to_numpy()
            v_cache_values = v.permute(0, 2, 1, 3).to_numpy()

            self._write_kv_batch_to_cache(
                block_manager,
                seq_ids,
                0,
                k_cache_values,
                v_cache_values,
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

        if self.gpu_resident_kv:
            k_new_device = k.permute(0, 2, 1, 3).contiguous()
            v_new_device = v.permute(0, 2, 1, 3).contiguous()
            self._ensure_kernel_runtime(block_manager, seq_ids)
        else:
            k_new = k.permute(0, 2, 1, 3).to_numpy()[:, 0, :, :]
            v_new = v.permute(0, 2, 1, 3).to_numpy()[:, 0, :, :]

        if self.decode_backend == "cuda" and not self.gpu_resident_kv:
            prior_context_lens = [
                block_manager.get_context_len(seq_id) - 1 for seq_id in seq_ids
            ]
            self._ensure_runtime_synced_for_sequences(
                block_manager,
                seq_ids,
                prior_context_lens,
            )

        for batch_idx, seq_id in enumerate(seq_ids):
            token_index = block_manager.get_context_len(seq_id) - 1
            block_id, slot_idx = block_manager.get_physical_location(seq_id, token_index)
            if self.gpu_resident_kv:
                row_elems = self.n_head * self.head_dim
                itemsize = np.dtype(datatype).itemsize
                offset = batch_idx * row_elems * itemsize
                self._kernel.update_slot_from_device(
                    block_id,
                    slot_idx,
                    self._device_ptr(k_new_device) + offset,
                    self._device_ptr(v_new_device) + offset,
                )
                self._runtime_valid_block_ids.add(block_id)
                self._runtime_block_manager_id = id(block_manager)
            else:
                block_manager.write_kv_slot(
                    block_id,
                    slot_idx,
                    k_new[batch_idx],
                    v_new[batch_idx],
                    layer=self.layer_id,
                )
                if self.decode_backend == "cuda":
                    self._kernel.update_slot(
                        block_id,
                        slot_idx,
                        k_new[batch_idx],
                        v_new[batch_idx],
                    )
                    self._runtime_valid_block_ids.add(block_id)
                    self._runtime_block_manager_id = id(block_manager)

        if self.decode_backend == "cuda":
            output = self._decode_attention_kernel(q, block_manager, seq_ids)
            if self.compare_to_ref and not self.gpu_resident_kv:
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

    def close_decode_runtime(self) -> None:
        if self._kernel is not None:
            self._kernel.close()
        self._runtime_block_manager_id = None
        self._runtime_valid_block_ids.clear()
