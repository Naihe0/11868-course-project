"""
Block-based memory manager for KV cache in PagedAttention.

This version keeps the architecture aligned with paged attention:
the BlockManager owns global key/value caches with shape
``(num_blocks, block_size, n_head, head_dim)``, while each KVBlock stores
only metadata and each sequence keeps a BlockTable of physical block ids.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_BLOCK_SIZE = 16  # Number of tokens per block
CACHE_DTYPE = np.float32


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


class KVBlock:
    """Metadata for a physical KV cache block.

    The actual K/V vectors live in BlockManager.key_cache/value_cache.
    """

    def __init__(self, block_id: int, block_size: int, n_head: int, head_dim: int):
        self.block_id = block_id
        self.block_size = block_size
        self.ref_count = 0
        self.num_filled = 0
        self.n_head = n_head
        self.head_dim = head_dim

    @property
    def is_full(self) -> bool:
        return self.num_filled >= self.block_size

    @property
    def num_empty_slots(self) -> int:
        return self.block_size - self.num_filled

    def __repr__(self) -> str:
        return (
            f"KVBlock(id={self.block_id}, filled={self.num_filled}/"
            f"{self.block_size}, refs={self.ref_count})"
        )


class BlockTable:
    """Maps a sequence's logical block indices to physical block ids."""

    def __init__(self, seq_id: int, block_ids: Optional[List[int]] = None):
        self.seq_id = seq_id
        self.block_ids: List[int] = list(block_ids or [])

    @property
    def num_blocks(self) -> int:
        return len(self.block_ids)

    def append_block(self, block_id: int) -> None:
        self.block_ids.append(block_id)

    def __repr__(self) -> str:
        return f"BlockTable(seq={self.seq_id}, blocks={self.block_ids})"


# ---------------------------------------------------------------------------
# Block Manager
# ---------------------------------------------------------------------------


class BlockManager:
    """Manages global KV cache storage and per-sequence block tables."""

    def __init__(
        self,
        num_blocks: int,
        block_size: int = DEFAULT_BLOCK_SIZE,
        n_head: int = 8,
        head_dim: int = 64,
        cache_dtype: np.dtype = CACHE_DTYPE,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.n_head = n_head
        self.head_dim = head_dim
        self.cache_dtype = cache_dtype

        # Physical block metadata pool.
        self.blocks: Dict[int, KVBlock] = {
            block_id: KVBlock(block_id, block_size, n_head, head_dim)
            for block_id in range(num_blocks)
        }
        self.free_block_ids: List[int] = list(range(num_blocks))

        # Per-sequence metadata.
        self.block_tables: Dict[int, BlockTable] = {}
        self.context_lens: Dict[int, int] = {}

        # Global K/V caches indexed by physical block id.
        cache_shape = (num_blocks, block_size, n_head, head_dim)
        self.key_cache = np.zeros(cache_shape, dtype=cache_dtype)
        self.value_cache = np.zeros(cache_shape, dtype=cache_dtype)

    # ----- Capacity queries ------------------------------------------------

    @property
    def num_free_blocks(self) -> int:
        """Return the number of unallocated physical blocks."""
        return len(self.free_block_ids)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - self.num_free_blocks

    def can_allocate(self, num_required: int = 1) -> bool:
        """Check whether `num_required` more blocks can be allocated."""
        return self.num_free_blocks >= num_required

    # ----- Allocation / deallocation ---------------------------------------

    def allocate_block(self) -> KVBlock:
        """Allocate a single physical block from the free pool.

        This should:
          1. Pop a block id from `free_block_ids`
          2. Reset that block's metadata (`ref_count`, `num_filled`)
          3. Optionally zero the corresponding cache slices
          4. Return the KVBlock metadata object
        """
        # TODO: Implement block allocation for the global-cache architecture.
        raise NotImplementedError

    def allocate_blocks_for_sequence(self, seq_id: int, num_tokens: int) -> BlockTable:
        """Allocate enough blocks to hold `num_tokens` tokens for a sequence.

        This should create a BlockTable and initialize `context_lens[seq_id]`.
        The actual K/V vectors will later be written into `key_cache` and
        `value_cache`, not into the KVBlock objects themselves.
        """
        # TODO: Implement sequence block allocation.
        raise NotImplementedError

    def append_token_to_sequence(self, seq_id: int) -> KVBlock:
        """Reserve one new token slot for an existing sequence.

        If the current tail block is full, allocate a new block and append its
        id to the sequence's BlockTable.
        """
        # TODO: Implement single-token append.
        raise NotImplementedError

    def free_sequence(self, seq_id: int) -> None:
        """Free all blocks associated with `seq_id`.

        This should update ref counts, return reusable block ids to the free
        pool, and optionally clear the corresponding global cache slices.
        """
        # TODO: Implement sequence deallocation.
        raise NotImplementedError

    # ----- Cache access helpers -------------------------------------------

    def get_context_len(self, seq_id: int) -> int:
        """Return the current token count for a sequence."""
        return self.context_lens[seq_id]

    def get_physical_location(self, seq_id: int, token_index: int) -> Tuple[int, int]:
        """Map a token position to `(block_id, slot_idx)` in the global cache."""
        # TODO: Implement logical-position -> physical-cache mapping.
        raise NotImplementedError

    def write_kv_slot(
        self,
        block_id: int,
        slot_idx: int,
        key: np.ndarray,
        value: np.ndarray,
    ) -> None:
        """Write one token's K/V vectors into the global cache."""
        # TODO: Validate shapes and write into `key_cache` / `value_cache`.
        raise NotImplementedError

    def write_token_kv(
        self,
        seq_id: int,
        key: np.ndarray,
        value: np.ndarray,
    ) -> Tuple[int, int]:
        """Append one token and write its K/V vectors.

        Expected flow:
          1. Reserve a slot via `append_token_to_sequence`
          2. Determine the slot index within that block
          3. Call `write_kv_slot`
          4. Return `(block_id, slot_idx)`
        """
        # TODO: Implement append-and-write helper.
        raise NotImplementedError

    def get_block_table_array(
        self, seq_ids: List[int], pad_value: int = -1
    ) -> np.ndarray:
        """Build a padded block-table array suitable for kernel input."""
        # TODO: Implement padded block-table export for CUDA / reference paths.
        raise NotImplementedError

    # ----- Introspection ---------------------------------------------------

    def get_block_table(self, seq_id: int) -> BlockTable:
        """Return the BlockTable for a given sequence."""
        return self.block_tables[seq_id]

    def get_block_table_tensor(self, seq_id: int) -> List[int]:
        """Return block ids as a flat list (for passing to a kernel)."""
        return self.block_tables[seq_id].block_ids

    def compute_fragmentation(self) -> Dict[str, float]:
        """Compute internal and external fragmentation metrics."""
        # TODO: Implement fragmentation metrics for the paged-cache design.
        raise NotImplementedError

    def __repr__(self) -> str:
        return (
            f"BlockManager(total={self.num_blocks}, free={self.num_free_blocks}, "
            f"sequences={len(self.block_tables)})"
        )
