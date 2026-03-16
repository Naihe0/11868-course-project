"""
Block-based memory manager for KV cache in PagedAttention.

Manages KV cache memory in fixed-size blocks (pages) that do not need
to be contiguous, inspired by OS virtual memory paging.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple

from .tensor import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_BLOCK_SIZE = 16  # Number of tokens per block


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

class KVBlock:
    """A single fixed-size block that stores key and value vectors for a
    contiguous subsequence of tokens.

    Attributes:
        block_id: Unique identifier for this physical block.
        block_size: Maximum number of tokens this block can hold.
        ref_count: Number of sequences currently referencing this block.
        key_data: Tensor storage for key vectors  (block_size, n_head, head_dim).
        value_data: Tensor storage for value vectors (block_size, n_head, head_dim).
        num_filled: How many token slots are currently occupied.
    """

    def __init__(self, block_id: int, block_size: int, n_head: int, head_dim: int):
        self.block_id = block_id
        self.block_size = block_size
        self.ref_count = 1
        self.num_filled = 0
        self.n_head = n_head
        self.head_dim = head_dim
        # Placeholder – actual GPU tensors will be allocated during implementation
        self.key_data: Optional[np.ndarray] = None
        self.value_data: Optional[np.ndarray] = None

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
    """Maps a sequence's logical block indices to physical KVBlock ids.

    Each sequence in a batch maintains its own BlockTable, analogous to a
    page table in virtual memory.
    """

    def __init__(self, seq_id: int):
        self.seq_id = seq_id
        self.block_ids: List[int] = []  # Ordered list of physical block ids

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
    """Manages allocation and deallocation of KV cache blocks.

    Maintains a pool of physical blocks and assigns them to sequences
    on demand.  Supports:
      - Allocating new blocks for a sequence.
      - Freeing blocks when a sequence completes.
      - Querying available capacity.

    Args:
        num_blocks: Total number of physical blocks in the pool.
        block_size: Number of tokens each block can store.
        n_head: Number of attention heads.
        head_dim: Dimension of each attention head.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int = DEFAULT_BLOCK_SIZE,
        n_head: int = 8,
        head_dim: int = 64,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.n_head = n_head
        self.head_dim = head_dim

        # Physical block pool
        self.blocks: Dict[int, KVBlock] = {}
        self.free_block_ids: List[int] = list(range(num_blocks))

        # Mapping from sequence id -> BlockTable
        self.block_tables: Dict[int, BlockTable] = {}

    # ----- Capacity queries ------------------------------------------------

    @property
    def num_free_blocks(self) -> int:
        """Return the number of unallocated blocks."""
        return len(self.free_block_ids)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - self.num_free_blocks

    def can_allocate(self, num_required: int = 1) -> bool:
        """Check whether *num_required* blocks can be allocated."""
        return self.num_free_blocks >= num_required

    # ----- Allocation / deallocation ---------------------------------------

    def allocate_block(self) -> KVBlock:
        """Allocate a single physical block from the free pool.

        Returns:
            The newly allocated KVBlock.

        Raises:
            RuntimeError: If no free blocks are available.
        """
        # TODO: Implement block allocation
        raise NotImplementedError

    def allocate_blocks_for_sequence(self, seq_id: int, num_tokens: int) -> BlockTable:
        """Allocate enough blocks to hold *num_tokens* for a new sequence.

        Creates a new BlockTable for the sequence and allocates
        ceil(num_tokens / block_size) blocks.

        Args:
            seq_id: Unique identifier for the sequence.
            num_tokens: Number of tokens (e.g. prompt length).

        Returns:
            The BlockTable mapping for this sequence.
        """
        # TODO: Implement sequence block allocation
        raise NotImplementedError

    def append_token_to_sequence(self, seq_id: int) -> KVBlock:
        """Append a single new token to a sequence's KV cache.

        If the last block for this sequence is full, allocate a new block.

        Args:
            seq_id: Sequence identifier.

        Returns:
            The KVBlock where the new token should be written.
        """
        # TODO: Implement single-token append
        raise NotImplementedError

    def free_sequence(self, seq_id: int) -> None:
        """Free all blocks associated with *seq_id*.

        Decrements reference counts and returns blocks with zero
        references back to the free pool.

        Args:
            seq_id: Sequence to free.
        """
        # TODO: Implement sequence deallocation
        raise NotImplementedError

    # ----- Introspection ---------------------------------------------------

    def get_block_table(self, seq_id: int) -> BlockTable:
        """Return the BlockTable for a given sequence."""
        return self.block_tables[seq_id]

    def get_block_table_tensor(self, seq_id: int) -> List[int]:
        """Return block ids as a flat list (for passing to CUDA kernel)."""
        return self.block_tables[seq_id].block_ids

    def compute_fragmentation(self) -> Dict[str, float]:
        """Compute internal and external fragmentation metrics.

        Returns:
            Dict with 'internal' and 'external' fragmentation ratios.
        """
        # TODO: Implement fragmentation metrics
        raise NotImplementedError

    def __repr__(self) -> str:
        return (
            f"BlockManager(total={self.num_blocks}, free={self.num_free_blocks}, "
            f"sequences={len(self.block_tables)})"
        )
