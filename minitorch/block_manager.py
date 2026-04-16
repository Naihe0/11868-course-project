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
import math

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
        num_layers: int = None,
    ):
        if num_layers is None:
            raise ValueError("BlockManager requires an explicit num_layers")
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
        self.num_layers = num_layers
        self.key_cache = [np.zeros(cache_shape, dtype=cache_dtype) for _ in range(num_layers)]
        self.value_cache = [np.zeros(cache_shape, dtype=cache_dtype) for _ in range(num_layers)]

    # ----- Capacity queries ------------------------------------------------

    @property
    def num_free_blocks(self) -> int:
        """Return the number of unallocated physical blocks."""
        return len(self.free_block_ids)
    def get_key_cache(self, layer: int) -> np.ndarray:
        return self.key_cache[layer]
    def get_value_cache(self, layer: int) -> np.ndarray:
        return self.value_cache[layer]
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
 
        if self.num_free_blocks == 0:
            raise RuntimeError("No free blocks available")

        block_id = self.free_block_ids.pop(0)
        block = self.blocks[block_id]
        block.ref_count = 1
        block.num_filled = 0
        for layer in range(self.num_layers):
            self.key_cache[layer][block_id].fill(0)
            self.value_cache[layer][block_id].fill(0)
        return block

    def allocate_blocks_for_sequence(self, seq_id: int, num_tokens: int) -> BlockTable:
        """Allocate enough blocks to hold `num_tokens` tokens for a sequence.

        This should create a BlockTable and initialize `context_lens[seq_id]`.
        The actual K/V vectors will later be written into `key_cache` and
        `value_cache`, not into the KVBlock objects themselves.
        """
        if num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")
        if seq_id in self.block_tables:
            raise ValueError(f"Sequence {seq_id} already has allocated blocks")

        num_blocks = math.ceil(num_tokens / self.block_size)
        if not self.can_allocate(num_blocks):
            raise RuntimeError(
                f"Not enough free blocks for sequence {seq_id}: "
                f"need {num_blocks}, have {self.num_free_blocks}"
            )

        block_table = BlockTable(seq_id)
        remaining_tokens = num_tokens
        allocated_block_ids = []
        try:
            for _ in range(num_blocks):
                block = self.allocate_block()
                allocated_block_ids.append(block.block_id)
                block_table.append_block(block.block_id)
                if remaining_tokens > self.block_size:
                    block.num_filled = self.block_size
                else:
                    block.num_filled = remaining_tokens
                remaining_tokens -= block.num_filled
        except Exception:
            for block_id in allocated_block_ids:
                self.blocks[block_id].ref_count = 0
                self.blocks[block_id].num_filled = 0
                for layer_id in range(self.num_layers):
                    self.key_cache[layer_id][block_id].fill(0)
                    self.value_cache[layer_id][block_id].fill(0)
                self.free_block_ids.append(block_id)
            self.free_block_ids.sort()
            raise

        self.context_lens[seq_id] = num_tokens
        self.block_tables[seq_id] = block_table
        return block_table

    def append_token_to_sequence(self, seq_id: int) -> KVBlock:
        """Reserve one new token slot for an existing sequence.

        If the current tail block is full, allocate a new block and append its
        id to the sequence's BlockTable.
        """
        block_table = self.block_tables[seq_id]
        if not block_table.block_ids:
            block = self.allocate_block()
            block_id = block.block_id
            block_table.append_block(block.block_id)
        else:
            block_id = block_table.block_ids[-1]
            if self.blocks[block_id].is_full:
                block = self.allocate_block()
                block_id = block.block_id
                block_table.append_block(block.block_id)
        self.blocks[block_id].num_filled += 1
        self.context_lens[seq_id] += 1
        return self.blocks[block_id]

    def free_sequence(self, seq_id: int) -> None:
        """Free all blocks associated with `seq_id`.

        This should update ref counts, return reusable block ids to the free
        pool, and optionally clear the corresponding global cache slices.
        """
        block_table = self.block_tables[seq_id]
        for block_id in block_table.block_ids:
            self.blocks[block_id].ref_count -= 1
            if self.blocks[block_id].ref_count == 0:
                self.blocks[block_id].num_filled = 0
                for layer_id in range(self.num_layers):
                    self.key_cache[layer_id][block_id].fill(0)
                    self.value_cache[layer_id][block_id].fill(0)
                self.free_block_ids.append(block_id)
        del self.block_tables[seq_id]
        del self.context_lens[seq_id]

    # ----- Cache access helpers -------------------------------------------

    def get_context_len(self, seq_id: int) -> int:
        """Return the current token count for a sequence."""
        return self.context_lens[seq_id]

    def get_physical_location(self, seq_id: int, token_index: int) -> Tuple[int, int]:
        """Map a token position to `(block_id, slot_idx)` in the global cache."""
        block_table = self.block_tables[seq_id]
        block_id = block_table.block_ids[token_index // self.block_size]
        slot_idx = token_index % self.block_size
        return block_id, slot_idx

    def write_kv_slot(
        self,
        block_id: int,
        slot_idx: int,
        key: np.ndarray,
        value: np.ndarray,
        layer: int = 0,
    ) -> None:
        """Write one token's K/V vectors into the global cache."""
        if key.shape != (self.n_head, self.head_dim):
            raise ValueError(f"Key shape must be ({self.n_head}, {self.head_dim})")
        if value.shape != (self.n_head, self.head_dim):
            raise ValueError(f"Value shape must be ({self.n_head}, {self.head_dim})")
        self.key_cache[layer][block_id, slot_idx, :, :] = key
        self.value_cache[layer][block_id, slot_idx, :, :] = value

    def write_token_kv(
        self,
        seq_id: int,
        key: np.ndarray,
        value: np.ndarray,
        layer: int,
    ) -> Tuple[int, int]:
        """Append one token and write its K/V vectors.

        Expected flow:
          1. Reserve a slot via `append_token_to_sequence`
          2. Determine the slot index within that block
          3. Call `write_kv_slot`
          4. Return `(block_id, slot_idx)`
        """
        # TODO: Implement append-and-write helper.
        block = self.append_token_to_sequence(seq_id)
        slot_idx = block.num_filled - 1
        self.write_kv_slot(block.block_id, slot_idx, key, value, layer)
        return block.block_id, slot_idx

    def get_block_table_array(
        self, seq_ids: List[int], pad_value: int = -1
    ) -> np.ndarray:
        """Build a padded block-table array suitable for kernel input."""
        block_tables = [self.block_tables[seq_id] for seq_id in seq_ids]
        max_blocks_per_seq = max(
            (block_table.num_blocks for block_table in block_tables),
            default=0,
        )
        out = np.full(
            (len(seq_ids), max_blocks_per_seq),
            pad_value,
            dtype=np.int32,
        )
        for row, block_table in enumerate(block_tables):
            out[row, : block_table.num_blocks] = np.array(
                block_table.block_ids,
                dtype=np.int32,
            )
        return out

    # ----- Introspection ---------------------------------------------------

    def get_block_table(self, seq_id: int) -> BlockTable:
        """Return the BlockTable for a given sequence."""
        return self.block_tables[seq_id]

    def get_block_table_tensor(self, seq_id: int) -> List[int]:
        """Return block ids as a flat list (for passing to a kernel)."""
        return self.block_tables[seq_id].block_ids

    def compute_fragmentation(self) -> Dict[str, float]:
        """Compute internal and external fragmentation metrics."""
        used_blocks = [
            block for block in self.blocks.values() if block.num_filled > 0
        ]

        if not used_blocks:
            return {"internal": 0.0, "external": 0.0}

        allocated_capacity = len(used_blocks) * self.block_size
        used_slots = sum(block.num_filled for block in used_blocks)
        wasted_slots = allocated_capacity - used_slots
        internal_fragmentation = wasted_slots / allocated_capacity

        used_block_ids = sorted(block.block_id for block in used_blocks)
        if len(used_block_ids) <= 1 or self.num_free_blocks == 0:
            external_fragmentation = 0.0
        else:
            low = used_block_ids[0]
            high = used_block_ids[-1]
            free_between = sum(
                1
                for block_id in range(low, high + 1)
                if self.blocks[block_id].num_filled == 0
            )
            external_fragmentation = free_between / self.num_free_blocks

        return {
            "internal": internal_fragmentation,
            "external": external_fragmentation,
        }

    def __repr__(self) -> str:
        return (
            f"BlockManager(total={self.num_blocks}, free={self.num_free_blocks}, "
            f"sequences={len(self.block_tables)})"
        )
