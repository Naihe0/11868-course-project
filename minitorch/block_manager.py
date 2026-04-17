"""
Block-based memory manager for KV cache in PagedAttention.

This version keeps the architecture aligned with paged attention:
the BlockManager owns global key/value caches with shape
``(num_blocks, block_size, n_head, head_dim)``, while each KVBlock stores
only metadata and each sequence keeps a BlockTable of physical block ids.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
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
        # Prefix-cache metadata. A block may outlive a sequence if it has been
        # published into the prefix cache and can be reused by future requests.
        self.block_hash: Optional[Tuple[object, ...]] = None
        self.is_prefix_cached = False

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


@dataclass
class PrefixCacheMatch:
    """Result of a prefix-cache lookup.

    Attributes:
        block_ids: Physical block ids that can be reused for the request prefix.
        cached_token_count: Number of prefix tokens covered by those blocks.
    """

    block_ids: List[int]
    cached_token_count: int


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
        # Prefix-cache bookkeeping.
        self.block_hash_to_block_id: Dict[Tuple[object, ...], int] = {}
        self.block_id_to_hash: Dict[int, Tuple[object, ...]] = {}
        self.cached_block_ids: set[int] = set()
        self.cached_free_lru: "OrderedDict[int, None]" = OrderedDict()
        self.seq_prefix_cache_info: Dict[int, PrefixCacheMatch] = {}

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
        return (self.num_free_blocks + len(self.cached_free_lru)) >= num_required

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
            self.evict_cached_block_if_needed()
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

    # ----- Prefix-cache helpers -------------------------------------------

    def compute_block_hash_chain(
        self,
        token_ids: np.ndarray,
        extra_hash: object = None,
    ) -> List[Tuple[object, ...]]:
        """Compute a vLLM-style full-block hash chain for a token sequence.

        Expected behavior for the final implementation:
          1. Split `token_ids` into full blocks of size `self.block_size`
          2. Skip any tail block that is not completely full
          3. For block 0, compute a hash from `(None, block_tokens, extra_hash)`
          4. For block i>0, compute a hash from `(prev_hash, block_tokens, extra_hash)`
          5. Return one hash per full block in prefix order

        This helper only computes logical block hashes. It does not allocate,
        publish, or mutate any block metadata.
        """
        token_ids = np.asarray(token_ids)
        if token_ids.ndim != 1:
            raise ValueError("token_ids must be a rank-1 array")

        hash_chain: List[Tuple[object, ...]] = []
        prev_hash: Optional[str] = None
        extra_hash_repr = repr(extra_hash)

        num_full_blocks = token_ids.shape[0] // self.block_size
        for block_idx in range(num_full_blocks):
            start = block_idx * self.block_size
            end = start + self.block_size
            block_tokens = tuple(int(token) for token in token_ids[start:end].tolist())

            hasher = hashlib.sha256()
            hasher.update(b"parent=")
            hasher.update((prev_hash if prev_hash is not None else "<ROOT>").encode("utf-8"))
            hasher.update(b"|tokens=")
            hasher.update(repr(block_tokens).encode("utf-8"))
            hasher.update(b"|extra=")
            hasher.update(extra_hash_repr.encode("utf-8"))

            curr_hash = hasher.hexdigest()
            hash_chain.append((curr_hash,))
            prev_hash = curr_hash

        return hash_chain

    def lookup_prefix_blocks(
        self,
        token_ids: np.ndarray,
        extra_hash: object = None,
    ) -> PrefixCacheMatch:
        """Find the longest cached prefix that can be reused.

        Expected behavior for the final implementation:
          1. Call `compute_block_hash_chain(...)`
          2. Walk the chain in order until the first cache miss
          3. Collect matching physical block ids from `block_hash_to_block_id`
          4. Return both the block ids and the number of tokens they cover

        Notes:
          - Only full blocks should be considered cacheable / reusable
          - The returned prefix should be contiguous in logical order
          - Tail partial blocks should not be included in the match
        """
        hash_chain = self.compute_block_hash_chain(token_ids, extra_hash)
        matched_block_ids: List[int] = []

        for block_hash in hash_chain:
            block_id = self.block_hash_to_block_id.get(block_hash)
            if block_id is None:
                break
            matched_block_ids.append(block_id)

        return PrefixCacheMatch(
            block_ids=matched_block_ids,
            cached_token_count=len(matched_block_ids) * self.block_size,
        )

    def allocate_sequence_with_prefix(
        self,
        seq_id: int,
        num_tokens: int,
        prefix_block_ids: List[int],
    ) -> BlockTable:
        """Allocate a sequence whose prefix reuses cached blocks.

        Expected behavior for the final implementation:
          1. Create a new `BlockTable` for `seq_id`
          2. Reuse `prefix_block_ids` by appending them to the table and
             incrementing each block's `ref_count`
          3. Allocate any additional suffix blocks needed for `num_tokens`
          4. Record `context_lens[seq_id]` and `seq_prefix_cache_info[seq_id]`

        Important follow-up detail:
          - Shared prefix blocks are logically read-only. If decode later wants
            to append into a shared tail block, you will need copy-on-write.
        """
        if num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")
        if seq_id in self.block_tables:
            raise ValueError(f"Sequence {seq_id} already has allocated blocks")

        prefix_token_count = len(prefix_block_ids) * self.block_size
        if prefix_token_count > num_tokens:
            raise ValueError("prefix blocks cover more tokens than requested")

        remaining_tokens = num_tokens - prefix_token_count
        suffix_blocks_needed = math.ceil(remaining_tokens / self.block_size) if remaining_tokens > 0 else 0
        if not self.can_allocate(suffix_blocks_needed):
            raise RuntimeError(
                f"Not enough free blocks for sequence {seq_id}: "
                f"need {suffix_blocks_needed}, have {self.num_free_blocks}"
            )

        block_table = BlockTable(seq_id)
        allocated_suffix_ids: List[int] = []

        try:
            for block_id in prefix_block_ids:
                block_table.append_block(block_id)
                self.blocks[block_id].ref_count += 1
                self.cached_free_lru.pop(block_id, None)

            tokens_left = remaining_tokens
            for _ in range(suffix_blocks_needed):
                block = self.allocate_block()
                allocated_suffix_ids.append(block.block_id)
                block_table.append_block(block.block_id)
                block.num_filled = min(tokens_left, self.block_size)
                tokens_left -= block.num_filled
        except Exception:
            for block_id in prefix_block_ids:
                self.blocks[block_id].ref_count -= 1
            for block_id in allocated_suffix_ids:
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
        self.seq_prefix_cache_info[seq_id] = PrefixCacheMatch(
            block_ids=list(prefix_block_ids),
            cached_token_count=prefix_token_count,
        )
        return block_table
    def publish_sequence_prefix_blocks(
        self,
        seq_id: int,
        token_ids: np.ndarray,
        extra_hash: object = None,
    ) -> None:
        """Publish full blocks from a finished prefill into the prefix cache.

        Expected behavior for the final implementation:
          1. Compute the block hash chain for `token_ids`
          2. Map each full logical block for `seq_id` to its physical block id
          3. Mark the block as cached (`is_prefix_cached = True`)
          4. Populate `block_hash_to_block_id` and `block_id_to_hash`

        Publication should happen only after all layers have finished writing
        K/V for those blocks; otherwise later reuse would observe incomplete KV.
        """
        hash_chain = self.compute_block_hash_chain(token_ids, extra_hash)
        if seq_id not in self.block_tables:
            raise ValueError(f"Sequence {seq_id} has no allocated blocks")

        block_table = self.block_tables[seq_id]
        num_full_blocks = len(hash_chain)
        if num_full_blocks > block_table.num_blocks:
            raise ValueError(
                f"Sequence {seq_id} has only {block_table.num_blocks} physical blocks "
                f"but {num_full_blocks} full logical blocks were computed"
            )

        for logical_block_idx, block_hash in enumerate(hash_chain):
            block_id = block_table.block_ids[logical_block_idx]
            block = self.blocks[block_id]

            old_hash = block.block_hash
            if old_hash is not None and old_hash != block_hash:
                self.block_hash_to_block_id.pop(old_hash, None)

            block.block_hash = block_hash
            block.is_prefix_cached = True
            self.block_hash_to_block_id[block_hash] = block_id
            self.block_id_to_hash[block_id] = block_hash
            self.cached_block_ids.add(block_id)
            self.cached_free_lru.pop(block_id, None)

    def evict_cached_block_if_needed(self) -> Optional[int]:
        """Evict one cacheable block from the prefix cache if capacity requires it.

        Expected behavior for the final implementation:
          1. Choose a zero-ref cached block from `cached_free_lru`
          2. Remove its hash entries from the prefix-cache dictionaries
          3. Clear its cached flag / hash metadata
          4. Zero its K/V slots and return it to the ordinary free list
          5. Return the evicted `block_id`, or `None` if nothing was evicted

        A simple LRU policy is sufficient for the first version.
        """
        if not self.cached_free_lru:
            return None

        block_id, _ = self.cached_free_lru.popitem(last=False)
        block = self.blocks[block_id]

        block_hash = self.block_id_to_hash.pop(block_id, None)
        if block_hash is not None:
            self.block_hash_to_block_id.pop(block_hash, None)

        block.block_hash = None
        block.is_prefix_cached = False
        block.num_filled = 0
        block.ref_count = 0
        self.cached_block_ids.discard(block_id)

        for layer_id in range(self.num_layers):
            self.key_cache[layer_id][block_id].fill(0)
            self.value_cache[layer_id][block_id].fill(0)

        if block_id not in self.free_block_ids:
            self.free_block_ids.append(block_id)
            self.free_block_ids.sort()

        return block_id

    def append_token_to_sequence(self, seq_id: int) -> KVBlock:
        """Reserve one new token slot for an existing sequence.

        If the current tail block is full, allocate a new block and append its
        id to the sequence's BlockTable.

        Prefix-caching note:
          - If the tail block is shared (`ref_count > 1`) and still has space,
            the final implementation should clone it before appending. That is a
            copy-on-write requirement for shared-prefix reuse.
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

        Prefix-caching note:
          - Cached blocks should not necessarily be zeroed and returned to the
            ordinary free list when `ref_count` reaches zero. A later
            implementation should instead move them into `cached_free_lru` for
            potential reuse and evict them lazily.
        """
        block_table = self.block_tables[seq_id]
        for block_id in block_table.block_ids:
            self.blocks[block_id].ref_count -= 1
            if self.blocks[block_id].ref_count == 0:
                if self.blocks[block_id].is_prefix_cached:
                    self.cached_free_lru[block_id] = None
                    self.cached_free_lru.move_to_end(block_id, last=True)
                else:
                    self.blocks[block_id].num_filled = 0
                    for layer_id in range(self.num_layers):
                        self.key_cache[layer_id][block_id].fill(0)
                        self.value_cache[layer_id][block_id].fill(0)
                    self.free_block_ids.append(block_id)
        del self.block_tables[seq_id]
        del self.context_lens[seq_id]
        self.seq_prefix_cache_info.pop(seq_id, None)

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
