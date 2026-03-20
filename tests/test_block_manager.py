"""
Unit tests for the BlockManager.

Tests cover:
  - Block allocation and deallocation
  - Sequence lifecycle (allocate → append → free)
  - Edge cases (OOM, double free)
  - Fragmentation metrics
"""

import pytest
import sys

sys.path.insert(0, ".")

from minitorch.block_manager import BlockManager, BlockTable, KVBlock, DEFAULT_BLOCK_SIZE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_manager():
    """BlockManager with 8 blocks, block_size=4, 2 heads, head_dim=8."""
    return BlockManager(num_blocks=8, block_size=4, n_head=2, head_dim=8)


# ---------------------------------------------------------------------------
# Basic allocation
# ---------------------------------------------------------------------------

class TestBlockAllocation:
    def test_initial_state(self, small_manager):
        assert small_manager.num_free_blocks == 8
        assert small_manager.num_used_blocks == 0
        assert small_manager.key_cache.shape == (8, 4, 2, 8)
        assert small_manager.value_cache.shape == (8, 4, 2, 8)

    def test_allocate_single_block(self, small_manager):
        # TODO: Test allocating a single block in the global-cache design
        pass

    def test_allocate_returns_unique_blocks(self, small_manager):
        # TODO: Test that consecutive allocations return different block ids
        pass

    def test_allocate_when_empty_raises(self, small_manager):
        # TODO: Exhaust all blocks, then verify RuntimeError on next allocate
        pass


# ---------------------------------------------------------------------------
# Sequence management
# ---------------------------------------------------------------------------

class TestSequenceManagement:
    def test_allocate_blocks_for_sequence(self, small_manager):
        # TODO: Allocate blocks for a sequence of N tokens,
        #       verify correct number of blocks allocated
        pass

    def test_append_token_within_block(self, small_manager):
        # TODO: Append tokens that fit in existing block
        pass

    def test_append_token_triggers_new_block(self, small_manager):
        # TODO: Fill a block, then append another token — should allocate new block
        pass

    def test_free_sequence(self, small_manager):
        # TODO: Allocate and then free a sequence,
        #       verify blocks returned to free pool
        pass

    def test_write_token_kv_writes_global_cache(self, small_manager):
        # TODO: Verify that appended token KV is written into the global cache
        pass


# ---------------------------------------------------------------------------
# Block table
# ---------------------------------------------------------------------------

class TestBlockTable:
    def test_block_table_ordering(self, small_manager):
        # TODO: Verify block table preserves logical ordering
        pass

    def test_get_block_table_tensor(self, small_manager):
        # TODO: Verify flat list of block ids matches block table
        pass

    def test_get_physical_location(self, small_manager):
        # TODO: Verify logical token positions map to the right block/slot pair
        pass


# ---------------------------------------------------------------------------
# Fragmentation metrics
# ---------------------------------------------------------------------------

class TestFragmentation:
    def test_no_fragmentation_when_full(self, small_manager):
        # TODO: Fill blocks completely, verify 0 internal fragmentation
        pass

    def test_internal_fragmentation(self, small_manager):
        # TODO: Partially fill last block, measure internal fragmentation
        pass

    def test_external_fragmentation(self, small_manager):
        # TODO: Allocate and free interleaved sequences,
        #       measure external fragmentation
        pass
