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
import numpy as np

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
        assert small_manager.free_block_ids == list(range(8))

    def test_allocate_single_block(self, small_manager):
        block = small_manager.allocate_block()

        assert isinstance(block, KVBlock)
        assert block.block_id == 0
        assert block.num_filled == 0
        assert small_manager.num_free_blocks == 7
        assert small_manager.num_used_blocks == 1
        assert 0 not in small_manager.free_block_ids

    def test_allocate_returns_unique_blocks(self, small_manager):
        block_0 = small_manager.allocate_block()
        block_1 = small_manager.allocate_block()

        assert block_0.block_id != block_1.block_id
        assert [block_0.block_id, block_1.block_id] == [0, 1]
        assert small_manager.num_free_blocks == 6

    def test_allocate_when_empty_raises(self, small_manager):
        for _ in range(8):
            small_manager.allocate_block()

        with pytest.raises(RuntimeError):
            small_manager.allocate_block()


# ---------------------------------------------------------------------------
# Sequence management
# ---------------------------------------------------------------------------

class TestSequenceManagement:
    def test_allocate_blocks_for_sequence(self, small_manager):
        table = small_manager.allocate_blocks_for_sequence(seq_id=7, num_tokens=10)

        assert isinstance(table, BlockTable)
        assert table.seq_id == 7
        assert table.block_ids == [0, 1, 2]
        assert small_manager.block_tables[7] is table
        assert small_manager.context_lens[7] == 10
        assert small_manager.blocks[0].num_filled == 4
        assert small_manager.blocks[1].num_filled == 4
        assert small_manager.blocks[2].num_filled == 2
        assert small_manager.num_free_blocks == 5

    def test_append_token_within_block(self, small_manager):
        small_manager.allocate_blocks_for_sequence(seq_id=1, num_tokens=3)

        block = small_manager.append_token_to_sequence(1)

        assert block.block_id == 0
        assert block.num_filled == 4
        assert small_manager.context_lens[1] == 4
        assert small_manager.block_tables[1].block_ids == [0]
        assert small_manager.num_free_blocks == 7

    def test_append_token_triggers_new_block(self, small_manager):
        small_manager.allocate_blocks_for_sequence(seq_id=1, num_tokens=4)

        block = small_manager.append_token_to_sequence(1)

        assert block.block_id == 1
        assert block.num_filled == 1
        assert small_manager.context_lens[1] == 5
        assert small_manager.block_tables[1].block_ids == [0, 1]
        assert small_manager.num_free_blocks == 6

    def test_free_sequence(self, small_manager):
        small_manager.allocate_blocks_for_sequence(seq_id=3, num_tokens=6)

        small_manager.free_sequence(3)

        assert 3 not in small_manager.block_tables
        assert 3 not in small_manager.context_lens
        assert sorted(small_manager.free_block_ids) == list(range(8))
        assert small_manager.num_free_blocks == 8

    def test_write_token_kv_writes_global_cache(self, small_manager):
        small_manager.allocate_blocks_for_sequence(seq_id=5, num_tokens=0)
        key = np.ones((2, 8), dtype=np.float32)
        value = np.full((2, 8), 3.0, dtype=np.float32)

        block_id, slot_idx = small_manager.write_token_kv(5, key, value)

        assert (block_id, slot_idx) == (0, 0)
        np.testing.assert_allclose(small_manager.key_cache[0, 0], key)
        np.testing.assert_allclose(small_manager.value_cache[0, 0], value)
        assert small_manager.context_lens[5] == 1


# ---------------------------------------------------------------------------
# Block table
# ---------------------------------------------------------------------------

class TestBlockTable:
    def test_block_table_ordering(self, small_manager):
        table = small_manager.allocate_blocks_for_sequence(seq_id=9, num_tokens=10)
        assert table.block_ids == [0, 1, 2]

    def test_get_block_table_tensor(self, small_manager):
        small_manager.allocate_blocks_for_sequence(seq_id=11, num_tokens=8)
        assert small_manager.get_block_table_tensor(11) == [0, 1]

    def test_get_physical_location(self, small_manager):
        small_manager.allocate_blocks_for_sequence(seq_id=12, num_tokens=6)

        assert small_manager.get_physical_location(12, 0) == (0, 0)
        assert small_manager.get_physical_location(12, 3) == (0, 3)
        assert small_manager.get_physical_location(12, 4) == (1, 0)
        assert small_manager.get_physical_location(12, 5) == (1, 1)

    def test_get_block_table_array_pads_rows(self, small_manager):
        small_manager.allocate_blocks_for_sequence(seq_id=1, num_tokens=8)   # [0, 1]
        small_manager.allocate_blocks_for_sequence(seq_id=2, num_tokens=4)   # [2]
        small_manager.allocate_blocks_for_sequence(seq_id=3, num_tokens=10)  # [3, 4, 5]

        block_table_array = small_manager.get_block_table_array([1, 2, 3], pad_value=-1)

        expected = np.array(
            [
                [0, 1, -1],
                [2, -1, -1],
                [3, 4, 5],
            ],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(block_table_array, expected)


# ---------------------------------------------------------------------------
# Fragmentation metrics
# ---------------------------------------------------------------------------

class TestFragmentation:
    def test_no_fragmentation_when_full(self, small_manager):
        small_manager.allocate_blocks_for_sequence(seq_id=1, num_tokens=8)
        fragmentation = small_manager.compute_fragmentation()

        assert fragmentation["internal"] == pytest.approx(0.0)
        assert fragmentation["external"] == pytest.approx(0.0)

    def test_internal_fragmentation(self, small_manager):
        small_manager.allocate_blocks_for_sequence(seq_id=2, num_tokens=5)
        fragmentation = small_manager.compute_fragmentation()

        # 2 allocated blocks -> capacity 8, valid tokens 5, wasted 3
        assert fragmentation["internal"] == pytest.approx(3 / 8)

    def test_external_fragmentation(self, small_manager):
        small_manager.allocate_blocks_for_sequence(seq_id=1, num_tokens=4)   # [0]
        small_manager.allocate_blocks_for_sequence(seq_id=2, num_tokens=4)   # [1]
        small_manager.allocate_blocks_for_sequence(seq_id=3, num_tokens=4)   # [2]
        small_manager.free_sequence(2)

        fragmentation = small_manager.compute_fragmentation()

        # Free block 1 is trapped between used blocks 0 and 2.
        assert fragmentation["external"] > 0.0
