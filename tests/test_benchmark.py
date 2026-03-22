"""
Performance regression tests.

These are lightweight tests meant to catch accidental performance
regressions, NOT full benchmarks. For comprehensive benchmarks, use
project/run_benchmark.py.
"""

import pytest
import sys
import time

sys.path.insert(0, ".")

from minitorch.block_manager import BlockManager


class TestBlockManagerPerformance:
    def test_allocation_speed(self):
        """Allocating 1000 blocks should complete quickly."""
        manager = BlockManager(num_blocks=1000, block_size=16, n_head=8, head_dim=64)
        start = time.perf_counter()
        for i in range(100):
            manager.allocate_blocks_for_sequence(seq_id=i, num_tokens=128)
        elapsed = time.perf_counter() - start
        # Sanity check: should be well under 1 second
        assert elapsed < 1.0, f"Block allocation too slow: {elapsed:.3f}s"

    def test_free_and_reallocate(self):
        """Free-then-reallocate cycle should not degrade performance."""
        manager = BlockManager(num_blocks=256, block_size=16, n_head=8, head_dim=64)
        num_sequences = 16
        tokens_per_sequence = 128
        num_cycles = 10

        start = time.perf_counter()
        for cycle in range(num_cycles):
            seq_ids = [
                cycle * num_sequences + i
                for i in range(num_sequences)
            ]

            for seq_id in seq_ids:
                manager.allocate_blocks_for_sequence(
                    seq_id=seq_id,
                    num_tokens=tokens_per_sequence,
                )

            assert manager.num_free_blocks == 128

            for seq_id in seq_ids:
                manager.free_sequence(seq_id)

            assert manager.num_free_blocks == 256
            assert manager.block_tables == {}
            assert manager.context_lens == {}

        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"Free/reallocate cycle too slow: {elapsed:.3f}s"
