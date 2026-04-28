"""
Transformer model with PagedAttention support for MiniTorch.

Extends the hw3 transformer with two operating modes:
  - Standard mode (prefill): processes full prompt with contiguous KV.
  - Paged mode (decode): autoregressive generation using paged KV cache.
"""

import builtins
import numpy as np
from .tensor import tensor, tensor_from_numpy
from .module import Module, Parameter
from .modules_basic import (
    Embedding,
    Dropout,
    LayerNorm1d,
    Linear,
)
from .tensor_ops import TensorBackend
from .nn import (
    max,
    softmax,
    dropout,
    GELU,
)
from .block_manager import BlockManager, DEFAULT_BLOCK_SIZE
from .paged_attention import PagedMultiHeadAttention
from typing import Any, Dict, List, Optional, Tuple

datatype = np.float32


# ---------------------------------------------------------------------------
# Feed-Forward Network (same as hw3)
# ---------------------------------------------------------------------------

class FeedForward(Module):
    def __init__(
        self,
        n_embd: int,
        middle_dim: int = 256,
        p_dropout: float = 0.1,
        bias: bool = True,
        backend: TensorBackend = None,
    ):
        super().__init__()
        self.linear_in = Linear(n_embd, middle_dim, bias=bias, backend=backend)
        self.linear_out = Linear(middle_dim, n_embd, bias=bias, backend=backend)
        self.dropout = Dropout(p_dropout)

    def forward(self, x):
        batch_size, seq_len, n_embd = x.shape
        x = GELU(self.linear_in(x.view(batch_size * seq_len, n_embd)))
        x = self.dropout(self.linear_out(x)).view(batch_size, seq_len, n_embd)
        return x


# ---------------------------------------------------------------------------
# Transformer Layer with PagedAttention
# ---------------------------------------------------------------------------

class PagedTransformerLayer(Module):
    """Single transformer layer that uses PagedMultiHeadAttention.

    Supports both prefill and decode phases via its attention module.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int = DEFAULT_BLOCK_SIZE,
        p_dropout: float = 0.1,
        ln_eps: float = 1e-5,
        bias: bool = True,
        backend: TensorBackend = None,
        layer_id: int = 0,
        decode_backend: str = "ref",
        compare_to_ref: bool = False,
        compare_tolerance: float = 1e-4,
        gpu_resident_kv: bool = False,
    ):
        super().__init__()
        self.ln_1 = LayerNorm1d(n_embd, ln_eps, backend)
        self.ln_2 = LayerNorm1d(n_embd, ln_eps, backend)
        self.attention = PagedMultiHeadAttention(
            n_embd, n_head, block_size=block_size,
            p_dropout=p_dropout, bias=bias, backend=backend,
            layer_id=layer_id,
            decode_backend=decode_backend,
            compare_to_ref=compare_to_ref,
            compare_tolerance=compare_tolerance,
            gpu_resident_kv=gpu_resident_kv,
        )
        self.ff = FeedForward(
            n_embd, middle_dim=4 * n_embd,
            p_dropout=p_dropout, bias=bias, backend=backend,
        )

    def forward_prefill(self, x, block_manager: BlockManager, seq_ids: List[int]):
        """Prefill pass through a single transformer layer."""
        batch_size, seq_len, n_embd = x.shape
        # Pre-LN: LayerNorm → Attention → Residual
        norm_x = self.ln_1(x.view(batch_size * seq_len, n_embd)).view(
            batch_size, seq_len, n_embd
        )
        x = x + self.attention.forward_prefill(norm_x, block_manager, seq_ids)
        # Pre-LN: LayerNorm → FeedForward → Residual
        norm_x = self.ln_2(x.view(batch_size * seq_len, n_embd)).view(
            batch_size, seq_len, n_embd
        )
        x = x + self.ff(norm_x)
        return x

    def forward_decode(self, x, block_manager: BlockManager, seq_ids: List[int]):
        """Decode pass through a single transformer layer."""
        batch_size, seq_len, n_embd = x.shape
        # Pre-LN: LayerNorm → Attention → Residual
        norm_x = self.ln_1(x.view(batch_size * seq_len, n_embd)).view(
            batch_size, seq_len, n_embd
        )
        x = x + self.attention.forward_decode(norm_x, block_manager, seq_ids)
        # Pre-LN: LayerNorm → FeedForward → Residual
        norm_x = self.ln_2(x.view(batch_size * seq_len, n_embd)).view(
            batch_size, seq_len, n_embd
        )
        x = x + self.ff(norm_x)
        return x


# ---------------------------------------------------------------------------
# Decoder Language Model with PagedAttention
# ---------------------------------------------------------------------------

class PagedDecoderLM(Module):
    """Decoder-only transformer language model with PagedAttention.

    Args:
        n_vocab:      Vocabulary size.
        n_embd:       Embedding dimension.
        n_head:       Number of attention heads.
        n_positions:  Maximum sequence length.
        n_layers:     Number of transformer layers.
        block_size:   KV cache block size.
        p_dropout:    Dropout probability.
        ln_eps:       LayerNorm epsilon.
        bias:         Whether to use bias in linear layers.
        backend:      MiniTorch tensor backend.
    """

    def __init__(
        self,
        n_vocab: int,
        n_embd: int,
        n_head: int,
        n_positions: int,
        n_layers: int = 4,
        block_size: int = DEFAULT_BLOCK_SIZE,
        p_dropout: float = 0.1,
        ln_eps: float = 1e-5,
        bias: bool = True,
        backend: TensorBackend = None,
        decode_backend: str = "ref",
        compare_to_ref: bool = False,
        compare_tolerance: float = 1e-4,
        gpu_resident_kv: bool = False,
    ):
        super().__init__()
        self.backend = backend
        self.n_embd = n_embd
        self.n_vocab = n_vocab
        self.n_layers = n_layers

        # Embeddings
        self.token_embeddings = Embedding(n_vocab, n_embd, backend)
        self.position_embeddings = Embedding(n_positions, n_embd, backend)

        # Transformer layers
        self.layers = []
        for layer_id in range(n_layers):
            layer = PagedTransformerLayer(
                n_embd, n_head, block_size=block_size,
                p_dropout=p_dropout, ln_eps=ln_eps,
                bias=bias, backend=backend,
                layer_id=layer_id,
                decode_backend=decode_backend,
                compare_to_ref=compare_to_ref,
                compare_tolerance=compare_tolerance,
                gpu_resident_kv=gpu_resident_kv,
            )
            setattr(self, f"layer_{layer_id}", layer)
            self.layers.append(layer)

        # Output head
        self.dropout = Dropout(p_dropout)
        self.ln = LayerNorm1d(n_embd, ln_eps, backend)
        self.lm_head = Linear(n_embd, n_vocab, bias=False, backend=backend)

    def _embed(self, idx, start_pos: int = 0):
        """Compute token + positional embeddings.

        Args:
            idx: Token indices (batch, seq_len).
            start_pos: Starting position for positional embeddings (used in decode).

        Returns:
            Embedded tensor (batch, seq_len, n_embd).
        """
        batch_size, seq_len = idx.shape
        tok_emb = self.token_embeddings(idx)
        pos_ids = tensor_from_numpy(
            np.arange(start_pos, start_pos + seq_len)
            .reshape(1, seq_len)
            .astype(np.float32),
            backend=self.backend,
        )
        pos_emb = self.position_embeddings(pos_ids)
        return self.dropout(tok_emb + pos_emb)

    def _forward_prefill_group_with_prefix(
        self,
        idx,
        block_manager: BlockManager,
        seq_ids: List[int],
        cached_token_count: int,
    ):
        """Run prefix-aware prefill for a group sharing one prefix length."""
        batch_size, seq_len = idx.shape
        if batch_size != len(seq_ids):
            raise ValueError("Batch size must match seq_ids length")
        if seq_len <= 0:
            raise ValueError("Prefill expects at least one prompt token")

        work_start = cached_token_count if cached_token_count < seq_len else seq_len - 1
        write_kv_to_cache = cached_token_count < seq_len
        idx_np = idx.to_numpy()
        idx_work = tensor_from_numpy(
            idx_np[:, work_start:].astype(np.float32),
            backend=self.backend,
        )

        x = self._embed(idx_work, start_pos=work_start)
        prefix_token_count = work_start

        for layer in self.layers:
            if hasattr(layer, "attention") and hasattr(layer.attention, "forward_prefill_with_prefix_batch"):
                batch_size, work_len, n_embd = x.shape
                norm_x = layer.ln_1(x.view(batch_size * work_len, n_embd)).view(
                    batch_size, work_len, n_embd
                )
                x = x + layer.attention.forward_prefill_with_prefix_batch(
                    norm_x,
                    block_manager,
                    seq_ids,
                    prefix_token_count,
                    cached_token_count=cached_token_count,
                    write_kv_to_cache=write_kv_to_cache,
                )
                norm_x = layer.ln_2(x.view(batch_size * work_len, n_embd)).view(
                    batch_size, work_len, n_embd
                )
                x = x + layer.ff(norm_x)
            elif hasattr(layer, "forward_prefill_with_prefix_batch"):
                x = layer.forward_prefill_with_prefix_batch(
                    x,
                    block_manager,
                    seq_ids,
                    prefix_token_count,
                    cached_token_count=cached_token_count,
                    write_kv_to_cache=write_kv_to_cache,
                )
            else:
                x = layer.forward_prefill(x, block_manager, seq_ids)

        batch_size, work_len, _ = x.shape
        x = self.ln(x.view(batch_size * work_len, self.n_embd)).view(
            batch_size, work_len, self.n_embd
        )
        logits_work = self.lm_head(
            x.view(batch_size * work_len, self.n_embd)
        ).view(batch_size, work_len, self.n_vocab)

        logits_np = np.zeros((batch_size, seq_len, self.n_vocab), dtype=np.float32)
        logits_np[:, work_start:, :] = logits_work.to_numpy()
        return tensor_from_numpy(logits_np, backend=self.backend)

    def forward_prefill(
        self,
        idx,
        block_manager: BlockManager,
        seq_ids: List[int],
    ):
        """Prefill phase: process full prompt tokens.

        Args:
            idx: Token indices (batch, seq_len).
            block_manager: Block manager for KV cache allocation.
            seq_ids: Sequence identifiers for each batch item.

        Returns:
            Logits (batch, seq_len, n_vocab).
        """
        batch_size, seq_len = idx.shape
        idx_np = idx.to_numpy().astype(np.int32)

        prefix_matches = []
        for batch_idx, seq_id in enumerate(seq_ids):
            if seq_id in block_manager.block_tables:
                raise ValueError(
                    f"Sequence {seq_id} is already active; free it before prefill"
                )
            seq_tokens = idx_np[batch_idx]
            match = block_manager.lookup_prefix_blocks(seq_tokens)
            prefix_matches.append(match)
            if match.block_ids:
                block_manager.allocate_sequence_with_prefix(
                    seq_id,
                    seq_len,
                    match.block_ids,
                )
            else:
                block_manager.allocate_blocks_for_sequence(seq_id, seq_len)

        if all(match.cached_token_count == 0 for match in prefix_matches):
            x = self._embed(idx, start_pos=0)
            for layer in self.layers:
                x = layer.forward_prefill(x, block_manager, seq_ids)

            for batch_idx, seq_id in enumerate(seq_ids):
                # Publish only after all layers have finished populating K/V so the
                # cached prefix is fully materialized across the whole network.
                block_manager.publish_sequence_prefix_blocks(seq_id, idx_np[batch_idx])

            x = self.ln(x.view(batch_size * seq_len, self.n_embd)).view(
                batch_size, seq_len, self.n_embd
            )
            logits = self.lm_head(
                x.view(batch_size * seq_len, self.n_embd)
            ).view(batch_size, seq_len, self.n_vocab)
            return logits

        logits_np = np.zeros((batch_size, seq_len, self.n_vocab), dtype=np.float32)

        no_hit_indices = [
            batch_idx
            for batch_idx, match in enumerate(prefix_matches)
            if match.cached_token_count == 0
        ]
        if no_hit_indices:
            idx_no_hit = tensor_from_numpy(
                idx_np[no_hit_indices].astype(np.float32),
                backend=self.backend,
            )
            seq_ids_no_hit = [seq_ids[batch_idx] for batch_idx in no_hit_indices]
            x = self._embed(idx_no_hit, start_pos=0)
            for layer in self.layers:
                x = layer.forward_prefill(x, block_manager, seq_ids_no_hit)
            x = self.ln(x.view(len(no_hit_indices) * seq_len, self.n_embd)).view(
                len(no_hit_indices), seq_len, self.n_embd
            )
            logits_no_hit = self.lm_head(
                x.view(len(no_hit_indices) * seq_len, self.n_embd)
            ).view(len(no_hit_indices), seq_len, self.n_vocab).to_numpy()
            logits_np[no_hit_indices] = logits_no_hit
            for batch_idx in no_hit_indices:
                block_manager.publish_sequence_prefix_blocks(seq_ids[batch_idx], idx_np[batch_idx])

        hit_groups: Dict[int, List[int]] = {}
        for batch_idx, match in enumerate(prefix_matches):
            if match.cached_token_count > 0:
                hit_groups.setdefault(match.cached_token_count, []).append(batch_idx)

        for cached_token_count, group_indices in hit_groups.items():
            idx_group = tensor_from_numpy(
                idx_np[group_indices].astype(np.float32),
                backend=self.backend,
            )
            seq_ids_group = [seq_ids[batch_idx] for batch_idx in group_indices]
            logits_group = self._forward_prefill_group_with_prefix(
                idx_group,
                block_manager,
                seq_ids_group,
                cached_token_count,
            ).to_numpy()
            logits_np[group_indices] = logits_group
            for batch_idx in group_indices:
                block_manager.publish_sequence_prefix_blocks(seq_ids[batch_idx], idx_np[batch_idx])

        return tensor_from_numpy(logits_np, backend=self.backend)

    def forward_decode(
        self,
        idx,
        block_manager: BlockManager,
        seq_ids: List[int],
        start_pos: int = 0,
    ):
        """Decode phase: process a single new token per sequence.

        Args:
            idx: Token indices (batch, 1).
            block_manager: Block manager for KV cache.
            seq_ids: Sequence identifiers for each batch item.
            start_pos: Position index for the new token.

        Returns:
            Logits (batch, 1, n_vocab).
        """
        batch_size, seq_len = idx.shape
        assert seq_len == 1, "Decode processes one token at a time"
        x = self._embed(idx, start_pos=start_pos)
        for seq_id in seq_ids:
            if seq_id not in block_manager.block_tables:
                block_manager.allocate_blocks_for_sequence(seq_id, 0)
            block_manager.append_token_to_sequence(seq_id)
        for layer in self.layers:
            x = layer.forward_decode(x, block_manager, seq_ids)

        x = self.ln(x.view(batch_size, self.n_embd)).view(batch_size, 1, self.n_embd)
        logits = self.lm_head(x.view(batch_size, self.n_embd)).view(
            batch_size, 1, self.n_vocab
        )
        return logits

    def close_decode_runtime(self) -> None:
        for layer in self.layers:
            if hasattr(layer, "attention") and hasattr(layer.attention, "close_decode_runtime"):
                layer.attention.close_decode_runtime()
            elif hasattr(layer, "close_decode_runtime"):
                layer.close_decode_runtime()

    @staticmethod
    def generate(
        model: "PagedDecoderLM",
        idx,
        max_new_tokens: int,
        block_manager: BlockManager,
        seq_ids: List[int],
        temperature: float = 1.0,
    ):
        """Autoregressive generation loop using PagedAttention.

        Args:
            model: The PagedDecoderLM instance.
            idx: Prompt token indices (batch, prompt_len).
            max_new_tokens: Number of tokens to generate.
            block_manager: Block manager instance.
            seq_ids: Sequence identifiers.
            temperature: Sampling temperature.

        Returns:
            Generated token indices (batch, prompt_len + max_new_tokens).
        """
        model.eval()
        batch_size, prompt_len = idx.shape
        generated = idx.to_numpy().tolist()  # list of lists
        # 1. Prefill: process the full prompt
        logits = model.forward_prefill(idx, block_manager, seq_ids)

        def _sample(logits_np: np.ndarray) -> np.ndarray:
            """Temperature-scaled sampling. Returns (batch,) int array."""
            if temperature <= 0:
                return np.argmax(logits_np, axis=-1)
            scaled = logits_np / temperature
            scaled = scaled - np.max(scaled, axis=-1, keepdims=True)
            probs = np.exp(scaled) / np.sum(np.exp(scaled), axis=-1, keepdims=True)
            tokens = np.array([
                np.random.choice(len(p), p=p) for p in probs
            ])
            return tokens

        try:
            if max_new_tokens <= 0:
                return tensor_from_numpy(
                    np.array(generated, dtype=datatype),
                    backend=model.backend,
                )

            # 2. Sample next token from last prompt position logits
            last_logits_np = logits.to_numpy()[:, -1, :]  # (batch, n_vocab)
            next_tokens = _sample(last_logits_np)  # (batch,)
            for b in range(batch_size):
                generated[b].append(int(next_tokens[b]))

            # 3. Decode loop
            for step in range(1, max_new_tokens):
                # Build input tensor (batch, 1)
                token_input = tensor_from_numpy(
                    next_tokens.reshape(batch_size, 1).astype(datatype),
                    backend=model.backend,
                )
                start_pos = prompt_len + step - 1
                logits = model.forward_decode(
                    token_input, block_manager, seq_ids, start_pos=start_pos
                )
                logits_np = logits.to_numpy()[:, 0, :]  # (batch, n_vocab)
                next_tokens = _sample(logits_np)
                for b in range(batch_size):
                    generated[b].append(int(next_tokens[b]))
        finally:
            for seq_id in seq_ids:
                if seq_id in block_manager.block_tables:
                    block_manager.free_sequence(seq_id)
            model.close_decode_runtime()

        return tensor_from_numpy(
            np.array(generated, dtype=datatype),
            backend=model.backend,
        )

    @staticmethod
    def generate_parallel_sampling(
        model: "PagedDecoderLM",
        idx,
        max_new_tokens: int,
        num_outputs: int,
        block_manager: BlockManager,
        seq_ids: List[int],
        temperature: float = 1.0,
    ):
        """Generate multiple sampled continuations that share one prompt prefix.

        Current implementation targets batch_size == 1. It prefills the prompt
        once, forks additional sequences that share the prompt KV blocks, and
        then decodes all continuations together.
        """
        model.eval()
        batch_size, prompt_len = idx.shape
        if batch_size != 1 or len(seq_ids) != 1:
            raise NotImplementedError("Parallel sampling currently supports batch_size == 1")
        if num_outputs <= 0:
            raise ValueError("num_outputs must be positive")

        def _sample_many(logits_np: np.ndarray, k: int) -> np.ndarray:
            if temperature <= 0:
                top = int(np.argmax(logits_np))
                return np.full((k,), top, dtype=np.int32)
            scaled = logits_np / temperature
            scaled = scaled - np.max(scaled)
            probs = np.exp(scaled) / np.sum(np.exp(scaled))
            return np.array([np.random.choice(len(probs), p=probs) for _ in range(k)], dtype=np.int32)

        base_seq_id = seq_ids[0]
        prompt_tokens = idx.to_numpy().astype(np.int32)[0].tolist()
        next_seq_id = builtins.max(
            [base_seq_id]
            + list(block_manager.block_tables.keys())
            + list(block_manager.context_lens.keys())
        ) + 1

        logits = model.forward_prefill(idx, block_manager, [base_seq_id])
        try:
            if max_new_tokens <= 0:
                return tensor_from_numpy(
                    np.array([prompt_tokens], dtype=datatype),
                    backend=model.backend,
                )

            active_seq_ids = [base_seq_id]
            for _ in range(num_outputs - 1):
                child_seq_id = next_seq_id
                next_seq_id += 1
                block_manager.fork_sequence(base_seq_id, child_seq_id)
                active_seq_ids.append(child_seq_id)

            generated = [list(prompt_tokens) for _ in range(num_outputs)]
            first_logits = logits.to_numpy()[0, -1, :]
            next_tokens = _sample_many(first_logits, num_outputs)
            for out_idx in range(num_outputs):
                generated[out_idx].append(int(next_tokens[out_idx]))

            for step in range(1, max_new_tokens):
                token_input = tensor_from_numpy(
                    next_tokens.reshape(num_outputs, 1).astype(datatype),
                    backend=model.backend,
                )
                start_pos = prompt_len + step - 1
                logits = model.forward_decode(
                    token_input, block_manager, active_seq_ids, start_pos=start_pos
                )
                logits_np = logits.to_numpy()[:, 0, :]
                sampled = []
                for out_idx in range(num_outputs):
                    sampled.append(_sample_many(logits_np[out_idx], 1)[0])
                next_tokens = np.array(sampled, dtype=np.int32)
                for out_idx in range(num_outputs):
                    generated[out_idx].append(int(next_tokens[out_idx]))
        finally:
            for seq_id in list(block_manager.block_tables.keys()):
                if seq_id in block_manager.block_tables:
                    block_manager.free_sequence(seq_id)
            model.close_decode_runtime()

        return tensor_from_numpy(
            np.array(generated, dtype=datatype),
            backend=model.backend,
        )

    @staticmethod
    def generate_beam_search(
        model: "PagedDecoderLM",
        idx,
        max_new_tokens: int,
        beam_width: int,
        block_manager: BlockManager,
        seq_ids: List[int],
        eos_token_id: Optional[int] = None,
    ):
        """Beam-search generation loop.

        Current implementation targets the common single-request serving case
        (`batch_size == 1`) and relies on BlockManager live-forking so multiple
        beams can share the same KV prefix blocks.
        """
        model.eval()
        batch_size, prompt_len = idx.shape
        if batch_size != 1 or len(seq_ids) != 1:
            raise NotImplementedError("Beam search currently supports batch_size == 1")
        if beam_width <= 0:
            raise ValueError("beam_width must be positive")

        def _log_softmax(logits_np: np.ndarray) -> np.ndarray:
            shifted = logits_np - np.max(logits_np, axis=-1, keepdims=True)
            logsumexp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
            return shifted - logsumexp

        base_seq_id = seq_ids[0]
        prompt_tokens = idx.to_numpy().astype(np.int32)[0].tolist()
        next_seq_id = builtins.max(
            [base_seq_id]
            + list(block_manager.block_tables.keys())
            + list(block_manager.context_lens.keys())
        ) + 1

        def _allocate_child_seq_id() -> int:
            nonlocal next_seq_id
            while next_seq_id in block_manager.block_tables:
                next_seq_id += 1
            child_seq_id = next_seq_id
            next_seq_id += 1
            return child_seq_id

        logits = model.forward_prefill(idx, block_manager, [base_seq_id])
        try:
            if max_new_tokens <= 0:
                return tensor_from_numpy(
                    np.array([prompt_tokens], dtype=datatype),
                    backend=model.backend,
                )

            initial_log_probs = _log_softmax(logits.to_numpy()[0, -1:, :])[0]
            initial_topk = np.argsort(initial_log_probs)[-beam_width:][::-1]

            beams = []
            for rank, token_id in enumerate(initial_topk):
                if rank == 0:
                    seq_id = base_seq_id
                else:
                    seq_id = _allocate_child_seq_id()
                    block_manager.fork_sequence(base_seq_id, seq_id)
                beams.append(
                    {
                        "seq_id": seq_id,
                        "tokens": prompt_tokens + [int(token_id)],
                        "score": float(initial_log_probs[token_id]),
                        "finished": eos_token_id is not None and int(token_id) == eos_token_id,
                    }
                )

            for step in range(1, max_new_tokens):
                active_beams = [beam for beam in beams if not beam["finished"]]
                finished_beams = [beam for beam in beams if beam["finished"]]
                if not active_beams:
                    break

                token_input = tensor_from_numpy(
                    np.array(
                        [[beam["tokens"][-1]] for beam in active_beams],
                        dtype=datatype,
                    ),
                    backend=model.backend,
                )
                active_seq_ids = [beam["seq_id"] for beam in active_beams]
                decode_logits = model.forward_decode(
                    token_input,
                    block_manager,
                    active_seq_ids,
                    start_pos=prompt_len + step - 1,
                )
                decode_log_probs = _log_softmax(decode_logits.to_numpy()[:, 0, :])

                candidates = [
                    {
                        "seq_id": beam["seq_id"],
                        "tokens": list(beam["tokens"]),
                        "score": beam["score"],
                        "finished": True,
                        "carry": True,
                    }
                    for beam in finished_beams
                ]

                for beam_idx, beam in enumerate(active_beams):
                    topk = np.argsort(decode_log_probs[beam_idx])[-beam_width:][::-1]
                    for token_id in topk:
                        candidates.append(
                            {
                                "parent_seq_id": beam["seq_id"],
                                "parent_tokens": beam["tokens"],
                                "token_id": int(token_id),
                                "score": beam["score"] + float(decode_log_probs[beam_idx, token_id]),
                                "finished": eos_token_id is not None and int(token_id) == eos_token_id,
                                "carry": False,
                            }
                        )

                ordered = sorted(candidates, key=lambda item: item["score"], reverse=True)[:beam_width]
                next_beams = []
                parent_use_count: Dict[int, int] = {}
                retained_seq_ids = set()

                for candidate in ordered:
                    if candidate["carry"]:
                        seq_id = candidate["seq_id"]
                        tokens = candidate["tokens"]
                    else:
                        parent_seq_id = candidate["parent_seq_id"]
                        use_count = parent_use_count.get(parent_seq_id, 0)
                        if use_count == 0:
                            seq_id = parent_seq_id
                        else:
                            seq_id = _allocate_child_seq_id()
                            block_manager.fork_sequence(parent_seq_id, seq_id)
                        parent_use_count[parent_seq_id] = use_count + 1
                        tokens = list(candidate["parent_tokens"]) + [candidate["token_id"]]

                    next_beams.append(
                        {
                            "seq_id": seq_id,
                            "tokens": tokens,
                            "score": candidate["score"],
                            "finished": candidate["finished"],
                        }
                    )
                    retained_seq_ids.add(seq_id)

                old_seq_ids = {beam["seq_id"] for beam in beams}
                for seq_id in sorted(old_seq_ids - retained_seq_ids):
                    if seq_id in block_manager.block_tables:
                        block_manager.free_sequence(seq_id)

                beams = next_beams

            beams = sorted(beams, key=lambda beam: beam["score"], reverse=True)
            return tensor_from_numpy(
                np.array([beam["tokens"] for beam in beams], dtype=datatype),
                backend=model.backend,
            )
        finally:
            live_seq_ids = sorted({beam["seq_id"] for beam in locals().get("beams", [])})
            if not live_seq_ids and base_seq_id in block_manager.block_tables:
                live_seq_ids = [base_seq_id]
            for seq_id in live_seq_ids:
                if seq_id in block_manager.block_tables:
                    block_manager.free_sequence(seq_id)
            model.close_decode_runtime()

    @staticmethod
    def generate_beam_search_naive(
        model: "PagedDecoderLM",
        idx,
        max_new_tokens: int,
        beam_width: int,
        block_manager: BlockManager,
        seq_ids: List[int],
        eos_token_id: Optional[int] = None,
    ):
        """Beam search baseline without KV sharing.

        Child beams physically duplicate the parent's current KV cache using
        `clone_sequence(...)` instead of sharing blocks with `fork_sequence(...)`.
        """
        model.eval()
        batch_size, prompt_len = idx.shape
        if batch_size != 1 or len(seq_ids) != 1:
            raise NotImplementedError("Naive beam search currently supports batch_size == 1")
        if beam_width <= 0:
            raise ValueError("beam_width must be positive")

        def _log_softmax(logits_np: np.ndarray) -> np.ndarray:
            shifted = logits_np - np.max(logits_np, axis=-1, keepdims=True)
            logsumexp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
            return shifted - logsumexp

        base_seq_id = seq_ids[0]
        prompt_tokens = idx.to_numpy().astype(np.int32)[0].tolist()
        next_seq_id = builtins.max(
            [base_seq_id]
            + list(block_manager.block_tables.keys())
            + list(block_manager.context_lens.keys())
        ) + 1

        def _allocate_child_seq_id() -> int:
            nonlocal next_seq_id
            while next_seq_id in block_manager.block_tables:
                next_seq_id += 1
            child_seq_id = next_seq_id
            next_seq_id += 1
            return child_seq_id

        logits = model.forward_prefill(idx, block_manager, [base_seq_id])
        try:
            if max_new_tokens <= 0:
                return tensor_from_numpy(
                    np.array([prompt_tokens], dtype=datatype),
                    backend=model.backend,
                )

            initial_log_probs = _log_softmax(logits.to_numpy()[0, -1:, :])[0]
            initial_topk = np.argsort(initial_log_probs)[-beam_width:][::-1]

            beams = []
            for rank, token_id in enumerate(initial_topk):
                if rank == 0:
                    seq_id = base_seq_id
                else:
                    seq_id = _allocate_child_seq_id()
                    block_manager.clone_sequence(base_seq_id, seq_id)
                beams.append(
                    {
                        "seq_id": seq_id,
                        "tokens": prompt_tokens + [int(token_id)],
                        "score": float(initial_log_probs[token_id]),
                        "finished": eos_token_id is not None and int(token_id) == eos_token_id,
                    }
                )

            for step in range(1, max_new_tokens):
                active_beams = [beam for beam in beams if not beam["finished"]]
                finished_beams = [beam for beam in beams if beam["finished"]]
                if not active_beams:
                    break

                token_input = tensor_from_numpy(
                    np.array([[beam["tokens"][-1]] for beam in active_beams], dtype=datatype),
                    backend=model.backend,
                )
                active_seq_ids = [beam["seq_id"] for beam in active_beams]
                decode_logits = model.forward_decode(
                    token_input,
                    block_manager,
                    active_seq_ids,
                    start_pos=prompt_len + step - 1,
                )
                decode_log_probs = _log_softmax(decode_logits.to_numpy()[:, 0, :])

                candidates = [
                    {
                        "seq_id": beam["seq_id"],
                        "tokens": list(beam["tokens"]),
                        "score": beam["score"],
                        "finished": True,
                        "carry": True,
                    }
                    for beam in finished_beams
                ]

                for beam_idx, beam in enumerate(active_beams):
                    topk = np.argsort(decode_log_probs[beam_idx])[-beam_width:][::-1]
                    for token_id in topk:
                        candidates.append(
                            {
                                "parent_seq_id": beam["seq_id"],
                                "parent_tokens": beam["tokens"],
                                "token_id": int(token_id),
                                "score": beam["score"] + float(decode_log_probs[beam_idx, token_id]),
                                "finished": eos_token_id is not None and int(token_id) == eos_token_id,
                                "carry": False,
                            }
                        )

                ordered = sorted(candidates, key=lambda item: item["score"], reverse=True)[:beam_width]
                next_beams = []
                parent_use_count: Dict[int, int] = {}
                retained_seq_ids = set()

                for candidate in ordered:
                    if candidate["carry"]:
                        seq_id = candidate["seq_id"]
                        tokens = candidate["tokens"]
                    else:
                        parent_seq_id = candidate["parent_seq_id"]
                        use_count = parent_use_count.get(parent_seq_id, 0)
                        if use_count == 0:
                            seq_id = parent_seq_id
                        else:
                            seq_id = _allocate_child_seq_id()
                            block_manager.clone_sequence(parent_seq_id, seq_id)
                        parent_use_count[parent_seq_id] = use_count + 1
                        tokens = list(candidate["parent_tokens"]) + [candidate["token_id"]]

                    next_beams.append(
                        {
                            "seq_id": seq_id,
                            "tokens": tokens,
                            "score": candidate["score"],
                            "finished": candidate["finished"],
                        }
                    )
                    retained_seq_ids.add(seq_id)

                old_seq_ids = {beam["seq_id"] for beam in beams}
                for seq_id in sorted(old_seq_ids - retained_seq_ids):
                    if seq_id in block_manager.block_tables:
                        block_manager.free_sequence(seq_id)

                beams = next_beams

            beams = sorted(beams, key=lambda beam: beam["score"], reverse=True)
            return tensor_from_numpy(
                np.array([beam["tokens"] for beam in beams], dtype=datatype),
                backend=model.backend,
            )
        finally:
            live_seq_ids = sorted({beam["seq_id"] for beam in locals().get("beams", [])})
            if not live_seq_ids and base_seq_id in block_manager.block_tables:
                live_seq_ids = [base_seq_id]
            for seq_id in live_seq_ids:
                if seq_id in block_manager.block_tables:
                    block_manager.free_sequence(seq_id)
            model.close_decode_runtime()
