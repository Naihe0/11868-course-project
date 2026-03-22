"""
Transformer model with PagedAttention support for MiniTorch.

Extends the hw3 transformer with two operating modes:
  - Standard mode (prefill): processes full prompt with contiguous KV.
  - Paged mode (decode): autoregressive generation using paged KV cache.
"""

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
    ):
        super().__init__()
        self.ln_1 = LayerNorm1d(n_embd, ln_eps, backend)
        self.ln_2 = LayerNorm1d(n_embd, ln_eps, backend)
        self.attention = PagedMultiHeadAttention(
            n_embd, n_head, block_size=block_size,
            p_dropout=p_dropout, bias=bias, backend=backend,
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
        self.layers = [
            PagedTransformerLayer(
                n_embd, n_head, block_size=block_size,
                p_dropout=p_dropout, ln_eps=ln_eps,
                bias=bias, backend=backend,
            )
            for _ in range(n_layers)
        ]

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
        x = self._embed(idx, start_pos=0)

        for layer in self.layers:
            x = layer.forward_prefill(x, block_manager, seq_ids)

        x = self.ln(x.view(batch_size * seq_len, self.n_embd)).view(
            batch_size, seq_len, self.n_embd
        )
        logits = self.lm_head(
            x.view(batch_size * seq_len, self.n_embd)
        ).view(batch_size, seq_len, self.n_vocab)
        return logits

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

        for layer in self.layers:
            x = layer.forward_decode(x, block_manager, seq_ids)

        x = self.ln(x.view(batch_size, self.n_embd)).view(batch_size, 1, self.n_embd)
        logits = self.lm_head(x.view(batch_size, self.n_embd)).view(
            batch_size, 1, self.n_vocab
        )
        return logits

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
        # Take logits at the last prompt position
        last_logits_np = logits.to_numpy()[:, -1, :]  # (batch, n_vocab)

        # 2. Sample next token from last position logits
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

        # 4. Free sequences
        for seq_id in seq_ids:
            block_manager.free_sequence(seq_id)

        return tensor_from_numpy(
            np.array(generated, dtype=datatype),
            backend=model.backend,
        )
