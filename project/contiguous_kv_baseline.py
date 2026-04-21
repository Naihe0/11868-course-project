"""
Contiguous KV Cache baseline (HuggingFace-style) that shares weights with a
PagedDecoderLM.

The point of this module is to give Figure 3 a *fair algorithmic baseline*
for decode speed.  The existing "no-cache" baseline (``DecoderLM.forward``)
re-runs the full prompt through the network on every new token, so its per-
step cost grows as O(T).  A real serving system instead keeps K/V in a
contiguous pre-allocated buffer and only projects the newest token — that is
the cost model paged attention should be compared against.

This module implements that baseline using minitorch tensors (so attention
runs on the same backend as the paged model), while storing the cache as a
NumPy array so updates are trivial `arr[..., t, :] = k_new`.  Per-step
overhead is one small host→device copy of the active cache slice, which is
negligible relative to the attention matmul cost.

Only inference is supported.  Weights (Q/K/V/out projections, FFN, layer
norms, embeddings, lm_head) are borrowed by reference from a source
``PagedDecoderLM`` so we measure exactly the paged model's network, with
paged replaced by a contiguous KV path.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

import minitorch
from minitorch.tensor import tensor_from_numpy
from minitorch.paged_attention import standard_attention
from minitorch.transformer import PagedDecoderLM

datatype = np.float32


class ContiguousKVDecoderLM:
    """HuggingFace-style contiguous KV cache wrapper around a PagedDecoderLM.

    The wrapped model provides all weights (embeddings, per-layer Q/K/V/out
    projections, FFN, layer norms, final norm, lm head).  This wrapper holds
    a pre-allocated per-layer K cache and V cache as NumPy arrays and walks
    the layers manually, so attention is computed over the live slice of the
    cache rather than gathered from paged blocks.

    Args:
        paged_model: Source model whose weights will be reused.
        max_batch_size: Upper bound on the number of active sequences.
        max_seq_len: Upper bound on ``prompt_len + max_new_tokens`` per seq.
    """

    def __init__(
        self,
        paged_model: PagedDecoderLM,
        max_batch_size: int,
        max_seq_len: int,
    ) -> None:
        self.model = paged_model
        self.backend = paged_model.backend
        self.n_embd = paged_model.n_embd
        self.n_vocab = paged_model.n_vocab
        self.n_layers = paged_model.n_layers
        first_attn = paged_model.layers[0].attention
        self.n_head = first_attn.n_head
        self.head_dim = first_attn.head_dim
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        # Per-layer contiguous KV caches — one contiguous slab per layer,
        # shape [max_batch, n_head, max_seq_len, head_dim].  This is exactly
        # the memory layout a naive serving system would use, with static
        # over-reservation per sequence.
        cache_shape = (max_batch_size, self.n_head, max_seq_len, self.head_dim)
        self._k_cache = [
            np.zeros(cache_shape, dtype=datatype) for _ in range(self.n_layers)
        ]
        self._v_cache = [
            np.zeros(cache_shape, dtype=datatype) for _ in range(self.n_layers)
        ]
        self._context_len = 0  # number of tokens currently valid in cache
        self._batch_size = 0  # number of sequences in the batch

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Drop any cached K/V so the object can be reused for a new request."""
        self._context_len = 0
        self._batch_size = 0
        # We deliberately don't zero the arrays — slicing only reads the
        # valid prefix, and zeroing a big buffer every reset is wasted work.

    def reserved_bytes(self) -> int:
        """Bytes statically reserved by the contiguous cache."""
        per_slab = (
            self.max_batch_size
            * self.n_head
            * self.max_seq_len
            * self.head_dim
            * np.dtype(datatype).itemsize
        )
        # One K slab + one V slab per layer.
        return 2 * self.n_layers * per_slab

    def used_bytes(self) -> int:
        """Bytes actually holding live K/V data."""
        per_slab = (
            self._batch_size
            * self.n_head
            * self._context_len
            * self.head_dim
            * np.dtype(datatype).itemsize
        )
        return 2 * self.n_layers * per_slab

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    def _embed(self, idx, start_pos: int = 0):
        batch_size, seq_len = idx.shape
        tok_emb = self.model.token_embeddings(idx)
        pos_ids = tensor_from_numpy(
            np.arange(start_pos, start_pos + seq_len)
            .reshape(1, seq_len)
            .astype(datatype),
            backend=self.backend,
        )
        pos_emb = self.model.position_embeddings(pos_ids)
        return self.model.dropout(tok_emb + pos_emb)

    # ------------------------------------------------------------------
    # Per-layer primitives
    # ------------------------------------------------------------------
    def _project_qkv(self, attn, x, seq_len):
        """Run q_proj/k_proj/v_proj from a PagedMultiHeadAttention on ``x``.

        Returns (q, k, v) each shaped (batch, n_head, seq_len, head_dim).
        """
        batch_size = x.shape[0]
        flat_x = x.contiguous().view(batch_size * seq_len, self.n_embd)
        q = attn.q_proj(flat_x).view(batch_size, seq_len, self.n_head, self.head_dim)
        k = attn.k_proj(flat_x).view(batch_size, seq_len, self.n_head, self.head_dim)
        v = attn.v_proj(flat_x).view(batch_size, seq_len, self.n_head, self.head_dim)
        return q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)

    def _run_layer_prefill(self, layer, x, seq_len):
        """Run one paged transformer layer in prefill mode with contiguous KV.

        Writes this layer's K/V at [:, :, :seq_len, :] of the layer's cache.
        """
        attn = layer.attention
        batch_size = x.shape[0]

        norm_x = layer.ln_1(x.view(batch_size * seq_len, self.n_embd)).view(
            batch_size, seq_len, self.n_embd
        )
        q, k, v = self._project_qkv(attn, norm_x, seq_len)

        # Cache K/V for this layer so the decode path can find them.
        layer_idx = attn.layer_id
        self._k_cache[layer_idx][:batch_size, :, :seq_len, :] = k.to_numpy()
        self._v_cache[layer_idx][:batch_size, :, :seq_len, :] = v.to_numpy()

        # Standard contiguous attention with a causal mask.
        mask_np = np.triu(
            np.full((seq_len, seq_len), -1e9, dtype=datatype),
            k=1,
        ).reshape(1, 1, seq_len, seq_len)
        mask = tensor_from_numpy(mask_np, backend=self.backend)
        attn_out = standard_attention(q, k, v, mask)
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(
            batch_size * seq_len, self.n_embd
        )
        attn_out = attn.out_proj(attn_out).view(batch_size, seq_len, self.n_embd)
        x = x + attn_out

        # Feed-forward block (Pre-LN).
        norm_x = layer.ln_2(x.view(batch_size * seq_len, self.n_embd)).view(
            batch_size, seq_len, self.n_embd
        )
        x = x + layer.ff(norm_x)
        return x

    def _run_layer_decode(self, layer, x):
        """Run one paged transformer layer for a single new token with KV read
        from the contiguous cache."""
        attn = layer.attention
        batch_size = x.shape[0]

        norm_x = layer.ln_1(x.view(batch_size * 1, self.n_embd)).view(
            batch_size, 1, self.n_embd
        )
        q_new, k_new, v_new = self._project_qkv(attn, norm_x, 1)
        # q_new/k_new/v_new: (batch, n_head, 1, head_dim)

        layer_idx = attn.layer_id
        # Insert the new token's K/V at the current position.
        t = self._context_len
        self._k_cache[layer_idx][:batch_size, :, t : t + 1, :] = k_new.to_numpy()
        self._v_cache[layer_idx][:batch_size, :, t : t + 1, :] = v_new.to_numpy()

        # Grab the active slice [0, t+1) and lift it back to the GPU.
        k_full_np = self._k_cache[layer_idx][:batch_size, :, : t + 1, :]
        v_full_np = self._v_cache[layer_idx][:batch_size, :, : t + 1, :]
        k_full = tensor_from_numpy(np.ascontiguousarray(k_full_np), backend=self.backend)
        v_full = tensor_from_numpy(np.ascontiguousarray(v_full_np), backend=self.backend)

        # No causal mask needed for a single-token query against its own past.
        attn_out = standard_attention(q_new, k_full, v_full, mask=None)
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(
            batch_size * 1, self.n_embd
        )
        attn_out = attn.out_proj(attn_out).view(batch_size, 1, self.n_embd)
        x = x + attn_out

        norm_x = layer.ln_2(x.view(batch_size * 1, self.n_embd)).view(
            batch_size, 1, self.n_embd
        )
        x = x + layer.ff(norm_x)
        return x

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def forward_prefill(self, idx):
        """Consume the prompt, populate the KV cache, and return logits.

        Args:
            idx: Tensor (batch, seq_len) of token ids.

        Returns:
            Logits tensor (batch, seq_len, n_vocab).
        """
        batch_size, seq_len = idx.shape
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"batch_size {batch_size} exceeds max_batch_size {self.max_batch_size}"
            )
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )

        self._batch_size = batch_size
        self._context_len = seq_len

        x = self._embed(idx, start_pos=0)
        for layer in self.model.layers:
            x = self._run_layer_prefill(layer, x, seq_len)

        x_flat = self.model.ln(x.view(batch_size * seq_len, self.n_embd))
        logits = self.model.lm_head(x_flat).view(batch_size, seq_len, self.n_vocab)
        return logits

    def forward_decode(self, idx, start_pos: int):
        """Emit logits for a single new token per sequence using the cache.

        Args:
            idx: Tensor (batch, 1) of token ids.
            start_pos: Absolute position of the new token (== current cache
                length before inserting).

        Returns:
            Logits tensor (batch, 1, n_vocab).
        """
        batch_size, seq_len = idx.shape
        if seq_len != 1:
            raise ValueError("forward_decode expects exactly one new token")
        if batch_size != self._batch_size:
            raise ValueError(
                "batch_size mismatch between prefill and decode "
                f"({self._batch_size} vs {batch_size})"
            )
        if start_pos != self._context_len:
            raise ValueError(
                f"start_pos {start_pos} does not match cache length {self._context_len}"
            )
        if self._context_len >= self.max_seq_len:
            raise ValueError("Cache exhausted; raise max_seq_len")

        x = self._embed(idx, start_pos=start_pos)
        for layer in self.model.layers:
            x = self._run_layer_decode(layer, x)

        x_flat = self.model.ln(x.view(batch_size, self.n_embd))
        logits = self.model.lm_head(x_flat).view(batch_size, 1, self.n_vocab)

        self._context_len += 1
        return logits
