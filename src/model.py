"""
model.py — Custom Cross-Attention Fusion Layer
==============================================
Implements scaled dot-product cross-attention where:
  Q  =  projected text query embedding        shape (1, d_model)
  K  =  segment embeddings matrix             shape (N, d_model)
  V  =  segment embeddings matrix             shape (N, d_model)

  Attention(Q, K, V) = softmax( (Q @ K.T) / sqrt(d_k) ) @ V

The module is intentionally self-contained with no external utility imports
so it can be unit-tested and imported independently of the rest of the system.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    """
    Scaled dot-product cross-attention fusion layer.

    The text query embedding (Q) attends over all candidate segment
    embeddings (K, V) to produce a weighted context vector that represents
    the most query-aligned content, together with per-segment attention scores
    used for final ranking.

    Parameters
    ----------
    input_dim : int
        Raw embedding dimensionality fed into the layer.
        Default is 1024 (VISUAL_DIM + AUDIO_DIM from pipeline.py).
    d_model : int
        Internal projection dimensionality used for Q/K/V.
        Default is 256.
    num_heads : int
        Number of parallel attention heads. `d_model` must be divisible by
        this value. Default is 4.
    dropout : float
        Attention weight dropout probability during training. Default is 0.1.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        d_model: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})."
            )

        self.input_dim = input_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # per-head key dimension

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(input_dim, d_model, bias=False)
        self.W_k = nn.Linear(input_dim, d_model, bias=False)
        self.W_v = nn.Linear(input_dim, d_model, bias=False)

        # Output projection back to input_dim
        self.W_o = nn.Linear(d_model, input_dim, bias=False)

        # Layer norms for residual connections
        self.norm_q = nn.LayerNorm(input_dim)
        self.norm_out = nn.LayerNorm(input_dim)

        self.attn_dropout = nn.Dropout(p=dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform initialisation for all projection matrices."""
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape (batch, seq, d_model) → (batch, num_heads, seq, d_k).
        For query the seq dimension is 1; for keys/values it is N (num_segments).
        """
        batch, seq, _ = x.shape
        x = x.view(batch, seq, self.num_heads, self.d_k)
        return x.permute(0, 2, 1, 3)  # (batch, num_heads, seq, d_k)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape (batch, num_heads, seq, d_k) → (batch, seq, d_model).
        """
        batch, _, seq, _ = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch, seq, self.d_model)

    def forward(
        self,
        query_embed: torch.Tensor,
        segment_embeds: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cross-attention between text query and segment embeddings.

        Parameters
        ----------
        query_embed : torch.Tensor
            Text query embedding.  Shape: (input_dim,) or (1, input_dim).
        segment_embeds : torch.Tensor
            Stacked segment embeddings.  Shape: (N, input_dim).

        Returns
        -------
        context : torch.Tensor
            Attention-weighted context vector.  Shape: (input_dim,).
        attn_weights : torch.Tensor
            Per-segment attention scores (post-softmax, averaged over heads).
            Shape: (N,) — used directly for ranking.
        """
        # Normalise inputs
        if query_embed.dim() == 1:
            query_embed = query_embed.unsqueeze(0)   # (1, input_dim)
        query_embed = self.norm_q(query_embed)       # stable normalisation

        # Add batch dimension:  (1, 1, input_dim) and (1, N, input_dim)
        Q = query_embed.unsqueeze(0)                  # (1, 1, input_dim)
        K = segment_embeds.unsqueeze(0)               # (1, N, input_dim)
        V = segment_embeds.unsqueeze(0)               # (1, N, input_dim)

        # Linear projections
        Q = self.W_q(Q)   # (1, 1, d_model)
        K = self.W_k(K)   # (1, N, d_model)
        V = self.W_v(V)   # (1, N, d_model)

        # Split into heads
        Q = self._split_heads(Q)   # (1, num_heads, 1, d_k)
        K = self._split_heads(K)   # (1, num_heads, N, d_k)
        V = self._split_heads(V)   # (1, num_heads, N, d_k)

        # Scaled dot-product attention
        # scores: (1, num_heads, 1, N)
        scale = math.sqrt(self.d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

        attn_weights_heads = F.softmax(scores, dim=-1)  # (1, num_heads, 1, N)
        attn_weights_heads = self.attn_dropout(attn_weights_heads)

        # Context: (1, num_heads, 1, d_k)
        context = torch.matmul(attn_weights_heads, V)

        # Merge heads back: (1, 1, d_model)
        context = self._merge_heads(context)

        # Output projection: (1, 1, input_dim)
        context = self.W_o(context)

        # Residual + final norm (add query as residual)
        context = self.norm_out(context + query_embed.unsqueeze(0))

        # Collapse batch and query dims: (input_dim,)
        context = context.squeeze(0).squeeze(0)

        # Collapse heads → per-segment scalar weights: (N,)
        attn_weights = attn_weights_heads.squeeze(0).mean(dim=0).squeeze(0)  # (N,)

        return context, attn_weights

    def rank_segments(
        self,
        query_embed: torch.Tensor,
        segment_embeds: torch.Tensor,
        top_k: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience wrapper: forward pass + return top-k segment indices and scores.

        Parameters
        ----------
        query_embed : torch.Tensor
            Shape: (input_dim,)
        segment_embeds : torch.Tensor
            Shape: (N, input_dim)
        top_k : int
            Number of top segments to return.

        Returns
        -------
        top_indices : torch.Tensor  shape (top_k,)  — 0-based segment indices
        top_scores  : torch.Tensor  shape (top_k,)  — corresponding attention scores
        """
        self.eval()
        with torch.no_grad():
            _, attn_weights = self.forward(query_embed, segment_embeds)

        actual_k = min(top_k, segment_embeds.shape[0])
        top_scores, top_indices = torch.topk(attn_weights, k=actual_k)
        # Sort by descending score for clean presentation
        order = torch.argsort(top_scores, descending=True)
        return top_indices[order], top_scores[order]


# ─────────────────────────────────────────────────────────────────────────────
# Standalone validation block
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("CrossAttentionFusion — Dimension Alignment Validation")
    print("=" * 60)

    INPUT_DIM = 1024
    D_MODEL = 256
    NUM_HEADS = 4
    N_SEGMENTS = 10
    TOP_K = 5

    model = CrossAttentionFusion(
        input_dim=INPUT_DIM,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dropout=0.0,
    )
    model.eval()
    print(f"\nModel instantiated successfully.")
    print(f"  input_dim : {model.input_dim}")
    print(f"  d_model   : {model.d_model}")
    print(f"  num_heads : {model.num_heads}")
    print(f"  d_k       : {model.d_k}")

    # Mock tensors
    torch.manual_seed(42)
    query_embed = torch.randn(INPUT_DIM)
    segment_embeds = torch.randn(N_SEGMENTS, INPUT_DIM)

    print(f"\nInput shapes:")
    print(f"  query_embed    : {tuple(query_embed.shape)}")
    print(f"  segment_embeds : {tuple(segment_embeds.shape)}")

    context, attn_weights = model(query_embed, segment_embeds)

    print(f"\nOutput shapes:")
    print(f"  context      : {tuple(context.shape)}")
    print(f"  attn_weights : {tuple(attn_weights.shape)}")

    # Assertions
    assert context.shape == (INPUT_DIM,), (
        f"context shape mismatch: expected ({INPUT_DIM},), got {tuple(context.shape)}"
    )
    assert attn_weights.shape == (N_SEGMENTS,), (
        f"attn_weights shape mismatch: expected ({N_SEGMENTS},), got {tuple(attn_weights.shape)}"
    )
    assert abs(attn_weights.sum().item() - 1.0) < 1e-5, (
        f"Attention weights do not sum to 1.0: {attn_weights.sum().item()}"
    )

    top_indices, top_scores = model.rank_segments(query_embed, segment_embeds, top_k=TOP_K)
    print(f"\ntop_{TOP_K} segment indices : {top_indices.tolist()}")
    print(f"top_{TOP_K} attention scores : {[round(s, 6) for s in top_scores.tolist()]}")

    assert top_indices.shape == (TOP_K,), (
        f"top_indices shape mismatch: expected ({TOP_K},), got {tuple(top_indices.shape)}"
    )
    assert all(0 <= idx < N_SEGMENTS for idx in top_indices.tolist()), (
        "top_indices contains out-of-range segment index."
    )

    print(f"\n[PASS] All dimension and value assertions satisfied.")
    print("=" * 60)
    sys.exit(0)
