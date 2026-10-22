"""Transformer encoder components replacing the alpha/beta BiLSTMs.

Architecture:
  InputProjection       — project raw features to d_model + learnable PE
  EventEncoder          — nn.TransformerEncoder (self-attention over events)
  TemporalAttentionPool — single-head cross-attention → alpha_weights (Level-1)
  FeatureWeightingBlock — position-wise MLP + softmax → beta_weights (Level-1)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Input projection + learnable positional embeddings
# ---------------------------------------------------------------------------

class InputProjection(nn.Module):
    """Project raw input (B, T, D_in) → (B, T, d_model) with positional embeddings.

    Uses learnable (not sinusoidal) positional embeddings because event
    sequences have irregular inter-event timing and variable padding, making
    fixed uniform-interval PE inappropriate.  Padding positions (indicated by
    pad_mask) receive zero embedding so they don't inject position information.
    """

    def __init__(self, dim_dynamic: int, d_model: int, max_len: int, dropout: float = 0.15):
        super().__init__()
        self.proj = nn.Linear(dim_dynamic, d_model)
        # Learnable positional embeddings; shape (1, max_len, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self._max_len = max_len

    def forward(self, x: Tensor, pad_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x:        (B, T, D_in)
            pad_mask: (B, T) bool, True at padding positions

        Returns:
            (B, T, d_model)
        """
        B, T, _ = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        pe = self.pos_emb(positions)  # (1, T, d_model)

        # Zero out PE at padding positions so padding tokens don't get positional signal
        if pad_mask is not None:
            pe = pe * (~pad_mask).unsqueeze(-1).float()

        out = self.proj(x) + pe
        return self.dropout(out)


# ---------------------------------------------------------------------------
# Transformer encoder body
# ---------------------------------------------------------------------------

class EventEncoder(nn.Module):
    """Wrap nn.TransformerEncoder with batch_first=True and pre-norm."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float = 0.15,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # pre-norm for training stability
        )
        # Nested tensors are unsupported with norm_first; disable explicitly to avoid the warning.
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)

    def forward(self, x: Tensor, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x:                   (B, T, d_model)
            src_key_padding_mask:(B, T) bool — True at padding positions

        Returns:
            (B, T, d_model)
        """
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)


# ---------------------------------------------------------------------------
# Temporal attention pooling — Level-1 explainability (alpha equivalent)
# ---------------------------------------------------------------------------

class TemporalAttentionPool(nn.Module):
    """Single-head cross-attention with a learned query vector.

    Produces alpha_weights (B, T): the importance of each event timestep.
    These are softmax-normalized over non-padded positions, exactly matching
    the original alpha layer's Softmax(axis=1) over the BiLSTM output.

    The resulting context vector (B, d_model) replaces the original
    sum(alpha * beta * raw_input, axis=time) context.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.xavier_uniform_(self.query.view(1, d_model).unsqueeze(0))
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=1,
            batch_first=True,
            bias=False,
        )

    def forward(
        self, x: Tensor, key_padding_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x:               (B, T, d_model)
            key_padding_mask:(B, T) bool — True at padding positions

        Returns:
            context:      (B, d_model)
            alpha_weights:(B, T)  in [0,1], sums to 1 over non-padded positions
        """
        B = x.size(0)
        q = self.query.expand(B, -1, -1)  # (B, 1, d_model)

        context, alpha = self.attn(
            q,
            x,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,   # single head
        )
        # context: (B, 1, d_model) → (B, d_model)
        # alpha:   (B, 1, T)       → (B, T)
        return context.squeeze(1), alpha.squeeze(1)


# ---------------------------------------------------------------------------
# Feature weighting block — Level-1 explainability (beta equivalent)
# ---------------------------------------------------------------------------

class FeatureWeightingBlock(nn.Module):
    """Position-wise linear projection of encoder output → per-feature weights.

    Replicates the beta BiLSTM: for each timestep, produces a weight over the
    D_dynamic input feature dimensions. The softmax is applied over the time
    axis (axis=1), matching the original beta Softmax(axis=1).

    These weights are used together with alpha_weights and raw_input to
    compute the context vector:
        c_t = sum(alpha.unsqueeze(-1) * beta * raw_input, dim=T)
    """

    def __init__(self, d_model: int, dim_dynamic: int):
        super().__init__()
        self.proj = nn.Linear(d_model, dim_dynamic)

    def forward(self, x: Tensor, pad_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x:        (B, T, d_model)
            pad_mask: (B, T) bool — True at padding positions

        Returns:
            beta: (B, T, D_dynamic)  — softmax over time axis
        """
        beta = torch.tanh(self.proj(x))  # (B, T, D_dynamic)
        if pad_mask is not None:
            # Mask padding so softmax ignores those positions
            beta = beta.masked_fill(pad_mask.unsqueeze(-1), float("-inf"))
        return torch.softmax(beta, dim=1)  # softmax over time
