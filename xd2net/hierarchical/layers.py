"""Self-attention building blocks that expose their attention weights.

``nn.TransformerEncoderLayer`` calls MultiheadAttention with ``need_weights=False``,
so its attention cannot be read. ``AttnEncoderLayer`` is an equivalent pre-norm
layer that returns per-head weights, and ``cls_readout`` turns them into the
CLS-row explanation (head mean, optional rollout, self-mass renormalisation).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class AttnEncoderLayer(nn.Module):
    """Pre-norm Transformer encoder layer returning attention weights."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.15):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self, x: Tensor, key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x:                (N, L, d)
            key_padding_mask: (N, L) bool, True at padding positions
            attn_mask:        (L, L) bool, True where attention is not allowed

        Returns:
            out:  (N, L, d)
            attn: (N, H, L, L) per-head attention (rows sum to 1)
        """
        h = self.norm1(x)
        a, w = self.self_attn(
            h, h, h,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        x = x + self.dropout1(a)
        x = x + self.dropout2(self.ff(self.norm2(x)))
        return x, w


class AttnEncoder(nn.Module):
    """Stack of AttnEncoderLayer with a final LayerNorm (pre-norm convention)."""

    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, dropout: float = 0.15):
        super().__init__()
        self.layers = nn.ModuleList(
            AttnEncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self, x: Tensor, key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, List[Tensor]]:
        attns: List[Tensor] = []
        for layer in self.layers:
            x, w = layer(x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            attns.append(w)
        return self.final_norm(x), attns


def cls_attn_mask(length: int, cls_self_attend: bool, device=None) -> Optional[Tensor]:
    """(L, L) mask blocking the CLS token (position 0) from attending to itself.

    With the self cell blocked, the CLS attention row is a distribution over the
    input tokens only, and the CLS summary is a convex combination of their values.
    """
    if cls_self_attend:
        return None
    mask = torch.zeros(length, length, dtype=torch.bool, device=device)
    mask[0, 0] = True
    return mask


# ---------------------------------------------------------------------------
# Explanation readout
# ---------------------------------------------------------------------------

def attention_rollout(attns: List[Tensor]) -> Tensor:
    """Attention rollout (Abnar & Zuidema, 2020) on head-averaged attention.

    Each layer: Â = 0.5 · mean_h(A) + 0.5 · I (residual), row-normalised;
    R = Â_L · … · Â_1.

    Args:
        attns: list of (N, H, L, L)

    Returns:
        (N, L, L)
    """
    rollout: Optional[Tensor] = None
    for a in attns:
        a = a.mean(dim=1)
        eye = torch.eye(a.size(-1), device=a.device, dtype=a.dtype).expand_as(a)
        a = 0.5 * a + 0.5 * eye
        a = a / a.sum(dim=-1, keepdim=True)
        rollout = a if rollout is None else torch.bmm(a, rollout)
    return rollout


def cls_readout(
    attns: List[Tensor], mode: str = "rollout", token_mask: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor]:
    """Read the CLS (position 0) attention row as an explanation.

    Args:
        attns:      list of (N, H, L, L) per-layer attention
        mode:       "last" (head-mean of the last layer) or "rollout"
        token_mask: (N, L-1) bool, True at padded non-CLS tokens (weight forced to 0)

    Returns:
        weights:   (N, L-1) over non-CLS tokens, renormalised to sum to 1
        self_mass: (N,) head-mean attention the CLS token paid to itself in the
                   last layer (raw attention, not rollout: the rollout identity
                   term would put it at >= 0.5 by construction)
    """
    last = attns[-1].mean(dim=1)[:, 0, :]
    if mode == "last":
        row = last
    elif mode == "rollout":
        row = attention_rollout(attns)[:, 0, :]
    else:
        raise ValueError(f"Unknown readout mode {mode!r}")
    self_mass = last[:, 0]
    w = row[:, 1:]
    if token_mask is not None:
        w = w.masked_fill(token_mask, 0.0)
    w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return w, self_mass
