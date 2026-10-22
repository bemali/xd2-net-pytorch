"""Padding-mask helper shared by every XD2-Net variant."""

from __future__ import annotations

import torch
from torch import Tensor


def make_padding_mask(lengths: Tensor, max_len: int) -> Tensor:
    """Build (B, T) bool mask: True where position >= actual length (padding).

    Args:
        lengths: (B,) integer tensor of actual sequence lengths.
        max_len: total sequence length T.

    Returns:
        (B, T) bool tensor for use as src_key_padding_mask.
    """
    B = lengths.size(0)
    positions = torch.arange(max_len, device=lengths.device).unsqueeze(0).expand(B, -1)
    return positions >= lengths.unsqueeze(1)
