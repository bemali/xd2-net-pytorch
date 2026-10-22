"""Model configuration for the hierarchical self-attention XD2-Net.

Training, experiment and dataset configs are shared by every variant and live in
``xd2net.common.config``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HierModelConfig:
    d_model: int = 64
    nhead: int = 4
    event_layers: int = 1
    case_layers: int = 1
    dim_feedforward_mult: int = 4
    dropout: float = 0.15
    attn_readout: str = "rollout"   # "rollout" | "last"  (identical when layers == 1)
    cls_self_attend: bool = False   # False: [EVT]/[CASE] cannot attend to themselves
    dense_1_mult: float = 1.5

    def __post_init__(self):
        if self.d_model % self.nhead != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by nhead ({self.nhead}).")
        if self.attn_readout not in ("rollout", "last"):
            raise ValueError(f"attn_readout must be 'rollout' or 'last', got {self.attn_readout!r}.")
        if self.event_layers < 1 or self.case_layers < 1:
            raise ValueError("event_layers and case_layers must be >= 1.")
