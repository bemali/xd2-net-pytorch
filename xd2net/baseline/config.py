"""Model configuration for the baseline XD2-Net (Transformer + TemporalAttentionPool).

Training, experiment and dataset configs are shared by every variant and live in
``xd2net.common.config``. ``l2_reg`` moved to ``TrainingConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BaselineModelConfig:
    d_model: int = 140
    nhead: int = 4
    num_encoder_layers: int = 1
    dim_feedforward_mult: int = 4
    dropout: float = 0.15
    dense_1_mult: float = 1.5

    def __post_init__(self):
        if self.d_model % self.nhead != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by nhead ({self.nhead}).")
        if self.num_encoder_layers < 1:
            raise ValueError("num_encoder_layers must be >= 1.")
