"""XD2-Net: Transformer-based inherently explainable process outcome predictor.

Replaces the original alpha/beta BiLSTM backbone with a Transformer encoder
while preserving both levels of inherent explainability:

  Level 1 — temporal (event-level):
    alpha_weights (B, T):  which events mattered (from TemporalAttentionPool)
    beta_weights  (B, T, D): per-feature weight per event (from FeatureWeightingBlock)

  Level 2 — feature-level:
    thetas (B, exp_dim): which aggregated features mattered (from ThetaHead)

The prediction path is identical to the original:
    prediction = sigmoid(W * (thetas ⊙ input_exp))   with W initialized to ones

Forward returns (thetas, act_output, alpha_weights) so the caller has all
signals needed for explanation without a second forward pass.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from xd2net.model.encoder import (
    EventEncoder,
    FeatureWeightingBlock,
    InputProjection,
    TemporalAttentionPool,
    make_padding_mask,
)
from xd2net.model.heads import PredictionHead, ThetaHead
from xd2net.config import ModelConfig


class XD2Net(nn.Module):
    """XD2-Net with Transformer encoder backbone.

    Args:
        dim_dynamic: dimensionality of the dynamic (sequence) feature vector
        dim_static:  dimensionality of the static (case-level) feature vector
        exp_dim:     number of explainable (aggregated) features
        max_len:     maximum padded sequence length (prefix length for this bucket)
        cfg:         ModelConfig controlling architecture hyperparameters
    """

    def __init__(
        self,
        dim_dynamic: int,
        dim_static: int,
        exp_dim: int,
        max_len: int,
        cfg: Optional[ModelConfig] = None,
    ):
        super().__init__()
        if cfg is None:
            cfg = ModelConfig()

        self.dim_dynamic = dim_dynamic
        self.dim_static = dim_static
        self.exp_dim = exp_dim
        self.max_len = max_len
        self.cfg = cfg

        d_model = cfg.d_model
        dim_ff = d_model * cfg.dim_feedforward_mult

        # --- Encoder ---
        self.input_proj = InputProjection(dim_dynamic, d_model, max_len, cfg.dropout)
        self.encoder = EventEncoder(
            d_model=d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_encoder_layers,
            dim_feedforward=dim_ff,
            dropout=cfg.dropout,
        )

        # --- Level-1 explainability ---
        self.temporal_pool = TemporalAttentionPool(d_model)
        self.feature_weighting = FeatureWeightingBlock(d_model, dim_dynamic)

        # --- Level-2 explainability ---
        context_dim = dim_dynamic + dim_static   # c_t + static concatenation
        self.theta_head = ThetaHead(context_dim, exp_dim, cfg.dense_1_mult)
        self.prediction_head = PredictionHead(exp_dim)

    # ------------------------------------------------------------------

    def forward(
        self,
        raw_input: Tensor,
        raw_input_static: Tensor,
        input_exp: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass.

        Args:
            raw_input:        (B, T, D_dynamic)
            raw_input_static: (B, D_static)
            input_exp:        (B, exp_dim)
            lengths:          (B,) actual sequence lengths — used to build
                              padding mask. If None, no masking is applied.

        Returns:
            thetas:       (B, exp_dim)  — Level-2 explanation weights
            act_output:   (B,)          — prediction probabilities in (0, 1)
            alpha_weights:(B, T)        — Level-1 temporal attention weights
        """
        # Build padding mask
        pad_mask: Optional[Tensor] = None
        if lengths is not None:
            pad_mask = make_padding_mask(lengths, raw_input.size(1))

        # --- Encoder ---
        x = self.input_proj(raw_input, pad_mask=pad_mask)           # (B, T, d_model)
        enc_out = self.encoder(x, src_key_padding_mask=pad_mask)    # (B, T, d_model)

        # --- Level-1: temporal attention (alpha) ---
        _, alpha_weights = self.temporal_pool(enc_out, key_padding_mask=pad_mask)
        # alpha_weights: (B, T) — used only for explainability output

        # --- Level-1: feature weighting (beta) ---
        beta_weights = self.feature_weighting(enc_out, pad_mask=pad_mask)
        # beta_weights: (B, T, D_dynamic)

        # --- Context vector (same formula as original) ---
        # c_t = sum(alpha.unsqueeze(-1) * beta * raw_input, axis=T)
        c_t = (alpha_weights.unsqueeze(-1) * beta_weights * raw_input).sum(dim=1)
        # c_t: (B, D_dynamic)

        # Concatenate with static features
        c_full = torch.cat([c_t, raw_input_static], dim=-1)  # (B, D_dynamic + D_static)

        # --- Level-2: thetas ---
        thetas = self.theta_head(c_full)          # (B, exp_dim)

        # --- Prediction ---
        act_output = self.prediction_head(thetas, input_exp)  # (B,)

        return thetas, act_output, alpha_weights
