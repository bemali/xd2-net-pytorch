"""Theta and Prediction heads — Level-2 explainability.

These are ported directly from the original Keras model with no structural change:
  ThetaHead:      two dense layers with softmax, producing thetas (explanation weights)
  PredictionHead: linear(exp_dim → 1) with ones initializer, sigmoid output
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ThetaHead(nn.Module):
    """Compute softmax explanation weights (thetas) from the context vector.

    Matches the original:
        Dense(1.5 * exp_dim, softmax, glorot_uniform)
        Dense(exp_dim, softmax, glorot_uniform)  ← explanation_weights layer

    Args:
        in_dim:        dimensionality of the context + static concat vector
        exp_dim:       number of explainable features
        dense_1_mult:  multiplier for the intermediate layer size (default 1.5)
    """

    def __init__(self, in_dim: int, exp_dim: int, dense_1_mult: float = 1.5):
        super().__init__()
        dense_1_dim = round(exp_dim * dense_1_mult)
        self.dense_1 = nn.Linear(in_dim, dense_1_dim)
        self.dense_2 = nn.Linear(dense_1_dim, exp_dim)
        # Glorot uniform is the PyTorch default for nn.Linear, so no explicit init needed.

    def forward(self, c_full: Tensor) -> Tensor:
        """
        Args:
            c_full: (B, in_dim) — context + static concat

        Returns:
            thetas: (B, exp_dim) — softmax-normalized explanation weights
        """
        x = torch.softmax(self.dense_1(c_full), dim=-1)
        return torch.softmax(self.dense_2(x), dim=-1)


class PredictionHead(nn.Module):
    """Final prediction: sigmoid(W * (thetas ⊙ input_exp)).

    W is initialized to all-ones (matching the original Ones() initializer),
    making the initial prediction a simple weighted sum of features.

    Args:
        exp_dim: number of explainable features
    """

    def __init__(self, exp_dim: int):
        super().__init__()
        self.linear = nn.Linear(exp_dim, 1, bias=True)
        nn.init.ones_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, thetas: Tensor, input_exp: Tensor) -> Tensor:
        """
        Args:
            thetas:    (B, exp_dim) — explanation weights from ThetaHead
            input_exp: (B, exp_dim) — aggregated explainable features

        Returns:
            (B,) prediction probabilities in (0, 1)
        """
        reg = thetas * input_exp  # element-wise: (B, exp_dim)
        return torch.sigmoid(self.linear(reg)).squeeze(-1)
