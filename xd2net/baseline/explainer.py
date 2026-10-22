"""Explanations for the baseline XD2Net.

Level-2 (thetas) explanations and the attention-cell table come from the shared
``Explainer``. The baseline's Level-1 signal is the TemporalAttentionPool
weight per event (alpha); it has no per-field attention, so the FEATURE_ATTN /
JOINT_ATTN / self-attention columns of the cell table are NaN.
"""

from __future__ import annotations

import numpy as np

from xd2net.baseline.model import XD2Net
from xd2net.common.data.features import EventSchema
from xd2net.common.explainer import Explainer


class BaselineExplainer(Explainer):
    """Explainer for the baseline XD2Net."""

    def __init__(self, model: XD2Net, device: str = "cpu"):
        super().__init__(model, device=device)

    def _decode_events(self, raw_ev: np.ndarray, schema: EventSchema) -> np.ndarray:
        # One-hot block per categorical field (index 0 = "none"), then one column per numeric field.
        width = sum(schema.cardinalities) + schema.n_num
        if raw_ev.shape[-1] != width:
            raise ValueError(f"schema implies {width} one-hot/numeric columns, input has {raw_ev.shape[-1]}.")
        out = np.empty((raw_ev.shape[0], schema.n_fields), dtype=np.float32)
        start = 0
        for j, card in enumerate(schema.cardinalities):
            out[:, j] = raw_ev[:, start:start + card].argmax(axis=1) + 1   # shifted index
            start += card
        out[:, schema.n_cat:] = raw_ev[:, start:]
        return out
