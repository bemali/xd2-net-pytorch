"""Hierarchical attention explanations for XD2NetHier.

Level-2 (thetas) explanations and the attention-cell table come from the shared
``Explainer``. This module supplies the Level-1 hierarchy read from internal
self-attention:

    alpha[t]   case  → event         ([CASE] attention row)
    beta[t,f]  event → field         ([EVT] attention row)
    joint      alpha[t] · beta[t,f]  (sums to 1 per case)
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from torch import Tensor

from xd2net.common.data.features import EventSchema
from xd2net.common.explainer import Explainer
from xd2net.hierarchical.model import XD2NetHier


class HierExplainer(Explainer):
    """Explainer for XD2NetHier."""

    def __init__(self, model: XD2NetHier, device: str = "cpu"):
        super().__init__(model, device=device)

    def get_hierarchical_attention(
        self,
        raw_input: Tensor,
        raw_input_static: Tensor,
        input_exp: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Dict[str, np.ndarray]:
        """Return alpha (N,T), beta (N,T,F), joint (N,T,F), evt_self_attn (N,T), case_self_attn (N,)."""
        self.model.eval()
        with torch.no_grad():
            out = self.model(
                raw_input.to(self.device),
                raw_input_static.to(self.device),
                input_exp.to(self.device),
                lengths.to(self.device) if lengths is not None else None,
                return_attn=True,
            )
        return {
            "alpha": out.alpha.cpu().numpy(),
            "beta": out.beta.cpu().numpy(),
            "joint": out.joint.cpu().numpy(),
            "evt_self_attn": out.evt_self_attn.cpu().numpy(),
            "case_self_attn": out.case_self_attn.cpu().numpy(),
        }

    def _attention_batch(self, raw_input, raw_input_static, input_exp, lengths):
        att = self.get_hierarchical_attention(raw_input, raw_input_static, input_exp, lengths)
        del att["joint"]
        return att

    def _decode_events(self, raw_ev: np.ndarray, schema: EventSchema) -> np.ndarray:
        # Packed tokens already hold shifted category indices followed by numeric values.
        if raw_ev.shape[-1] != schema.n_fields:
            raise ValueError(f"schema has {schema.n_fields} fields, input has {raw_ev.shape[-1]}.")
        return raw_ev
