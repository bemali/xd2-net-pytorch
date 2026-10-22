"""XD2-Net with hierarchical self-attention.

  packed (B,T,F) → FieldTokenizer → (B·T, F, d) + [EVT]
    → Level A (event encoder)  → event rep (B,T,d),  [EVT] attention row → beta  (B,T,F)
  event rep + position emb + [CASE]
    → Level B (case encoder)   → case rep (B,d),     [CASE] attention row → alpha (B,T)
  [case rep, static] → ThetaHead → thetas → PredictionHead(thetas, input_exp) → prob

Level-2 heads are the unchanged baseline ThetaHead / PredictionHead.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import List, NamedTuple, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from xd2net.hierarchical.config import HierModelConfig
from xd2net.common.data.features import EventSchema
from xd2net.hierarchical.layers import AttnEncoder, cls_attn_mask, cls_readout
from xd2net.hierarchical.tokenizer import FieldTokenizer
from xd2net.common.masks import make_padding_mask
from xd2net.common.heads import PredictionHead, ThetaHead


class HierOutput(NamedTuple):
    thetas: Tensor               # (B, exp_dim)
    act_output: Tensor           # (B,) probabilities
    alpha: Tensor                # (B, T)    case → event; 0 at padding; sums to 1
    beta: Tensor                 # (B, T, F) event → field; 0 at padded events; sums to 1 per real event
    joint: Tensor                # (B, T, F) alpha · beta; sums to 1 per case
    evt_self_attn: Tensor        # (B, T)    [EVT] self-mass; 0 at padding
    case_self_attn: Tensor       # (B,)      [CASE] self-mass
    event_attn_raw: List[Tensor]  # per layer (B·T, H, 1+F, 1+F)
    case_attn_raw: List[Tensor]   # per layer (B, H, 1+T, 1+T)


def _cls_param(d_model: int) -> nn.Parameter:
    return nn.Parameter(torch.randn(1, 1, d_model) * 0.02)


class EventLevelEncoder(nn.Module):
    """Level A: self-attention over the field tokens of each event (shared across events)."""

    def __init__(
        self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, dropout: float,
        cls_self_attend: bool = False,
    ):
        super().__init__()
        self.cls_self_attend = cls_self_attend
        self.evt_token = _cls_param(d_model)
        self.encoder = AttnEncoder(d_model, nhead, num_layers, dim_feedforward, dropout)

    def forward(self, tokens: Tensor, readout: str) -> Tuple[Tensor, Tensor, Tensor, List[Tensor]]:
        B, T, F, d = tokens.shape
        x = tokens.reshape(B * T, F, d)
        x = torch.cat([self.evt_token.expand(B * T, -1, -1), x], dim=1)   # (B·T, 1+F, d)
        out, attns = self.encoder(x, attn_mask=cls_attn_mask(1 + F, self.cls_self_attend, x.device))
        evt_rep = out[:, 0].reshape(B, T, d)
        beta, evt_self = cls_readout(attns, readout)
        return evt_rep, beta.reshape(B, T, F), evt_self.reshape(B, T), attns


class CaseLevelEncoder(nn.Module):
    """Level B: self-attention over event representations with a [CASE] token."""

    def __init__(
        self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, dropout: float, max_len: int,
        cls_self_attend: bool = False,
    ):
        super().__init__()
        self.cls_self_attend = cls_self_attend
        self.case_token = _cls_param(d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.encoder = AttnEncoder(d_model, nhead, num_layers, dim_feedforward, dropout)

    def forward(
        self, evt_rep: Tensor, pad_mask: Optional[Tensor], readout: str
    ) -> Tuple[Tensor, Tensor, Tensor, List[Tensor]]:
        B, T, _ = evt_rep.shape
        pe = self.pos_emb(torch.arange(T, device=evt_rep.device)).unsqueeze(0)   # (1, T, d)
        if pad_mask is not None:
            pe = pe * (~pad_mask).unsqueeze(-1).to(pe.dtype)
        x = self.dropout(evt_rep + pe)
        x = torch.cat([self.case_token.expand(B, -1, -1), x], dim=1)            # (B, 1+T, d)
        kpm = None
        if pad_mask is not None:
            kpm = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=pad_mask.device), pad_mask], dim=1)
        mask = cls_attn_mask(1 + T, self.cls_self_attend, x.device)
        out, attns = self.encoder(x, key_padding_mask=kpm, attn_mask=mask)
        alpha, case_self = cls_readout(attns, readout, token_mask=pad_mask)
        return out[:, 0], alpha, case_self, attns


class XD2NetHier(nn.Module):
    """Hierarchical self-attention XD2-Net.

    Args:
        schema:       EventSchema describing the packed dynamic input
        dim_static:   width of the static (case-level) vector
        exp_dim:      number of explainable (aggregated) features
        max_len:      maximum prefix length (size of the position embedding)
        cfg:          HierModelConfig
        dataset_name: optional label stored in get_config()
    """

    def __init__(
        self,
        schema: EventSchema,
        dim_static: int,
        exp_dim: int,
        max_len: int,
        cfg: Optional[HierModelConfig] = None,
        dataset_name: Optional[str] = None,
    ):
        super().__init__()
        cfg = cfg or HierModelConfig()
        self.schema = schema
        self.cfg = cfg
        self.dim_static = dim_static
        self.exp_dim = exp_dim
        self.max_len = max_len
        self.dataset_name = dataset_name

        d = cfg.d_model
        ff = d * cfg.dim_feedforward_mult
        # Every encoder parameter lives under `encoder`: the baseline Trainer applies L2 to it.
        self.encoder = nn.ModuleDict({
            "tokenizer": FieldTokenizer(schema, d),
            "event": EventLevelEncoder(d, cfg.nhead, cfg.event_layers, ff, cfg.dropout, cfg.cls_self_attend),
            "case": CaseLevelEncoder(d, cfg.nhead, cfg.case_layers, ff, cfg.dropout, max_len, cfg.cls_self_attend),
        })
        self.theta_head = ThetaHead(d + dim_static, exp_dim, cfg.dense_1_mult)
        self.prediction_head = PredictionHead(exp_dim)

    def forward(
        self,
        raw_input: Tensor,
        raw_input_static: Tensor,
        input_exp: Tensor,
        lengths: Optional[Tensor] = None,
        return_attn: bool = False,
    ) -> Union[Tuple[Tensor, Tensor, Tensor], HierOutput]:
        """
        Args:
            raw_input:        (B, T, n_cat + n_num) packed tokens
            raw_input_static: (B, D_static)
            input_exp:        (B, exp_dim)
            lengths:          (B,) real sequence lengths; None = no padding
            return_attn:      return a HierOutput instead of (thetas, act_output, alpha)
        """
        T = raw_input.size(1)
        pad_mask = make_padding_mask(lengths, T) if lengths is not None else None
        readout = self.cfg.attn_readout

        tokens = self.encoder["tokenizer"](raw_input)
        evt_rep, beta, evt_self, ev_attns = self.encoder["event"](tokens, readout)
        case_rep, alpha, case_self, case_attns = self.encoder["case"](evt_rep, pad_mask, readout)

        c_full = torch.cat([case_rep, raw_input_static], dim=-1)
        thetas = self.theta_head(c_full)
        act_output = self.prediction_head(thetas, input_exp)

        if not return_attn:
            return thetas, act_output, alpha

        if pad_mask is not None:
            beta = beta.masked_fill(pad_mask.unsqueeze(-1), 0.0)
            evt_self = evt_self.masked_fill(pad_mask, 0.0)
        joint = alpha.unsqueeze(-1) * beta
        return HierOutput(thetas, act_output, alpha, beta, joint, evt_self, case_self, ev_attns, case_attns)

    # ------------------------------------------------------------------

    def get_config(self) -> dict:
        """Everything needed to rebuild the module (pair with a state_dict)."""
        return {
            "dataset_name": self.dataset_name,
            "schema": self.schema.to_dict(),
            "dim_static": self.dim_static,
            "exp_dim": self.exp_dim,
            "max_len": self.max_len,
            "model": asdict(self.cfg),
        }

    @classmethod
    def from_config(cls, config: dict) -> "XD2NetHier":
        return cls(
            schema=EventSchema.from_dict(config["schema"]),
            dim_static=config["dim_static"],
            exp_dim=config["exp_dim"],
            max_len=config["max_len"],
            cfg=HierModelConfig(**config["model"]),
            dataset_name=config.get("dataset_name"),
        )
