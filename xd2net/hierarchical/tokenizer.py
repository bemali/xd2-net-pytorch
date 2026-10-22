"""Value-aware field tokenizer (FT-Transformer style).

One token per field per event; the token encodes the (field, value) pair:
    categorical: E_field[f] + E_val_f[v]
    numeric:     E_field[f] + x · w_f + b_f
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from xd2net.common.data.features import EventSchema


class FieldTokenizer(nn.Module):
    def __init__(self, schema: EventSchema, d_model: int):
        super().__init__()
        self.n_cat = schema.n_cat
        self.n_num = schema.n_num
        self.field_emb = nn.Embedding(schema.n_fields, d_model)
        self.val_emb = nn.ModuleList(
            nn.Embedding(card + 1, d_model, padding_idx=0) for card in schema.cardinalities
        )
        self.num_w = nn.Parameter(torch.randn(self.n_num, d_model) * 0.02)
        self.num_b = nn.Parameter(torch.zeros(self.n_num, d_model))

    def forward(self, packed: Tensor) -> Tensor:
        """
        Args:
            packed: (B, T, n_cat + n_num) float — shifted category indices then numeric values

        Returns:
            tokens: (B, T, n_fields, d_model)
        """
        toks = [emb(packed[..., i].long()) for i, emb in enumerate(self.val_emb)]
        if self.n_num:
            num = packed[..., self.n_cat:].unsqueeze(-1)            # (B, T, n_num, 1)
            toks.extend((num * self.num_w + self.num_b).unbind(dim=2))
        return torch.stack(toks, dim=2) + self.field_emb.weight
