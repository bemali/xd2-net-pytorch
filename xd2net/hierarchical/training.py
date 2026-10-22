"""Trainer for XD2NetHier.

Uses the shared ``Trainer`` loop and narrows the L2 penalty: the attention
query/key projections, LayerNorm parameters and biases are not regularised.
"""

from __future__ import annotations

import torch

from xd2net.common.training import Trainer


class HierTrainer(Trainer):
    def _l2_penalty(self) -> torch.Tensor:
        """Sum of squares over encoder weights that do not set attention sharpness.

        Excluded: attention query/key projections, LayerNorm parameters and biases.
        Shrinking W_q / W_k or the LayerNorm gain feeding them scales the attention
        logits towards zero, which flattens the attention (and with it the Level-1
        explanation) to uniform. Value/output projections, FFN weights and
        embeddings are still regularised.
        """
        total = torch.zeros((), device=self.device)
        for name, p in self.model.encoder.named_parameters():
            if "norm" in name or (name.endswith("bias") and "in_proj" not in name):
                continue
            if name.endswith("self_attn.in_proj_weight"):
                p = p[2 * p.size(1):]          # keep only the V rows of [W_q; W_k; W_v]
            elif name.endswith("self_attn.in_proj_bias"):
                p = p[2 * (p.size(0) // 3):]
            total = total + p.pow(2).sum()
        return total
