"""Explainability extraction for XD2-Net.

Extracts Level-1 (temporal alpha_weights) and Level-2 (thetas) explanations
and formats them as DataFrames matching the original output CSVs.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from xd2net.model.xd2net import XD2Net

logger = logging.getLogger(__name__)


class Explainer:
    """Extract explanations and predictions from a trained XD2-Net.

    Args:
        model:  Trained XD2Net instance.
        device: Device string ('cpu', 'cuda', 'mps') or 'auto'.
    """

    def __init__(self, model: XD2Net, device: str = "cpu"):
        self.model = model
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # High-level: process a DataLoader, return explanation DataFrames
    # ------------------------------------------------------------------

    def explain_loader(
        self,
        loader: DataLoader,
        caseids: np.ndarray,
        feature_names: List[str],
        bucket: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run inference over a DataLoader and build explanation DataFrames.

        Args:
            loader:        DataLoader yielding (raw_input, static, exp, label, lengths)
            caseids:       (N,) case identifiers matching loader order
            feature_names: ordered list of explainable feature names (length = exp_dim)
            bucket:        bucket label string (e.g. '1_15') for output column

        Returns:
            df_predictions:  per-case predictions (CASEID, PREDICTION_PROB, PREDICTION, GROUND_TRUTH, BUCKET)
            df_explanations: per-case feature importances (CASEID, FEATURE, FEATURE_VALUE, FEATURE_IMPORTANCE, BUCKET)
        """
        all_thetas: List[Tensor] = []
        all_probs: List[Tensor] = []
        all_labels: List[Tensor] = []
        all_exp: List[Tensor] = []

        with torch.no_grad():
            for batch in loader:
                raw_input, raw_input_static, input_exp, labels, lengths = (
                    t.to(self.device) for t in batch
                )
                thetas, act_output, _ = self.model(
                    raw_input, raw_input_static, input_exp, lengths
                )
                all_thetas.append(thetas.cpu())
                all_probs.append(act_output.cpu())
                all_labels.append(labels.cpu())
                all_exp.append(input_exp.cpu())

        thetas_np = torch.cat(all_thetas).numpy()        # (N, exp_dim)
        probs_np = torch.cat(all_probs).numpy()          # (N,)
        labels_np = torch.cat(all_labels).numpy()        # (N,)
        exp_np = torch.cat(all_exp).numpy()              # (N, exp_dim)

        # Feature importance = thetas * final_layer_weight (same as original)
        final_w = self.model.prediction_head.linear.weight.detach().cpu().numpy().squeeze()
        importance = thetas_np * final_w  # (N, exp_dim)

        # ------ df_predictions ------
        df_pred = pd.DataFrame(
            {
                "CASEID": caseids,
                "PREDICTION_PROB": probs_np,
                "PREDICTION": (probs_np >= 0.5).astype(int),
                "GROUND_TRUTH": labels_np.astype(int),
                "BUCKET": bucket,
            }
        )

        # ------ df_explanations (long format) ------
        rows = []
        for i in range(len(caseids)):
            for j, fname in enumerate(feature_names):
                rows.append(
                    {
                        "CASEID": caseids[i],
                        "FEATURE": fname,
                        "FEATURE_VALUE": float(exp_np[i, j]),
                        "FEATURE_IMPORTANCE": float(importance[i, j]),
                        "BUCKET": bucket,
                    }
                )
        df_exp = pd.DataFrame(rows)

        return df_pred, df_exp

    # ------------------------------------------------------------------
    # Level-1: temporal attention
    # ------------------------------------------------------------------

    def get_temporal_attention(
        self,
        raw_input: Tensor,
        raw_input_static: Tensor,
        input_exp: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> np.ndarray:
        """Return alpha_weights (N, T) for event-level attention visualization.

        Args:
            raw_input:        (N, T, D) dynamic input tensor
            raw_input_static: (N, S) static input tensor
            input_exp:        (N, E) explainable input tensor
            lengths:          (N,) actual sequence lengths

        Returns:
            alpha_weights: (N, T) numpy array
        """
        self.model.eval()
        with torch.no_grad():
            ri = raw_input.to(self.device)
            rs = raw_input_static.to(self.device)
            ie = input_exp.to(self.device)
            ln = lengths.to(self.device) if lengths is not None else None
            _, _, alpha = self.model(ri, rs, ie, ln)
        return alpha.cpu().numpy()

    # ------------------------------------------------------------------
    # Global summary
    # ------------------------------------------------------------------

    def global_explanation(
        self,
        df_explanations: pd.DataFrame,
        feature_names: List[str],
        agg: str = "mean",
    ) -> pd.DataFrame:
        """Aggregate per-case explanations into global feature importances.

        Args:
            df_explanations: output of explain_loader()
            feature_names:   ordered feature names
            agg:             'mean' or 'median'

        Returns:
            DataFrame with FEATURE and IMPORTANCE columns, sorted descending.
        """
        fn = df_explanations.groupby("FEATURE")["FEATURE_IMPORTANCE"]
        importance = fn.mean() if agg == "mean" else fn.median()
        df_global = importance.reset_index().rename(columns={"FEATURE_IMPORTANCE": "IMPORTANCE"})
        df_global = df_global.sort_values("IMPORTANCE", ascending=False).reset_index(drop=True)
        return df_global
