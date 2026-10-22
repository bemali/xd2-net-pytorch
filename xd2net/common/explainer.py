"""Explanation extraction shared by every XD2-Net variant.

* Level 2 (thetas): ``explain_loader`` / ``global_explanation``, identical for all
  variants because they share ThetaHead / PredictionHead.
* Level 1 (attention): ``explain_attention_loader`` writes one model-independent
  cell table. Variants plug in through ``_attention_batch`` (which attention they
  have) and ``_decode_events`` (how their dynamic input encodes event values).
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, SequentialSampler

from xd2net.common.data.features import EventSchema

logger = logging.getLogger(__name__)


class Explainer:
    """Extract explanations and predictions from a trained XD2-Net variant.

    Args:
        model:  trained model (forward returns (thetas, prob, alpha); has ``prediction_head``
                and, for attention tables, a ``schema`` EventSchema).
        device: Device string ('cpu', 'cuda', 'mps') or 'auto'.
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
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
    # Level-2: predictions + theta explanations
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
        if len(caseids) != len(probs_np):
            raise ValueError(f"caseids has {len(caseids)} entries but the loader produced {len(probs_np)} cases.")

        # Feature importance = thetas * final_layer_weight (same as original)
        final_w = self.model.prediction_head.linear.weight.detach().cpu().numpy().reshape(-1)
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

        # ------ df_explanations (long format: case-major, feature order) ------
        n, e = exp_np.shape
        df_exp = pd.DataFrame(
            {
                "CASEID": np.repeat(np.asarray(caseids, dtype=object), e),
                "FEATURE": np.tile(np.asarray(feature_names, dtype=object), n),
                "FEATURE_VALUE": exp_np.reshape(-1).astype(float),
                "FEATURE_IMPORTANCE": importance.reshape(-1).astype(float),
                "BUCKET": bucket,
            }
        )

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
    # Level-1: attention cells (one row per case, real event, field)
    # ------------------------------------------------------------------

    def _attention_batch(
        self, raw_input: Tensor, raw_input_static: Tensor, input_exp: Tensor, lengths: Tensor
    ) -> Dict[str, Optional[np.ndarray]]:
        """Variant hook: return ``alpha`` (B,T) and, when the model has them,
        ``beta`` (B,T,F), ``evt_self_attn`` (B,T), ``case_self_attn`` (B,); else None."""
        alpha = self.get_temporal_attention(raw_input, raw_input_static, input_exp, lengths)
        return {"alpha": alpha, "beta": None, "evt_self_attn": None, "case_self_attn": None}

    def _decode_events(self, raw_ev: np.ndarray, schema: EventSchema) -> np.ndarray:
        """Variant hook: (n_ev, D) dynamic-input rows -> (n_ev, F) values, with
        categorical fields as shifted indices (see EventSchema) and numerics as is."""
        raise NotImplementedError

    def explain_attention_loader(
        self, loader: DataLoader, caseids: np.ndarray, bucket: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Build Level-1 attention tables in a model-independent schema.

        Returns:
            df_cells:       one row per (case, real event, field): CASEID, EVENT_POS
                            (0 = most recent), EVENT_INDEX (0 = first event),
                            EVENT_ACTIVITY, EVENT_ROLE, FIELD, VALUE_LABEL, VALUE_NUM,
                            EVENT_ATTN (alpha), FEATURE_ATTN (beta), JOINT_ATTN,
                            EVT_SELF_ATTN, BUCKET. Columns a model does not produce are NaN.
            df_case_fields: one row per (case, field) with JOINT_ATTN summed over events.
        """
        schema = getattr(self.model, "schema", None)
        if schema is None:
            raise ValueError("The model has no EventSchema; build it with schema=bucket['event_schema'].")
        if not isinstance(getattr(loader, "sampler", None), SequentialSampler):
            warnings.warn("Loader is not sequential; rows may not align with caseids.", stacklevel=2)

        acc: Dict[str, list] = {"raw": [], "lengths": []}
        for raw_input, raw_input_static, input_exp, _y, lengths in loader:
            for k, v in self._attention_batch(raw_input, raw_input_static, input_exp, lengths).items():
                acc.setdefault(k, []).append(v)
            acc["raw"].append(raw_input.cpu().numpy())
            acc["lengths"].append(lengths.cpu().numpy())
        data = {k: (None if any(x is None for x in v) else np.concatenate(v)) for k, v in acc.items()}

        alpha, lengths = data["alpha"], data["lengths"]
        N, T = alpha.shape
        F = schema.n_fields
        if len(caseids) != N:
            raise ValueError(f"caseids has {len(caseids)} entries but the loader produced {N} cases.")
        if data["beta"] is not None and data["beta"].shape[-1] != F:
            raise ValueError(f"schema has {F} fields, model attention has {data['beta'].shape[-1]}.")

        n_idx, t_idx = np.nonzero(np.arange(T)[None, :] < lengths[:, None])   # real events
        n_ev = len(n_idx)
        values = self._decode_events(data["raw"][n_idx, t_idx], schema)     # (n_ev, F)
        alpha_ev = alpha[n_idx, t_idx]
        beta_ev = data["beta"][n_idx, t_idx] if data["beta"] is not None else np.full((n_ev, F), np.nan)
        evt_self = (data["evt_self_attn"][n_idx, t_idx] if data["evt_self_attn"] is not None
                    else np.full(n_ev, np.nan))

        event_cols = {
            "CASEID": np.asarray(caseids, dtype=object)[n_idx],
            "EVENT_POS": t_idx,                                               # 0 = most recent
            "EVENT_INDEX": lengths[n_idx] - 1 - t_idx,                        # 0 = first event
        }
        # Label columns for the event's activity / role, for readability.
        for f_name in ("ACTIVITY", "ROLE"):
            if f_name in schema.cat_fields:
                j = schema.cat_fields.index(f_name)
                event_cols[f"EVENT_{f_name}"] = _labels(values[:, j], schema.index_to_label[f_name])

        frames = []
        for j, f_name in enumerate(schema.field_names):
            is_cat = j < schema.n_cat
            frames.append(pd.DataFrame({
                **event_cols,
                "FIELD": f_name,
                "VALUE_LABEL": _labels(values[:, j], schema.index_to_label[f_name]) if is_cat else "",
                "VALUE_NUM": np.full(n_ev, np.nan) if is_cat else values[:, j],
                "EVENT_ATTN": alpha_ev,
                "FEATURE_ATTN": beta_ev[:, j],
                "JOINT_ATTN": alpha_ev * beta_ev[:, j],
                "EVT_SELF_ATTN": evt_self,
                "BUCKET": bucket,
            }))
        df_cells = pd.concat(frames, ignore_index=True)
        # Order: loader case order, chronological event, schema field order.
        case_rank = np.tile(n_idx, F)
        field_rank = np.repeat(np.arange(F), n_ev)
        order = np.lexsort((field_rank, df_cells["EVENT_INDEX"].to_numpy(), case_rank))
        df_cells = df_cells.iloc[order].reset_index(drop=True)

        df_case = (
            df_cells.groupby(["CASEID", "FIELD"], sort=False)["JOINT_ATTN"].sum(min_count=1)
            .rename("CASE_FIELD_ATTN").reset_index()
        )
        case_self = data["case_self_attn"] if data["case_self_attn"] is not None else np.full(N, np.nan)
        df_case["CASE_SELF_ATTN"] = df_case["CASEID"].map(
            pd.Series(case_self, index=pd.Index(caseids, name="CASEID"))
        )
        df_case["BUCKET"] = bucket
        return df_cells, df_case

    # ------------------------------------------------------------------
    # Global summaries
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

    @staticmethod
    def global_attention(df_cells: pd.DataFrame, by: str = "field") -> pd.DataFrame:
        """Mean per-case attention by field, or by (field, label).

        by="field":       mean over cases of sum_t JOINT_ATTN per field
        by="field_label": mean over cases of sum_{t: label=v} JOINT_ATTN per (field, label),
                          plus N_CASES = number of cases containing the label
                          (numeric fields are reported with an empty label)
        Only meaningful for models with field attention (JOINT_ATTN not NaN).
        """
        n_cases = df_cells["CASEID"].nunique()
        if by == "field":
            keys = ["FIELD"]
        elif by == "field_label":
            keys = ["FIELD", "VALUE_LABEL"]
        else:
            raise ValueError("by must be 'field' or 'field_label'")
        per_case = df_cells.groupby(["CASEID", *keys], sort=False)["JOINT_ATTN"].sum().reset_index()
        out = per_case.groupby(keys)["JOINT_ATTN"].agg(ATTENTION="sum", N_CASES="count").reset_index()
        out["ATTENTION"] = out["ATTENTION"] / max(n_cases, 1)   # mean over all cases (absent = 0)
        return out.sort_values("ATTENTION", ascending=False).reset_index(drop=True)


def _labels(shifted_idx: np.ndarray, index_to_label: Dict[int, str]) -> np.ndarray:
    return pd.Series(shifted_idx.astype(np.int64)).map(index_to_label).fillna("<pad>").to_numpy()
