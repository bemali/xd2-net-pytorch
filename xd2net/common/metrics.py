"""Classification and ranking metrics for XD2-Net evaluation."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    target_names: Optional[List[str]] = None,
) -> Dict:
    """Compute classification metrics from probabilities and ground truth.

    Args:
        y_true:       (N,) integer labels (0 or 1)
        y_prob:       (N,) predicted probabilities in [0, 1]
        threshold:    binarization threshold (default 0.5)
        target_names: class names for the report

    Returns:
        dict with keys: accuracy, precision, recall, f1, roc_auc, report
    """
    y_pred = (y_prob >= threshold).astype(int)
    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "report": report,
    }


def feature_agreement(
    thetas: np.ndarray,
    baseline_coefs: np.ndarray,
    top_k: int = 5,
) -> Dict[str, float]:
    """Compute feature agreement and rank agreement between XD2-Net and a baseline.

    Args:
        thetas:          (N, F) per-case feature importance from XD2-Net
        baseline_coefs:  (F,) global feature coefficients from the baseline (e.g., LR)
        top_k:           how many top features to compare

    Returns:
        dict with 'feature_agreement' and 'rank_agreement' in [0, 1]
    """
    mean_thetas = thetas.mean(axis=0)  # (F,)
    xd2_top = set(np.argsort(mean_thetas)[::-1][:top_k])
    bl_top = set(np.argsort(np.abs(baseline_coefs))[::-1][:top_k])

    fa = len(xd2_top & bl_top) / top_k

    xd2_rank = np.argsort(np.argsort(mean_thetas)[::-1])
    bl_rank = np.argsort(np.argsort(np.abs(baseline_coefs))[::-1])
    rank_diffs = np.abs(xd2_rank[:top_k] - bl_rank[:top_k])
    ra = 1.0 - float(rank_diffs.mean()) / len(mean_thetas)

    return {"feature_agreement": fa, "rank_agreement": max(ra, 0.0)}
