"""Feature engineering for the explainable input vector (input_exp).

Refactored from original/functions/generate_inputs.py with:
- No TensorFlow dependency (to_categorical replaced with numpy)
- align_exp_dims extracted as its own function
- Typed signatures
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# One-hot helpers (no TF dependency)
# ---------------------------------------------------------------------------

def _to_categorical(indices: np.ndarray, num_classes: int) -> np.ndarray:
    """One-hot encode integer index array.  Shape: (*indices.shape, num_classes)."""
    out = np.zeros(indices.shape + (num_classes,), dtype=np.float32)
    np.put_along_axis(
        out,
        np.expand_dims(indices.astype(np.int64), axis=-1),
        1.0,
        axis=-1,
    )
    return out


# ---------------------------------------------------------------------------
# Build raw dynamic input  (B, T, D_dynamic)
# ---------------------------------------------------------------------------

def build_dynamic_input(
    vec: dict,
    indexes: Dict,
    max_len: int,
    cat_dynamic: List[str],
    num_dynamic: List[str],
) -> np.ndarray:
    """Build (N, max_len, D_dynamic) raw input for the Transformer encoder."""
    parts = []
    for feat in cat_dynamic:
        feat_index = indexes["dynamic_indexes"][f"{feat}_index"]
        ohe = _to_categorical(
            vec["prefixes"][feat][:, :max_len].astype(np.int64),
            num_classes=len(feat_index),
        )  # (N, T, C)
        if ohe.ndim < 3:
            ohe = ohe.reshape(ohe.shape[0], 1, ohe.shape[1])
        parts.append(ohe)

    for feat in num_dynamic:
        num = vec["prefixes"][feat][:, :max_len]  # (N, T)
        parts.append(num.reshape(num.shape[0], num.shape[1], 1))

    return np.concatenate(parts, axis=2)  # (N, T, D_dynamic)


# ---------------------------------------------------------------------------
# Build static input  (B, D_static)
# ---------------------------------------------------------------------------

def build_static_input(
    vec: dict,
    indexes: Dict,
    cat_static: List[str],
    num_static: List[str],
) -> np.ndarray:
    """Build (N, D_static) static input vector."""
    parts = []
    for feat in cat_static:
        feat_index = indexes["static_indexes"][f"{feat}_index"]
        ohe = _to_categorical(
            vec["statics"][feat].astype(np.int64),
            num_classes=len(feat_index),
        )  # (N, C)
        parts.append(ohe)

    for feat in num_static:
        num = vec["statics"][feat].reshape(-1, 1)
        parts.append(num)

    if not parts:
        n = len(vec["outcome"])
        return np.zeros((n, 0), dtype=np.float32)

    return np.concatenate(parts, axis=1)  # (N, D_static)


# ---------------------------------------------------------------------------
# Build explainable (aggregated) input  (B, exp_dim)
# ---------------------------------------------------------------------------

def build_exp_input(
    vec: dict,
    indexes: Dict,
    max_len: int,
    cat_dynamic: List[str],
    num_dynamic: List[str],
    cat_static: List[str],
    num_static: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """Build (N, exp_dim) aggregated feature vector and corresponding feature names.

    Categorical dynamic features: summed one-hot over time (occurrence counts).
    Numerical dynamic features:
      - TIME_ELAPSED: max over time
      - others (TASK_DURATION): mean over time
    Static features: appended as-is.
    """
    agg_parts = []
    feature_names: List[str] = []

    for feat in cat_dynamic:
        feat_index = indexes["dynamic_indexes"][f"{feat}_index"]
        ohe = _to_categorical(
            vec["prefixes"][feat][:, :max_len].astype(np.int64),
            num_classes=len(feat_index),
        )  # (N, T, C)
        if ohe.ndim < 3:
            ohe = ohe.reshape(ohe.shape[0], 1, ohe.shape[1])
        agg_parts.append(ohe.sum(axis=1))  # (N, C) — occurrence counts

        if feat == "ROLE":
            names = [
                f"Participation of {feat}_{k}"
                for k in feat_index
            ]
        else:
            names = [f"Occurrence of {feat}_{k}" for k in feat_index]
        feature_names.extend(names)

    for feat in num_dynamic:
        num = vec["prefixes"][feat][:, :max_len]  # (N, T)
        if feat == "TIME_ELAPSED":
            agg = num.max(axis=1, keepdims=True)  # (N, 1)
            feature_names.append(f"Overall {feat}")
        else:
            agg = num.mean(axis=1, keepdims=True)  # (N, 1)
            feature_names.append(f"Average {feat}")
        agg_parts.append(agg)

    raw_input_exp_ag = np.concatenate(agg_parts, axis=1)  # (N, D_dynamic_agg)

    # Static features
    static_input = build_static_input(vec, indexes, cat_static, num_static)

    for feat in cat_static:
        feat_index = indexes["static_indexes"][f"{feat}_index"]
        feature_names.extend([f"{feat}_{k}" for k in feat_index])
    for feat in num_static:
        feature_names.append(feat)

    if static_input.shape[1] > 0:
        input_exp = np.concatenate([raw_input_exp_ag, static_input], axis=1)
    else:
        input_exp = raw_input_exp_ag

    return input_exp.astype(np.float32), feature_names


# ---------------------------------------------------------------------------
# Align feature dimensions between train and test buckets
# ---------------------------------------------------------------------------

def align_exp_dims(
    inputs_train: dict,
    inputs_test: dict,
    features_train: List[str],
    features_test: List[str],
) -> Tuple[dict, dict, List[str]]:
    """Zero-pad exp_input to the union of features across train and test.

    When a train bucket has more activity categories than the test bucket
    (or vice versa), the shorter matrix is padded with zeros so both have
    the same exp_dim.

    Returns:
        (inputs_train, inputs_test, aligned_feature_names)
    """
    dim_train = inputs_train["input_exp"].shape[1]
    dim_test = inputs_test["input_exp"].shape[1]

    if dim_train == dim_test:
        return inputs_train, inputs_test, features_train

    if dim_train > dim_test:
        pad = np.zeros((inputs_test["input_exp"].shape[0], dim_train - dim_test), dtype=np.float32)
        inputs_test["input_exp"] = np.concatenate([inputs_test["input_exp"], pad], axis=1)
        return inputs_train, inputs_test, features_train
    else:
        pad = np.zeros((inputs_train["input_exp"].shape[0], dim_test - dim_train), dtype=np.float32)
        inputs_train["input_exp"] = np.concatenate([inputs_train["input_exp"], pad], axis=1)
        return inputs_train, inputs_test, features_test


# ---------------------------------------------------------------------------
# High-level: generate all inputs + outputs for a bucket
# ---------------------------------------------------------------------------

def generate_inputs(
    vec: dict,
    indexes: Dict,
    max_len: int,
    cat_dynamic: List[str],
    num_dynamic: List[str],
    cat_static: List[str],
    num_static: List[str],
) -> Tuple[dict, dict, int, List[str]]:
    """Build all model inputs and outputs for a single prefix-length bucket.

    Returns:
        inputs:       dict with 'input_dynamic', 'input_static', 'input_exp'
        outputs:      dict with 'act_output'
        exp_dim:      dimensionality of input_exp
        feature_names: ordered list of exp feature names
    """
    inputs: dict = {}

    inputs["input_dynamic"] = build_dynamic_input(
        vec, indexes, max_len, cat_dynamic, num_dynamic
    )
    inputs["input_static"] = build_static_input(vec, indexes, cat_static, num_static)

    input_exp, feature_names = build_exp_input(
        vec, indexes, max_len, cat_dynamic, num_dynamic, cat_static, num_static
    )
    inputs["input_exp"] = input_exp

    outputs: dict = {"act_output": vec["outcome"].astype(np.int64)}

    return inputs, outputs, int(input_exp.shape[1]), list(feature_names)
