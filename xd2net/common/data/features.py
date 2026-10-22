"""Model-independent inputs built from the ``vec`` dict of ``preprocessing.vectorize``.

* ``EventSchema`` describes the event fields (categorical then numeric) and their labels.
* ``build_static_input`` builds the case-level vector (one-hot / value).
* ``build_exp_input`` builds the explainable aggregated vector used by the
  Level-2 (theta) explanation: dataset-agnostic names, padding excluded.

Sequences in ``vec`` are reversed (position 0 = most recent event) and
right-padded; ``vec["lengths"]`` gives the real lengths.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np

AGGREGATIONS = ("mean", "max", "min", "sum", "last", "first")


# ---------------------------------------------------------------------------
# Event schema
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EventSchema:
    """Event fields of a prepared log: categorical fields first, then numeric ones.

    ``index_to_label`` is keyed by *shifted* index: 0 = PAD, ``k + 1`` = original
    category index k (as produced by ``create_index(reserve_zero=True)``, where
    k = 0 is ``"none"``). Each variant encodes the fields its own way.
    """

    cat_fields: List[str]
    num_fields: List[str]
    cardinalities: List[int]                    # per cat field, excluding PAD
    index_to_label: Dict[str, Dict[int, str]]   # shifted index (>= 1) -> label

    @property
    def n_cat(self) -> int:
        return len(self.cat_fields)

    @property
    def n_num(self) -> int:
        return len(self.num_fields)

    @property
    def field_names(self) -> List[str]:
        return [*self.cat_fields, *self.num_fields]

    @property
    def n_fields(self) -> int:
        return self.n_cat + self.n_num

    def to_dict(self) -> dict:
        d = asdict(self)
        # JSON keys must be strings
        d["index_to_label"] = {f: {str(k): v for k, v in m.items()} for f, m in self.index_to_label.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "EventSchema":
        return cls(
            cat_fields=list(d["cat_fields"]),
            num_fields=list(d["num_fields"]),
            cardinalities=[int(c) for c in d["cardinalities"]],
            index_to_label={f: {int(k): v for k, v in m.items()} for f, m in d["index_to_label"].items()},
        )


def build_event_schema(indexes: Dict, cat_fields: List[str], num_fields: List[str]) -> EventSchema:
    """Build an EventSchema from the baseline ``indexes['dynamic_indexes']`` dict."""
    cards, i2l = [], {}
    for feat in cat_fields:
        index = indexes["dynamic_indexes"][f"{feat}_index"]
        cards.append(len(index))
        i2l[feat] = {int(idx) + 1: str(label) for label, idx in index.items()}
    return EventSchema(list(cat_fields), list(num_fields), cards, i2l)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def valid_mask(lengths: np.ndarray, max_len: int) -> np.ndarray:
    """(N, T) bool, True at real (non-padded) positions."""
    return np.arange(max_len)[None, :] < np.asarray(lengths)[:, None]


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
# Static (case-level) input  (N, D_static)
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
# Explainable (aggregated) input  (N, exp_dim)
# ---------------------------------------------------------------------------

def default_agg(feat: str) -> str:
    return "max" if feat == "TIME_SINCE_START" else "mean"


def _aggregate(num: np.ndarray, valid: np.ndarray, lengths: np.ndarray, how: str) -> np.ndarray:
    n = num.shape[0]
    lens = np.maximum(np.asarray(lengths), 1)
    if how == "mean":
        return np.where(valid, num, 0.0).sum(1) / lens
    if how == "sum":
        return np.where(valid, num, 0.0).sum(1)
    if how == "max":
        return np.where(valid, num, -np.inf).max(1)
    if how == "min":
        return np.where(valid, num, np.inf).min(1)
    if how == "last":   # most recent event = position 0 (reversed sequences)
        return num[:, 0]
    if how == "first":  # chronologically first event = position length-1
        return num[np.arange(n), lens - 1]
    raise ValueError(f"Unknown aggregation {how!r}; use one of {AGGREGATIONS}.")


def build_exp_input(
    vec: dict,
    indexes: Dict,
    max_len: int,
    cat_fields: List[str],
    num_fields: List[str],
    case_cat: List[str],
    case_num: List[str],
    num_agg: Dict[str, str] | None = None,
) -> Tuple[np.ndarray, List[str]]:
    """Build ``(N, exp_dim)`` aggregated features and their names.

    * categorical event field: occurrence count per label, padding excluded
      → ``"Occurrence of FIELD=label"``
    * numeric event field: aggregation (default mean; TIME_SINCE_START max)
      → ``"Mean FIELD"``, ``"Max FIELD"``, ...
    * case-level fields: appended as in the baseline (one-hot / value)
      → ``"FIELD=label"`` / ``"FIELD"``
    """
    num_agg = num_agg or {}
    lengths = vec["lengths"]
    valid = valid_mask(lengths, max_len)
    parts: List[np.ndarray] = []
    names: List[str] = []

    for feat in cat_fields:
        index = indexes["dynamic_indexes"][f"{feat}_index"]
        idx = vec["prefixes"][feat][:, :max_len].astype(np.int64)
        labels = sorted(index.items(), key=lambda kv: kv[1])
        counts = np.stack([((idx == v) & valid).sum(1) for _, v in labels], axis=1)
        parts.append(counts.astype(np.float32))
        names.extend(f"Occurrence of {feat}={label}" for label, _ in labels)

    for feat in num_fields:
        how = num_agg.get(feat, default_agg(feat))
        agg = _aggregate(vec["prefixes"][feat][:, :max_len], valid, lengths, how)
        parts.append(agg.reshape(-1, 1).astype(np.float32))
        names.append(f"{how.capitalize()} {feat}")

    static = build_static_input(vec, indexes, case_cat, case_num)
    if static.shape[1]:
        parts.append(static.astype(np.float32))
        for feat in case_cat:
            index = indexes["static_indexes"][f"{feat}_index"]
            names.extend(f"{feat}={label}" for label, _ in sorted(index.items(), key=lambda kv: kv[1]))
        names.extend(case_num)

    exp = np.concatenate(parts, axis=1).astype(np.float32)
    assert exp.shape[1] == len(names), (exp.shape, len(names))
    return exp, names
