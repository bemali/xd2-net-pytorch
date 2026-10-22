"""Data loading, encoding, and vectorization for XD2-Net.

Refactored from original/functions/data_proc.py with:
- Pre-allocated NumPy arrays in vectorize() (was O(N^2) concatenation)
- normalize_events() fits on train, applies to test (original re-computed on test)
- RoleDiscovery class wraps KMeans fit/transform
- No TensorFlow dependency
- Typed signatures, logging instead of print
"""

from __future__ import annotations

import itertools
import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------

def extract_prefixes(log_df: pd.DataFrame, n_size: int) -> pd.DataFrame:
    """Return rows where the case has at least n_size events, truncated to n_size."""
    temp = (
        log_df.groupby(["CASEID", "OUTCOME"])["ACTIVITY_INDEX"]
        .max()
        .reset_index()
    )
    temp = temp[temp["ACTIVITY_INDEX"] >= n_size].drop(
        columns=["OUTCOME", "ACTIVITY_INDEX"]
    )
    temp = temp.merge(log_df, on="CASEID")
    temp = temp[temp["ACTIVITY_INDEX"] <= n_size]
    return temp


# ---------------------------------------------------------------------------
# Balancing
# ---------------------------------------------------------------------------

def balance_dataset(
    log_df: pd.DataFrame,
    target: str,
    minority_label: Optional[str] = None,
    id_col: str = "CASEID",
    seed: int = 42,
) -> pd.DataFrame:
    """Undersample the majority class to match the minority class case count.

    The minority class is the label with the fewest cases. ``minority_label`` is
    optional; if it is given but is not actually the minority, the real minority
    is used instead (undersampling the smaller class would need frac > 1).
    """
    case_counts = log_df.groupby(id_col)[target].first().value_counts()
    actual = case_counts.idxmin()
    if minority_label is not None and minority_label != actual:
        logger.warning(
            "balance_dataset: %r is not the minority class (counts %s); using %r.",
            minority_label, case_counts.to_dict(), actual,
        )
    minority_label = actual
    df_min = log_df[log_df[target] == minority_label]
    df_maj = log_df[log_df[target] != minority_label]

    fraction = df_min[id_col].nunique() / df_maj[id_col].nunique()

    cases_maj = pd.DataFrame(df_maj[id_col].unique(), columns=[id_col])
    cases_maj = cases_maj.sample(frac=fraction, random_state=seed).reset_index(drop=True)

    df_maj = df_maj.merge(cases_maj, on=id_col)
    df_maj = df_maj.sort_values([id_col, "ACTIVITY_INDEX"])

    df = pd.concat([df_min, df_maj], ignore_index=True)
    cases = pd.DataFrame(df[id_col].unique(), columns=[id_col])
    cases = cases.sample(frac=1, random_state=seed).reset_index(drop=True)
    cases["_order"] = cases.index

    df = cases.merge(df, on=id_col)
    df = df.sort_values(["_order", "ACTIVITY_INDEX"]).drop(columns="_order")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def split_train_test(
    df: pd.DataFrame, test_frac: float, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Case-level random split; returns (train_df, test_df)."""
    cases = pd.DataFrame(df["CASEID"].unique(), columns=["CASEID"])
    cases = cases.sample(frac=1, random_state=seed).reset_index(drop=True)

    n_test = int(round(len(cases) * test_frac))
    test_cases = cases.iloc[:n_test].copy()
    train_cases = cases.iloc[n_test:].copy()

    def _order_merge(case_df: pd.DataFrame) -> pd.DataFrame:
        case_df = case_df.copy()
        case_df["_order"] = range(len(case_df))
        merged = case_df.merge(df, on="CASEID")
        merged = merged.sort_values(["_order", "ACTIVITY_INDEX"]).drop(columns="_order")
        return merged.reset_index(drop=True)

    return _order_merge(train_cases), _order_merge(test_cases)


# ---------------------------------------------------------------------------
# Role discovery
# ---------------------------------------------------------------------------

class RoleDiscovery:
    """Assign resources to roles via KMeans on activity co-occurrence profiles.

    Fits on the training set. Resources seen at fit time keep their training
    role on any later frame; resources that are missing (NaN) or unseen at fit
    time get the role ``"none"`` (index 0 after ``create_index``), so no new
    categories leak in from the test set and no events are dropped.

    Expects ``RESOURCE`` and ``MILESTONE`` columns; adds a string ``ROLE`` column.
    """

    MISSING = "none"

    def __init__(self, n_clusters: int = 5, random_state: int = 0):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self._model: Optional[KMeans] = None
        self._role_map: Optional[Dict[str, str]] = None

    @staticmethod
    def _resources(df: pd.DataFrame) -> pd.Series:
        """RESOURCE as strings (NaN kept). Integer-valued floats lose the ``.0``, so an
        id column that became float because of missing values matches its int form."""
        def as_str(v):
            if isinstance(v, (float, np.floating)) and float(v).is_integer():
                return str(int(v))
            return str(v)

        res = df["RESOURCE"]
        return res.map(as_str).where(res.notna())

    def fit(self, df_train: pd.DataFrame) -> pd.DataFrame:
        """Fit on train, return a copy of df_train with a ROLE column added."""
        res = self._resources(df_train)
        known = res.notna() & (res != self.MISSING)
        # Binary resource × activity profile (did the resource ever perform the activity?)
        profiles = pd.crosstab(res[known], df_train.loc[known, "MILESTONE"]).clip(upper=1)
        if len(profiles) < self.n_clusters:
            raise ValueError(
                f"Role discovery needs at least n_clusters={self.n_clusters} resources, "
                f"found {len(profiles)}."
            )
        self._model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        labels = self._model.fit_predict(profiles.to_numpy())
        self._role_map = {r: f"role_{c}" for r, c in zip(profiles.index, labels)}
        return self.transform(df_train)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of df with a ROLE column, using the fitted role map."""
        if self._role_map is None:
            raise RuntimeError("Call fit() before transform().")
        res = self._resources(df)
        roles = res.map(self._role_map)
        n_unseen = int((roles.isna() & res.notna() & (res != self.MISSING)).sum())
        if n_unseen:
            logger.warning(
                "Role discovery: %d events from resources unseen at fit time get ROLE='%s'.",
                n_unseen, self.MISSING,
            )
        out = df.copy()
        out["ROLE"] = roles.fillna(self.MISSING).astype(str)
        return out


# ---------------------------------------------------------------------------
# Category index encoding
# ---------------------------------------------------------------------------

def create_index(
    log_df: pd.DataFrame, column: str, reserve_zero: bool = False
) -> Dict[str, int]:
    """Build a string → int index for a categorical column.

    Args:
        log_df: DataFrame containing the column.
        column: Column name to index.
        reserve_zero: If True, index starts at 1 and 'none' maps to 0.

    Returns:
        Sorted dict mapping category values to integers.
    """
    values = log_df[log_df[column] != "none"][column].dropna().unique()
    values = sorted(values.tolist())
    if reserve_zero:
        alias = {v: i + 1 for i, v in enumerate(values)}
        alias["none"] = 0
    else:
        alias = {v: i for i, v in enumerate(values)}
    return {k: v for k, v in sorted(alias.items(), key=lambda x: x[1])}


# ---------------------------------------------------------------------------
# Numerical normalization  (fit on train, apply to test — no leakage)
# ---------------------------------------------------------------------------

def normalize_events(
    log_df: pd.DataFrame,
    features: List[str],
    method: str = "max",
    scalers: Optional[Dict[str, StandardScaler]] = None,
) -> Tuple[pd.DataFrame, Dict[str, StandardScaler]]:
    """Normalize numerical features.

    Args:
        log_df: DataFrame with raw feature values.
        features: List of column names to normalize.
        method: 'max' (z-score) or 'lognorm' (log1p then z-score).
        scalers: Pre-fitted scalers to apply (for test set). Pass None to fit new ones.

    Returns:
        (df_out, scalers) where df_out has *_norm columns and scalers can be
        reused on the test set.
    """
    log_df = log_df.copy()
    fitted: Dict[str, StandardScaler] = {}

    for feat in features:
        if method == "lognorm":
            log_df[f"{feat}_log"] = log_df[feat].apply(math.log1p)
            src_col = f"{feat}_log"
        else:
            src_col = feat

        scaler = scalers[feat] if scalers is not None else StandardScaler()
        if scalers is None:
            scaler.fit(log_df[[src_col]])
        log_df[f"{feat}_norm"] = scaler.transform(log_df[[src_col]])
        fitted[feat] = scaler

    return log_df, fitted


# ---------------------------------------------------------------------------
# Event reformatting into trace dictionaries
# ---------------------------------------------------------------------------

def reformat_events(
    log_df: pd.DataFrame,
    indexes: Dict,
    cat_dynamic: List[str],
    num_dynamic: List[str],
    cat_static: List[str],
    num_static: List[str],
    reverse_order: bool = True,
) -> List[dict]:
    """Group events by case and build per-trace dicts.

    Dynamic categorical features are encoded to integer indices.
    Dynamic numerical features use the *_norm columns.
    Sequence order is reversed (most recent event first) when reverse_order=True,
    matching the original implementation.

    Returns:
        List of trace dicts with keys: caseid, dynamic, static, outcome.
    """
    records = log_df.to_dict("records")
    records = sorted(records, key=lambda x: (x["CASEID"], x["ACTIVITY_INDEX"]))

    temp_data = []
    for key, group in itertools.groupby(records, key=lambda x: x["CASEID"]):
        trace = list(group)
        dynamic: Dict[str, list] = {}
        static: Dict[str, float] = {}

        for feat in cat_dynamic:
            seq = [x[f"{feat}_index"] for x in trace]
            if reverse_order:
                seq = list(reversed(seq))
            dynamic[feat] = seq

        for feat in num_dynamic:
            seq = [x[f"{feat}_norm"] for x in trace]
            if reverse_order:
                seq = list(reversed(seq))
            dynamic[feat] = seq

        for feat in cat_static:
            static[feat] = max(x[f"{feat}_index"] for x in trace)

        for feat in num_static:
            static[feat] = max(x[f"{feat}_norm"] for x in trace)

        outcome = max(x["oc_index"] for x in trace)

        temp_data.append(
            {"caseid": key, "dynamic": dynamic, "static": static, "outcome": outcome}
        )

    return temp_data


# ---------------------------------------------------------------------------
# Vectorization  (pre-allocated — no O(N^2) concatenation)
# ---------------------------------------------------------------------------

def vectorize(
    log: List[dict],
    trc_len: int,
    cat_dynamic: List[str],
    num_dynamic: List[str],
    cat_static: List[str],
    num_static: List[str],
    activity_feature: str = "ACTIVITY",
) -> dict:
    """Convert a list of trace dicts into padded NumPy arrays.

    Categorical and numerical dynamic features are right-zero-padded to trc_len.
    The lengths array records the actual (unpadded) sequence length per trace,
    which is required to build padding masks for the Transformer.

    Returns:
        vec dict with keys:
          prefixes[feat]: (N, trc_len) padded feature arrays
          statics[feat]:  (N,) static feature arrays
          outcome:        (N,) binary labels
          caseid:         (N,) case identifiers
          lengths:        (N,) actual sequence lengths
    """
    n = len(log)
    prefixes: Dict[str, np.ndarray] = {}
    statics: Dict[str, np.ndarray] = {}

    for feat in cat_dynamic:
        prefixes[feat] = np.zeros((n, trc_len), dtype=np.float32)
    for feat in num_dynamic:
        prefixes[feat] = np.zeros((n, trc_len), dtype=np.float32)
    for feat in cat_static:
        statics[feat] = np.zeros(n, dtype=np.float32)
    for feat in num_static:
        statics[feat] = np.zeros(n, dtype=np.float32)

    outcome = np.zeros(n, dtype=np.int64)
    caseid = np.empty(n, dtype=object)
    lengths = np.zeros(n, dtype=np.int64)

    for i, trace in enumerate(log):
        seq_len = len(trace["dynamic"][activity_feature])
        lengths[i] = seq_len

        for feat in cat_dynamic:
            seq = trace["dynamic"][feat]
            prefixes[feat][i, :len(seq)] = seq
        for feat in num_dynamic:
            seq = trace["dynamic"][feat]
            prefixes[feat][i, :len(seq)] = seq
        for feat in cat_static:
            statics[feat][i] = trace["static"][feat]
        for feat in num_static:
            statics[feat][i] = trace["static"][feat]

        outcome[i] = trace["outcome"]
        caseid[i] = trace["caseid"]

    return {
        "prefixes": prefixes,
        "statics": statics,
        "outcome": outcome,
        "caseid": caseid,
        "lengths": lengths,
    }


# ---------------------------------------------------------------------------
# Helper: compute max trace length and case count from a trace list
# ---------------------------------------------------------------------------

def trace_stats(log: List[dict], activity_feature: str = "ACTIVITY") -> Tuple[int, int]:
    """Return (max_trace_len, n_cases)."""
    trc_len = max(len(t["dynamic"][activity_feature]) for t in log)
    return trc_len, len(log)
