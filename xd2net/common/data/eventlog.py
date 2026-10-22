"""Generic event-log loading, shared by every XD2-Net variant.

Turns any event log (caseid, timestamp, activity, role/resource, plus declared
case- and event-level variables) into the internal frame expected by the
preprocessing helpers:

    CASEID, TIMESTAMP, ACTIVITY, [ROLE | RESOURCE], ACTIVITY_INDEX,
    OUTCOME (label string), oc_index (0/1), derived time features,
    and every declared variable under its own name.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

from xd2net.common.config import TIME_UNITS, DatasetConfig

logger = logging.getLogger(__name__)

_OUTCOME_RAW = "_OUTCOME_RAW"
_ROW = "_ROW"
MISSING_CAT = "none"   # baseline create_index(reserve_zero=True) maps this to index 0


def read_table(path: str) -> pd.DataFrame:
    """Read a CSV or Parquet event log."""
    lower = path.lower()
    if lower.endswith(".parquet") or lower.endswith(".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def load_event_log(ds: DatasetConfig, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Load, rename, sort, label and validate an event log.

    Args:
        ds: dataset config.
        df: optional pre-loaded frame (used by tests); read from ``ds.path`` otherwise.
    """
    if df is None:
        logger.info("[%s] reading %s", ds.name, ds.path)
        df = read_table(ds.path)

    df = _select_and_rename(df, ds)
    df[_ROW] = np.arange(len(df))

    # --- timestamps + ordering ---------------------------------------------
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], format=ds.timestamp_format, errors="coerce")
    n_bad = int(df["TIMESTAMP"].isna().sum())
    if n_bad:
        if n_bad == len(df):
            raise ValueError(f"[{ds.name}] no timestamp could be parsed; set timestamp_format.")
        logger.warning("[%s] dropping %d events with unparseable timestamps.", ds.name, n_bad)
        df = df[df["TIMESTAMP"].notna()]

    df["CASEID"] = df["CASEID"].astype(str)
    order = ["CASEID", "TIMESTAMP", _ROW] if ds.event_order == "timestamp" else ["CASEID", _ROW]
    df = df.sort_values(order, kind="mergesort").reset_index(drop=True)

    # --- categorical / numeric typing --------------------------------------
    cat_cols = ["ACTIVITY", *ds.event_cat, *ds.case_cat]
    if "ROLE" in df.columns:
        cat_cols.append("ROLE")
    if "RESOURCE" in df.columns:
        cat_cols.append("RESOURCE")
    for col in cat_cols:
        df[col] = _to_category_str(df[col])
    for col in [*ds.event_num, *ds.case_num]:
        df[col] = _to_numeric(df[col], col, ds.name)

    if ds.drop_activities:
        drop = df["ACTIVITY"].isin(set(map(str, ds.drop_activities)))
        logger.info("[%s] dropping %d events with activities %s", ds.name, int(drop.sum()), ds.drop_activities)
        df = df[~drop].reset_index(drop=True)

    _derive_time_features(df, ds)
    df = _apply_outcome(df, ds)
    _validate_case_level(df, ds)

    df["ACTIVITY_INDEX"] = df.groupby("CASEID").cumcount() + 1
    df = df.drop(columns=[_ROW])

    n_pos = int(df.groupby("CASEID")["oc_index"].first().sum())
    n_cases = df["CASEID"].nunique()
    logger.info(
        "[%s] %d events, %d cases, positive rate %.3f", ds.name, len(df), n_cases, n_pos / max(n_cases, 1)
    )
    return df


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def _select_and_rename(df: pd.DataFrame, ds: DatasetConfig) -> pd.DataFrame:
    cols = ds.columns
    mapping: Dict[str, str] = {
        cols.caseid: "CASEID",
        cols.timestamp: "TIMESTAMP",
        cols.activity: "ACTIVITY",
    }
    if cols.role is not None:
        mapping[cols.role] = "ROLE"
    else:
        mapping[cols.resource] = "RESOURCE"
    if ds.outcome.mode == "column":
        mapping[ds.outcome.column] = _OUTCOME_RAW

    extras = [*ds.event_cat, *ds.event_num, *ds.case_cat, *ds.case_num]
    overlap = set(mapping) & set(extras)
    if overlap:
        raise ValueError(f"[{ds.name}] columns used both as core column and variable: {sorted(overlap)}")
    if len(set(mapping)) != len(mapping) or len(set(mapping.values())) != len(mapping):
        raise ValueError(f"[{ds.name}] a source column is mapped to more than one role: {mapping}")

    wanted = list(mapping) + extras
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        raise KeyError(f"[{ds.name}] columns not found in log: {missing}. Available: {list(df.columns)}")

    return df[wanted].rename(columns=mapping).copy()


def _to_category_str(s: pd.Series) -> pd.Series:
    out = s.astype(object).where(s.notna(), MISSING_CAT).astype(str)
    return out.replace({"": MISSING_CAT})


def _to_numeric(s: pd.Series, col: str, name: str) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce")
    coerced = int(out.isna().sum() - s.isna().sum())
    if coerced:
        logger.warning("[%s] %s: %d non-numeric values treated as missing.", name, col, coerced)
    if out.notna().sum() == 0:
        raise ValueError(f"[{name}] numeric variable {col!r} has no numeric values.")
    return out.astype(float)


def _derive_time_features(df: pd.DataFrame, ds: DatasetConfig) -> None:
    tf = ds.time_features
    unit = TIME_UNITS[tf.unit]
    ts = df["TIMESTAMP"]
    grp = ts.groupby(df["CASEID"])
    if tf.since_prev:
        df["TIME_SINCE_PREV"] = (grp.diff().dt.total_seconds() / unit).fillna(0.0)
    if tf.since_start:
        df["TIME_SINCE_START"] = (ts - grp.transform("min")).dt.total_seconds() / unit
    if tf.hour:
        df["HOUR"] = ts.dt.hour.astype(str)
    if tf.weekday:
        df["WEEKDAY"] = ts.dt.day_name()


def _apply_outcome(df: pd.DataFrame, ds: DatasetConfig) -> pd.DataFrame:
    oc = ds.outcome
    neg_label, pos_label = oc.labels

    if oc.mode == "column":
        raw = df[_OUTCOME_RAW]
        varying = raw.groupby(df["CASEID"]).nunique(dropna=True) > 1
        share = float(varying.mean())
        if share > ds.case_level_tolerance:
            raise ValueError(
                f"[{ds.name}] outcome column {oc.column!r} varies within {int(varying.sum())} cases "
                f"({share:.1%}); it must be constant per case."
            )
        per_case = raw.groupby(df["CASEID"]).transform("last")
        is_pos = (per_case == oc.positive) | (per_case.astype(str) == str(oc.positive))
        df["oc_index"] = is_pos.astype(int)
        df = df.drop(columns=[_OUTCOME_RAW])
    else:
        hit = df["ACTIVITY"].isin(set(map(str, oc.activities)))
        # Position of each event within its case, and first hit position per case.
        pos = df.groupby("CASEID").cumcount()
        first_hit = pos.where(hit).groupby(df["CASEID"]).transform("min")
        df["oc_index"] = first_hit.notna().astype(int)
        # Leakage guard: drop the target activity and everything after it.
        keep = first_hit.isna() | (pos < first_hit)
        n_before = df["CASEID"].nunique()
        df = df[keep].copy()
        n_dropped = n_before - df["CASEID"].nunique()
        if n_dropped:
            logger.warning(
                "[%s] %d cases start with a target activity and have no usable prefix; dropped.",
                ds.name, n_dropped,
            )

    if df["oc_index"].nunique() < 2:
        raise ValueError(f"[{ds.name}] outcome has a single class after labelling; check the outcome config.")
    df["OUTCOME"] = np.where(df["oc_index"] == 1, pos_label, neg_label)
    return df


def _validate_case_level(df: pd.DataFrame, ds: DatasetConfig) -> None:
    """Check declared case-level variables are constant per case, then broadcast."""
    for col in [*ds.case_cat, *ds.case_num]:
        values = df[col]
        if col in ds.case_cat:
            values = values.where(values != MISSING_CAT)
        varying = values.groupby(df["CASEID"]).nunique(dropna=True) > 1
        share = float(varying.mean())
        if share > ds.case_level_tolerance:
            examples = list(varying[varying].index[:5])
            raise ValueError(
                f"[{ds.name}] case-level variable {col!r} varies within {int(varying.sum())} cases "
                f"({share:.1%}, e.g. {examples}). Declare it as event-level or raise case_level_tolerance."
            )
        if share > 0:
            logger.warning("[%s] %s varies within %.2f%% of cases; using the first value.", ds.name, col, 100 * share)
        first = values.groupby(df["CASEID"]).transform("first")
        df[col] = first.fillna(MISSING_CAT) if col in ds.case_cat else first
