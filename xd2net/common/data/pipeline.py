"""Dataset preparation and per-bucket input assembly, shared by every XD2-Net variant.

Wraps the preprocessing helpers (split, role discovery, normalisation, index
encoding, prefix extraction, balancing, reformatting, vectorisation) behind a
dataset config, so any event log can be fed to any model with the same settings.

Only the per-event (dynamic) input differs between variants; ``build_bucket``
takes the variant's encoder function for it (``ModelSpec.encode_dynamic``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from xd2net.common.config import DatasetConfig, PrefixConfig
from xd2net.common.data.eventlog import MISSING_CAT, load_event_log
from xd2net.common.data.features import (
    EventSchema,
    build_event_schema,
    build_exp_input,
    build_static_input,
)
from xd2net.common.data.preprocessing import (
    RoleDiscovery,
    balance_dataset,
    create_index,
    extract_prefixes,
    normalize_events,
    reformat_events,
    split_train_test,
    trace_stats,
    vectorize,
)
logger = logging.getLogger(__name__)

# (vec, schema, max_len) -> input_dynamic (N, max_len, D)
DynamicEncoder = Callable[[dict, EventSchema, int], np.ndarray]


@dataclass
class PreparedLog:
    ds: DatasetConfig
    train: pd.DataFrame
    test: pd.DataFrame
    indexes: Dict
    scalers: Dict
    medians: Dict[str, float]
    event_cat: List[str]
    event_num: List[str]
    case_cat: List[str]
    case_num: List[str]
    case_lengths_train: np.ndarray = field(repr=False)

    def summary(self) -> dict:
        """JSON-friendly description of the prepared dataset."""
        return {
            "name": self.ds.name,
            "n_train_cases": int(self.train["CASEID"].nunique()),
            "n_test_cases": int(self.test["CASEID"].nunique()),
            "event_cat": self.event_cat,
            "event_num": self.event_num,
            "case_cat": self.case_cat,
            "case_num": self.case_num,
            "cardinalities": {
                f: len(self.indexes["dynamic_indexes"][f"{f}_index"]) for f in self.event_cat
            },
            "case_length_quantiles": {
                str(q): float(np.quantile(self.case_lengths_train, q)) for q in (0.1, 0.5, 0.9, 1.0)
            },
        }


def prepare_dataset(
    ds: DatasetConfig, seed: int = 42, df: Optional[pd.DataFrame] = None
) -> PreparedLog:
    """Load, split, assign roles, impute, normalise and index-encode a log."""
    log = load_event_log(ds, df=df)
    train, test = split_train_test(log, ds.test_split, seed=seed)

    # --- roles ---------------------------------------------------------------
    if "ROLE" not in log.columns:
        logger.info("[%s] discovering %d roles from resources", ds.name, ds.role_discovery_clusters)
        train["MILESTONE"] = train["ACTIVITY"]
        test["MILESTONE"] = test["ACTIVITY"]
        disc = RoleDiscovery(n_clusters=ds.role_discovery_clusters, random_state=seed)
        train = disc.fit(train)
        test = disc.transform(test)

    event_cat = ds.event_cat_fields
    event_num = ds.event_num_fields
    case_cat, case_num = list(ds.case_cat), list(ds.case_num)

    # --- numeric imputation (train medians) + normalisation -----------------
    medians: Dict[str, float] = {}
    for feat in [*event_num, *case_num]:
        med = train[feat].median()
        medians[feat] = 0.0 if pd.isna(med) else float(med)
        n_missing = int(train[feat].isna().sum() + test[feat].isna().sum())
        if n_missing:
            logger.info("[%s] %s: imputing %d missing values with %.4g", ds.name, feat, n_missing, medians[feat])
        train[feat] = train[feat].fillna(medians[feat])
        test[feat] = test[feat].fillna(medians[feat])

    scalers: Dict = {}
    by_method: Dict[str, List[str]] = {}
    for feat in [*event_num, *case_num]:
        by_method.setdefault(ds.norm.method_for(feat), []).append(feat)
    for method, feats in by_method.items():
        train, fitted = normalize_events(train, feats, method)
        test, _ = normalize_events(test, feats, method, scalers=fitted)
        scalers.update(fitted)

    # --- categorical indexes (fit on train; unseen → 0 / "none") -------------
    indexes: Dict = {"dynamic_indexes": {}, "static_indexes": {}}
    for kind, feats in (("dynamic_indexes", event_cat), ("static_indexes", case_cat)):
        for feat in feats:
            index = create_index(train, feat, reserve_zero=True)
            indexes[kind][f"{feat}_index"] = index
            for frame in (train, test):
                frame[f"{feat}_index"] = frame[feat].map(index).fillna(0).astype(int)
            n_unseen = int((~test[feat].isin(index.keys()) & (test[feat] != MISSING_CAT)).sum())
            if n_unseen:
                logger.info("[%s] %s: %d test events with unseen labels mapped to 'none'", ds.name, feat, n_unseen)
    indexes["oc_index"] = {0: ds.outcome.labels[0], 1: ds.outcome.labels[1]}

    lengths = train.groupby("CASEID").size().to_numpy()
    return PreparedLog(
        ds=ds, train=train, test=test, indexes=indexes, scalers=scalers, medians=medians,
        event_cat=event_cat, event_num=event_num, case_cat=case_cat, case_num=case_num,
        case_lengths_train=lengths,
    )


def resolve_prefix_lengths(prepared: PreparedLog, prefix: Optional[PrefixConfig] = None) -> List[int]:
    """Explicit list, or ``auto`` = min_len .. min(max_len, quantile of train case length)."""
    prefix = prefix or prepared.ds.prefix
    if prefix.lengths != "auto":
        if isinstance(prefix.lengths, str):
            raise ValueError(f"prefix.lengths must be 'auto' or a list of ints, got {prefix.lengths!r}")
        return sorted(int(n) for n in prefix.lengths)
    upper = min(prefix.max_len, int(np.quantile(prepared.case_lengths_train, prefix.quantile)))
    lengths = list(range(prefix.min_len, upper + 1))
    if not lengths:
        raise ValueError(
            f"[{prepared.ds.name}] auto prefix lengths are empty (min_len={prefix.min_len}, upper={upper})."
        )
    return lengths


def _vectorize_frame(df: pd.DataFrame, p: PreparedLog, max_len: Optional[int] = None):
    log = reformat_events(df, p.indexes, p.event_cat, p.event_num, p.case_cat, p.case_num)
    trc_len, _ = trace_stats(log)
    max_len = max_len or trc_len
    vec = vectorize(log, max_len, p.event_cat, p.event_num, p.case_cat, p.case_num, activity_feature="ACTIVITY")
    return vec, trc_len


def _inputs(vec: dict, p: PreparedLog, schema: EventSchema, max_len: int, encode_dynamic: DynamicEncoder):
    exp, names = build_exp_input(
        vec, p.indexes, max_len, p.event_cat, p.event_num, p.case_cat, p.case_num, p.ds.num_agg
    )
    inputs = {
        "input_dynamic": encode_dynamic(vec, schema, max_len),
        "input_static": build_static_input(vec, p.indexes, p.case_cat, p.case_num).astype(np.float32),
        "input_exp": exp,
    }
    outputs = {"act_output": vec["outcome"].astype(np.int64)}
    return inputs, outputs, names


def build_bucket(
    prepared: PreparedLog, n_size: int, encode_dynamic: DynamicEncoder, seed: int = 42
) -> Optional[dict]:
    """Assemble model inputs for prefix-length bucket ``1_{n_size}``.

    ``encode_dynamic`` builds the variant's per-event input from the vectorised
    prefixes (e.g. ``build_dynamic_onehot`` or ``build_dynamic_tokens``).
    Returns None when the bucket has too few cases or a single class.
    """
    p, ds = prepared, prepared.ds
    key = f"1_{n_size}"
    tr = extract_prefixes(p.train, n_size)
    te = extract_prefixes(p.test, n_size)

    n_tr, n_te = tr["CASEID"].nunique(), te["CASEID"].nunique()
    if n_tr < ds.prefix.min_cases or n_te == 0:
        logger.warning("[%s] bucket %s: too few cases (train=%d, test=%d), skipping.", ds.name, key, n_tr, n_te)
        return None
    case_labels = tr.groupby("CASEID")["OUTCOME"].first()
    if case_labels.nunique() < 2:
        logger.warning("[%s] bucket %s: single class in train, skipping.", ds.name, key)
        return None

    if ds.balance_train:
        tr = balance_dataset(tr, "OUTCOME", id_col="CASEID", seed=seed)

    vec_tr, len_tr = _vectorize_frame(tr, p)
    vec_te, len_te = _vectorize_frame(te, p)
    max_len = max(len_tr, len_te)
    if len_tr != max_len:
        vec_tr, _ = _vectorize_frame(tr, p, max_len)
    if len_te != max_len:
        vec_te, _ = _vectorize_frame(te, p, max_len)

    schema = build_event_schema(p.indexes, p.event_cat, p.event_num)
    inputs_tr, outputs_tr, names = _inputs(vec_tr, p, schema, max_len, encode_dynamic)
    inputs_te, outputs_te, names_te = _inputs(vec_te, p, schema, max_len, encode_dynamic)
    assert names == names_te, "train/test exp feature names diverged"

    return {
        "dataset": ds.name,
        "bucket_key": key,
        "inputs_train": inputs_tr,
        "outputs_train": outputs_tr,
        "lengths_train": vec_tr["lengths"],
        "caseids_train": vec_tr["caseid"],
        "inputs_test": inputs_te,
        "outputs_test": outputs_te,
        "lengths_test": vec_te["lengths"],
        "caseids_test": vec_te["caseid"],
        "exp_dim": inputs_tr["input_exp"].shape[1],
        "max_len": max_len,
        "feature_names": names,
        "dim_dynamic": inputs_tr["input_dynamic"].shape[2],
        "dim_static": inputs_tr["input_static"].shape[1],
        "event_schema": schema,
        "n_train": len(vec_tr["caseid"]),
        "n_test": len(vec_te["caseid"]),
    }
