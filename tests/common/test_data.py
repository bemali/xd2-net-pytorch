"""Tests for the generic event-log pipeline (config, loader, features, buckets)."""

import numpy as np
import pandas as pd
import pytest

from tests.synthetic import dataset_dict, make_log
from xd2net.common.config import dataset_config_from_dict
from xd2net.common.data import (
    EventSchema,
    build_bucket,
    build_event_schema,
    build_exp_input,
    load_event_log,
    prepare_dataset,
    resolve_prefix_lengths,
)
from xd2net.hierarchical.data import build_dynamic_tokens


def _cfg(mode="column", **overrides):
    d = dataset_dict("unused.csv", mode)
    d.update(overrides)
    return dataset_config_from_dict(d)


def _tiny_log():
    """Two cases, rows out of order, one timestamp tie (resolved by row order)."""
    return pd.DataFrame({
        "case_id": ["B", "A", "A", "B", "A", "B"],
        "ts": ["2024-01-01 10:00", "2024-01-01 09:00", "2024-01-01 08:00",
               "2024-01-01 10:00", "2024-01-01 11:00", "2024-01-01 12:30"],
        "task": ["b1", "a2", "a1", "b2", "a3", "Close_Won"],
        "team": ["x", "y", "x", None, "y", "x"],
        "channel": ["phone", "web", "web", "phone", "web", "phone"],
        "amount": [5.0, 1.0, 1.0, 5.0, 1.0, 5.0],
        "priority": ["low", "high", "low", "low", "low", "low"],
        "cost": [1, 2, 3, "n/a", 5, 6],
        "result": ["won", "lost", "lost", "won", "lost", "won"],
    })


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_config_rejects_unknown_keys():
    d = dataset_dict("x.csv")
    d["evnt_cat"] = ["typo"]
    with pytest.raises(ValueError, match="unknown keys"):
        dataset_config_from_dict(d)


def test_config_rejects_reserved_and_duplicate_names():
    with pytest.raises(ValueError, match="reserved"):
        _cfg(event_cat=["ROLE"])
    with pytest.raises(ValueError, match="more than once"):
        _cfg(event_num=["cost"], case_num=["cost"])


def test_field_order():
    cfg = _cfg()
    assert cfg.event_cat_fields == ["ACTIVITY", "ROLE", "priority", "HOUR"]
    assert cfg.event_num_fields == ["cost", "TIME_SINCE_PREV", "TIME_SINCE_START"]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def test_sorting_ties_and_time_features():
    cfg = _cfg(drop_activities=[])
    df = load_event_log(cfg, df=_tiny_log())
    a = df[df.CASEID == "A"]
    assert a.ACTIVITY.tolist() == ["a1", "a2", "a3"]
    assert a.ACTIVITY_INDEX.tolist() == [1, 2, 3]
    assert a.TIME_SINCE_PREV.tolist() == [0.0, 1.0, 2.0]          # hours
    assert a.TIME_SINCE_START.tolist() == [0.0, 1.0, 3.0]
    b = df[df.CASEID == "B"]
    assert b.ACTIVITY.tolist() == ["b1", "b2", "Close_Won"]        # tie keeps file order
    assert b.TIME_SINCE_PREV.iloc[1] == 0.0


def test_file_order_option():
    cfg = _cfg(drop_activities=[], event_order="file")
    df = load_event_log(cfg, df=_tiny_log())
    assert df[df.CASEID == "A"].ACTIVITY.tolist() == ["a2", "a1", "a3"]


def test_missing_values_and_column_outcome():
    cfg = _cfg(drop_activities=[])
    df = load_event_log(cfg, df=_tiny_log())
    assert (df.ROLE == "none").sum() == 1
    assert df.cost.isna().sum() == 1                               # "n/a" → NaN, imputed later
    assert df.groupby("CASEID").oc_index.first().to_dict() == {"A": 0, "B": 1}
    assert set(df.OUTCOME) == {"lost", "won"}


def test_drop_activities():
    df = load_event_log(_cfg(), df=_tiny_log())
    assert "Close_Won" not in set(df.ACTIVITY)


def test_activity_outcome_cuts_prefix():
    log = make_log(200, seed=3)
    cfg = _cfg("activity")
    df = load_event_log(cfg, df=log)
    assert not df.ACTIVITY.isin(["Close_Won", "Close_Lost"]).any()
    truth = log.groupby("case_id").task.apply(lambda s: "Close_Won" in set(s))
    got = df.groupby("CASEID").oc_index.first().astype(bool)
    assert (got == truth.loc[got.index]).all()


def test_case_level_validation():
    log = _tiny_log()
    log.loc[log.case_id == "A", "amount"] = [1.0, 2.0, 1.0]       # varies within case A
    with pytest.raises(ValueError, match="case-level variable 'amount'"):
        load_event_log(_cfg(drop_activities=[]), df=log)
    df = load_event_log(_cfg(drop_activities=[], case_level_tolerance=0.5), df=log)
    assert df[df.CASEID == "A"].amount.nunique() == 1


def test_missing_column_reports_available():
    log = _tiny_log().drop(columns=["priority"])
    with pytest.raises(KeyError, match="priority"):
        load_event_log(_cfg(), df=log)


def test_single_class_raises():
    log = _tiny_log()
    log["result"] = "won"
    with pytest.raises(ValueError, match="single class"):
        load_event_log(_cfg(), df=log)


# ---------------------------------------------------------------------------
# Prepared dataset / buckets
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def prepared():
    return prepare_dataset(_cfg(), df=make_log(400, seed=1))


def test_unseen_categories_map_to_none(prepared):
    index = prepared.indexes["dynamic_indexes"]["ACTIVITY_index"]
    assert index["none"] == 0
    test = prepared.test.copy()
    test.loc[test.index[0], "ACTIVITY"] = "never_seen"
    assert test["ACTIVITY"].map(index).fillna(0).iloc[0] == 0


def test_no_missing_after_imputation(prepared):
    for f in prepared.event_num + prepared.case_num:
        assert prepared.train[f"{f}_norm"].notna().all()
        assert prepared.test[f"{f}_norm"].notna().all()


def test_role_discovery_path():
    log = make_log(150, seed=2).rename(columns={"team": "resource"})
    d = dataset_dict("unused.csv")
    d["columns"] = {"caseid": "case_id", "timestamp": "ts", "activity": "task", "role": None, "resource": "resource"}
    d["role_discovery_clusters"] = 2
    p = prepare_dataset(dataset_config_from_dict(d), df=log)
    assert p.train["ROLE"].str.startswith("role_").all()
    assert "ROLE_index" in p.train.columns


def test_auto_prefix_lengths(prepared):
    lengths = resolve_prefix_lengths(prepared)
    q90 = int(np.quantile(prepared.case_lengths_train, 0.9))
    assert lengths == list(range(2, min(8, q90) + 1))


def test_bucket_shapes_and_schema(prepared):
    b = build_bucket(prepared, 4, build_dynamic_tokens)
    schema: EventSchema = b["event_schema"]
    x = b["inputs_train"]["input_dynamic"]
    assert x.shape == (b["n_train"], b["max_len"], schema.n_fields)
    assert schema.field_names == prepared.event_cat + prepared.event_num
    assert b["inputs_train"]["input_exp"].shape[1] == len(b["feature_names"]) == b["exp_dim"]
    assert b["inputs_test"]["input_exp"].shape[1] == b["exp_dim"]
    assert set(np.unique(b["outputs_train"]["act_output"])) == {0, 1}
    # balanced train set
    assert b["outputs_train"]["act_output"].mean() == pytest.approx(0.5)


def test_bucket_skipped_when_too_few_cases(prepared):
    assert build_bucket(prepared, 50, build_dynamic_tokens) is None


# ---------------------------------------------------------------------------
# Feature builders on a hand-made vec
# ---------------------------------------------------------------------------

def _vec():
    # Two traces, reversed order (pos 0 = most recent), right-padded; lengths 3 and 1.
    return {
        "prefixes": {
            "ACTIVITY": np.array([[2, 1, 0, 0], [0, 0, 0, 0]], dtype=np.float32),
            "x": np.array([[1.0, 2.0, 3.0, 0.0], [4.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        },
        "statics": {},
        "outcome": np.array([0, 1]),
        "lengths": np.array([3, 1]),
    }


INDEXES = {"dynamic_indexes": {"ACTIVITY_index": {"none": 0, "a": 1, "b": 2}}, "static_indexes": {}}


def test_dynamic_tokens_shift_and_pad():
    schema = build_event_schema(INDEXES, ["ACTIVITY"], ["x"])
    assert schema.cardinalities == [3]
    assert schema.index_to_label["ACTIVITY"] == {1: "none", 2: "a", 3: "b"}
    tok = build_dynamic_tokens(_vec(), schema, 4)
    assert tok.shape == (2, 4, 2)
    assert tok[0, :, 0].tolist() == [3, 2, 1, 0]                  # +1 shift; "none" stays real (1)
    assert tok[1, :, 0].tolist() == [1, 0, 0, 0]                  # PAD only beyond length
    assert tok[0, :, 1].tolist() == [1.0, 2.0, 3.0, 0.0]


def test_event_schema_roundtrip():
    schema = build_event_schema(INDEXES, ["ACTIVITY"], ["x"])
    assert EventSchema.from_dict(schema.to_dict()) == schema


def test_exp_input_excludes_padding():
    exp, names = build_exp_input(_vec(), INDEXES, 4, ["ACTIVITY"], ["x"], [], [], {"x": "mean"})
    assert names == ["Occurrence of ACTIVITY=none", "Occurrence of ACTIVITY=a",
                     "Occurrence of ACTIVITY=b", "Mean x"]
    # trace 0: none,a,b once each; trace 1: one "none" event (padding not counted)
    assert exp[:, :3].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert exp[:, 3].tolist() == pytest.approx([2.0, 4.0])


@pytest.mark.parametrize("how,expected", [
    ("max", [3.0, 4.0]), ("min", [1.0, 4.0]), ("sum", [6.0, 4.0]),
    ("last", [1.0, 4.0]), ("first", [3.0, 4.0]),
])
def test_exp_input_aggregations(how, expected):
    exp, names = build_exp_input(_vec(), INDEXES, 4, [], ["x"], [], [], {"x": how})
    assert names == [f"{how.capitalize()} x"]
    assert exp[:, 0].tolist() == pytest.approx(expected)


def test_shipped_configs_parse():
    from pathlib import Path

    from xd2net.common.config import load_dataset_config, load_experiment_config
    from xd2net.common.registry import available_models, get_spec

    root = Path(__file__).resolve().parents[2]
    for name in available_models():
        spec = get_spec(name)
        exp = load_experiment_config(str(root / spec.default_config), spec.model_config_cls)
        assert exp.model.d_model % exp.model.nhead == 0
    root = root / "configs"
    for path in sorted((root / "datasets").glob("*.yaml")):
        load_dataset_config(str(path))
