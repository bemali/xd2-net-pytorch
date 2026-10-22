"""Tests for HierExplainer on a real (synthetic) bucket."""

import numpy as np
import pytest
import torch

from tests.synthetic import dataset_dict, make_log
from xd2net.common.data.dataset import make_test_loader
from xd2net.common.config import dataset_config_from_dict
from xd2net.hierarchical.config import HierModelConfig
from xd2net.common.data import build_bucket, prepare_dataset
from xd2net.hierarchical.data import build_dynamic_tokens
from xd2net.hierarchical.explainer import HierExplainer
from xd2net.hierarchical.model import XD2NetHier


@pytest.fixture(scope="module")
def setup():
    p = prepare_dataset(dataset_config_from_dict(dataset_dict("unused.csv")), df=make_log(300, seed=4))
    b = build_bucket(p, 4, build_dynamic_tokens)
    torch.manual_seed(0)
    m = XD2NetHier(b["event_schema"], b["dim_static"], b["exp_dim"], b["max_len"],
                   HierModelConfig(d_model=16, nhead=2))
    loader = make_test_loader(b["inputs_test"], b["outputs_test"], b["lengths_test"], batch_size=32)
    return b, HierExplainer(m, "cpu"), loader


def test_cells_table(setup):
    b, ex, loader = setup
    schema = b["event_schema"]
    cells, case_fields = ex.explain_attention_loader(loader, b["caseids_test"], b["bucket_key"])
    assert len(cells) == int(b["lengths_test"].sum()) * schema.n_fields
    assert np.allclose(cells.groupby("CASEID").JOINT_ATTN.sum(), 1.0, atol=1e-5)
    per_event = cells.groupby(["CASEID", "EVENT_INDEX"]).FEATURE_ATTN.sum()
    assert np.allclose(per_event, 1.0, atol=1e-5)
    # Chronological index and recency position agree.
    assert ((cells.EVENT_POS + cells.EVENT_INDEX + 1) == cells.CASEID.map(
        dict(zip(b["caseids_test"], b["lengths_test"])))).all()
    # Categorical cells carry real labels; numeric cells carry values.
    cat = cells[cells.FIELD.isin(schema.cat_fields)]
    assert cat.VALUE_LABEL.ne("<pad>").all() and cat.VALUE_NUM.isna().all()
    assert cells[cells.FIELD.isin(schema.num_fields)].VALUE_NUM.notna().all()
    act = cells[cells.FIELD == "ACTIVITY"]
    assert (act.VALUE_LABEL == act.EVENT_ACTIVITY).all()
    # Case order follows the loader.
    assert list(dict.fromkeys(cells.CASEID)) == list(b["caseids_test"])
    assert np.allclose(case_fields.groupby("CASEID").CASE_FIELD_ATTN.sum(), 1.0, atol=1e-5)


def test_event_activity_matches_source_log(setup):
    b, ex, loader = setup
    cells, _ = ex.explain_attention_loader(loader, b["caseids_test"], b["bucket_key"])
    first = cells[(cells.EVENT_INDEX == 0) & (cells.FIELD == "ACTIVITY")]
    # Every synthetic case starts with "Register".
    assert (first.VALUE_LABEL == "Register").all()


def test_global_attention(setup):
    b, ex, loader = setup
    cells, _ = ex.explain_attention_loader(loader, b["caseids_test"], b["bucket_key"])
    g = ex.global_attention(cells)
    assert set(g.FIELD) == set(b["event_schema"].field_names)
    assert g.ATTENTION.sum() == pytest.approx(1.0, abs=1e-5)
    gl = ex.global_attention(cells, by="field_label")
    assert gl.ATTENTION.sum() == pytest.approx(1.0, abs=1e-5)
    assert (gl.N_CASES <= cells.CASEID.nunique()).all()


def test_caseid_mismatch_raises(setup):
    b, ex, loader = setup
    with pytest.raises(ValueError, match="caseids"):
        ex.explain_attention_loader(loader, b["caseids_test"][:-1], "x")


def test_inherited_level2_explanations(setup):
    b, ex, loader = setup
    df_pred, df_exp = ex.explain_loader(loader, b["caseids_test"], b["feature_names"], b["bucket_key"])
    assert list(df_pred.columns) == ["CASEID", "PREDICTION_PROB", "PREDICTION", "GROUND_TRUTH", "BUCKET"]
    assert len(df_exp) == len(df_pred) * b["exp_dim"]
