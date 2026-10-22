"""Every registered model runs the same bucket → model → train → explain path."""

import numpy as np
import pytest
import torch

from tests.synthetic import dataset_dict, make_log
from xd2net.common.config import TrainingConfig, dataset_config_from_dict
from xd2net.common.data import build_bucket, make_dataloaders, make_test_loader, prepare_dataset
from xd2net.common.registry import available_models, get_spec

CELL_COLUMNS = ["CASEID", "EVENT_POS", "EVENT_INDEX", "EVENT_ACTIVITY", "EVENT_ROLE", "FIELD",
                "VALUE_LABEL", "VALUE_NUM", "EVENT_ATTN", "FEATURE_ATTN", "JOINT_ATTN",
                "EVT_SELF_ATTN", "BUCKET"]


@pytest.fixture(scope="module")
def prepared():
    return prepare_dataset(dataset_config_from_dict(dataset_dict("unused.csv")), df=make_log(240, seed=5))


@pytest.mark.parametrize("name", available_models())
def test_spec_end_to_end(name, prepared, tmp_path):
    spec = get_spec(name)
    b = build_bucket(prepared, 4, spec.encode_dynamic)
    cfg = spec.model_config_cls(d_model=16, nhead=2)
    torch.manual_seed(0)
    model = spec.build_model(b, cfg, "synthetic")

    tl, vl = make_dataloaders(b["inputs_train"], b["outputs_train"], b["lengths_train"], batch_size=64)
    tr = spec.trainer_cls(model, TrainingConfig(epochs=1, device="cpu"), str(tmp_path), bucket="1_4")
    tr.train(tl, vl)
    model = tr.load_best_model()

    # rebuild from the saved config
    clone = spec.model_cls.from_config(model.get_config())
    clone.load_state_dict(model.state_dict())

    ex = spec.explainer_cls(clone, "cpu")
    te = make_test_loader(b["inputs_test"], b["outputs_test"], b["lengths_test"], batch_size=64)
    pred, exp = ex.explain_loader(te, b["caseids_test"], b["feature_names"], "1_4")
    assert len(pred) == b["n_test"] and len(exp) == b["n_test"] * b["exp_dim"]

    cells, case_fields = ex.explain_attention_loader(te, b["caseids_test"], "1_4")
    schema = b["event_schema"]
    assert list(cells.columns) == CELL_COLUMNS
    assert len(cells) == int(b["lengths_test"].sum()) * schema.n_fields
    alpha = cells.drop_duplicates(["CASEID", "EVENT_POS"]).groupby("CASEID")["EVENT_ATTN"].sum()
    assert np.allclose(alpha.to_numpy(), 1.0, atol=1e-4)
    assert len(case_fields) == b["n_test"] * schema.n_fields


def test_both_models_see_the_same_bucket(prepared):
    buckets = {n: build_bucket(prepared, 4, get_spec(n).encode_dynamic) for n in available_models()}
    keys = ["caseids_train", "caseids_test", "lengths_train", "lengths_test", "feature_names"]
    a, b = buckets.values()
    for k in keys:
        assert list(a[k]) == list(b[k]), k
    assert np.array_equal(a["inputs_train"]["input_exp"], b["inputs_train"]["input_exp"])
    assert a["event_schema"] == b["event_schema"]
