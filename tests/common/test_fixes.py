"""Regression tests for the issues fixed in the former baseline code (DEV_SPEC §13)."""

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from xd2net.common.config import TrainingConfig
from xd2net.common.data import make_dataloaders
from xd2net.common.data.features import build_event_schema, build_exp_input
from xd2net.common.data.preprocessing import RoleDiscovery, balance_dataset
from xd2net.common.heads import PredictionHead, ThetaHead
from xd2net.common.training import Trainer


# ---------------------------------------------------------------------------
# RoleDiscovery: pandas 3, NaN resources, unseen resources, no dropped events
# ---------------------------------------------------------------------------

def _resource_log():
    acts = ["a", "b", "c", "d"]
    rows = []
    for i, res in enumerate([101, 102, 103, 104, 105, 106]):          # numeric ids (pandas-3 failure case)
        for j, act in enumerate(acts[: 2 + i % 3]):
            rows.append({"CASEID": f"c{i}", "RESOURCE": res, "MILESTONE": act, "ACTIVITY_INDEX": j + 1})
    rows.append({"CASEID": "c0", "RESOURCE": np.nan, "MILESTONE": "a", "ACTIVITY_INDEX": 9})
    return pd.DataFrame(rows)


def test_role_discovery_keeps_every_event_and_maps_missing_to_none():
    train = _resource_log()
    disc = RoleDiscovery(n_clusters=2, random_state=0)
    out = disc.fit(train)
    assert len(out) == len(train)                                      # inner merge used to drop the NaN row
    assert out.loc[train["RESOURCE"].isna().to_numpy(), "ROLE"].eq("none").all()
    assert out["ROLE"].map(type).eq(str).all()

    test = pd.DataFrame({"CASEID": ["t", "t"], "RESOURCE": [101, 999], "MILESTONE": ["a", "b"],
                         "ACTIVITY_INDEX": [1, 2]})
    roles = disc.transform(test)["ROLE"].tolist()
    assert roles[0] == out.loc[train["RESOURCE"].eq(101).to_numpy(), "ROLE"].iloc[0]   # seen: train role
    assert roles[1] == "none"                                                          # unseen: none


# ---------------------------------------------------------------------------
# balance_dataset uses the actual minority
# ---------------------------------------------------------------------------

def test_balance_dataset_uses_actual_minority():
    rows = [{"CASEID": f"p{i}", "OUTCOME": "pos", "ACTIVITY_INDEX": 1} for i in range(8)]
    rows += [{"CASEID": f"n{i}", "OUTCOME": "neg", "ACTIVITY_INDEX": 1} for i in range(3)]
    df = pd.DataFrame(rows)
    out = balance_dataset(df, "OUTCOME", "pos")                        # wrong hint: pos is the majority
    counts = out.groupby("CASEID")["OUTCOME"].first().value_counts()
    assert counts.to_dict() == {"pos": 3, "neg": 3}


# ---------------------------------------------------------------------------
# Exp-input counts exclude padding
# ---------------------------------------------------------------------------

def test_exp_counts_exclude_padding():
    indexes = {"dynamic_indexes": {"ACTIVITY_index": {"none": 0, "a": 1}}, "static_indexes": {}}
    vec = {"prefixes": {"ACTIVITY": np.array([[1, 0, 0]], dtype=np.float32)},
           "statics": {}, "outcome": np.array([1]), "lengths": np.array([1])}
    exp, names = build_exp_input(vec, indexes, 3, ["ACTIVITY"], [], [], [])
    assert dict(zip(names, exp[0])) == {"Occurrence of ACTIVITY=none": 0, "Occurrence of ACTIVITY=a": 1}


# ---------------------------------------------------------------------------
# Trainer: current torch, config fields, best checkpoint only, weights_only load
# ---------------------------------------------------------------------------

class _Tiny(nn.Module):
    def __init__(self, D=3, E=4):
        super().__init__()
        self.encoder = nn.Linear(D, 8)
        self.theta_head = ThetaHead(8, E, 1.5)
        self.prediction_head = PredictionHead(E)

    def forward(self, x, s, e, lengths=None):
        h = self.encoder(x).mean(1)
        th = self.theta_head(h)
        return th, self.prediction_head(th, e), torch.zeros(x.shape[:2])


def _loaders(n=64, T=5, D=3, E=4):
    rng = np.random.default_rng(0)
    inputs = {"input_dynamic": rng.random((n, T, D)).astype(np.float32),
              "input_static": np.zeros((n, 0), np.float32),
              "input_exp": rng.random((n, E)).astype(np.float32)}
    return make_dataloaders(inputs, {"act_output": rng.integers(0, 2, n)}, np.full(n, T), batch_size=16)


def test_trainer_keeps_only_best_checkpoint(tmp_path):
    torch.manual_seed(0)
    tl, vl = _loaders()
    tr = Trainer(_Tiny(), TrainingConfig(epochs=6, device="cpu", learning_rate=0.05), str(tmp_path), bucket="b")
    tr.train(tl, vl)
    ckpts = list(tmp_path.glob("model_b_*.pt"))
    assert len(ckpts) == 1 and str(ckpts[0]) == tr._best_checkpoint
    best = torch.load(tr._best_checkpoint, weights_only=True)
    model = tr.load_best_model()
    for k, v in model.state_dict().items():
        assert torch.equal(v, best[k])


def test_trainer_without_checkpoints_restores_best_in_memory():
    torch.manual_seed(0)
    tl, vl = _loaders()
    tr = Trainer(_Tiny(), TrainingConfig(epochs=4, device="cpu", save_checkpoints=False), "unused", bucket="b")
    tr.train(tl, vl)
    assert tr._best_checkpoint is None and tr._best_state is not None
    model = tr.load_best_model()
    for k, v in model.state_dict().items():
        assert torch.equal(v, tr._best_state[k])


def test_train_loader_shuffle_is_seeded():
    order = []
    for _ in range(2):
        tl, _ = _loaders()
        order.append(torch.cat([b[2] for b in tl]))
    assert torch.equal(order[0], order[1])


def test_event_schema_from_indexes():
    indexes = {"dynamic_indexes": {"ROLE_index": {"none": 0, "r1": 1}}}
    schema = build_event_schema(indexes, ["ROLE"], ["x"])
    assert schema.field_names == ["ROLE", "x"] and schema.cardinalities == [2]


@pytest.mark.parametrize("field", ["l2_reg", "save_checkpoints", "seed"])
def test_training_config_has_fields_trainer_reads(field):
    assert hasattr(TrainingConfig(), field)
