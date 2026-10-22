"""Tests for the baseline XD2Net, its one-hot input and BaselineTrainer."""

import numpy as np
import pytest
import torch

from xd2net.baseline.config import BaselineModelConfig
from xd2net.baseline.data.features import build_dynamic_onehot, dynamic_feature_names
from xd2net.baseline.model import XD2Net
from xd2net.baseline.training import BaselineTrainer
from xd2net.common.config import TrainingConfig
from xd2net.common.data.dataset import make_dataloaders
from xd2net.common.data.features import EventSchema, build_event_schema

INDEXES = {"dynamic_indexes": {"ACTIVITY_index": {"none": 0, "a": 1, "b": 2}}, "static_indexes": {}}
B, T, S, E = 4, 6, 3, 7
LENGTHS = torch.tensor([1, 3, 6, 4])
SCHEMA = EventSchema(
    cat_fields=["ACTIVITY"], num_fields=["x"], cardinalities=[3],
    index_to_label={"ACTIVITY": {1: "none", 2: "a", 3: "b"}},
)
D = 4   # 3 one-hot + 1 numeric


def _vec():
    return {
        "prefixes": {
            "ACTIVITY": np.array([[2, 1, 0, 0], [0, 0, 0, 0]], dtype=np.float32),
            "x": np.array([[1.0, 2.0, 3.0, 0.0], [4.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        },
        "statics": {}, "outcome": np.array([0, 1]), "lengths": np.array([3, 1]),
    }


def test_onehot_layout_and_zero_padding():
    schema = build_event_schema(INDEXES, ["ACTIVITY"], ["x"])
    x = build_dynamic_onehot(_vec(), schema, 4)
    assert x.shape == (2, 4, 4)
    assert x[0, :3, :3].tolist() == [[0, 0, 1], [0, 1, 0], [1, 0, 0]]   # b, a, none (real "none" event kept)
    assert x[0, :3, 3].tolist() == [1.0, 2.0, 3.0]
    assert not x[0, 3].any() and not x[1, 1:].any()                      # padded steps are all-zero
    assert dynamic_feature_names(schema) == ["ACTIVITY=none", "ACTIVITY=a", "ACTIVITY=b", "x"]


def _inputs(seed=0, t=T, lengths=LENGTHS):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, 3, (B, t), generator=g)
    x = torch.cat([torch.nn.functional.one_hot(idx, 3).float(), torch.randn(B, t, 1, generator=g)], -1)
    x = x.masked_fill((torch.arange(t)[None, :] >= lengths[:, None])[..., None], 0.0)
    return x, torch.randn(B, S, generator=g), torch.rand(B, E, generator=g)


def _model():
    torch.manual_seed(0)
    return XD2Net(D, S, E, max_len=10, cfg=BaselineModelConfig(d_model=16, nhead=2, dropout=0.0),
                  schema=SCHEMA, dataset_name="toy").eval()


def test_forward_shapes_and_alpha():
    thetas, prob, alpha = _model()(*_inputs(), LENGTHS)
    assert thetas.shape == (B, E) and prob.shape == (B,) and alpha.shape == (B, T)
    assert torch.allclose(alpha.sum(1), torch.ones(B), atol=1e-5)
    assert (alpha[torch.arange(T)[None, :] >= LENGTHS[:, None]] == 0).all()


def test_mask_invariance():
    m = _model()
    x, s, e = _inputs()
    _, p1, a1 = m(x, s, e, LENGTHS)
    x2 = torch.cat([x, torch.zeros(B, 3, D)], dim=1)                     # extra padding
    _, p2, a2 = m(x2, s, e, LENGTHS)
    assert torch.allclose(p1, p2, atol=1e-5)
    assert torch.allclose(a1, a2[:, :T], atol=1e-5)


def test_config_roundtrip():
    m = _model()
    m2 = XD2Net.from_config(m.get_config())
    m2.load_state_dict(m.state_dict())
    m2.eval()
    x, s, e = _inputs()
    assert torch.allclose(m(x, s, e, LENGTHS)[1], m2(x, s, e, LENGTHS)[1])
    assert m2.schema == SCHEMA and m2.dataset_name == "toy"


def test_trainer_runs_and_checkpoints(tmp_path):
    rng = np.random.default_rng(0)
    n = 48
    inputs = {"input_dynamic": rng.random((n, T, D)).astype(np.float32),
              "input_static": rng.random((n, S)).astype(np.float32),
              "input_exp": rng.random((n, E)).astype(np.float32)}
    tl, vl = make_dataloaders(inputs, {"act_output": rng.integers(0, 2, n)}, np.full(n, T), batch_size=16)
    m = _model()
    tr = BaselineTrainer(m, TrainingConfig(epochs=2, device="cpu"), str(tmp_path), bucket="t")
    hist = tr.train(tl, vl)
    assert len(hist["train_loss"]) == 2 and all(np.isfinite(hist["train_loss"]))
    assert tr.load_best_model() is m
    assert len(list(tmp_path.glob("model_t_*.pt"))) == 1


def test_l2_penalty_covers_all_encoder_params():
    m = _model()
    tr = BaselineTrainer(m, TrainingConfig(device="cpu", save_checkpoints=False), "unused")
    expected = sum(p.pow(2).sum() for p in m.encoder.parameters())
    assert tr._l2_penalty().item() == pytest.approx(expected.item())
