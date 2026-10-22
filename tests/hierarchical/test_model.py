"""Tests for XD2NetHier and HierTrainer."""

import numpy as np
import pytest
import torch

from xd2net.common.data.dataset import make_dataloaders
from xd2net.common.config import TrainingConfig
from xd2net.hierarchical.config import HierModelConfig
from xd2net.common.data.features import EventSchema
from xd2net.hierarchical.model import HierOutput, XD2NetHier
from xd2net.hierarchical.training import HierTrainer

SCHEMA = EventSchema(
    cat_fields=["ACTIVITY", "ROLE"],
    num_fields=["x", "y"],
    cardinalities=[5, 3],
    index_to_label={"ACTIVITY": {i: f"a{i}" for i in range(1, 6)}, "ROLE": {i: f"r{i}" for i in range(1, 4)}},
)
B, T, F, S, E = 4, 6, 4, 3, 7
LENGTHS = torch.tensor([1, 3, 6, 4])


def _packed(b=B, t=T, lengths=LENGTHS, seed=0):
    g = torch.Generator().manual_seed(seed)
    act = torch.randint(1, 6, (b, t), generator=g).float()
    role = torch.randint(1, 4, (b, t), generator=g).float()
    num = torch.randn(b, t, 2, generator=g)
    x = torch.cat([act[..., None], role[..., None], num], dim=-1)
    pad = torch.arange(t)[None, :] >= lengths[:, None]
    return x.masked_fill(pad[..., None], 0.0)


def _model(**cfg):
    torch.manual_seed(0)
    c = HierModelConfig(d_model=16, nhead=2, dropout=0.0, **cfg)
    return XD2NetHier(SCHEMA, S, E, max_len=10, cfg=c).eval()


def _inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    return _packed(seed=seed), torch.randn(B, S, generator=g), torch.rand(B, E, generator=g)


@pytest.fixture
def out():
    x, s, e = _inputs()
    return _model()(x, s, e, LENGTHS, return_attn=True)


def _real(t=T):
    return torch.arange(t)[None, :] < LENGTHS[:, None]


def test_output_types_and_shapes(out):
    m = _model()
    x, s, e = _inputs()
    thetas, prob, alpha = m(x, s, e, LENGTHS)
    assert thetas.shape == (B, E) and prob.shape == (B,) and alpha.shape == (B, T)
    assert isinstance(out, HierOutput)
    assert out.beta.shape == out.joint.shape == (B, T, F)
    assert out.evt_self_attn.shape == (B, T) and out.case_self_attn.shape == (B,)
    assert out.event_attn_raw[0].shape == (B * T, 2, 1 + F, 1 + F)
    assert out.case_attn_raw[0].shape == (B, 2, 1 + T, 1 + T)


def test_beta_sums_to_one_per_real_event(out):
    real = _real()
    assert torch.allclose(out.beta.sum(-1)[real], torch.ones(int(real.sum())), atol=1e-5)
    assert torch.all(out.beta[~real] == 0)


def test_alpha_sums_to_one_and_zero_on_padding(out):
    assert torch.allclose(out.alpha.sum(-1), torch.ones(B), atol=1e-5)
    assert torch.all(out.alpha[~_real()] == 0)


def test_joint_sums_to_one_per_case(out):
    assert torch.allclose(out.joint.sum((1, 2)), torch.ones(B), atol=1e-5)


def test_cls_self_mass_is_zero_by_default(out):
    assert torch.all(out.case_self_attn == 0)
    assert torch.all(out.evt_self_attn == 0)


def test_cls_self_attend_option():
    x, s, e = _inputs()
    o = _model(cls_self_attend=True)(x, s, e, LENGTHS, return_attn=True)
    assert torch.all(o.case_self_attn > 0)
    assert torch.allclose(o.joint.sum((1, 2)), torch.ones(B), atol=1e-5)


@pytest.mark.parametrize("readout,layers", [("rollout", 1), ("rollout", 3), ("last", 2)])
def test_mask_invariance(readout, layers):
    """Extra padded events with arbitrary content must not change outputs."""
    m = _model(event_layers=layers, case_layers=layers, attn_readout=readout)
    x, s, e = _inputs()
    o1 = m(x, s, e, LENGTHS, return_attn=True)
    junk = _packed(t=T + 3, lengths=torch.full((B,), T + 3), seed=9)[:, T:]
    o2 = m(torch.cat([x, junk], dim=1), s, e, LENGTHS, return_attn=True)
    real = _real()
    assert torch.allclose(o1.act_output, o2.act_output, atol=1e-5)
    assert torch.allclose(o1.alpha, o2.alpha[:, :T], atol=1e-5)
    assert torch.allclose(o1.beta[real], o2.beta[:, :T][real], atol=1e-5)
    assert torch.all(o2.alpha[:, T:] == 0)


def test_batch_size_one_and_no_lengths():
    m = _model()
    x, s, e = _inputs()
    thetas, prob, alpha = m(x[:1], s[:1], e[:1], LENGTHS[:1])
    assert prob.shape == (1,)
    o = m(x, s, e, None, return_attn=True)
    assert torch.allclose(o.alpha.sum(-1), torch.ones(B), atol=1e-5)


def test_all_encoder_params_under_encoder():
    m = _model()
    n_total = sum(p.numel() for p in m.parameters())
    n_split = sum(p.numel() for mod in (m.encoder, m.theta_head, m.prediction_head) for p in mod.parameters())
    assert n_total == n_split


def test_config_roundtrip():
    m = _model(event_layers=2)
    m2 = XD2NetHier.from_config(m.get_config())
    m2.load_state_dict(m.state_dict())
    m2.eval()
    x, s, e = _inputs()
    assert torch.allclose(m(x, s, e, LENGTHS)[1], m2(x, s, e, LENGTHS)[1])


def _arrays(n=80, seed=0):
    rng = np.random.default_rng(seed)
    lengths = rng.integers(1, T + 1, n)
    x = _packed(n, T, torch.from_numpy(lengths), seed).numpy()
    inputs = {
        "input_dynamic": x,
        "input_static": rng.normal(size=(n, S)).astype(np.float32),
        "input_exp": rng.random((n, E)).astype(np.float32),
    }
    return inputs, {"act_output": rng.integers(0, 2, n)}, lengths


def test_trainer_runs_and_checkpoints(tmp_path):
    inputs, outputs, lengths = _arrays()
    tl, vl = make_dataloaders(inputs, outputs, lengths, batch_size=16)
    m = _model()
    tr = HierTrainer(m, TrainingConfig(epochs=2, device="cpu"), str(tmp_path), bucket="t")
    hist = tr.train(tl, vl)
    assert len(hist["train_loss"]) == 2 and all(np.isfinite(hist["train_loss"]))
    assert tr.load_best_model() is m
    assert list(tmp_path.glob("model_t_*.pt"))


def test_l2_penalty_skips_query_key_and_norms():
    m = _model()
    tr = HierTrainer(m, TrainingConfig(device="cpu", save_checkpoints=False), "unused")
    base = tr._l2_penalty().item()
    attn = m.encoder["event"].encoder.layers[0].self_attn
    d = attn.in_proj_weight.size(1)
    with torch.no_grad():
        attn.in_proj_weight[: 2 * d] *= 10                          # Q, K: not penalised
        m.encoder["event"].encoder.layers[0].norm1.weight *= 10    # LayerNorm: not penalised
    assert tr._l2_penalty().item() == pytest.approx(base, rel=1e-6)
    with torch.no_grad():
        attn.in_proj_weight[2 * d:] *= 10                           # V: penalised
    assert tr._l2_penalty().item() > base
