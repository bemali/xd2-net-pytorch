"""Tests for attention layers and explanation readout."""

import torch

from xd2net.hierarchical.layers import (
    AttnEncoder,
    AttnEncoderLayer,
    attention_rollout,
    cls_attn_mask,
    cls_readout,
)


def test_layer_returns_per_head_attention():
    torch.manual_seed(0)
    layer = AttnEncoderLayer(16, 2, 32, 0.0).eval()
    out, w = layer(torch.randn(3, 5, 16))
    assert out.shape == (3, 5, 16)
    assert w.shape == (3, 2, 5, 5)
    assert torch.allclose(w.sum(-1), torch.ones(3, 2, 5), atol=1e-6)


def test_cls_mask_blocks_self_cell():
    torch.manual_seed(0)
    enc = AttnEncoder(16, 2, 2, 32, 0.0).eval()
    _, attns = enc(torch.randn(3, 5, 16), attn_mask=cls_attn_mask(5, cls_self_attend=False))
    for a in attns:
        assert torch.all(a[:, :, 0, 0] == 0)
        assert torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)), atol=1e-6)
    assert cls_attn_mask(5, cls_self_attend=True) is None


def test_head_mean_readout_matches_manual():
    a = torch.softmax(torch.randn(4, 3, 6, 6), dim=-1)
    w, self_mass = cls_readout([a], mode="last")
    manual = a.mean(1)[:, 0, :]
    assert torch.allclose(self_mass, manual[:, 0])
    assert torch.allclose(w, manual[:, 1:] / manual[:, 1:].sum(-1, keepdim=True))
    assert torch.allclose(w.sum(-1), torch.ones(4))


def test_rollout_equals_last_for_one_layer():
    a = torch.softmax(torch.randn(4, 3, 6, 6), dim=-1)
    w_last, s_last = cls_readout([a], mode="last")
    w_roll, s_roll = cls_readout([a], mode="rollout")
    assert torch.allclose(w_last, w_roll, atol=1e-6)
    assert torch.allclose(s_last, s_roll)                          # self-mass is always raw last-layer


def test_rollout_rows_sum_to_one():
    attns = [torch.softmax(torch.randn(2, 4, 5, 5), dim=-1) for _ in range(3)]
    r = attention_rollout(attns)
    assert r.shape == (2, 5, 5)
    assert torch.allclose(r.sum(-1), torch.ones(2, 5), atol=1e-6)


def test_readout_token_mask_zeroes_padding():
    a = torch.softmax(torch.randn(2, 2, 5, 5), dim=-1)
    mask = torch.tensor([[False, False, True, True], [False, False, False, False]])
    w, _ = cls_readout([a], mode="last", token_mask=mask)
    assert torch.all(w[0, 2:] == 0)
    assert torch.allclose(w.sum(-1), torch.ones(2))
