"""The GPU code paths, exercised on CPU: device-resident loaders, mixed precision, resume."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import SequentialSampler

import xd2net.common.data.dataset as dataset_mod
from tests.synthetic import dataset_dict, make_log
from xd2net.common.config import TrainingConfig, dataset_config_from_dict
from xd2net.common.data import build_bucket, make_dataloaders, make_test_loader, prepare_dataset
from xd2net.common.data.dataset import TensorBatchLoader
from xd2net.common.registry import available_models, get_spec

ROOT = Path(__file__).resolve().parents[2]


def _arrays(n=50, T=4, D=3, E=5):
    rng = np.random.default_rng(0)
    inputs = {"input_dynamic": rng.random((n, T, D)).astype(np.float32),
              "input_static": np.zeros((n, 0), np.float32),
              "input_exp": np.arange(n * E, dtype=np.float32).reshape(n, E)}   # row id in column 0 / E
    return inputs, {"act_output": rng.integers(0, 2, n)}, np.full(n, T)


@pytest.fixture
def force_tensor_loader(monkeypatch):
    """Route device=... through the TensorBatchLoader path on CPU."""
    monkeypatch.setattr(dataset_mod, "_on_accelerator", lambda d: None if d is None else torch.device("cpu"))


def _row_ids(loader):
    return torch.cat([b[2][:, 0] for b in loader]).long().div(5, rounding_mode="floor").tolist()


def test_tensor_loader_same_split_as_dataloader(force_tensor_loader):
    inputs, outputs, lengths = _arrays()
    tl_a, vl_a = make_dataloaders(inputs, outputs, lengths, batch_size=8, seed=3)                 # DataLoader
    tl_b, vl_b = make_dataloaders(inputs, outputs, lengths, batch_size=8, seed=3, device="cuda")  # forced
    assert isinstance(tl_b, TensorBatchLoader) and isinstance(vl_b, TensorBatchLoader)
    assert sorted(_row_ids(tl_a)) == sorted(_row_ids(tl_b))
    assert _row_ids(vl_a) == _row_ids(vl_b)                    # validation order is sequential in both
    assert len(tl_a) == len(tl_b) and len(vl_a) == len(vl_b)


def test_tensor_loader_shuffle_is_seeded_and_complete():
    t = torch.arange(20).float()
    a = TensorBatchLoader([t], batch_size=6, shuffle=True, seed=1)
    b = TensorBatchLoader([t], batch_size=6, shuffle=True, seed=1)
    ea, eb = [torch.cat([x[0] for x in a]) for _ in range(2)], [torch.cat([x[0] for x in b]) for _ in range(2)]
    assert all(torch.equal(x, y) for x, y in zip(ea, eb))     # same sequence of epochs per seed
    assert not torch.equal(ea[0], ea[1])                      # epochs reshuffle
    assert sorted(ea[0].tolist()) == list(range(20)) and len(a) == 4


def test_test_loader_is_sequential(force_tensor_loader):
    inputs, outputs, lengths = _arrays()
    loader = make_test_loader(inputs, outputs, lengths, batch_size=16, device="cuda")
    assert isinstance(loader.sampler, SequentialSampler)
    assert _row_ids(loader) == list(range(50))


@pytest.mark.parametrize("name", available_models())
def test_explainer_with_device_loader(name, force_tensor_loader):
    p = prepare_dataset(dataset_config_from_dict(dataset_dict("unused.csv")), df=make_log(160, seed=7))
    spec = get_spec(name)
    b = build_bucket(p, 4, spec.encode_dynamic)
    model = spec.build_model(b, spec.model_config_cls(d_model=16, nhead=2), "syn")
    ex = spec.explainer_cls(model, "cpu")
    ref = make_test_loader(b["inputs_test"], b["outputs_test"], b["lengths_test"], batch_size=32)
    dev = make_test_loader(b["inputs_test"], b["outputs_test"], b["lengths_test"], batch_size=32, device="cuda")
    cells_ref, _ = ex.explain_attention_loader(ref, b["caseids_test"], "1_4")
    cells_dev, _ = ex.explain_attention_loader(dev, b["caseids_test"], "1_4")
    pd.testing.assert_frame_equal(cells_ref, cells_dev)


@pytest.mark.parametrize("name", available_models())
def test_trainer_amp_on_cpu(name, tmp_path):
    p = prepare_dataset(dataset_config_from_dict(dataset_dict("unused.csv")), df=make_log(160, seed=7))
    spec = get_spec(name)
    b = build_bucket(p, 4, spec.encode_dynamic)
    torch.manual_seed(0)
    model = spec.build_model(b, spec.model_config_cls(d_model=16, nhead=2), "syn")
    tl, vl = make_dataloaders(b["inputs_train"], b["outputs_train"], b["lengths_train"], batch_size=32)
    tr = spec.trainer_cls(model, TrainingConfig(epochs=1, device="cpu", amp=True), str(tmp_path), bucket="b")
    assert tr.amp_dtype == torch.bfloat16 and not tr.scaler.is_enabled()
    hist = tr.train(tl, vl)
    assert np.isfinite(hist["train_loss"][0]) and np.isfinite(hist["val_loss"][0])


def _run(args, cwd):
    env = {**os.environ, "PYTHONWARNINGS": "ignore", "PYTHONIOENCODING": "utf-8"}
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "run_experiment.py"), *args],
                          cwd=cwd, capture_output=True, text=True, encoding="utf-8", env=env)


def test_runner_skip_existing_resumes(tmp_path):
    import yaml

    make_log(200, seed=8).to_csv(tmp_path / "log.csv", index=False)
    (tmp_path / "ds.yaml").write_text(yaml.safe_dump(dataset_dict(str(tmp_path / "log.csv"), name="syn")))
    ws = tmp_path / "ws"
    args = ["--model", *available_models(), "--datasets", str(tmp_path / "ds.yaml"), "--epochs", "1",
            "--buckets", "1_3", "1_4", "--workspace", str(ws), "--skip-existing", "--device", "cpu"]

    first = _run(args, tmp_path)
    assert first.returncode == 0, first.stderr[-2000:]
    ckpts = sorted(ws.rglob("model_1_*.pt"))
    assert len(ckpts) == 2 * len(available_models())
    mtimes = [c.stat().st_mtime_ns for c in ckpts]
    results = pd.read_csv(ws / "results_all.csv")

    second = _run(args, tmp_path)
    assert second.returncode == 0, second.stderr[-2000:]
    assert second.stderr.count("already done, skipping") == 2 * len(available_models())
    assert [c.stat().st_mtime_ns for c in sorted(ws.rglob("model_1_*.pt"))] == mtimes
    pd.testing.assert_frame_equal(pd.read_csv(ws / "results_all.csv"), results)
