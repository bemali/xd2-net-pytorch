#!/usr/bin/env python
"""Run one or more XD2-Net variants on one or more event logs.

Each model keeps its own model + training settings, applied unchanged to every
dataset. Each dataset is prepared once (same split, roles, normalisation) and
shared by all models, so their results are directly comparable.

Usage:
    python scripts/run_experiment.py \
        --model baseline hierarchical \
        --datasets configs/datasets/bpic2012.yaml configs/datasets/my_log.yaml \
        [--model-config configs/models/baseline.yaml configs/models/hierarchical.yaml] \
        [--buckets 1_5 1_10] [--seed 42] [--epochs 5] [--workspace ./outputs] [--fail-fast]

``--model-config`` defaults to ``configs/models/<model>.yaml`` for each model.

Outputs, under <workspace>/<model>/<dataset>/:
    dataset_config.yaml, experiment_config.yaml, dataset_summary.json
    checkpoints/model_<bucket>_*.pt, checkpoints/model_config_<bucket>.json
    explanations/predictions_<bucket>.csv           Level-2
    explanations/explanations_<bucket>.csv          Level-2 (thetas)
    explanations/attention_cells_<bucket>.csv       Level-1: (case, event, field) α [, β, α·β]
    explanations/attention_case_fields_<bucket>.csv
    explanations/attention_global_<bucket>.csv      per field and per (field, label); models with β only
    results.csv
and a combined <workspace>/results_all.csv with one row per (model, dataset, bucket).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from xd2net.common.config import (
    DatasetConfig,
    ExperimentConfig,
    load_dataset_config,
    load_experiment_config,
    save_yaml,
)
from xd2net.common.data import build_bucket, make_dataloaders, make_test_loader, prepare_dataset, resolve_prefix_lengths
from xd2net.common.data.pipeline import PreparedLog
from xd2net.common.metrics import evaluate_predictions
from xd2net.common.registry import ModelSpec, available_models, get_spec

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_experiment")

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_bucket(spec: ModelSpec, exp: ExperimentConfig, ds: DatasetConfig, bucket: dict, out_dir: str) -> dict:
    key = bucket["bucket_key"]
    tcfg = exp.training
    seed_everything(tcfg.seed)

    model = spec.build_model(bucket, exp.model, ds.name)
    train_loader, val_loader = make_dataloaders(
        bucket["inputs_train"], bucket["outputs_train"], bucket["lengths_train"],
        val_split=tcfg.val_split, batch_size=tcfg.batch_size, seed=tcfg.seed,
    )
    test_loader = make_test_loader(
        bucket["inputs_test"], bucket["outputs_test"], bucket["lengths_test"], batch_size=tcfg.batch_size,
    )

    ckpt_dir = os.path.join(out_dir, "checkpoints")
    trainer = spec.trainer_cls(model, tcfg, ckpt_dir, bucket=key)
    history = trainer.train(train_loader, val_loader)
    model = trainer.load_best_model()
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, f"model_config_{key}.json"), "w") as f:
        json.dump(model.get_config(), f, indent=2)

    explainer = spec.explainer_cls(model, device=tcfg.device)
    df_pred, df_exp = explainer.explain_loader(test_loader, bucket["caseids_test"], bucket["feature_names"], key)
    cells, case_fields = explainer.explain_attention_loader(test_loader, bucket["caseids_test"], key)

    exp_dir = os.path.join(out_dir, "explanations")
    os.makedirs(exp_dir, exist_ok=True)
    df_pred.to_csv(os.path.join(exp_dir, f"predictions_{key}.csv"), index=False)
    df_exp.to_csv(os.path.join(exp_dir, f"explanations_{key}.csv"), index=False)
    cells.to_csv(os.path.join(exp_dir, f"attention_cells_{key}.csv"), index=False)
    case_fields.to_csv(os.path.join(exp_dir, f"attention_case_fields_{key}.csv"), index=False)
    if cells["JOINT_ATTN"].notna().any():
        g_field = explainer.global_attention(cells, by="field").assign(LEVEL="field")
        g_label = explainer.global_attention(cells, by="field_label").assign(LEVEL="field_label")
        pd.concat([g_field, g_label], ignore_index=True).to_csv(
            os.path.join(exp_dir, f"attention_global_{key}.csv"), index=False
        )

    row = {"model": spec.name, "dataset": ds.name, "bucket": key, "n_train": bucket["n_train"],
           "n_test": bucket["n_test"], "epochs": len(history.get("train_loss", []))}
    try:
        m = evaluate_predictions(
            df_pred["GROUND_TRUTH"].values, df_pred["PREDICTION_PROB"].values, target_names=ds.outcome.labels
        )
        row.update({k: m[k] for k in METRICS})
    except ValueError as e:   # e.g. single class in the test bucket
        logger.warning("[%s/%s] bucket %s: metrics unavailable (%s)", spec.name, ds.name, key, e)
        row.update({k: float("nan") for k in METRICS})
    logger.info("[%s/%s] bucket %s — AUC %.3f  F1 %.3f  Acc %.3f",
                spec.name, ds.name, key, row["roc_auc"], row["f1"], row["accuracy"])
    return row


def run_dataset(
    models: List[tuple], ds: DatasetConfig, workspace: str, seed: int, buckets: Optional[List[str]]
) -> List[dict]:
    """Prepare ``ds`` once, then train/explain every model on every bucket."""
    prepared: PreparedLog = prepare_dataset(ds, seed=seed)
    prefix_lengths = resolve_prefix_lengths(prepared)
    summary = {**prepared.summary(), "prefix_lengths": prefix_lengths}
    logger.info("[%s] prefix lengths: %s", ds.name, prefix_lengths)

    out_dirs = {}
    for spec, exp in models:
        out_dir = os.path.join(workspace, spec.name, ds.name)
        os.makedirs(out_dir, exist_ok=True)
        save_yaml(ds, os.path.join(out_dir, "dataset_config.yaml"))
        save_yaml(exp, os.path.join(out_dir, "experiment_config.yaml"))
        with open(os.path.join(out_dir, "dataset_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        out_dirs[spec.name] = out_dir

    rows: List[dict] = []
    for n in prefix_lengths:
        key = f"1_{n}"
        if buckets and key not in buckets:
            continue
        for spec, exp in models:
            bucket = build_bucket(prepared, n, spec.encode_dynamic, seed=exp.training.seed)
            if bucket is None:
                break           # skipping is a property of the data, the same for every model
            rows.append(run_bucket(spec, exp, ds, bucket, out_dirs[spec.name]))
            model_rows = [r for r in rows if r["model"] == spec.name]
            pd.DataFrame(model_rows).to_csv(os.path.join(out_dirs[spec.name], "results.csv"), index=False)
    return rows


def main():
    ap = argparse.ArgumentParser(description="XD2-Net multi-model, multi-dataset runner")
    ap.add_argument("--model", nargs="+", default=["baseline"], choices=available_models(),
                    help="One or more model variants")
    ap.add_argument("--model-config", nargs="+",
                    help="One YAML per --model, in the same order (default configs/models/<model>.yaml)")
    ap.add_argument("--datasets", nargs="+", required=True, help="Dataset YAML files")
    ap.add_argument("--buckets", nargs="*", help="Subset of buckets, e.g. 1_5 1_10")
    ap.add_argument("--seed", type=int, help="Override training.seed (also used for the data split)")
    ap.add_argument("--epochs", type=int, help="Override training.epochs")
    ap.add_argument("--workspace", help="Override workspace_dir")
    ap.add_argument("--fail-fast", action="store_true", help="Stop at the first failing dataset")
    args = ap.parse_args()

    if args.model_config and len(args.model_config) != len(args.model):
        ap.error("--model-config needs one file per --model.")
    config_paths = args.model_config or [None] * len(args.model)

    models = []
    for name, path in zip(args.model, config_paths):
        spec = get_spec(name)
        exp = load_experiment_config(path or str(REPO_ROOT / spec.default_config), spec.model_config_cls)
        overrides = {k: v for k, v in (("seed", args.seed), ("epochs", args.epochs)) if v is not None}
        if overrides:
            exp.training = dataclasses.replace(exp.training, **overrides)
        models.append((spec, exp))

    workspace = args.workspace or models[0][1].workspace_dir
    for _, exp in models:
        exp.workspace_dir = workspace
    os.makedirs(workspace, exist_ok=True)
    seed = models[0][1].training.seed

    all_rows, failed = [], []
    for path in args.datasets:
        try:
            ds = load_dataset_config(path)
            logger.info("=" * 70)
            logger.info("Dataset %s (%s)", ds.name, path)
            all_rows.extend(run_dataset(models, ds, workspace, seed, args.buckets))
        except Exception:
            logger.exception("Dataset %s failed", path)
            failed.append(path)
            if args.fail_fast:
                raise
        if all_rows:
            pd.DataFrame(all_rows).to_csv(os.path.join(workspace, "results_all.csv"), index=False)

    if all_rows:
        df = pd.DataFrame(all_rows)
        print(df.to_string(index=False))
        print("\nMean AUC per model and dataset:")
        print(df.groupby(["model", "dataset"])["roc_auc"].mean().unstack("model").to_string())
    if failed:
        logger.error("Failed datasets: %s", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
