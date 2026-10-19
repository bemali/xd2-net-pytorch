#!/usr/bin/env python
"""Headless CLI for running a full XD2-Net experiment.

Usage:
    python scripts/run_experiment.py --config configs/bpic2012.yaml
    python scripts/run_experiment.py --config configs/bpic2012.yaml --buckets 1_10 1_15
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from xd2net.config import load_config, XD2Config
from xd2net.data.preprocessing import (
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
from xd2net.data.feature_engineering import generate_inputs, align_exp_dims
from xd2net.data.dataset import make_dataloaders, make_test_loader
from xd2net.model.xd2net import XD2Net
from xd2net.training.trainer import Trainer
from xd2net.training.metrics import evaluate_predictions
from xd2net.explain.explainer import Explainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_experiment")


# ---------------------------------------------------------------------------

def load_and_preprocess(cfg: XD2Config):
    """Load the CSV, split, encode, and bucket."""
    logger.info("Loading data from %s", cfg.data.csv_path)
    df = pd.read_csv(cfg.data.csv_path)

    # Rename columns to internal names expected by the pipeline
    col_map = {
        cfg.data.caseid_col: "CASEID",
        cfg.data.activity_col: "MILESTONE",
        cfg.data.outcome_col: "OUTCOME",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k != v})

    # Binary outcome label
    df["oc_index"] = (df["OUTCOME"] == cfg.data.approved_label).astype(int)

    # Activity index (used for prefix extraction)
    df = df.sort_values(["CASEID", "Complete Timestamp"]) if "Complete Timestamp" in df.columns else df
    df["ACTIVITY_INDEX"] = df.groupby("CASEID").cumcount() + 1

    # Add ACTIVITY column (alias of MILESTONE)
    df["ACTIVITY"] = df["MILESTONE"]

    # Train / test split (before role discovery to prevent leakage)
    df_train_full, df_test_full = split_train_test(df, cfg.data.test_split)
    logger.info("Train cases: %d  Test cases: %d", df_train_full["CASEID"].nunique(), df_test_full["CASEID"].nunique())

    # Role discovery (fit on train only)
    role_disc = RoleDiscovery(n_clusters=cfg.data.role_discovery_clusters)
    df_train_full = role_disc.fit(df_train_full)
    df_test_full = role_disc.transform(df_test_full)

    # Normalize numerical features (fit on train, apply to test)
    all_num = cfg.data.num_dynamic + cfg.data.num_static
    df_train_full, scalers = normalize_events(df_train_full, all_num, cfg.data.norm_method)
    df_test_full, _ = normalize_events(df_test_full, all_num, cfg.data.norm_method, scalers=scalers)

    # Categorical index encoding (from train)
    indexes: dict = {"dynamic_indexes": {}, "static_indexes": {}}
    for feat in cfg.data.cat_dynamic:
        indexes["dynamic_indexes"][f"{feat}_index"] = create_index(df_train_full, feat, reserve_zero=True)
        df_train_full[f"{feat}_index"] = df_train_full[feat].map(indexes["dynamic_indexes"][f"{feat}_index"]).fillna(0).astype(int)
        df_test_full[f"{feat}_index"] = df_test_full[feat].map(indexes["dynamic_indexes"][f"{feat}_index"]).fillna(0).astype(int)

    for feat in cfg.data.cat_static:
        indexes["static_indexes"][f"{feat}_index"] = create_index(df_train_full, feat, reserve_zero=True)
        df_train_full[f"{feat}_index"] = df_train_full[feat].map(indexes["static_indexes"][f"{feat}_index"]).fillna(0).astype(int)
        df_test_full[f"{feat}_index"] = df_test_full[feat].map(indexes["static_indexes"][f"{feat}_index"]).fillna(0).astype(int)

    indexes["oc_index"] = {0: "A_DECLINED", 1: "A_APPROVED"}

    return df_train_full, df_test_full, indexes, scalers


def process_bucket(
    cfg: XD2Config,
    df_train_full: pd.DataFrame,
    df_test_full: pd.DataFrame,
    indexes: dict,
    n_size: int,
) -> dict | None:
    """Prepare data for one prefix-length bucket."""
    bucket_key = f"1_{n_size}"

    df_bucket_train = extract_prefixes(df_train_full, n_size)
    df_bucket_test = extract_prefixes(df_test_full, n_size)

    if df_bucket_train["CASEID"].nunique() < 10:
        logger.warning("Bucket %s: too few training cases, skipping.", bucket_key)
        return None

    if cfg.data.balance_train:
        df_bucket_train = balance_dataset(
            df_bucket_train, "OUTCOME", cfg.data.approved_label, "CASEID"
        )

    log_train = reformat_events(df_bucket_train, indexes, cfg.data.cat_dynamic,
                                cfg.data.num_dynamic, cfg.data.cat_static, cfg.data.num_static)
    log_test = reformat_events(df_bucket_test, indexes, cfg.data.cat_dynamic,
                               cfg.data.num_dynamic, cfg.data.cat_static, cfg.data.num_static)

    max_len_train, cases_train = trace_stats(log_train)
    max_len_test, _ = trace_stats(log_test)
    max_len = max(max_len_train, max_len_test)

    vec_train = vectorize(log_train, max_len, cfg.data.cat_dynamic, cfg.data.num_dynamic,
                          cfg.data.cat_static, cfg.data.num_static)
    vec_test = vectorize(log_test, max_len, cfg.data.cat_dynamic, cfg.data.num_dynamic,
                         cfg.data.cat_static, cfg.data.num_static)

    inputs_train, outputs_train, exp_dim, feature_names = generate_inputs(
        vec_train, indexes, max_len, cfg.data.cat_dynamic, cfg.data.num_dynamic,
        cfg.data.cat_static, cfg.data.num_static,
    )
    inputs_test, outputs_test, exp_dim_test, feature_names_test = generate_inputs(
        vec_test, indexes, max_len, cfg.data.cat_dynamic, cfg.data.num_dynamic,
        cfg.data.cat_static, cfg.data.num_static,
    )
    inputs_train, inputs_test, feature_names = align_exp_dims(
        inputs_train, inputs_test, feature_names, feature_names_test
    )

    return {
        "bucket_key": bucket_key,
        "inputs_train": inputs_train,
        "outputs_train": outputs_train,
        "lengths_train": vec_train["lengths"],
        "caseids_train": vec_train["caseid"],
        "inputs_test": inputs_test,
        "outputs_test": outputs_test,
        "lengths_test": vec_test["lengths"],
        "caseids_test": vec_test["caseid"],
        "exp_dim": inputs_train["input_exp"].shape[1],
        "max_len": max_len,
        "feature_names": feature_names,
        "dim_dynamic": inputs_train["input_dynamic"].shape[2],
        "dim_static": inputs_train["input_static"].shape[1],
    }


def run(cfg: XD2Config, buckets_to_run: list[str] | None = None):
    """Run the full experiment pipeline."""
    os.makedirs(cfg.output.workspace_dir, exist_ok=True)
    ckpt_dir = os.path.join(cfg.output.workspace_dir, "checkpoints")
    results_rows = []

    df_train_full, df_test_full, indexes, scalers = load_and_preprocess(cfg)

    for n_size in cfg.data.prefix_lengths:
        bucket_key = f"1_{n_size}"
        if buckets_to_run and bucket_key not in buckets_to_run:
            continue

        logger.info("=" * 60)
        logger.info("Bucket %s", bucket_key)
        bucket = process_bucket(cfg, df_train_full, df_test_full, indexes, n_size)
        if bucket is None:
            continue

        # Model
        model = XD2Net(
            dim_dynamic=bucket["dim_dynamic"],
            dim_static=bucket["dim_static"],
            exp_dim=bucket["exp_dim"],
            max_len=bucket["max_len"],
            cfg=cfg.model,
        )

        # DataLoaders
        train_loader, val_loader = make_dataloaders(
            bucket["inputs_train"],
            bucket["outputs_train"],
            bucket["lengths_train"],
            val_split=cfg.training.val_split,
            batch_size=cfg.training.batch_size,
        )
        test_loader = make_test_loader(
            bucket["inputs_test"],
            bucket["outputs_test"],
            bucket["lengths_test"],
            batch_size=cfg.training.batch_size,
        )

        # Train
        trainer = Trainer(model, cfg.training, ckpt_dir, bucket=bucket_key)
        history = trainer.train(train_loader, val_loader)
        model = trainer.load_best_model()

        # Evaluate
        explainer = Explainer(model, device=cfg.training.device)
        df_pred, df_exp = explainer.explain_loader(
            test_loader,
            bucket["caseids_test"],
            bucket["feature_names"],
            bucket=bucket_key,
        )
        metrics = evaluate_predictions(
            df_pred["GROUND_TRUTH"].values,
            df_pred["PREDICTION_PROB"].values,
            target_names=["A_DECLINED", "A_APPROVED"],
        )
        logger.info(
            "Bucket %s — Accuracy: %.3f  AUC: %.3f  F1: %.3f",
            bucket_key, metrics["accuracy"], metrics["roc_auc"], metrics["f1"],
        )

        results_rows.append({
            "bucket": bucket_key,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
        })

        # Save explanation outputs
        out_dir = os.path.join(cfg.output.workspace_dir, "explanations")
        os.makedirs(out_dir, exist_ok=True)
        df_pred.to_csv(os.path.join(out_dir, f"predictions_{bucket_key}.csv"), index=False)
        df_exp.to_csv(os.path.join(out_dir, f"explanations_{bucket_key}.csv"), index=False)

    # Save results summary
    df_results = pd.DataFrame(results_rows)
    results_path = os.path.join(cfg.output.workspace_dir, "results.csv")
    df_results.to_csv(results_path, index=False)
    logger.info("Results saved to %s", results_path)
    print(df_results.to_string(index=False))


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="XD2-Net experiment runner")
    parser.add_argument("--config", default="configs/bpic2012.yaml", help="Path to YAML config")
    parser.add_argument("--buckets", nargs="*", help="Subset of buckets to run (e.g. 1_10 1_15)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run(cfg, buckets_to_run=args.buckets)


if __name__ == "__main__":
    main()
