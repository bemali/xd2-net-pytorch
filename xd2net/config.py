"""Configuration dataclasses and YAML helpers for XD2-Net."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import yaml


@dataclass
class DataConfig:
    csv_path: str = "data/BPIC_2012_OUTCOME.csv"
    caseid_col: str = "CASEID"
    activity_col: str = "ACTIVITY"
    outcome_col: str = "OUTCOME"
    approved_label: str = "A_APPROVED"
    cat_dynamic: List[str] = field(default_factory=lambda: ["ACTIVITY", "ROLE"])
    num_dynamic: List[str] = field(default_factory=lambda: ["TASK_DURATION", "TIME_ELAPSED"])
    cat_static: List[str] = field(default_factory=list)
    num_static: List[str] = field(default_factory=lambda: ["AMOUNT_REQ"])
    norm_method: str = "max"
    test_split: float = 0.30
    prefix_lengths: List[int] = field(default_factory=lambda: list(range(5, 21)))
    role_discovery_clusters: int = 5
    balance_train: bool = True


@dataclass
class ModelConfig:
    d_model: int = 140
    nhead: int = 4
    num_encoder_layers: int = 1
    dim_feedforward_mult: int = 4
    dropout: float = 0.15
    dense_1_mult: float = 1.5
    l2_reg: float = 0.0001


@dataclass
class TrainingConfig:
    optimizer: str = "Adam"
    learning_rate: float = 0.01
    batch_size: int = 256
    epochs: int = 100
    val_split: float = 0.15
    early_stopping_patience: int = 10
    lr_reduce_patience: int = 10
    lr_reduce_factor: float = 0.5
    lr_min: float = 1e-7
    device: str = "auto"


@dataclass
class OutputConfig:
    workspace_dir: str = "./outputs"
    experiment_name: str = "AG"
    log_name: str = "BPIC_2012"
    save_checkpoints: bool = True
    explain_buckets: List[str] = field(
        default_factory=lambda: ["1_5", "1_10", "1_15", "1_20"]
    )


@dataclass
class XD2Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _from_dict(cls, d: dict):
    """Recursively construct a dataclass from a nested dict."""
    import dataclasses

    if not dataclasses.is_dataclass(cls):
        return d
    field_types = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs = {}
    for k, v in d.items():
        if k not in field_types:
            continue
        kwargs[k] = v
    return cls(**kwargs)


def load_config(path: str) -> XD2Config:
    """Load an XD2Config from a YAML file."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return XD2Config(
        data=_from_dict(DataConfig, raw.get("data", {})),
        model=_from_dict(ModelConfig, raw.get("model", {})),
        training=_from_dict(TrainingConfig, raw.get("training", {})),
        output=_from_dict(OutputConfig, raw.get("output", {})),
    )


def save_config(cfg: XD2Config, path: str) -> None:
    """Serialize an XD2Config to a YAML file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(asdict(cfg), f, default_flow_style=False, sort_keys=False)
