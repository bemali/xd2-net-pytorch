"""Configuration shared by every XD2-Net variant.

Two kinds of YAML file are used:

* an **experiment config** per model (``model`` + ``training`` + ``workspace_dir``),
  e.g. ``configs/models/baseline.yaml``; the ``model`` section is validated by the
  variant's own model config class (see ``xd2net.common.registry``);
* one **dataset config** per event log describing its schema, outcome and
  prefix settings, e.g. ``configs/datasets/bpic2012.yaml``.

Unknown keys raise an error so that typos in a YAML file are caught early.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Union

import yaml


# ---------------------------------------------------------------------------
# Training / experiment
# ---------------------------------------------------------------------------

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
    l2_reg: float = 0.0001          # weight of the trainer's L2 penalty (see Trainer._l2_penalty)
    save_checkpoints: bool = True
    seed: int = 42
    device: str = "auto"


@dataclass
class ExperimentConfig:
    model: Any = None               # the variant's model config dataclass
    training: TrainingConfig = field(default_factory=TrainingConfig)
    workspace_dir: str = "./outputs"


# ---------------------------------------------------------------------------
# Dataset (one per event log)
# ---------------------------------------------------------------------------

@dataclass
class ColumnsConfig:
    caseid: str = "CASEID"
    timestamp: str = "TIMESTAMP"
    activity: str = "ACTIVITY"
    role: Optional[str] = "ROLE"        # set to null to discover roles from `resource`
    resource: Optional[str] = None      # only used when role is null


@dataclass
class OutcomeConfig:
    mode: str = "column"                # "column" | "activity"
    column: Optional[str] = None        # column mode: outcome column (constant per case)
    positive: Optional[Union[str, int, float, bool]] = None  # column mode: positive value
    activities: List[str] = field(default_factory=list)      # activity mode: target activities
    labels: List[str] = field(default_factory=lambda: ["negative", "positive"])  # [neg, pos]

    def __post_init__(self):
        if self.mode not in ("column", "activity"):
            raise ValueError(f"outcome.mode must be 'column' or 'activity', got {self.mode!r}.")
        if self.mode == "column" and (self.column is None or self.positive is None):
            raise ValueError("outcome.mode 'column' requires outcome.column and outcome.positive.")
        if self.mode == "activity" and not self.activities:
            raise ValueError("outcome.mode 'activity' requires a non-empty outcome.activities list.")
        if len(self.labels) != 2:
            raise ValueError("outcome.labels must be [negative_label, positive_label].")


@dataclass
class TimeFeatureConfig:
    since_prev: bool = True             # TIME_SINCE_PREV   (event numeric)
    since_start: bool = True            # TIME_SINCE_START  (event numeric)
    hour: bool = False                  # HOUR              (event categorical)
    weekday: bool = False               # WEEKDAY           (event categorical)
    unit: str = "hours"                 # seconds | minutes | hours | days

    def __post_init__(self):
        if self.unit not in TIME_UNITS:
            raise ValueError(f"time_features.unit must be one of {sorted(TIME_UNITS)}.")


TIME_UNITS = {"seconds": 1.0, "minutes": 60.0, "hours": 3600.0, "days": 86400.0}


@dataclass
class NormConfig:
    default: str = "max"                # baseline normalize_events: 'max' (z-score) | 'lognorm'
    overrides: Dict[str, str] = field(default_factory=dict)

    def method_for(self, feat: str) -> str:
        return self.overrides.get(feat, self.default)


@dataclass
class PrefixConfig:
    lengths: Union[str, List[int]] = "auto"
    min_len: int = 2
    max_len: int = 20
    quantile: float = 0.9
    min_cases: int = 10


@dataclass
class DatasetConfig:
    name: str = "dataset"
    path: str = ""
    columns: ColumnsConfig = field(default_factory=ColumnsConfig)
    timestamp_format: Optional[str] = None
    event_order: str = "timestamp"      # "timestamp" (stable sort) | "file" (keep row order per case)
    drop_activities: List[str] = field(default_factory=list)  # removed before prefixing (e.g. outcome-revealing terminal events)
    role_discovery_clusters: int = 5
    event_cat: List[str] = field(default_factory=list)
    event_num: List[str] = field(default_factory=list)
    case_cat: List[str] = field(default_factory=list)
    case_num: List[str] = field(default_factory=list)
    num_agg: Dict[str, str] = field(default_factory=dict)   # exp-input aggregation overrides
    outcome: Optional[OutcomeConfig] = None                  # required
    time_features: TimeFeatureConfig = field(default_factory=TimeFeatureConfig)
    norm: NormConfig = field(default_factory=NormConfig)
    prefix: PrefixConfig = field(default_factory=PrefixConfig)
    test_split: float = 0.30
    balance_train: bool = True
    case_level_tolerance: float = 0.0

    def __post_init__(self):
        if self.outcome is None:
            raise ValueError(f"[{self.name}] an outcome config is required.")
        if self.columns.role is None and self.columns.resource is None:
            raise ValueError(f"[{self.name}] columns.role or columns.resource must be set.")
        if self.event_order not in ("timestamp", "file"):
            raise ValueError(f"[{self.name}] event_order must be 'timestamp' or 'file'.")
        if self.outcome.mode == "activity" and set(self.outcome.activities) & set(self.drop_activities):
            raise ValueError(f"[{self.name}] outcome.activities must not appear in drop_activities.")
        declared = self.event_cat + self.event_num + self.case_cat + self.case_num
        dupes = {c for c in declared if declared.count(c) > 1}
        if dupes:
            raise ValueError(f"[{self.name}] variables declared more than once: {sorted(dupes)}")
        clash = set(declared) & RESERVED_NAMES
        if clash:
            raise ValueError(f"[{self.name}] variable names clash with reserved names: {sorted(clash)}")

    # Field lists in the order used by the model ---------------------------

    @property
    def event_cat_fields(self) -> List[str]:
        out = ["ACTIVITY", "ROLE", *self.event_cat]
        if self.time_features.hour:
            out.append("HOUR")
        if self.time_features.weekday:
            out.append("WEEKDAY")
        return out

    @property
    def event_num_fields(self) -> List[str]:
        out = list(self.event_num)
        if self.time_features.since_prev:
            out.append("TIME_SINCE_PREV")
        if self.time_features.since_start:
            out.append("TIME_SINCE_START")
        return out


# Internal column names used by the pipeline; user variables must not reuse them.
RESERVED_NAMES = {
    "CASEID", "TIMESTAMP", "ACTIVITY", "ROLE", "RESOURCE", "MILESTONE", "OUTCOME",
    "oc_index", "ACTIVITY_INDEX", "TIME_SINCE_PREV", "TIME_SINCE_START", "HOUR", "WEEKDAY",
}


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def _build(cls, d: Optional[dict], where: str):
    """Construct dataclass `cls` from dict `d`, rejecting unknown keys."""
    d = d or {}
    if not isinstance(d, dict):
        raise TypeError(f"{where}: expected a mapping, got {type(d).__name__}")
    known = {f.name for f in fields(cls)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}; allowed: {sorted(known)}")
    return cls(**d)


def load_experiment_config(path: str, model_config_cls) -> ExperimentConfig:
    """Load ``model`` / ``training`` / ``workspace_dir`` from YAML.

    ``model_config_cls`` is the variant's model config dataclass, which validates
    the ``model`` section.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    unknown = set(raw) - {"model", "training", "workspace_dir"}
    if unknown:
        raise ValueError(f"{path}: unknown top-level keys {sorted(unknown)}")
    return ExperimentConfig(
        model=_build(model_config_cls, raw.get("model"), f"{path}:model"),
        training=_build(TrainingConfig, raw.get("training"), f"{path}:training"),
        workspace_dir=raw.get("workspace_dir", "./outputs"),
    )


_NESTED = {
    "columns": ColumnsConfig,
    "outcome": OutcomeConfig,
    "time_features": TimeFeatureConfig,
    "norm": NormConfig,
    "prefix": PrefixConfig,
}


def dataset_config_from_dict(raw: dict, where: str = "dataset") -> DatasetConfig:
    raw = dict(raw or {})
    for key, cls in _NESTED.items():
        if key in raw:
            raw[key] = _build(cls, raw[key], f"{where}:{key}")
    return _build(DatasetConfig, raw, where)


def load_dataset_config(path: str) -> DatasetConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    return dataset_config_from_dict(raw, where=path)


def save_yaml(obj, path: str) -> None:
    """Serialise a dataclass (or dict) to YAML."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = asdict(obj) if hasattr(obj, "__dataclass_fields__") else obj
    with open(path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
