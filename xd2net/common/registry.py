"""Model registry: one ``ModelSpec`` per XD2-Net variant.

A spec bundles everything the runner needs to treat the variants the same way:
the model config class, the dynamic-input encoder passed to ``build_bucket``,
how to build the model from a bucket, and the trainer / explainer classes.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Type

import torch.nn as nn

from xd2net.common.data.pipeline import DynamicEncoder

# name -> module defining SPEC
_SPEC_MODULES: Dict[str, str] = {
    "baseline": "xd2net.baseline.spec",
    "hierarchical": "xd2net.hierarchical.spec",
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_config_cls: Type
    model_cls: Type[nn.Module]                                  # has get_config / from_config
    encode_dynamic: DynamicEncoder                              # (vec, schema, max_len) -> input_dynamic
    build_model: Callable[[dict, Any, str], nn.Module]          # (bucket, model_cfg, dataset_name) -> model
    trainer_cls: Type
    explainer_cls: Type
    default_config: str                                         # path relative to the repo root


def available_models() -> List[str]:
    return list(_SPEC_MODULES)


def get_spec(name: str) -> ModelSpec:
    if name not in _SPEC_MODULES:
        raise ValueError(f"Unknown model {name!r}; choose from {available_models()}.")
    return importlib.import_module(_SPEC_MODULES[name]).SPEC
