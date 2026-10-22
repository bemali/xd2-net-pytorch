"""Registry entry for the hierarchical self-attention XD2NetHier."""

from __future__ import annotations

from xd2net.common.registry import ModelSpec
from xd2net.hierarchical.config import HierModelConfig
from xd2net.hierarchical.data.features import build_dynamic_tokens
from xd2net.hierarchical.explainer import HierExplainer
from xd2net.hierarchical.model import XD2NetHier
from xd2net.hierarchical.training import HierTrainer


def build_model(bucket: dict, cfg: HierModelConfig, dataset_name: str) -> XD2NetHier:
    return XD2NetHier(
        bucket["event_schema"],
        bucket["dim_static"],
        bucket["exp_dim"],
        bucket["max_len"],
        cfg,
        dataset_name=dataset_name,
    )


SPEC = ModelSpec(
    name="hierarchical",
    model_config_cls=HierModelConfig,
    model_cls=XD2NetHier,
    encode_dynamic=build_dynamic_tokens,
    build_model=build_model,
    trainer_cls=HierTrainer,
    explainer_cls=HierExplainer,
    default_config="configs/models/hierarchical.yaml",
)
