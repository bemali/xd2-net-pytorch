"""Registry entry for the baseline XD2Net."""

from __future__ import annotations

from xd2net.baseline.config import BaselineModelConfig
from xd2net.baseline.data.features import build_dynamic_onehot
from xd2net.baseline.explainer import BaselineExplainer
from xd2net.baseline.model import XD2Net
from xd2net.baseline.training import BaselineTrainer
from xd2net.common.registry import ModelSpec


def build_model(bucket: dict, cfg: BaselineModelConfig, dataset_name: str) -> XD2Net:
    return XD2Net(
        dim_dynamic=bucket["dim_dynamic"],
        dim_static=bucket["dim_static"],
        exp_dim=bucket["exp_dim"],
        max_len=bucket["max_len"],
        cfg=cfg,
        schema=bucket["event_schema"],
        dataset_name=dataset_name,
    )


SPEC = ModelSpec(
    name="baseline",
    model_config_cls=BaselineModelConfig,
    model_cls=XD2Net,
    encode_dynamic=build_dynamic_onehot,
    build_model=build_model,
    trainer_cls=BaselineTrainer,
    explainer_cls=BaselineExplainer,
    default_config="configs/models/baseline.yaml",
)
