"""Baseline XD2-Net: Transformer encoder + external TemporalAttentionPool."""

from xd2net.baseline.config import BaselineModelConfig
from xd2net.baseline.explainer import BaselineExplainer
from xd2net.baseline.model import XD2Net
from xd2net.baseline.training import BaselineTrainer

__all__ = ["BaselineModelConfig", "BaselineExplainer", "XD2Net", "BaselineTrainer"]
