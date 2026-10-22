"""XD2-Net: Inherently Explainable Process Outcome Prediction.

Packages:
    xd2net.common        shared configs, data pipeline, heads, trainer, explainer, registry
    xd2net.baseline      Transformer + TemporalAttentionPool (XD2Net)
    xd2net.hierarchical  hierarchical self-attention (XD2NetHier)
"""

from xd2net.baseline.model import XD2Net
from xd2net.common.registry import available_models, get_spec
from xd2net.hierarchical.model import XD2NetHier

__all__ = ["XD2Net", "XD2NetHier", "available_models", "get_spec"]
