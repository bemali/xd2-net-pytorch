"""XD2-Net: Inherently Explainable Process Outcome Prediction."""

from xd2net.config import XD2Config, load_config, save_config
from xd2net.model.xd2net import XD2Net

__all__ = ["XD2Config", "load_config", "save_config", "XD2Net"]
