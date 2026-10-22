"""Device and mixed-precision helpers shared by the trainer, runner and notebooks."""

from __future__ import annotations

from typing import Optional

import torch


def resolve_device(device_str: str = "auto") -> torch.device:
    """``"auto"`` → CUDA, then MPS, then CPU; anything else is passed to ``torch.device``."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def amp_dtype(device: torch.device) -> Optional[torch.dtype]:
    """Autocast dtype for ``device``: bf16 where supported (CPU, Ampere+/L4), fp16 on older CUDA (T4);
    None where autocast is not used (MPS)."""
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "cpu":
        return torch.bfloat16
    return None


def accelerated(device: torch.device) -> bool:
    """True for devices where keeping the data on the device pays off."""
    return device.type in ("cuda", "mps")
