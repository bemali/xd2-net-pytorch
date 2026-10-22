"""Trainer for the baseline XD2Net.

The shared ``Trainer`` already applies the baseline's regulariser (L2 over every
``model.encoder`` parameter), so this subclass only exists to mirror the
hierarchical package layout.
"""

from __future__ import annotations

from xd2net.common.training import Trainer


class BaselineTrainer(Trainer):
    pass
