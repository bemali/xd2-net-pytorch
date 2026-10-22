"""Training loop shared by every XD2-Net variant (early stopping, LR scheduling, checkpoints).

Variants subclass ``Trainer`` and override ``_l2_penalty`` when they need a
different regulariser (see ``xd2net.hierarchical.training``).
"""

from __future__ import annotations

import copy
import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from xd2net.common.config import TrainingConfig
from xd2net.common.device import amp_dtype, resolve_device

logger = logging.getLogger(__name__)


def _build_optimizer(
    model: nn.Module, cfg: TrainingConfig
) -> torch.optim.Optimizer:
    opt_name = cfg.optimizer.lower()
    lr = cfg.learning_rate
    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    if opt_name == "nadam":
        return torch.optim.NAdam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-7)
    if opt_name == "adamax":
        return torch.optim.Adamax(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-7)
    if opt_name == "adagrad":
        return torch.optim.Adagrad(model.parameters(), lr=lr, initial_accumulator_value=0.1, eps=1e-7)
    logger.warning("Unknown optimizer '%s', falling back to Adam.", cfg.optimizer)
    return torch.optim.Adam(model.parameters(), lr=lr)


class Trainer:
    """Trains a single XD2-Net model for one prefix-length bucket.

    The model must expose an ``encoder`` module (the L2 target) and a forward
    ``(raw_input, raw_static, input_exp, lengths) -> (thetas, prob, alpha)``.

    Args:
        model:      model instance (on CPU; will be moved to device)
        cfg:        TrainingConfig
        output_dir: directory where checkpoints are saved
        bucket:     bucket identifier string for checkpoint naming (e.g. '1_15')
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: TrainingConfig,
        output_dir: str,
        bucket: str = "bucket",
    ):
        self.cfg = cfg
        self.output_dir = output_dir
        self.bucket = bucket
        self.device = resolve_device(cfg.device)

        # Mixed precision: autocast the forward pass; the loss is computed in fp32.
        self.amp_dtype = amp_dtype(self.device) if cfg.amp else None
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_dtype == torch.float16)

        self.model = model.to(self.device)
        self.optimizer = _build_optimizer(self.model, cfg)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=cfg.lr_reduce_factor,
            patience=cfg.lr_reduce_patience,
            min_lr=cfg.lr_min,
        )

        self._best_val_loss = float("inf")
        self._best_checkpoint: Optional[str] = None
        self._best_state: Optional[dict] = None   # in-memory copy when checkpoints are off

        if cfg.save_checkpoints:
            os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, List[float]]:
        """Run the full training loop.

        Returns:
            history dict: {'train_loss', 'val_loss', 'val_auc'}
        """
        history: Dict[str, List[float]] = defaultdict(list)
        patience_counter = 0

        for epoch in range(1, self.cfg.epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_loss, val_auc = self._val_epoch(val_loader)

            self.scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_auc"].append(val_auc)

            logger.info(
                "Epoch %3d | train_loss=%.4f | val_loss=%.4f | val_auc=%.4f | lr=%.2e",
                epoch,
                train_loss,
                val_loss,
                val_auc,
                self.optimizer.param_groups[0]["lr"],
            )

            if val_loss < self._best_val_loss - 1e-4:
                self._best_val_loss = val_loss
                patience_counter = 0
                if self.cfg.save_checkpoints:
                    self._save_checkpoint(epoch, val_loss)
                else:
                    self._best_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.early_stopping_patience:
                    logger.info("Early stopping at epoch %d.", epoch)
                    break

        return dict(history)

    def load_best_model(self) -> nn.Module:
        """Load the weights with the best validation loss."""
        if self._best_checkpoint is not None:
            state = torch.load(self._best_checkpoint, map_location=self.device, weights_only=True)
        elif self._best_state is not None:
            state = self._best_state
        else:
            return self.model
        self.model.load_state_dict(state)
        return self.model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _autocast(self):
        return torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None)

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = torch.zeros((), device=self.device)
        for batch in loader:
            raw_input, raw_input_static, input_exp, labels, lengths = self._to_device(batch)

            self.optimizer.zero_grad()
            with self._autocast():
                thetas, act_output, _ = self.model(raw_input, raw_input_static, input_exp, lengths)

            # binary_cross_entropy is not autocast-safe: compute the loss in fp32.
            bce = F.binary_cross_entropy(act_output.float(), labels)
            loss = bce + self.cfg.l2_reg * self._l2_penalty()

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.detach()

        return total_loss.item() / max(len(loader), 1)

    def _l2_penalty(self) -> Tensor:
        """Sum of squares over all encoder parameters (matches the original l2(0.0001))."""
        return sum(p.pow(2).sum() for p in self.model.encoder.parameters())

    def _val_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        all_labels: List[Tensor] = []
        all_probs: List[Tensor] = []

        with torch.no_grad():
            for batch in loader:
                raw_input, raw_input_static, input_exp, labels, lengths = self._to_device(batch)
                with self._autocast():
                    thetas, act_output, _ = self.model(
                        raw_input, raw_input_static, input_exp, lengths
                    )
                act_output = act_output.float()
                total_loss += F.binary_cross_entropy(act_output, labels)
                all_labels.append(labels)
                all_probs.append(act_output)

        y_true = torch.cat(all_labels).cpu().numpy()
        y_prob = torch.cat(all_probs).cpu().numpy()

        val_loss = float(total_loss) / max(len(loader), 1)
        try:
            from sklearn.metrics import roc_auc_score
            val_auc = roc_auc_score(y_true, y_prob)
        except Exception:
            val_auc = 0.0

        return val_loss, val_auc

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        fname = os.path.join(
            self.output_dir, f"model_{self.bucket}_ep{epoch:03d}_vl{val_loss:.4f}.pt"
        )
        torch.save(self.model.state_dict(), fname)
        previous, self._best_checkpoint = self._best_checkpoint, fname
        if previous and previous != fname and os.path.exists(previous):
            os.remove(previous)
        logger.debug("Checkpoint saved: %s", fname)

    def _to_device(self, batch: Tuple) -> Tuple:
        return tuple(t.to(self.device, non_blocking=True) for t in batch)
