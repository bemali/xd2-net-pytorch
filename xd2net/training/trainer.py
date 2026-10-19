"""Training loop for XD2-Net with early stopping and LR scheduling."""

from __future__ import annotations

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

from xd2net.config import TrainingConfig
from xd2net.model.xd2net import XD2Net
from xd2net.training.metrics import evaluate_predictions

logger = logging.getLogger(__name__)


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def _build_optimizer(
    model: XD2Net, cfg: TrainingConfig
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

    Args:
        model:      XD2Net instance (on CPU; will be moved to device)
        cfg:        TrainingConfig
        output_dir: directory where checkpoints are saved
        bucket:     bucket identifier string for checkpoint naming (e.g. '1_15')
    """

    def __init__(
        self,
        model: XD2Net,
        cfg: TrainingConfig,
        output_dir: str,
        bucket: str = "bucket",
    ):
        self.cfg = cfg
        self.output_dir = output_dir
        self.bucket = bucket
        self.device = _resolve_device(cfg.device)

        self.model = model.to(self.device)
        self.optimizer = _build_optimizer(self.model, cfg)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=cfg.lr_reduce_factor,
            patience=cfg.lr_reduce_patience,
            min_lr=cfg.lr_min,
            verbose=False,
        )

        self._best_val_loss = float("inf")
        self._best_checkpoint: Optional[str] = None

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
                patience_counter += 1
                if patience_counter >= self.cfg.early_stopping_patience:
                    logger.info("Early stopping at epoch %d.", epoch)
                    break

        return dict(history)

    def load_best_model(self) -> XD2Net:
        """Load the checkpoint with the best validation loss."""
        if self._best_checkpoint is None:
            return self.model
        self.model.load_state_dict(
            torch.load(self._best_checkpoint, map_location=self.device)
        )
        return self.model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in loader:
            raw_input, raw_input_static, input_exp, labels, lengths = self._to_device(batch)

            self.optimizer.zero_grad()
            thetas, act_output, _ = self.model(raw_input, raw_input_static, input_exp, lengths)

            bce = F.binary_cross_entropy(act_output, labels)
            # L2 regularization on encoder params (matches original l2(0.0001))
            l2 = sum(p.pow(2).sum() for p in self.model.encoder.parameters())
            loss = bce + self.cfg.l2_reg * l2

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / max(len(loader), 1)

    def _val_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        all_labels: List[Tensor] = []
        all_probs: List[Tensor] = []

        with torch.no_grad():
            for batch in loader:
                raw_input, raw_input_static, input_exp, labels, lengths = self._to_device(batch)
                thetas, act_output, _ = self.model(
                    raw_input, raw_input_static, input_exp, lengths
                )
                bce = F.binary_cross_entropy(act_output, labels)
                total_loss += bce.item()
                all_labels.append(labels.cpu())
                all_probs.append(act_output.cpu())

        y_true = torch.cat(all_labels).numpy()
        y_prob = torch.cat(all_probs).numpy()

        val_loss = total_loss / max(len(loader), 1)
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
        self._best_checkpoint = fname
        logger.debug("Checkpoint saved: %s", fname)

    def _to_device(self, batch: Tuple) -> Tuple:
        return tuple(t.to(self.device) for t in batch)
