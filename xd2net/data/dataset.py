"""PyTorch Dataset and DataLoader helpers for XD2-Net."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split


class EventLogDataset(Dataset):
    """Dataset for a single prefix-length bucket.

    Each item is a tuple:
        (raw_input, raw_input_static, input_exp, label, prefix_length)

    prefix_length is the actual (unpadded) sequence length, used by the model
    to construct the src_key_padding_mask for the Transformer encoder.
    """

    def __init__(self, inputs: dict, outputs: dict, lengths: np.ndarray):
        """
        Args:
            inputs:  dict with 'input_dynamic' (N,T,D), 'input_static' (N,S),
                     'input_exp' (N,E) — numpy float32 arrays.
            outputs: dict with 'act_output' (N,) — numpy int64 array.
            lengths: (N,) integer array of actual sequence lengths.
        """
        self.raw_input = torch.from_numpy(inputs["input_dynamic"]).float()
        self.raw_input_static = torch.from_numpy(inputs["input_static"]).float()
        self.input_exp = torch.from_numpy(inputs["input_exp"]).float()
        self.labels = torch.from_numpy(outputs["act_output"]).float()
        self.lengths = torch.from_numpy(lengths.astype(np.int64)).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        return (
            self.raw_input[idx],
            self.raw_input_static[idx],
            self.input_exp[idx],
            self.labels[idx],
            self.lengths[idx],
        )


def make_dataloaders(
    inputs_train: dict,
    outputs_train: dict,
    lengths_train: np.ndarray,
    val_split: float = 0.15,
    batch_size: int = 256,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders from a single bucket's arrays.

    The validation split is applied at the case level (random), matching the
    original Keras validation_split=0.15 behaviour.

    Returns:
        (train_loader, val_loader)
    """
    full_ds = EventLogDataset(inputs_train, outputs_train, lengths_train)
    n_val = int(round(len(full_ds) * val_split))
    n_train = len(full_ds) - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def make_test_loader(
    inputs_test: dict,
    outputs_test: dict,
    lengths_test: np.ndarray,
    batch_size: int = 256,
    num_workers: int = 0,
) -> DataLoader:
    """Build a test-only DataLoader (no shuffling)."""
    ds = EventLogDataset(inputs_test, outputs_test, lengths_test)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
