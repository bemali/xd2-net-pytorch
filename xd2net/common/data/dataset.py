"""PyTorch Dataset and DataLoader helpers for XD2-Net."""

from __future__ import annotations

from typing import Iterator, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, SequentialSampler, random_split

# Pinned memory only helps (and only avoids a warning) when a GPU is present.
_PIN_MEMORY = torch.cuda.is_available()


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


class TensorBatchLoader:
    """Minimal DataLoader replacement over tensors that already live on the target device.

    Yields ``(raw_input, raw_input_static, input_exp, label, prefix_length)`` batches by
    index slicing, with no per-item ``__getitem__`` / collate and no host→device copies.
    ``shuffle`` uses a seeded generator, so the order is reproducible per seed (it differs
    from DataLoader's order). ``sampler`` mirrors DataLoader's attribute so callers can
    check for sequential order.
    """

    def __init__(
        self,
        tensors: Sequence[Tensor],
        batch_size: int = 256,
        shuffle: bool = False,
        seed: int = 42,
    ):
        self.tensors = tuple(tensors)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = len(self.tensors[0])
        self._generator = torch.Generator().manual_seed(seed)
        self.sampler = None if shuffle else SequentialSampler(range(self.n))

    def __len__(self) -> int:
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Tuple[Tensor, ...]]:
        device = self.tensors[0].device
        if self.shuffle:
            order = torch.randperm(self.n, generator=self._generator).to(device)
        else:
            order = torch.arange(self.n, device=device)
        for start in range(0, self.n, self.batch_size):
            idx = order[start:start + self.batch_size]
            yield tuple(t[idx] for t in self.tensors)


Loader = Union[DataLoader, TensorBatchLoader]


def _device_tensors(ds: EventLogDataset, device: torch.device, indices=None) -> Tuple[Tensor, ...]:
    tensors = (ds.raw_input, ds.raw_input_static, ds.input_exp, ds.labels, ds.lengths)
    if indices is not None:
        idx = torch.as_tensor(list(indices), dtype=torch.long)
        tensors = tuple(t[idx] for t in tensors)
    return tuple(t.to(device) for t in tensors)


def _on_accelerator(device) -> Optional[torch.device]:
    if device is None:
        return None
    device = torch.device(device)
    return device if device.type in ("cuda", "mps") else None


def make_dataloaders(
    inputs_train: dict,
    outputs_train: dict,
    lengths_train: np.ndarray,
    val_split: float = 0.15,
    batch_size: int = 256,
    num_workers: int = 0,
    seed: int = 42,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[Loader, Loader]:
    """Build train and validation loaders from a single bucket's arrays.

    The validation split is applied at the case level (random), matching the
    original Keras validation_split=0.15 behaviour.

    ``device``: when a CUDA/MPS device is given, the tensors are moved there once and
    ``TensorBatchLoader``s are returned (same split, no per-batch copies). Otherwise
    (None / CPU) regular DataLoaders are returned.

    Returns:
        (train_loader, val_loader)
    """
    full_ds = EventLogDataset(inputs_train, outputs_train, lengths_train)
    n_val = int(round(len(full_ds) * val_split))
    n_train = len(full_ds) - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=generator)

    accel = _on_accelerator(device)
    if accel is not None:
        return (
            TensorBatchLoader(_device_tensors(full_ds, accel, train_ds.indices), batch_size, shuffle=True, seed=seed),
            TensorBatchLoader(_device_tensors(full_ds, accel, val_ds.indices), batch_size, shuffle=False),
        )

    # Seeded generator so the shuffle order is reproducible for a given seed.
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=_PIN_MEMORY,
        drop_last=False,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=_PIN_MEMORY,
    )
    return train_loader, val_loader


def make_test_loader(
    inputs_test: dict,
    outputs_test: dict,
    lengths_test: np.ndarray,
    batch_size: int = 256,
    num_workers: int = 0,
    device: Optional[Union[str, torch.device]] = None,
) -> Loader:
    """Build a test-only loader (no shuffling); see ``make_dataloaders`` for ``device``."""
    ds = EventLogDataset(inputs_test, outputs_test, lengths_test)
    accel = _on_accelerator(device)
    if accel is not None:
        return TensorBatchLoader(_device_tensors(ds, accel), batch_size, shuffle=False)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=_PIN_MEMORY,
    )
