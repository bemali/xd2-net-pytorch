"""Dynamic input for the hierarchical XD2-Net: packed per-field tokens.

``build_dynamic_tokens`` packs, per event, the shifted category index of each
categorical field (0 = PAD) followed by the numeric values, giving a float
tensor ``(N, T, n_cat + n_num)`` laid out as described by ``EventSchema`` and
consumed by ``FieldTokenizer``.
"""

from __future__ import annotations

import numpy as np

from xd2net.common.data.features import EventSchema, valid_mask

_FLOAT_EXACT = 2 ** 24


def build_dynamic_tokens(vec: dict, schema: EventSchema, max_len: int) -> np.ndarray:
    """Return packed float32 ``(N, max_len, n_cat + n_num)``."""
    valid = valid_mask(vec["lengths"], max_len)
    parts = []
    for feat, card in zip(schema.cat_fields, schema.cardinalities):
        idx = vec["prefixes"][feat][:, :max_len].astype(np.int64) + 1
        idx = np.where(valid, idx, 0)
        if idx.max(initial=0) > card or idx.max(initial=0) >= _FLOAT_EXACT:
            raise ValueError(f"{feat}: index out of range for cardinality {card}.")
        if (idx[valid] == 0).any():
            raise ValueError(f"{feat}: PAD index found at a real event position.")
        parts.append(idx.astype(np.float32))
    for feat in schema.num_fields:
        num = vec["prefixes"][feat][:, :max_len].astype(np.float32)
        parts.append(np.where(valid, num, 0.0).astype(np.float32))
    return np.stack(parts, axis=-1)
