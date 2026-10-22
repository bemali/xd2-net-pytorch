"""Dynamic input for the baseline XD2-Net: one-hot categories + numeric values.

``build_dynamic_onehot`` returns ``(N, T, D_dynamic)`` where each categorical
field contributes ``cardinality`` one-hot columns (index 0 = ``"none"``) and
each numeric field one column. Padded time steps are all-zero rows (the
original encoding set the category-0 bit there).
"""

from __future__ import annotations

from typing import List

import numpy as np

from xd2net.common.data.features import EventSchema, _to_categorical, valid_mask


def build_dynamic_onehot(vec: dict, schema: EventSchema, max_len: int) -> np.ndarray:
    """Return float32 ``(N, max_len, D_dynamic)``; padded steps are zero."""
    valid = valid_mask(vec["lengths"], max_len)[..., None]          # (N, T, 1)
    parts = []
    for feat, card in zip(schema.cat_fields, schema.cardinalities):
        idx = vec["prefixes"][feat][:, :max_len].astype(np.int64)
        if idx.max(initial=0) >= card:
            raise ValueError(f"{feat}: index out of range for cardinality {card}.")
        parts.append(_to_categorical(idx, card))
    for feat in schema.num_fields:
        parts.append(vec["prefixes"][feat][:, :max_len, None].astype(np.float32))
    return np.where(valid, np.concatenate(parts, axis=-1), 0.0).astype(np.float32)


def dynamic_feature_names(schema: EventSchema) -> List[str]:
    """Column names of ``build_dynamic_onehot`` (``FIELD=label`` / ``FIELD``)."""
    names: List[str] = []
    for feat, card in zip(schema.cat_fields, schema.cardinalities):
        names.extend(f"{feat}={schema.index_to_label[feat][k + 1]}" for k in range(card))
    names.extend(schema.num_fields)
    return names
