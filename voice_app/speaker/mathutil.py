"""Small numeric helpers for the speaker tracker, in plain numpy."""
from __future__ import annotations

import numpy as np


def unit_rows(values) -> np.ndarray:
    """Scale every row (or a single vector) to length 1."""
    array = np.asarray(values, dtype=np.float64)
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(norm, 1e-12)
