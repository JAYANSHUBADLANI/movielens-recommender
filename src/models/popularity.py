"""Non-personalised baseline: rank items by raw interaction count in train+val."""
from __future__ import annotations

import numpy as np


def fit_popularity(interaction_matrix) -> np.ndarray:
    """interaction_matrix: sparse (n_users x n_items) binary matrix.
    Returns a length-n_items popularity score array (higher = more popular).
    """
    return np.asarray(interaction_matrix.sum(axis=0)).ravel()
