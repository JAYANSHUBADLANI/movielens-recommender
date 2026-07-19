"""Item-item collaborative filtering.

Items are binary vectors over users; cosine similarity between those vectors,
truncated to the top-K neighbours per item, gives a sparse similarity matrix S.
A user's score for item j is the sum of similarities between j and their
history, which makes every recommendation decomposable into the neighbours
that drove it.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize


def fit_item_knn(interaction_matrix: sp.csr_matrix, top_k: int = 100) -> sp.csr_matrix:
    """interaction_matrix: sparse (n_users x n_items) binary matrix.
    Returns a sparse (n_items x n_items) similarity matrix, top_k neighbours per item.
    """
    item_vectors = normalize(interaction_matrix.T.tocsr(), norm="l2", axis=1)
    sim = item_vectors @ item_vectors.T
    sim = sim.tolil()
    sim.setdiag(0)  # an item is not its own neighbour
    sim = sim.tocsr()

    n_items = sim.shape[0]
    rows, cols, data = [], [], []
    for i in range(n_items):
        start, end = sim.indptr[i], sim.indptr[i + 1]
        row_data = sim.data[start:end]
        row_idx = sim.indices[start:end]
        if len(row_data) > top_k:
            keep = np.argpartition(-row_data, top_k)[:top_k]
            row_data, row_idx = row_data[keep], row_idx[keep]
        rows.extend([i] * len(row_idx))
        cols.extend(row_idx.tolist())
        data.extend(row_data.tolist())

    return sp.csr_matrix((data, (rows, cols)), shape=(n_items, n_items))


def score_users(interaction_matrix: sp.csr_matrix, sim: sp.csr_matrix) -> np.ndarray:
    """Score every (user, item) pair as the sum of similarities to the user's history."""
    return np.asarray((interaction_matrix @ sim).todense())
