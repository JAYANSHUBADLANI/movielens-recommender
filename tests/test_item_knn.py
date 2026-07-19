import numpy as np
import scipy.sparse as sp

from src.models.item_knn import fit_item_knn, score_users


def test_identical_items_get_similarity_one():
    # users 0,1,2 all interact with items 0 and 1 identically; item 2 is different.
    dense = np.array([
        [1, 1, 0],
        [1, 1, 0],
        [1, 1, 1],
        [0, 0, 1],
    ])
    mat = sp.csr_matrix(dense)
    sim = fit_item_knn(mat, top_k=5)
    assert sim.shape == (3, 3)
    # item 0 and item 1 are rated by exactly the same users -> cosine similarity 1.
    assert np.isclose(sim[0, 1], 1.0, atol=1e-6)
    assert np.isclose(sim[1, 0], 1.0, atol=1e-6)
    # self-similarity is explicitly zeroed out.
    assert sim[0, 0] == 0.0


def test_top_k_truncation_keeps_only_strongest_neighbours():
    rng = np.random.default_rng(0)
    dense = (rng.random((50, 30)) > 0.7).astype(float)
    mat = sp.csr_matrix(dense)
    sim_full = fit_item_knn(mat, top_k=30)  # effectively no truncation
    sim_trunc = fit_item_knn(mat, top_k=3)
    # every row of the truncated matrix should have at most 3 nonzero entries.
    nnz_per_row = np.diff(sim_trunc.tocsr().indptr)
    assert nnz_per_row.max() <= 3
    assert sim_trunc.nnz <= sim_full.nnz


def test_score_users_sums_similarities_over_history():
    dense = np.array([
        [1, 1, 0],
        [1, 1, 0],
        [1, 1, 1],
        [0, 0, 1],
    ])
    mat = sp.csr_matrix(dense)
    sim = fit_item_knn(mat, top_k=5)
    scores = score_users(mat, sim)
    assert scores.shape == (4, 3)
    # a user who has interacted with item 0 should score item 1 highly
    # (since item 0 and item 1 are near-identical neighbours here).
    assert scores[0, 1] > 0
