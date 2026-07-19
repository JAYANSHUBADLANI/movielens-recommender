import numpy as np
import scipy.sparse as sp

from src.models.popularity import fit_popularity


def test_popularity_counts_interactions_per_item():
    dense = np.array([
        [1, 0, 1],
        [1, 1, 1],
        [0, 0, 1],
    ])
    mat = sp.csr_matrix(dense)
    pop = fit_popularity(mat)
    assert list(pop) == [2, 1, 3]


def test_popularity_shape_matches_item_count():
    mat = sp.csr_matrix((10, 25))
    pop = fit_popularity(mat)
    assert pop.shape == (25,)
    assert (pop == 0).all()
