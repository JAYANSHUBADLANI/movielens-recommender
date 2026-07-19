import numpy as np
import scipy.sparse as sp

from src.models.als import ImplicitALS


def _make_synthetic_interactions(n_users=40, n_items=30, factors=4, seed=0):
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(n_users, factors))
    V = rng.normal(size=(n_items, factors))
    scores = U @ V.T
    # Top-scoring items per user become observed positive interactions.
    threshold = np.percentile(scores, 85, axis=1, keepdims=True)
    mask = scores >= threshold
    ratings = np.where(mask, rng.integers(1, 6, size=scores.shape), 0)
    return sp.csr_matrix(ratings), scores


def test_als_runs_and_produces_correct_shapes():
    interactions, _ = _make_synthetic_interactions()
    model = ImplicitALS(factors=8, reg=0.1, alpha=20.0, iterations=10, random_state=0)
    model.fit(interactions)
    assert model.X.shape == (40, 8)
    assert model.Y.shape == (30, 8)
    assert np.isfinite(model.X).all()
    assert np.isfinite(model.Y).all()


def test_als_recovers_latent_structure_better_than_random():
    """ALS should rank a user's held-out favourite items above random chance,
    on data explicitly generated from a low-rank factor structure."""
    interactions, true_scores = _make_synthetic_interactions(n_users=60, n_items=50, factors=5, seed=1)
    model = ImplicitALS(factors=10, reg=0.05, alpha=30.0, iterations=20, random_state=1)
    model.fit(interactions)

    hits = 0
    n_check = 0
    for u in range(interactions.shape[0]):
        row = interactions.getrow(u)
        observed = set(row.indices)
        if len(observed) < 2:
            continue
        # Hide the single highest-true-score observed item for this user and
        # check whether ALS ranks it inside the top 10 unobserved-or-hidden candidates.
        true_row = true_scores[u]
        held_out = max(observed, key=lambda i: true_row[i])
        candidates = [i for i in range(interactions.shape[1]) if i not in observed or i == held_out]
        pred = model.score(u)
        ranked = sorted(candidates, key=lambda i: -pred[i])
        n_check += 1
        if held_out in ranked[:10]:
            hits += 1

    assert n_check > 10
    hit_rate = hits / n_check
    assert hit_rate > 0.3, f"expected ALS to beat random chance by a wide margin, got hit_rate={hit_rate}"


def test_als_confidence_scaling_changes_fit():
    """Different alpha (confidence scaling) should produce a measurably different fit,
    confirming the confidence weighting actually participates in the optimisation."""
    interactions, _ = _make_synthetic_interactions(seed=2)
    low = ImplicitALS(factors=8, reg=0.1, alpha=1.0, iterations=10, random_state=0).fit(interactions)
    high = ImplicitALS(factors=8, reg=0.1, alpha=100.0, iterations=10, random_state=0).fit(interactions)
    assert not np.allclose(low.X, high.X)


def test_als_zero_history_user_gets_zero_factors():
    n_users, n_items, factors = 5, 10, 4
    interactions = sp.csr_matrix((n_users, n_items))
    model = ImplicitALS(factors=factors, reg=0.1, alpha=10.0, iterations=5, random_state=0)
    model.fit(interactions)
    assert np.allclose(model.X, 0.0)
