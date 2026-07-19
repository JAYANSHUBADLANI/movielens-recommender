import numpy as np

from src.evaluate import evaluate_user, mean_pop_rank, top_n_excluding


def test_top_n_excluding_skips_seen_items():
    scores = np.array([5, 4, 3, 2, 1])
    recs = top_n_excluding(scores, seen={0, 2}, n=2)
    assert recs == [1, 3]


def test_top_n_excluding_returns_fewer_if_catalog_exhausted():
    scores = np.array([3, 2, 1])
    recs = top_n_excluding(scores, seen={0, 1, 2}, n=5)
    assert recs == []


def test_evaluate_user_hand_computed():
    recs = [5, 3, 8, 1, 9, 2, 7, 4, 6, 0]
    relevant = {3, 9, 4}
    recall, ndcg, ap = evaluate_user(recs, relevant, n=10)

    assert recall == 1.0  # all 3 relevant items appear somewhere in the top 10
    assert np.isclose(ndcg, 0.62569, atol=1e-4)
    assert np.isclose(ap, 0.425, atol=1e-4)


def test_evaluate_user_no_relevant_items_returns_zero():
    recall, ndcg, ap = evaluate_user([1, 2, 3], relevant=set(), n=10)
    assert (recall, ndcg, ap) == (0.0, 0.0, 0.0)


def test_evaluate_user_perfect_ranking_gets_ndcg_one():
    recs = [10, 11, 12, 0, 0]
    relevant = {10, 11, 12}
    recall, ndcg, ap = evaluate_user(recs, relevant, n=5)
    assert recall == 1.0
    assert np.isclose(ndcg, 1.0)
    assert np.isclose(ap, 1.0)


def test_mean_pop_rank_averages_over_all_recommended_items():
    pop_rank = np.array([1, 2, 3, 4, 5])  # rank 1 = most popular
    all_recs = {0: [0, 1], 1: [4]}
    # recommended item pop ranks: 1, 2, 5 -> mean = 8/3
    assert np.isclose(mean_pop_rank(all_recs, pop_rank), 8 / 3)


def test_mean_pop_rank_empty_recs_is_nan():
    assert np.isnan(mean_pop_rank({}, np.array([1, 2, 3])))
