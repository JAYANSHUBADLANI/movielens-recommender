import pandas as pd

from src.split import temporal_split


def _make_user_history(user_id, n, start_ts=0):
    return pd.DataFrame({
        "user_id": [user_id] * n,
        "movie_id": list(range(n)),
        "rating": [5] * n,
        "timestamp": [start_ts + i for i in range(n)],
    })


def test_low_activity_user_stays_train_only():
    df = _make_user_history(1, n=10)  # below MIN_POSITIVES_FOR_SPLIT (15)
    train, val, test = temporal_split(df)
    assert len(train) == 10
    assert len(val) == 0
    assert len(test) == 0


def test_high_activity_user_gets_last_5_as_test_and_prior_5_as_val():
    df = _make_user_history(1, n=20)
    train, val, test = temporal_split(df)
    assert len(test) == 5
    assert len(val) == 5
    assert len(train) == 10
    # test items must be the chronologically latest ones.
    assert set(test["movie_id"]) == set(range(15, 20))
    assert set(val["movie_id"]) == set(range(10, 15))
    assert set(train["movie_id"]) == set(range(0, 10))


def test_no_temporal_leakage_across_users():
    df = pd.concat([
        _make_user_history(1, n=20, start_ts=0),
        _make_user_history(2, n=5, start_ts=1000),
    ], ignore_index=True)
    train, val, test = temporal_split(df)
    # user 2 is below threshold, entirely in train.
    assert set(train[train["user_id"] == 2]["movie_id"]) == set(range(5))
    assert len(test[test["user_id"] == 2]) == 0
    # user 1's test set is still exactly their last 5 by timestamp.
    assert set(test[test["user_id"] == 1]["movie_id"]) == set(range(15, 20))


def test_exact_threshold_boundary():
    df = _make_user_history(1, n=15)  # exactly MIN_POSITIVES_FOR_SPLIT
    train, val, test = temporal_split(df)
    assert len(test) == 5
    assert len(val) == 5
    assert len(train) == 5
