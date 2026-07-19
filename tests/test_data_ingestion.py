from src.data_ingestion import _generate_synthetic


def test_synthetic_data_has_expected_schema():
    ratings, movies = _generate_synthetic(n_users=50, n_movies=40, n_factors=4)
    assert list(ratings.columns) == ["user_id", "movie_id", "rating", "timestamp"]
    assert list(movies.columns) == ["movie_id", "title", "genres"]
    assert ratings["rating"].between(1, 5).all()
    assert ratings["user_id"].between(1, 50).all()
    assert ratings["movie_id"].isin(movies["movie_id"]).all()


def test_synthetic_data_is_deterministic_given_same_seed():
    r1, _ = _generate_synthetic(n_users=30, n_movies=20, n_factors=3)
    r2, _ = _generate_synthetic(n_users=30, n_movies=20, n_factors=3)
    assert r1.equals(r2)


def test_synthetic_data_sorted_by_user_then_time():
    ratings, _ = _generate_synthetic(n_users=20, n_movies=15, n_factors=3)
    for _, g in ratings.groupby("user_id"):
        assert (g["timestamp"].diff().dropna() >= 0).all()
