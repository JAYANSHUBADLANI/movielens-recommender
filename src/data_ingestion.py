"""Download MovieLens 1M (with a schema-identical synthetic fallback) and
write flat ratings/movies tables to data/raw/.
"""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import requests

from src.config import ML1M_URL, RAW_DIR, RANDOM_SEED, SYNTHETIC_MARKER

RATINGS_PATH = RAW_DIR / "ratings.csv"
MOVIES_PATH = RAW_DIR / "movies.csv"


def is_synthetic() -> bool:
    return SYNTHETIC_MARKER.exists()


def _parse_ml1m_zip(raw_bytes: bytes) -> tuple[pd.DataFrame, pd.DataFrame]:
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
        with zf.open("ml-1m/ratings.dat") as f:
            ratings = pd.read_csv(
                f, sep="::", engine="python", encoding="latin-1",
                names=["user_id", "movie_id", "rating", "timestamp"],
            )
        with zf.open("ml-1m/movies.dat") as f:
            movies = pd.read_csv(
                f, sep="::", engine="python", encoding="latin-1",
                names=["movie_id", "title", "genres"],
            )
    return ratings, movies


def _download_real() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    try:
        resp = requests.get(ML1M_URL, timeout=30)
        resp.raise_for_status()
        return _parse_ml1m_zip(resp.content)
    except Exception as exc:  # network blocked, host down, etc.
        print(f"Real MovieLens download failed ({exc}); falling back to synthetic data.")
        return None


def _generate_synthetic(
    n_users: int = 2000, n_movies: int = 1200, n_factors: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Schema-identical synthetic fallback: users and movies get random latent
    vectors in a low-dimensional space, and implicit ratings are generated from
    their dot product plus noise, so a factorization model has real signal to
    recover (not pure noise) while staying obviously synthetic.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    user_factors = rng.normal(size=(n_users, n_factors))
    item_factors = rng.normal(size=(n_movies, n_factors))
    item_popularity = rng.exponential(scale=1.0, size=n_movies)  # long-tail popularity

    genre_pool = [
        "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
        "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
        "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
    ]
    movie_ids = np.arange(1, n_movies + 1)
    movies = pd.DataFrame({
        "movie_id": movie_ids,
        "title": [f"Synthetic Movie {i} ({1970 + i % 55})" for i in movie_ids],
        "genres": [
            "|".join(rng.choice(genre_pool, size=rng.integers(1, 4), replace=False))
            for _ in movie_ids
        ],
    })

    # Each user rates a random subset of movies, biased toward popular ones and
    # toward movies their latent vector aligns with.
    rows = []
    base_ts = 946684800  # 2000-01-01, arbitrary anchor
    for u in range(n_users):
        n_rate = rng.integers(15, 120)
        probs = item_popularity / item_popularity.sum()
        rated = rng.choice(n_movies, size=min(n_rate, n_movies), replace=False, p=probs)
        affinity = item_factors[rated] @ user_factors[u]
        noise = rng.normal(scale=1.2, size=len(rated))
        score = affinity + noise
        # Map score to a 1-5 rating via quantile bucketing, so ~30-40% of a
        # user's history ends up a "positive" (>=4) implicit interaction.
        score_range = score.max() - score.min()
        ratings_1_5 = np.clip(np.round((score - score.min()) / (score_range + 1e-9) * 4) + 1, 1, 5)
        # Timestamps increasing with an index that roughly follows rating order,
        # so the temporal split has something meaningful to split on.
        order = np.argsort(rng.normal(size=len(rated)))
        timestamps = base_ts + np.arange(len(rated))[order] * 86400 + u * 37
        for m_idx, r, ts in zip(rated, ratings_1_5, timestamps):
            rows.append((u + 1, int(movie_ids[m_idx]), int(r), int(ts)))

    ratings = pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "timestamp"])
    ratings = ratings.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    return ratings, movies


def run() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    result = _download_real()

    if result is not None:
        ratings, movies = result
        if SYNTHETIC_MARKER.exists():
            SYNTHETIC_MARKER.unlink()
        print("Using real MovieLens 1M data.")
    else:
        ratings, movies = _generate_synthetic()
        SYNTHETIC_MARKER.write_text(
            "This dataset is synthetic. See src/data_ingestion.py for how it was generated.\n"
        )
        print("Using synthetic fallback data.")

    ratings.to_csv(RATINGS_PATH, index=False)
    movies.to_csv(MOVIES_PATH, index=False)
    print(f"Wrote {len(ratings):,} ratings and {len(movies):,} movies to {RAW_DIR}")


if __name__ == "__main__":
    run()
