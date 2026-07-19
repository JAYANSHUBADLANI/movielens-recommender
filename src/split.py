"""Leakage-safe temporal per-user split.

For each user with at least MIN_POSITIVES_FOR_SPLIT positive interactions
(rating >= POSITIVE_THRESHOLD), the chronologically last N_TEST_PER_USER
positives become the test set and the N_VAL_PER_USER before those become the
validation set; everything earlier is train. Low-activity users stay
train-only. A random split would let a user's future interactions leak into
training while asking the model to predict their past; splitting on each
user's own timeline avoids that.
"""
from __future__ import annotations

import pandas as pd

from src.config import (
    MIN_POSITIVES_FOR_SPLIT, N_TEST_PER_USER, N_VAL_PER_USER,
    POSITIVE_THRESHOLD, PROCESSED_DIR, RAW_DIR,
)
from src.data_ingestion import MOVIES_PATH, RATINGS_PATH


def temporal_split(positives: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_parts, val_parts, test_parts = [], [], []

    for _, group in positives.groupby("user_id", sort=False):
        group = group.sort_values("timestamp")
        n = len(group)
        if n >= MIN_POSITIVES_FOR_SPLIT:
            test_parts.append(group.iloc[n - N_TEST_PER_USER:])
            val_parts.append(group.iloc[n - N_TEST_PER_USER - N_VAL_PER_USER: n - N_TEST_PER_USER])
            train_parts.append(group.iloc[: n - N_TEST_PER_USER - N_VAL_PER_USER])
        else:
            train_parts.append(group)

    train = pd.concat(train_parts, ignore_index=True) if train_parts else positives.iloc[0:0]
    val = pd.concat(val_parts, ignore_index=True) if val_parts else positives.iloc[0:0]
    test = pd.concat(test_parts, ignore_index=True) if test_parts else positives.iloc[0:0]
    return train, val, test


def run() -> None:
    if not RATINGS_PATH.exists():
        raise FileNotFoundError(f"{RATINGS_PATH} not found, run `python -m src.data_ingestion` first")

    ratings = pd.read_csv(RATINGS_PATH)
    movies = pd.read_csv(MOVIES_PATH)

    positives = ratings[ratings["rating"] >= POSITIVE_THRESHOLD].copy()
    train, val, test = temporal_split(positives)

    n_eval_users = test["user_id"].nunique()
    n_train_only = positives["user_id"].nunique() - n_eval_users
    print(f"Users with >= {MIN_POSITIVES_FOR_SPLIT} positives (get val/test slices): {n_eval_users:,}")
    print(f"Train-only users (below the threshold): {n_train_only:,}")
    print(f"Rows: train={len(train):,} val={len(val):,} test={len(test):,}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)
    movies.to_csv(PROCESSED_DIR / "movies.csv", index=False)
    print(f"Wrote train/val/test/movies to {PROCESSED_DIR}")


if __name__ == "__main__":
    run()
