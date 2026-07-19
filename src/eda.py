"""EDA figures: rating distribution and the catalog's long tail."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, POSITIVE_THRESHOLD, RAW_DIR


def run() -> None:
    ratings = pd.read_csv(RAW_DIR / "ratings.csv")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    counts = ratings["rating"].value_counts().sort_index()
    colors = ["tomato" if r < POSITIVE_THRESHOLD else "steelblue" for r in counts.index]
    ax.bar(counts.index.astype(str), counts.values, color=colors)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    ax.set_title("Rating distribution (blue: treated as positive)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "rating_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    positives = ratings[ratings["rating"] >= POSITIVE_THRESHOLD]
    item_counts = positives.groupby("movie_id").size().sort_values(ascending=False).to_numpy()
    ax.plot(np.arange(1, len(item_counts) + 1), item_counts)
    ax.set_xlabel("Item rank (by popularity)")
    ax.set_ylabel("Positive interaction count")
    ax.set_title("Long tail of item popularity")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "long_tail.png", dpi=150)
    plt.close(fig)

    n_users = ratings["user_id"].nunique()
    n_items = ratings["movie_id"].nunique()
    n_positive = len(positives)
    print(f"Users: {n_users:,}  Items: {n_items:,}  Positive interactions: {n_positive:,}")
    print(f"Density: {n_positive / (n_users * n_items) * 100:.3f}%")


if __name__ == "__main__":
    run()
