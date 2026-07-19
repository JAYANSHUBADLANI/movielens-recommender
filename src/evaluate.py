"""Ranking metrics, hyperparameter tuning grid, and final test-set evaluation
with bootstrap significance testing.

Implicit-feedback top-N recommendation: models rank the full catalog per user
and are judged on their top TOP_N. Metrics:
  - Recall@N: fraction of a user's held-out positives that appear in their top-N.
  - NDCG@N: rank-aware version of the same (a hit at rank 1 counts more than rank 10).
  - MAP@N: mean average precision over the ranked top-N.
  - Coverage@N: the share of the whole catalog that appears in at least one
    user's top-N (an unpersonalised model that always recommends the same
    handful of blockbusters scores low here even with decent accuracy).
  - Mean pop. rank: average popularity rank of recommended items (1 = most
    popular); a model reaching deep into the tail has a high number here.
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.config import (
    ALS_ALPHA, ALS_FACTORS, ALS_ITERATIONS, ALS_REG, BOOTSTRAP_RESAMPLES,
    FIGURES_DIR, ITEM_KNN_TOP_K, PROCESSED_DIR, RANDOM_SEED, REPORTS_DIR, TOP_N,
)
from src.models.als import ImplicitALS
from src.models.item_knn import fit_item_knn, score_users as item_knn_score_users
from src.models.popularity import fit_popularity


def build_id_maps(*dfs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    user_ids = np.sort(pd.concat([d["user_id"] for d in dfs]).unique())
    item_ids = np.sort(pd.concat([d["movie_id"] for d in dfs]).unique())
    uidx = {u: i for i, u in enumerate(user_ids)}
    iidx = {m: i for i, m in enumerate(item_ids)}
    return user_ids, item_ids, uidx, iidx


def to_sparse(df: pd.DataFrame, uidx: dict, iidx: dict, n_users: int, n_items: int,
              value_col: str | None = None) -> sp.csr_matrix:
    rows = df["user_id"].map(uidx).to_numpy()
    cols = df["movie_id"].map(iidx).to_numpy()
    data = df[value_col].to_numpy(dtype=float) if value_col else np.ones(len(df))
    return sp.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))


def top_n_excluding(scores: np.ndarray, seen: set[int], n: int = TOP_N) -> list[int]:
    order = np.argsort(-scores, kind="stable")
    out = []
    for i in order:
        if int(i) not in seen:
            out.append(int(i))
            if len(out) == n:
                break
    return out


def evaluate_user(recs: list[int], relevant: set[int], n: int = TOP_N) -> tuple[float, float, float]:
    if not relevant:
        return 0.0, 0.0, 0.0
    hits = [1 if r in relevant else 0 for r in recs[:n]]
    recall = sum(hits) / len(relevant)

    dcg = sum(h / np.log2(rank + 2) for rank, h in enumerate(hits))
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(min(len(relevant), n)))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    hit_count, ap = 0, 0.0
    for rank, h in enumerate(hits):
        if h:
            hit_count += 1
            ap += hit_count / (rank + 1)
    ap = ap / min(len(relevant), n)
    return recall, ndcg, ap


def rank_all_users(
    scores_fn, eval_users: np.ndarray, seen_by_user: dict[int, set[int]],
    relevant_by_user: dict[int, set[int]], n_items: int, n: int = TOP_N,
) -> dict:
    recalls, ndcgs, maps, all_recs = [], [], [], {}
    for u in eval_users:
        scores = scores_fn(u)
        seen = seen_by_user.get(u, set())
        relevant = relevant_by_user.get(u, set())
        recs = top_n_excluding(scores, seen, n)
        r, nd, ap = evaluate_user(recs, relevant, n)
        recalls.append(r)
        ndcgs.append(nd)
        maps.append(ap)
        all_recs[u] = recs

    covered = {i for recs in all_recs.values() for i in recs}
    coverage = len(covered) / n_items
    return {
        "recall": np.array(recalls), "ndcg": np.array(ndcgs), "map": np.array(maps),
        "coverage": coverage, "recs": all_recs,
    }


def mean_pop_rank(all_recs: dict, pop_rank: np.ndarray) -> float:
    ranks = [pop_rank[i] for recs in all_recs.values() for i in recs]
    return float(np.mean(ranks)) if ranks else float("nan")


def _load_data():
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    val = pd.read_csv(PROCESSED_DIR / "val.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    return train, val, test


def _seen_and_relevant(history_df: pd.DataFrame, target_df: pd.DataFrame, uidx: dict, iidx: dict):
    seen_by_user, relevant_by_user = {}, {}
    for u, g in history_df.groupby("user_id"):
        seen_by_user[uidx[u]] = {iidx[m] for m in g["movie_id"] if m in iidx}
    for u, g in target_df.groupby("user_id"):
        relevant_by_user[uidx[u]] = {iidx[m] for m in g["movie_id"] if m in iidx}
    return seen_by_user, relevant_by_user


def _pop_rank_from(pop_scores: np.ndarray) -> np.ndarray:
    """rank 1 = most popular."""
    order = np.argsort(-pop_scores, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(1, len(order) + 1)
    return rank


def run_tune() -> None:
    train, val, _ = _load_data()
    user_ids, item_ids, uidx, iidx = build_id_maps(train, val)
    n_users, n_items = len(user_ids), len(item_ids)

    train_mat = to_sparse(train, uidx, iidx, n_users, n_items, value_col="rating")
    eval_users = np.array(sorted({uidx[u] for u in val["user_id"].unique()}))
    seen_by_user, relevant_by_user = _seen_and_relevant(train, val, uidx, iidx)

    base = {"factors": ALS_FACTORS, "reg": ALS_REG, "alpha": ALS_ALPHA}
    grids = {
        "factors": [16, 32, 64],
        "reg": [0.01, ALS_REG, 1.0],
        "alpha": [ALS_ALPHA, 5, 40],
    }

    lines = ["# Hyperparameter sensitivity (validation slice)\n"]
    lines.append(f"Base config: factors={base['factors']}, reg={base['reg']}, "
                 f"alpha={base['alpha']}, iterations={ALS_ITERATIONS}\n")

    results = {}
    for param, values in grids.items():
        row_scores = []
        for v in values:
            cfg = dict(base)
            cfg[param] = v
            model = ImplicitALS(
                factors=cfg["factors"], reg=cfg["reg"], alpha=cfg["alpha"],
                iterations=ALS_ITERATIONS, random_state=RANDOM_SEED,
            ).fit(train_mat)
            metrics = rank_all_users(
                lambda u: model.score(u), eval_users, seen_by_user, relevant_by_user, n_items,
            )
            recall = metrics["recall"].mean()
            row_scores.append(recall)
            print(f"{param}={v}: val Recall@{TOP_N} = {recall:.4f}")
        results[param] = dict(zip(values, row_scores))
        lines.append(f"\n## {param}\n")
        lines.append("| " + " | ".join(str(v) for v in values) + " |")
        lines.append("|" + "---|" * len(values))
        lines.append("| " + " | ".join(f"{s:.4f}" for s in row_scores) + " |")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "hyperparam_sensitivity.md").write_text("\n".join(lines) + "\n")
    with open(REPORTS_DIR / "hyperparam_sensitivity.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {REPORTS_DIR / 'hyperparam_sensitivity.md'}")


def run_test() -> None:
    train, val, test = _load_data()
    train_val = pd.concat([train, val], ignore_index=True)

    user_ids, item_ids, uidx, iidx = build_id_maps(train, val, test)
    n_users, n_items = len(user_ids), len(item_ids)

    train_val_mat_binary = to_sparse(train_val, uidx, iidx, n_users, n_items)
    train_val_mat_ratings = to_sparse(train_val, uidx, iidx, n_users, n_items, value_col="rating")

    eval_users = np.array(sorted({uidx[u] for u in test["user_id"].unique()}))
    seen_by_user, relevant_by_user = _seen_and_relevant(train_val, test, uidx, iidx)

    pop_scores = fit_popularity(train_val_mat_binary)
    pop_rank = _pop_rank_from(pop_scores)
    pop_order = np.argsort(-pop_scores, kind="stable")

    def pop_score_fn(u):
        return pop_scores

    pop_metrics = rank_all_users(pop_score_fn, eval_users, seen_by_user, relevant_by_user, n_items)

    sim = fit_item_knn(train_val_mat_binary, top_k=ITEM_KNN_TOP_K)
    knn_scores_all = item_knn_score_users(train_val_mat_binary, sim)

    def knn_score_fn(u):
        return knn_scores_all[u]

    knn_metrics = rank_all_users(knn_score_fn, eval_users, seen_by_user, relevant_by_user, n_items)

    als_model = ImplicitALS(
        factors=ALS_FACTORS, reg=ALS_REG, alpha=ALS_ALPHA,
        iterations=ALS_ITERATIONS, random_state=RANDOM_SEED,
    ).fit(train_val_mat_ratings)

    def als_score_fn(u):
        return als_model.score(u)

    als_metrics = rank_all_users(als_score_fn, eval_users, seen_by_user, relevant_by_user, n_items)

    rows = []
    for name, m in [("popularity", pop_metrics), ("item-knn", knn_metrics), ("als", als_metrics)]:
        rows.append({
            "model": name,
            "recall_at_10": round(m["recall"].mean(), 4),
            "ndcg_at_10": round(m["ndcg"].mean(), 4),
            "map_at_10": round(m["map"].mean(), 4),
            "coverage_at_10": round(m["coverage"], 4),
            "mean_pop_rank": round(mean_pop_rank(m["recs"], pop_rank), 1),
        })
    results_df = pd.DataFrame(rows)
    print(results_df.to_string(index=False))

    rng = np.random.default_rng(RANDOM_SEED)
    diffs = []
    n_eval = len(eval_users)
    for _ in range(BOOTSTRAP_RESAMPLES):
        sample_idx = rng.integers(0, n_eval, size=n_eval)
        diffs.append(als_metrics["recall"][sample_idx].mean() - knn_metrics["recall"][sample_idx].mean())
    diffs = np.array(diffs)
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    point_diff = als_metrics["recall"].mean() - knn_metrics["recall"].mean()
    print(f"ALS vs item-knn Recall@{TOP_N} diff: {point_diff:+.4f} [{ci_lo:+.4f}, {ci_hi:+.4f}]")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(REPORTS_DIR / "results.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    metrics_to_plot = ["recall_at_10", "ndcg_at_10", "map_at_10"]
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    for i, (_, row) in enumerate(results_df.iterrows()):
        ax.bar(x + i * width, [row[m] for m in metrics_to_plot], width, label=row["model"])
    ax.set_xticks(x + width)
    ax.set_xticklabels(["Recall@10", "NDCG@10", "MAP@10"])
    ax.set_title("Model comparison on the held-out test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)

    lines = ["# Results\n"]
    lines.append(
        f"Test protocol: each user's last {5} positives; models trained on "
        f"train+validation; {n_eval} evaluated users, {n_items}-item catalog. "
        f"ALS config: factors={ALS_FACTORS}, reg={ALS_REG}, alpha={ALS_ALPHA}, "
        f"iterations={ALS_ITERATIONS} (selected on the validation slice).\n"
    )
    lines.append("| Model | Recall@10 | NDCG@10 | MAP@10 | Coverage@10 | Mean pop. rank |")
    lines.append("|---|---|---|---|---|---|")
    for _, row in results_df.iterrows():
        lines.append(
            f"| {row['model']} | {row['recall_at_10']:.4f} | {row['ndcg_at_10']:.4f} | "
            f"{row['map_at_10']:.4f} | {row['coverage_at_10']*100:.1f}% | {row['mean_pop_rank']:.0f} |"
        )
    lines.append("")
    lines.append(
        f"ALS vs item-knn, Recall@10 difference: {point_diff:+.4f} "
        f"[{ci_lo:+.4f}, {ci_hi:+.4f}] ({BOOTSTRAP_RESAMPLES} bootstrap resamples over users)."
    )
    (REPORTS_DIR / "results.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {REPORTS_DIR / 'results.md'}")

    from src.config import ARTIFACTS_DIR
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(ARTIFACTS_DIR / "mappings.npz", user_ids=user_ids, item_ids=item_ids)
    np.savez(ARTIFACTS_DIR / "als.npz", X=als_model.X, Y=als_model.Y)
    sp.save_npz(ARTIFACTS_DIR / "item_sims.npz", sim)
    np.save(ARTIFACTS_DIR / "popularity.npy", pop_scores)
    print(f"Wrote model artifacts to {ARTIFACTS_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["tune", "test"])
    args = parser.parse_args()
    if args.mode == "tune":
        run_tune()
    else:
        run_test()


if __name__ == "__main__":
    main()
