"""Streamlit demo: pick a user, compare their history with each model's top-10.

Run the pipeline first so data/processed/artifacts/ exists (see README), then:

    streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
import streamlit as st

from src.config import ARTIFACTS_DIR, PROCESSED_DIR, SYNTHETIC_WARNING, TOP_N
from src.data_ingestion import is_synthetic

st.set_page_config(page_title="MovieLens Recommender", layout="wide")


@st.cache_resource(show_spinner="Loading model artifacts ...")
def load_state():
    if not (ARTIFACTS_DIR / "mappings.npz").exists():
        return None
    maps = np.load(ARTIFACTS_DIR / "mappings.npz")
    user_ids, item_ids = maps["user_ids"], maps["item_ids"]
    als = np.load(ARTIFACTS_DIR / "als.npz")
    S = sp.load_npz(ARTIFACTS_DIR / "item_sims.npz").tocsr()
    pop = np.load(ARTIFACTS_DIR / "popularity.npy")
    movies = pd.read_csv(PROCESSED_DIR / "movies.csv").set_index("movie_id")
    hist = pd.concat([pd.read_csv(PROCESSED_DIR / "train.csv"),
                      pd.read_csv(PROCESSED_DIR / "val.csv")],
                     ignore_index=True)
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    iidx = {int(m): k for k, m in enumerate(item_ids)}
    return {
        "user_ids": user_ids, "item_ids": item_ids, "iidx": iidx,
        "X": als["X"], "Y": als["Y"], "S": S, "S_csc": S.tocsc(),
        "pop": pop, "pop_order": np.argsort(-pop, kind="stable"),
        "movies": movies, "history": hist, "test": test,
        "eval_users": np.sort(test["user_id"].unique()),
    }


def top_n(scores: np.ndarray, seen: set, n: int = TOP_N) -> list[int]:
    order = np.argsort(-scores, kind="stable")
    out = []
    for i in order:
        if int(i) not in seen:
            out.append(int(i))
            if len(out) == n:
                break
    return out


def title_of(state, item_idx: int) -> str:
    mid = int(state["item_ids"][item_idx])
    if mid in state["movies"].index:
        return str(state["movies"].loc[mid, "title"])
    return f"movie {mid}"


def rec_table(state, rec_idx: list[int], test_idx: set) -> pd.DataFrame:
    rows = []
    for rank, i in enumerate(rec_idx, 1):
        mid = int(state["item_ids"][i])
        meta = state["movies"].loc[mid] if mid in state["movies"].index else None
        rows.append({
            "#": rank,
            "title": meta["title"] if meta is not None else f"movie {mid}",
            "genres": meta["genres"] if meta is not None else "?",
            "held-out hit": "*" if i in test_idx else "",
        })
    return pd.DataFrame(rows).set_index("#")


def main() -> None:
    st.title("MovieLens recommender: model comparison")
    if is_synthetic():
        st.warning(f"Data status: {SYNTHETIC_WARNING}. Run "
                   "`python -m src.data_ingestion` with network access and "
                   "re-run the pipeline for real results.")
    state = load_state()
    if state is None:
        st.error("No artifacts found. Run the pipeline first:\n\n"
                 "```\npython -m src.data_ingestion\npython -m src.split\n"
                 "python -m src.evaluate tune\npython -m src.evaluate test\n```")
        return

    user = st.sidebar.selectbox("User", state["eval_users"],
                                help="Users with a held-out test slice")
    st.sidebar.markdown(
        f"Ranking metrics live in `reports/results.md`. Recommendations "
        f"marked `*` hit one of the user's {TOP_N // 2} held-out test items.")

    hist = (state["history"][state["history"]["user_id"] == user]
            .sort_values("timestamp", ascending=False))
    hist_idx = [state["iidx"][m] for m in hist["movie_id"] if m in state["iidx"]]
    seen = set(hist_idx)
    test_mids = state["test"].loc[state["test"]["user_id"] == user, "movie_id"]
    test_idx = {state["iidx"][m] for m in test_mids if m in state["iidx"]}

    st.subheader(f"User {user}: watch history ({len(hist)} positives)")
    show = hist.head(15).merge(state["movies"], left_on="movie_id",
                               right_index=True, how="left")
    st.dataframe(show[["title", "genres", "rating"]].reset_index(drop=True),
                 use_container_width=True)

    # ----------------------------------------------------------- scoring --
    uarr = np.zeros(len(state["item_ids"]), dtype=np.float32)
    uarr[hist_idx] = 1.0
    knn_scores = np.asarray(uarr @ state["S"]).ravel()
    u_pos = int(np.searchsorted(state["user_ids"], user))
    als_scores = state["X"][u_pos] @ state["Y"].T
    recs = {
        "Popularity": top_n(state["pop"], seen),
        "Item-item CF": top_n(knn_scores, seen),
        "ALS": top_n(als_scores, seen),
    }

    st.subheader(f"Top-{TOP_N} recommendations")
    cols = st.columns(3)
    for col, (name, rec_idx) in zip(cols, recs.items()):
        with col:
            st.markdown(f"**{name}**")
            st.dataframe(rec_table(state, rec_idx, test_idx),
                         use_container_width=True)

    # ------------------------------------------------- why panel (item-knn) --
    with st.expander("Why did item-item CF recommend these?"):
        st.markdown("Each recommendation's score is a sum of cosine "
                    "similarities to the user's history; these are the "
                    "history items that contributed most.")
        S_csc = state["S_csc"]
        for i in recs["Item-item CF"]:
            col_i = S_csc.getcol(i)
            contrib = dict(zip(col_i.indices.tolist(), col_i.data.tolist()))
            drivers = sorted(((h, contrib[h]) for h in hist_idx if h in contrib),
                             key=lambda t: -t[1])[:3]
            why = ", ".join(f"{title_of(state, h)} (sim {s:.3f})"
                            for h, s in drivers) or "no overlapping neighbours"
            st.markdown(f"- **{title_of(state, i)}** ← {why}")


main()
