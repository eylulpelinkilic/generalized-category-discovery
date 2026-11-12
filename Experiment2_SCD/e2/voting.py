"""Vote-based cluster naming on fixed memberships (protocol B). Pure functions, no labels.

Each member image votes within the episode's available vocabulary (columns `cols`, ascending class ids):
  * `plurality` (k = 1): its highest-scoring available concept;
  * SCD-style top-k (k > 1): its k highest-scoring available concepts, one vote each (official `topk` voting).
Image-level ties go to the lowest column (= lowest class id).

The cluster's label is its most-voted concept. Ties between equally voted concepts go to the higher cluster-mean
similarity A_kj, then to the lowest column. This is the voting predictor's *own* label; its confidence scores are
computed relative to that label (Experiment 1's `vote_fraction` instead measures support for the mean-based label):

| score              | definition (n = members, v1 >= v2 the two largest vote counts)       | range  |
|--------------------|-----------------------------------------------------------------------|--------|
| `win_fraction`     | v1 / n: share of members whose vote(s) include the winner             | (0, 1] |
| `margin`           | (v1 - v2) / n; v2 = 0 when the vocabulary has one concept             | [0, 1] |
| `neg_entropy`      | 1 - H(votes / sum(votes)) / log |V|; 1 when |V| = 1                    | [0, 1] |
| `mean_sim`         | A_k,winner: cluster-mean cosine of the vote winner                   | [-1, 1]|

Higher = more confident = accept (UNMAPPED below the threshold). A cluster with no member in the population has
no prediction (label -1, scores NaN) and is excluded, as in Experiment 1. A single-member cluster scores 1 on the
first three scores (maximal confidence from one vote); this edge case is reported, not special-cased.
"""
from __future__ import annotations

import numpy as np

SCORES = ("win_fraction", "margin", "neg_entropy", "mean_sim")


def image_votes(R_cols: np.ndarray, k: int) -> np.ndarray:
    """[M, k] column indices voted by each image (highest first, ties to the lowest column)."""
    if k == 1:
        return np.argmax(R_cols, 1)[:, None]
    order = np.argsort(-R_cols, axis=1, kind="stable")
    return order[:, :min(k, R_cols.shape[1])]


def vote_matrix(R_cols: np.ndarray, member_group: np.ndarray, K: int, k: int) -> np.ndarray:
    m = R_cols.shape[1]
    votes = np.zeros((K, m))
    if len(R_cols):
        v = image_votes(R_cols, k)
        np.add.at(votes, (np.repeat(member_group, v.shape[1]), v.ravel()), 1.0)
    return votes


def vote_predict(votes: np.ndarray, A_cols: np.ndarray, n: np.ndarray) -> dict:
    """Winner per cluster and its confidence scores. `A_cols` breaks vote ties; `n` = members per cluster."""
    K, m = votes.shape
    pred = np.full(K, -1)
    out = {s: np.full(K, np.nan) for s in SCORES}
    v1 = np.zeros(K)
    v2 = np.zeros(K)
    valid = n > 0
    for c in np.flatnonzero(valid):
        top = votes[c].max()
        cand = np.flatnonzero(votes[c] == top)
        if len(cand) > 1:
            a = A_cols[c, cand]
            cand = cand[a == a.max()]
        w = int(cand[0])
        pred[c] = w
        v1[c] = votes[c, w]
        others = np.delete(votes[c], w)
        v2[c] = others.max() if len(others) else 0.0
        p = votes[c] / votes[c].sum()
        h = -(p[p > 0] * np.log(p[p > 0])).sum()
        out["neg_entropy"][c] = 1.0 - h / np.log(m) if m > 1 else 1.0
        out["mean_sim"][c] = A_cols[c, w]
    out["win_fraction"][valid] = v1[valid] / n[valid]
    out["margin"][valid] = (v1[valid] - v2[valid]) / n[valid]
    return {"pred": pred, "scores": out, "v1": v1, "v2": v2}


def score_votes(A: np.ndarray, member_R: np.ndarray, member_group: np.ndarray, cols: np.ndarray, k: int) -> dict:
    """All vote-based outputs for one episode. A [K, C] cluster means over all classes; member_R [M, C]."""
    K = len(A)
    n = np.bincount(member_group, minlength=K).astype(float) if len(member_group) else np.zeros(K)
    votes = vote_matrix(member_R[:, cols], member_group, K, k)
    res = vote_predict(votes, A[:, cols], n)
    res["votes"] = votes
    res["n"] = n
    return res


def distinct_names(member_R: np.ndarray, member_group: np.ndarray, K: int, cols: np.ndarray, k: int,
                   num_common_vote: int, num_common_linear: int) -> tuple[np.ndarray, dict]:
    """SCD distinct-name assignment on fixed memberships (one naming step, no reassignment).

    Valid only when the vocabulary has at least as many names as non-empty clusters (complete vocabulary);
    callers must not use it for incomplete vocabularies. Returns predicted column per cluster (-1 if unnamed)."""
    from scd.naming import NamingConfig, name_clusters

    clusters = sorted(set(member_group.tolist()))
    if len(cols) < len(clusters):
        raise ValueError(f"distinct-name assignment infeasible: {len(cols)} names for {len(clusters)} clusters")
    topk = image_votes(member_R[:, cols], k)                       # columns, used as vocabulary indices
    cfg = NamingConfig(topk_vote=topk.shape[1], num_common_vote=num_common_vote, num_common_linear=num_common_linear)
    step = name_clusters(topk, member_group, clusters, cfg, num_slots=len(clusters))
    pred = np.full(K, -1)
    for sl in step["slots"]:
        if sl["cluster"] is not None and sl["v"] is not None:
            pred[sl["cluster"]] = sl["v"]
    return pred, {k2: step[k2] for k2 in ("pool_size", "matrix_size", "total_weight", "repairs")} | {
        "n_zero_support": sum(sl["zero_support"] for sl in step["slots"])}
