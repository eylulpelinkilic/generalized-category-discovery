"""Image-level evaluation of cluster-level decisions.

A cluster's decision (label, accept/reject) applies to all of its member images. Everything is
computed from per-cluster counts, so images of one cluster always move together: tied clusters
are accepted together, AUROC treats images of one cluster as tied, and the bootstrap resamples
clusters.

Per (episode, cluster): n (images), n_avail (true class in V), n_correct (available and the
cluster label equals the true class), n_absent = n - n_avail.
"""
from __future__ import annotations

import numpy as np


def cluster_counts(img_group: np.ndarray, y_img: np.ndarray, pred_class: np.ndarray, vocab: set, K: int) -> dict:
    avail = np.isin(y_img, list(vocab))
    correct = avail & (pred_class[img_group] == y_img)
    n = np.bincount(img_group, minlength=K)
    return {"n": n, "n_avail": np.bincount(img_group, weights=avail, minlength=K),
            "n_correct": np.bincount(img_group, weights=correct, minlength=K)}


def risk_coverage(score: np.ndarray, n: np.ndarray, n_correct: np.ndarray, n_avail: np.ndarray | None = None) -> dict:
    """Whole-cluster acceptance in decreasing score; equal scores enter as one block.

    Error of an accepted image: its true class is absent from V, or the label is wrong.
    AURC = sum over blocks of (risk after the block) x (coverage added by the block).
    """
    keep = (n > 0) & np.isfinite(score)
    s, nn, nc = score[keep], n[keep].astype(float), n_correct[keep].astype(float)
    total = nn.sum()
    if total == 0:
        return {"coverage": np.zeros(0), "risk": np.zeros(0), "threshold": np.zeros(0), "aurc": np.nan}
    uniq = np.unique(s)[::-1]
    block = np.searchsorted(-uniq, -s)                       # block index of each cluster
    bn = np.bincount(block, weights=nn, minlength=len(uniq))
    bc = np.bincount(block, weights=nc, minlength=len(uniq))
    cum_n, cum_c = np.cumsum(bn), np.cumsum(bc)
    coverage = cum_n / total
    risk = 1.0 - cum_c / cum_n
    aurc = float(np.sum(risk * np.diff(np.concatenate([[0.0], coverage]))))
    return {"coverage": coverage, "risk": risk, "threshold": uniq, "aurc": aurc}


def auroc_clusters(score: np.ndarray, pos: np.ndarray, neg: np.ndarray, weights: np.ndarray | None = None):
    """Image-level AUROC when images inherit their cluster's score (ties count 1/2).

    pos/neg: per-cluster counts of positive/negative images. `weights` [B, K] gives bootstrap
    multiplicities; returns one value (or B values)."""
    keep = np.isfinite(score)
    s, p, q = score[keep], pos[keep].astype(float), neg[keep].astype(float)
    G = (s[:, None] > s[None, :]).astype(float) + 0.5 * (s[:, None] == s[None, :])
    if weights is None:
        den = p.sum() * q.sum()
        return float(p @ G @ q / den) if den > 0 else np.nan
    w = weights[:, keep]
    P, Q = w * p, w * q
    den = P.sum(1) * Q.sum(1)
    num = np.einsum("bk,kl,bl->b", P, G, Q)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def at_threshold(score: np.ndarray, n, n_avail, n_correct, tau: float) -> dict:
    acc = np.isfinite(score) & (score >= tau) & (n > 0)
    n, n_avail, n_correct = (np.asarray(x, float) for x in (n, n_avail, n_correct))
    n_absent = n - n_avail
    tot = n.sum()
    accepted = n[acc].sum()
    out = {"coverage": accepted / tot if tot else np.nan,
           "accuracy_among_accepted": n_correct[acc].sum() / accepted if accepted else np.nan,
           "false_acceptance_absent": n_absent[acc].sum() / n_absent.sum() if n_absent.sum() else np.nan,
           "true_acceptance_available": n_avail[acc].sum() / n_avail.sum() if n_avail.sum() else np.nan,
           "end_to_end_accuracy": (n_correct[acc].sum() + n_absent[~acc].sum()) / tot if tot else np.nan}
    return out


def choose_thresholds(score, n, n_avail, n_correct, target_tpr: float = 0.95) -> dict:
    """Calibration rules (PROTOCOL.md §7) from pooled (episode, cluster) rows.

    tau_e2e: maximises end-to-end accuracy (ties: the largest threshold);
    tau_95: largest threshold accepting >= target_tpr of label-available images.
    Candidates are the observed scores plus +inf (reject everything)."""
    keep = np.isfinite(score) & (n > 0)
    s, n, na, nc = score[keep], n[keep].astype(float), n_avail[keep].astype(float), n_correct[keep].astype(float)
    cand = np.concatenate([np.unique(s), [np.inf]])
    order = np.argsort(-s)
    s_sorted = s[order]
    e2e, tpr = [], []
    for t in cand:
        acc = s >= t
        e2e.append((nc[acc].sum() + (n - na)[~acc].sum()) / n.sum())
        tpr.append(na[acc].sum() / na.sum() if na.sum() else np.nan)
    e2e, tpr = np.array(e2e), np.array(tpr)
    best = np.flatnonzero(e2e == e2e.max())
    tau_e2e = float(cand[best].max())
    ok = np.flatnonzero(tpr >= target_tpr)
    tau_95 = float(cand[ok].max()) if len(ok) else float(s_sorted[-1])
    return {"tau_e2e": tau_e2e, "calibration_e2e_at_tau": float(e2e.max()), "tau_95": tau_95,
            "calibration_tpr_at_tau_95": float(tpr[np.flatnonzero(cand == tau_95)[0]]) if na.sum() else np.nan,
            "n_rows": int(keep.sum()), "n_images": float(n.sum())}


def bootstrap_weights(K: int, n_boot: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multinomial(K, np.full(K, 1.0 / K), size=n_boot).astype(float)


def ratio_ci(num: np.ndarray, den: np.ndarray, weights: np.ndarray) -> tuple[float, list]:
    """num/den are [E, K] per-episode per-cluster counts; metric = mean over episodes of the ratio."""
    point = np.nanmean(num.sum(1) / den.sum(1))
    reps = np.nanmean((weights @ num.T) / (weights @ den.T), axis=1)
    return float(point), [float(x) for x in np.nanpercentile(reps, [2.5, 97.5])]
