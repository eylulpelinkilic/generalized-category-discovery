"""Grounding baselines: one label (or a score for rejecting it) per cluster and episode.

Inputs are similarity matrices, member indices and episode columns only: no labels.
"""
from __future__ import annotations

import numpy as np

METHODS = ["forced_top1", "max_mean_sim", "margin", "msp_pilot", "msp_dev", "mcm_aggregate", "mcm_image_mean",
           "vote_fraction", "neg_softmax_entropy", "neg_vote_entropy"]
PILOT_T = 0.01      # experiments/baseline (evaluate_baselines.py)
MCM_T = 1.0         # deeplearning-wisc/MCM eval_ood_detection.py: --T default 1


def softmax(x: np.ndarray, T: float) -> np.ndarray:
    z = x / T
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def norm_entropy(p: np.ndarray) -> np.ndarray:
    m = p.shape[1]
    if m < 2:
        return np.zeros(len(p))
    h = -(p * np.log(np.clip(p, 1e-300, None))).sum(1)
    return h / np.log(m)


def score_clusters(A: np.ndarray, member_R: np.ndarray, member_group: np.ndarray, A_mcm: np.ndarray,
                   member_R_mcm: np.ndarray, cols: np.ndarray, T_dev: float) -> dict:
    """Scores for every method.

    A [K, C] and A_mcm [K, C]: cluster aggregates over all classes (selected prompt / MCM prompt).
    member_R [M, C], member_R_mcm [M, C]: image rows of the members used by this representation;
    member_group [M]: their group index 0..K-1. cols: the episode's class columns.
    Returns {method: (pred_column_index_into_cols [K], score [K])}; rows of empty groups are NaN/-1.
    Ties in the top-1 go to the lowest column index (np.argmax).
    """
    K, m = len(A), len(cols)
    Av, Amv = A[:, cols], A_mcm[:, cols]
    valid = ~np.isnan(Av[:, 0])
    out = {}
    pred = np.full(K, -1)
    pred_mcm = np.full(K, -1)
    pred[valid] = np.argmax(Av[valid], 1)
    pred_mcm[valid] = np.argmax(Amv[valid], 1)

    def put(name, p, s_valid):
        s = np.full(K, np.nan)
        s[valid] = s_valid
        out[name] = (p, s)

    srt = np.sort(Av[valid], 1)
    put("forced_top1", pred, np.ones(valid.sum()))
    put("max_mean_sim", pred, srt[:, -1])
    put("margin", pred, srt[:, -1] - srt[:, -2] if m > 1 else np.full(valid.sum(), np.inf))
    put("msp_pilot", pred, softmax(Av[valid], PILOT_T).max(1))
    p_dev = softmax(Av[valid], T_dev)
    put("msp_dev", pred, p_dev.max(1))
    put("neg_softmax_entropy", pred, 1.0 - norm_entropy(p_dev))
    put("mcm_aggregate", pred_mcm, softmax(Amv[valid], MCM_T).max(1))

    Rv, Rmv = member_R[:, cols], member_R_mcm[:, cols]
    n = np.bincount(member_group, minlength=K).astype(float)
    img_mcm = softmax(Rmv, MCM_T).max(1) if len(Rmv) else np.zeros(0)
    mcm_mean = np.bincount(member_group, weights=img_mcm, minlength=K) / np.maximum(n, 1)
    put("mcm_image_mean", pred_mcm, mcm_mean[valid])
    votes = np.zeros((K, m))
    if len(Rv):
        np.add.at(votes, (member_group, np.argmax(Rv, 1)), 1.0)
    frac = votes / np.maximum(n, 1)[:, None]
    put("vote_fraction", pred, frac[np.flatnonzero(valid), pred[valid]])
    put("neg_vote_entropy", pred, 1.0 - norm_entropy(frac[valid]))
    return out
