"""Evaluation of SCD runs. These functions take evaluation labels; the naming code never calls them.

* sACC (paper Eq. 3): exact match between the predicted name and the declared target name; image-weighted
  (`image`, the paper's definition and the official `sACC_all`) and class-averaged (`class`, official `sACC_avg`).
* Soft-sACC (paper Eq. 4): per image, the maximum Leacock-Chodorow similarity between any synset whose
  normalised first lemma is the predicted name and any synset whose normalised first lemma is the target name
  (official `calucate_dis_between_names`), averaged after normalisation. Two normalisations:
    - `official`: divide by the largest per-image score observed in the evaluated subset (main_*.py
      `matched_all / max(matched_all)`); the scale then depends on the subset (All, Old and New differ);
    - `max_possible`: divide by lch(x, x) = log(2 * 19) for WordNet 3.0 nouns (the paper's wording,
      "dividing with the maximum possible score").
* Clustering accuracy: one global Hungarian assignment between clusters and classes (GCD `v2`), All/Old/New
  read from it. Used for clustering evaluation only, never to name anything.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache

import numpy as np

from scd.assignment import max_weight_assignment
from scd.vocabulary import LCH_MAX


def semantic_accuracy(pred_names, target_names) -> dict:
    pred_names, target_names = list(pred_names), list(target_names)
    ok = np.array([p == t for p, t in zip(pred_names, target_names)], dtype=float)
    if len(ok) == 0:
        return {"image": float("nan"), "class": float("nan"), "n": 0}
    per = defaultdict(list)
    for t, o in zip(target_names, ok):
        per[t].append(o)
    return {"image": float(ok.mean()), "class": float(np.mean([np.mean(v) for v in per.values()])), "n": len(ok)}


class SoftScorer:
    """Leacock-Chodorow similarity between two normalised names (max over their synsets), cached."""

    def __init__(self, wnid_to_synset: dict, name_to_wnids: dict):
        self.syn = wnid_to_synset
        self.n2w = name_to_wnids

        @lru_cache(maxsize=None)
        def score(a: str, b: str):
            wa, wb = self.n2w.get(a, []), self.n2w.get(b, [])
            if not wa or not wb:
                return None
            return max(self.syn[x].lch_similarity(self.syn[y]) for x in wb for y in wa)

        self.score = score

    def per_image(self, pred_names, target_names) -> np.ndarray:
        vals = [self.score(p, t) for p, t in zip(pred_names, target_names)]
        if any(v is None for v in vals):
            missing = sorted({p for p, v in zip(pred_names, vals) if v is None})[:5]
            raise ValueError(f"names without WordNet synsets, e.g. {missing}")
        return np.array(vals, dtype=float)

    def soft_sacc(self, pred_names, target_names) -> dict:
        s = self.per_image(list(pred_names), list(target_names))
        if len(s) == 0:
            return {"official": float("nan"), "max_possible": float("nan"), "observed_max": float("nan")}
        return {"official": float((s / s.max()).mean()), "max_possible": float((s / LCH_MAX).mean()),
                "observed_max": float(s.max()), "lch_max": LCH_MAX}


def name_iou(pred_names, gt_names) -> float:
    a, b = set(pred_names), set(gt_names)
    return len(a & b) / len(a | b) if (a | b) else float("nan")


def cluster_acc_v2(y_true, y_pred, old_mask, solver: str = "munkres") -> dict:
    """GCD `split_cluster_acc_v2`: one Hungarian assignment on all images; All, Old, New from it."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    old_mask = np.asarray(old_mask, dtype=bool)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    np.add.at(w, (y_pred, y_true), 1)
    ind = max_weight_assignment(w, solver)
    mapped = dict((int(j), int(i)) for i, j in ind)          # true class -> cluster
    total = sum(w[i, j] for i, j in ind) / len(y_pred)
    out = {"all": float(total)}
    for tag, m in (("old", old_mask), ("new", ~old_mask)):
        cls = np.unique(y_true[m])
        n = sum(w[:, c].sum() for c in cls)
        out[tag] = float(sum(w[mapped[c], c] for c in cls) / n) if n else float("nan")
    return out
