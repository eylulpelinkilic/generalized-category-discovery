"""Image-level (R) and cluster-level (A, B) similarity matrices.

R_ij = <v_i, t_j> with unit vectors. For a cluster with member rows U:
A_kj = mean_{i in U} R_ij ;  B_kj = <m/|m|, t_j> with m = mean_{i in U} v_i ;  A = |m| * B.
Rows of A and B share their ranking when |m| > 0; magnitudes differ, which changes
threshold-based scores (max similarity, softmax) but not the top-1 label.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def similarity(image_emb: np.ndarray, class_emb: np.ndarray) -> np.ndarray:
    for name, x in (("image", image_emb), ("text", class_emb)):
        norms = np.linalg.norm(x, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4):
            raise ValueError(f"{name} embeddings are not unit-normalised (norms {norms.min():.4f}..{norms.max():.4f})")
    R = (image_emb @ class_emb.T).astype(np.float32)
    if R.size and (R.max() > 1 + 1e-5 or R.min() < -1 - 1e-5):
        raise ValueError("cosine similarity outside [-1, 1]")
    return R


def aggregate(R: np.ndarray, emb: np.ndarray, members: list[np.ndarray]) -> dict:
    """A, B, |m| and member counts for each group of row indices (NaN rows for empty groups)."""
    K, C = len(members), R.shape[1]
    A = np.full((K, C), np.nan, dtype=np.float32)
    B = np.full((K, C), np.nan, dtype=np.float32)
    norm = np.full(K, np.nan, dtype=np.float32)
    n = np.zeros(K, dtype=np.int64)
    for k, rows in enumerate(members):
        if len(rows) == 0:
            continue
        n[k] = len(rows)
        A[k] = R[rows].mean(0)
        m = emb[rows].mean(0)
        norm[k] = np.linalg.norm(m)
        if norm[k] > 0:
            B[k] = A[k] / norm[k]          # = <m/|m|, t_j>, since A_k = <m, t_j>
    return {"A": A, "B": B, "norm": norm, "n": n}


def select_members(rows_table: pd.DataFrame, group_col: str, groups: list, representation: str,
                   seed: int = 0) -> list[np.ndarray]:
    """Row indices per group for one representation, chosen without labels.

    `all`: every row; `centre<n>`: the n rows nearest to the GCD centre (sq_dist_to_centre, ties by
    sample id); `random<n>`: n rows drawn with a generator seeded per group.
    """
    out = []
    by_group = {g: t for g, t in rows_table.groupby(group_col)}
    for g in groups:
        t = by_group.get(g)
        if t is None or len(t) == 0:
            out.append(np.zeros(0, dtype=np.int64))
            continue
        if representation == "all":
            out.append(t["row"].to_numpy())
        elif representation.startswith("centre"):
            n = int(representation[len("centre"):])
            t = t.sort_values(["sq_dist_to_centre", "sample_id"], kind="stable")
            out.append(t["row"].to_numpy()[:n])
        elif representation.startswith("random"):
            n = int(representation[len("random"):])
            rows = np.sort(t["row"].to_numpy())
            rng = np.random.RandomState(seed + int(g) * 7919)
            out.append(np.sort(rng.choice(rows, size=min(n, len(rows)), replace=False)))
        else:
            raise ValueError(f"unknown representation {representation!r}")
    return out
