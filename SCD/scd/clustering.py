"""Initial clusterings of the two SCD settings.

* Unsupervised: `sklearn.cluster.KMeans(n_clusters=K, random_state=0)` on the L2-normalised features of the
  unlabelled images (official main_unsup.py; with scikit-learn 1.0.2, the authors' version, the defaults were
  n_init=10 and algorithm="auto" -> "elkan"; both are set explicitly here, Lloyd and Elkan give the same fixed
  points up to floating point).
* Partially supervised: constrained semi-supervised k-means ("CSS-k-means", paper Sec. 3.3.1; official
  local_utils/sskm_constrained.py), re-implemented. Labelled anchors stay in their class's cluster; unlabelled
  images are assigned by a min-cost flow in which every cluster receives between `size_min` and `size_max`
  *unlabelled* images; the flow cost is the Euclidean (not squared) distance times 1000, rounded to an integer.
  Centres are the means of all members. The official code calls it with random_state=None (unseeded) and
  returns a wrong iteration count (`i + 1` from an unrelated loop); here seeds are explicit.
"""
from __future__ import annotations

import numpy as np


def kmeans_official(X: np.ndarray, k: int, seed: int = 0, n_init: int = 10, algorithm: str = "lloyd"):
    import sklearn
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, random_state=seed, n_init=n_init, algorithm=algorithm).fit(X)
    return km.labels_.astype(np.int64), {"inertia": float(km.inertia_), "n_iter": int(km.n_iter_),
                                         "sklearn": sklearn.__version__, "n_init": n_init, "algorithm": algorithm,
                                         "random_state": seed}


def _sq_dist(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    d = (X * X).sum(1)[:, None] - 2 * X @ C.T + (C * C).sum(1)[None, :]
    return np.maximum(d, 0.0)


def _kpp(X: np.ndarray, pre_centers: np.ndarray, k: int, rs: np.random.RandomState) -> np.ndarray:
    """Official `kpp` with pre-centres: draw by D^2 using one uniform per new centre (`cum_prob >= r`)."""
    C = pre_centers.copy()
    while len(C) < k:
        d2 = _sq_dist(X, C).min(1)
        cum = np.cumsum(d2 / d2.sum())
        r = rs.rand()
        C = np.vstack([C, X[int(np.flatnonzero(cum >= r)[0])]])
    return C


def constrained_assignment(X: np.ndarray, C: np.ndarray, size_min: int, size_max: int):
    """Min-cost flow assignment with per-cluster size limits (the official graph, solved with OR-Tools).

    Nodes: images (supply 1) -> cluster copies C' (capacity 1 per arc, cost round(1000 * ||x - c||)) ->
    clusters C (capacity size_max, demand size_min) -> one artificial sink (demand n - k * size_min)."""
    from ortools.graph.python import min_cost_flow

    n, k = len(X), len(C)
    if n < k * size_min or n > k * size_max:
        raise ValueError(f"infeasible size limits: {n} points, {k} clusters, [{size_min}, {size_max}]")
    D = np.sqrt(_sq_dist(X, C))
    x_ix = np.arange(n)
    cd_ix = n + np.arange(k)
    c_ix = n + k + np.arange(k)
    art = n + 2 * k
    starts = np.concatenate([np.repeat(x_ix, k), cd_ix, c_ix]).astype(np.int32)
    ends = np.concatenate([np.tile(cd_ix, n), c_ix, np.full(k, art)]).astype(np.int32)
    caps = np.concatenate([np.ones(n * k), np.full(k, size_max), np.full(k, n)]).astype(np.int64)
    costs = np.concatenate([np.around(D.reshape(-1) * 1000, 0), np.zeros(2 * k)]).astype(np.int64)
    supplies = np.concatenate([np.ones(n), np.zeros(k), np.full(k, -size_min), [-(n - k * size_min)]]).astype(np.int64)
    smcf = min_cost_flow.SimpleMinCostFlow()
    arcs = smcf.add_arcs_with_capacity_and_unit_cost(starts, ends, caps, costs)
    smcf.set_nodes_supplies(np.arange(len(supplies)), supplies)
    status = smcf.solve()
    if status != smcf.OPTIMAL:
        raise RuntimeError(f"min cost flow status {status}")
    flow = smcf.flows(arcs[: n * k]).reshape(n, k)
    labels = flow.argmax(1).astype(np.int64)
    return labels, float((D[np.arange(n), labels] ** 2).sum())


def constrained_ss_kmeans(u_feats: np.ndarray, l_feats: np.ndarray, l_targets: np.ndarray, k: int,
                          size_min: int = 50, size_max: int = 1000, n_init: int = 10, max_iter: int = 10,
                          tol: float = 1e-4, seed: int = 0) -> dict:
    """Official `K_Means.fit_mix` (sskm_constrained.py) with an explicit seed.

    Labels of labelled points are the rank of their class among the sorted labelled classes. Per run: centres =
    labelled class means, completed by k-means++ on the unlabelled points; each iteration assigns unlabelled
    points by `constrained_assignment` (labelled stay fixed), inertia = unlabelled + labelled squared distances
    to the current centres, then centres are recomputed; the best-inertia labels (and the centres updated from
    them) are kept; stop when (sum of centre shifts)^2 < tol. The best of `n_init` runs (one random stream)."""
    rs = np.random.RandomState(seed)
    l_classes = np.unique(l_targets)
    cid = {c: i for i, c in enumerate(l_classes)}
    l_lab = np.array([cid[c] for c in l_targets], dtype=np.int64)
    l_centers = np.stack([l_feats[l_lab == i].mean(0) for i in range(len(l_classes))])
    cat = np.concatenate([l_feats, u_feats])
    best = None
    runs = []
    for run in range(n_init):
        C = _kpp(u_feats, l_centers, k, rs)
        best_run = None
        for it in range(max_iter):
            C_old = C.copy()
            u_lab, u_in = constrained_assignment(u_feats, C, size_min, size_max)
            l_in = float(((l_feats - C[l_lab]) ** 2).sum())
            inertia = u_in + l_in
            lab = np.concatenate([l_lab, u_lab])
            C = np.stack([cat[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
            if best_run is None or inertia < best_run[1]:
                best_run = (lab.copy(), inertia, C.copy(), it + 1)
            shift = np.sqrt(((C - C_old) ** 2).sum(1)).sum()
            if shift ** 2 < tol:
                break
        runs.append({"run": run, "inertia": best_run[1], "iterations": it + 1})
        if best is None or best_run[1] < best[1]:
            best = best_run
    lab, inertia, C, n_it = best
    return {"labels_labelled": lab[: len(l_feats)], "labels_unlabelled": lab[len(l_feats):], "centres": C,
            "inertia": inertia, "best_iteration": n_it, "runs": runs, "seed": seed,
            "settings": {"k": k, "size_min": size_min, "size_max": size_max, "n_init": n_init,
                         "max_iter": max_iter, "tol": tol, "cost": "round(1000 * euclidean distance)"}}
