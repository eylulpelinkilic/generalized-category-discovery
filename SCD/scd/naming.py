"""SCD naming and refinement (paper Sec. 3.2-3.3; official `main_unsup.py` / `main_ptsup.py`, commit 2ce64f7).

Sequence: initial clusters -> per-image top-k names over a large vocabulary -> cluster vote counters ->
candidate pool -> distinct-name assignment (Hungarian on vote counts) -> every image reassigned to its nearest
selected name -> repeat until the set of selected names is unchanged.

Index spaces are kept apart (see ../README.md): `v` vocabulary index, `s` pool index, `p` name position,
`c` cluster id. Nothing here reads evaluation labels.

Official behaviours re-created exactly (and tested):
  * per-cluster votes: `Counter` over the flattened top-k vocabulary indices of the member images, in data order;
    `most_common(n)` breaks ties by first appearance;
  * pool: `list(set(...))` of each cluster's `most_common(num_common_vote)` (`pool_order="official_set"`);
  * weights: w[row, s] += count for each cluster's `most_common(num_common_linear)`; square, padded with zero rows
    or columns; solved as `linear_assignment(w.max() - w)` with sklearn's Munkres; the first `num_slots` rows of the
    row-sorted result give the names (rows beyond the real clusters are padding rows and still contribute names);
  * a selected name is mapped back to a column with `nouns.index(name)` (first occurrence of the normalised name);
  * reassignment: argmax over the selected names (first maximum wins);
  * stop when the set of selected (voted) names equals the previous one.
Repairs (each recorded in the trace): iteration limit and cycle detection; a pool too small for the clusters (the
official code raises KeyError) is deepened; known names are filtered in vocabulary-index space (`known_filter`).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np

from scd.assignment import max_weight_assignment


@dataclass(frozen=True)
class NamingConfig:
    topk_vote: int = 3              # official scripts: --topk 3 (help text: imagenet 5, cub 3)
    num_common_vote: int = 10       # official scripts: --num_common_vote 10 (help: imagenet 20, cub 10)
    num_common_linear: int = 2      # official scripts: -num_common_linear 2 (typo; see INSPECTION.md); help: 4 / 2
    assignment: str = "hungarian"   # "hungarian" (distinct names) | "plurality" (Eq. 2: most common name, no matching)
    iterate: bool = True
    max_iter: int = 50              # official: no limit
    pool_order: str = "official_set"  # "official_set" | "first_appearance"
    solver: str = "munkres"         # "munkres" (official) | "scipy"
    known_filter: str = "official"  # partially supervised: "official" (index-space defect) | "fixed"
    insufficient_pool: str = "extend"  # official: KeyError ("error")

    def __post_init__(self):
        if self.num_common_linear > self.num_common_vote:
            raise ValueError("num_common_linear > num_common_vote: names outside the pool would enter the matrix")
        for name, ok in (("assignment", {"hungarian", "plurality"}), ("pool_order", {"official_set", "first_appearance"}),
                         ("solver", {"munkres", "scipy"}), ("known_filter", {"official", "fixed"}),
                         ("insufficient_pool", {"extend", "error"})):
            if getattr(self, name) not in ok:
                raise ValueError(f"{name} must be one of {sorted(ok)}")


class InsufficientCandidates(RuntimeError):
    pass


# ----------------------------------------------------------------------------- per-image names
def image_topk(image_emb: np.ndarray, text_emb: np.ndarray, k: int, columns=None, batch: int = 2048,
               logit_dtype: str = "float32", device: str = "cpu", logit_scale: float = 100.0):
    """Top-k vocabulary indices per image, highest score first; exact ties go to the lowest vocabulary index.

    `columns`: the vocabulary indices that may be predicted (all by default); results are vocabulary indices.
    `logit_dtype="float16"` rounds the logits (scale 100, as the official code) to fp16 before ranking, emulating
    the official GPU computation; ties it creates are broken by the lowest index instead of the CUDA kernel."""
    import torch

    cols = np.arange(len(text_emb)) if columns is None else np.unique(np.asarray(columns, dtype=np.int64))
    if len(cols) < k:
        raise ValueError(f"only {len(cols)} columns for top-{k}")
    T = torch.from_numpy(np.ascontiguousarray(text_emb[cols], dtype=np.float32)).to(device)
    extra = min(len(cols), k + 8)
    idx_out = np.zeros((len(image_emb), k), np.int64)
    val_out = np.zeros((len(image_emb), k), np.float32)
    n_boundary_ties = 0
    for s in range(0, len(image_emb), batch):
        x = torch.from_numpy(np.ascontiguousarray(image_emb[s:s + batch], dtype=np.float32)).to(device)
        logits = logit_scale * (x @ T.T)
        if logit_dtype == "float16":
            logits = logits.half().float()
        elif logit_dtype != "float32":
            raise ValueError(logit_dtype)
        val, ind = torch.topk(logits, extra, dim=1)
        val, ind = val.cpu().numpy(), ind.cpu().numpy()
        o = np.argsort(ind, axis=1, kind="stable")                       # lowest index first ...
        val, ind = np.take_along_axis(val, o, 1), np.take_along_axis(ind, o, 1)
        o = np.argsort(-val, axis=1, kind="stable")                      # ... then highest score
        val, ind = np.take_along_axis(val, o, 1), np.take_along_axis(ind, o, 1)
        # a tie straddling the candidate window would make the window order arbitrary: redo those rows exactly
        bad = np.flatnonzero((extra < len(cols)) & (val[:, k - 1] <= val[:, extra - 1]))
        for r in bad:
            row = logits[r].cpu().numpy()
            o = np.lexsort((np.arange(len(row)), -row))[:k]
            ind[r, :k], val[r, :k] = o, row[o]
        n_boundary_ties += int((val[:, k - 1] == val[:, min(k, extra - 1)]).sum()) if extra > k else 0
        idx_out[s:s + len(val)] = cols[ind[:, :k]]
        val_out[s:s + len(val)] = val[:, :k]
    return idx_out, val_out, {"n_rows_with_tie_at_k": n_boundary_ties, "logit_dtype": logit_dtype}


# ----------------------------------------------------------------------------- votes, pool, assignment
def cluster_counters(topk_idx: np.ndarray, labels: np.ndarray, clusters, k: int, exclude=None) -> dict:
    """Official vote counters: per cluster, a Counter of the members' top-k vocabulary indices (data order)."""
    excl = set(int(x) for x in exclude) if exclude is not None else None
    out = {}
    for c in clusters:
        flat = topk_idx[labels == c, :k].ravel().tolist()
        if excl:
            flat = [x for x in flat if x not in excl]
        out[c] = Counter(flat)
    return out


def candidate_pool(counters: dict, clusters, depth: int, order: str) -> list:
    voted = []
    for c in clusters:
        voted += [v for v, _ in counters[c].most_common(depth)]
    if order == "official_set":
        return list(set(voted))
    return list(dict.fromkeys(voted))


def assignment_weights(pool: list, counters: dict, clusters, num_common_linear: int) -> np.ndarray:
    col = {v: s for s, v in enumerate(pool)}
    D = max(len(pool), len(clusters))
    w = np.zeros((D, D), dtype=np.int64)
    for row, c in enumerate(clusters):
        for v, n in counters[c].most_common(num_common_linear):
            w[row, col[v]] += n
    return w


def name_clusters(topk_idx, labels, clusters, cfg: NamingConfig, num_slots: int, exclude=None) -> dict:
    """One naming step on fixed memberships. Returns per-slot vocabulary indices and diagnostics.

    `clusters` are named in the given order (rows of the matrix). Slots are the first `num_slots` rows."""
    counters = cluster_counters(topk_idx, labels, clusters, cfg.topk_vote, exclude)
    sizes = {c: int((labels == c).sum()) for c in clusters}
    repairs = []
    if not clusters:     # official: w.max() of an empty matrix raises ValueError
        return {"slots": [], "pool": [], "pool_size": 0, "matrix_size": 0, "total_weight": 0,
                "repairs": [{"repair": "no_clusters_to_name"}]}
    if cfg.assignment == "plurality":
        slots = []
        for row, c in enumerate(clusters):
            mc = counters[c].most_common(1)
            slots.append({"row": row, "cluster": c, "kind": "cluster", "v": mc[0][0] if mc else None,
                          "weight": mc[0][1] if mc else 0})
            if not mc:
                repairs.append({"repair": "cluster_without_votes", "cluster": c})
        pool, w = [], None
    else:
        depth = cfg.num_common_vote
        while True:
            pool = candidate_pool(counters, clusters, depth, cfg.pool_order)
            w = assignment_weights(pool, counters, clusters, cfg.num_common_linear)
            ind = max_weight_assignment(w, cfg.solver)[:num_slots]
            unnamed = [int(r) for r, s in ind if s >= len(pool)]
            if not unnamed:
                break
            if cfg.insufficient_pool == "error":
                raise InsufficientCandidates(f"{len(pool)} candidate names for {len(clusters)} clusters "
                                             f"(official code: KeyError)")
            longest = max((len(ct) for ct in counters.values()), default=0)
            if depth >= longest:          # every voted name is already in the pool
                repairs.append({"repair": "slots_left_unnamed", "rows": unnamed, "pool": len(pool)})
                break
            depth += 1
        if depth != cfg.num_common_vote:
            repairs.append({"repair": "pool_deepened", "depth": depth, "pool": len(pool),
                            "clusters": len(clusters)})
        slots = []
        for r, s in ind:
            r, s = int(r), int(s)
            real = r < len(clusters)
            slots.append({"row": r, "cluster": clusters[r] if real else None, "kind": "cluster" if real else "padding_row",
                          "v": pool[s] if s < len(pool) else None, "weight": int(w[r, s])})
    for sl in slots:
        c = sl["cluster"]
        sl["votes_in_cluster"] = int(counters[c][sl["v"]]) if (c is not None and sl["v"] is not None) else 0
        sl["cluster_size"] = sizes.get(c, 0) if c is not None else 0
        sl["zero_support"] = sl["v"] is not None and sl["weight"] == 0
        sl["top_votes"] = counters[c].most_common(5) if c is not None else []
    return {"slots": slots, "pool": pool, "pool_size": len(pool), "matrix_size": 0 if w is None else len(w),
            "total_weight": int(sum(sl["weight"] for sl in slots)), "repairs": repairs}


def reassign(image_emb: np.ndarray, text_emb: np.ndarray, sel_cols: list, batch: int = 8192):
    """Each image to its highest-scoring selected name (first maximum on ties). Returns positions and tie count."""
    T = text_emb[np.asarray(sel_cols, dtype=np.int64)].astype(np.float32)
    out = np.zeros(len(image_emb), np.int64)
    ties = 0
    for s in range(0, len(image_emb), batch):
        L = image_emb[s:s + batch].astype(np.float32) @ T.T
        out[s:s + len(L)] = L.argmax(1)
        ties += int(((L == L.max(1, keepdims=True)).sum(1) > 1).sum())
    return out, ties


# ----------------------------------------------------------------------------- the loop
def refine(image_emb, text_emb, vocab, topk_idx, init_labels, cfg: NamingConfig, *, n_cluster: int,
           cluster_ids=None, known_names=None, labelled_clusters=None):
    """Run SCD naming on images (rows of `image_emb`, `topk_idx`, `init_labels`; the unlabelled images).

    Unsupervised: `known_names=None`; `num_slots = n_cluster` (official `num_unlab_classes = args.n_cluster`).
    Partially supervised: `known_names` = normalised names of the labelled classes, in class order;
    `labelled_clusters` = initial cluster ids that contain labelled anchors (official `set(l_preds)`);
    `cluster_ids` = all initial cluster ids (official `set(all_preds)`); `num_slots = n_cluster - len(known_names)`.
    Returns {"labels": final name positions, "names": final selected names, "iterations": [...], ...}.
    """
    ptsup = known_names is not None
    known = list(known_names or [])
    for n in known:
        if n not in vocab.first_index:
            raise ValueError(f"known name {n!r} is not in the vocabulary (it must obey the episode vocabulary)")
    num_slots = n_cluster - len(known) if ptsup else n_cluster
    labels = np.asarray(init_labels).copy()
    all_ids = sorted(set(cluster_ids if cluster_ids is not None else labels.tolist()))
    if ptsup:
        clusters = sorted(set(all_ids) - set(labelled_clusters or []))
        exclude = ([vocab.first_index[n] for n in known] if cfg.known_filter == "official"
                   else sorted(v for n in known for v in vocab.columns_of[n]))
    else:
        clusters = sorted(set(labels.tolist()))
        exclude = None
    iterations, seen = [], []
    prev_names = None
    stop = None
    for it in range(1, cfg.max_iter + 1):
        step = name_clusters(topk_idx, labels, clusters, cfg, num_slots, exclude)
        cur_names = [vocab.names[sl["v"]] for sl in step["slots"] if sl["v"] is not None]
        if ptsup:
            cand = sorted(set(cur_names + known))
        else:
            cand = list(cur_names)
        sel_cols = [vocab.first_index[n] for n in cand]
        before = labels
        labels, ties = reassign(image_emb, text_emb, sel_cols)
        rec = {"iteration": it, "clusters_named": len(clusters), "num_slots": num_slots, **{k: step[k] for k in
               ("pool_size", "matrix_size", "total_weight", "repairs")},
               "slots": [{k: (v if k != "top_votes" else [[int(a), int(b)] for a, b in v]) for k, v in sl.items()}
                         for sl in step["slots"]],
               "voted_names": cur_names, "selected_names": cand, "selected_columns": sel_cols,
               "n_selected": len(cand), "n_unique_selected": len(set(cand)),
               "n_duplicate_voted_names": len(cur_names) - len(set(cur_names)),
               "n_zero_support_slots": sum(sl["zero_support"] for sl in step["slots"]),
               "n_padding_row_slots": sum(sl["kind"] == "padding_row" for sl in step["slots"]),
               "n_nonempty_after": int(len(np.unique(labels))), "reassignment_ties": ties,
               "labels_before": before, "labels_after": labels.copy(),
               "exclude_size": 0 if exclude is None else len(exclude)}
        iterations.append(rec)
        key = frozenset(cur_names)
        if not cfg.iterate:
            stop = "single_pass"
            break
        if prev_names is not None and key == prev_names:
            stop = "converged"
            break
        if key in seen[:-1]:
            stop = "cycle"
            rec["cycle_period"] = len(seen) - seen.index(key)
            break
        seen.append(key)
        prev_names = key
        # next iteration's clusters and filter (official index semantics)
        if ptsup:
            known_pos = [cand.index(n) for n in known]
            clusters = [p for p, n in enumerate(cand) if n not in set(known)]   # official: string-set order
            if cfg.known_filter == "official":
                exclude = known_pos            # positions in `cand`, compared with vocabulary indices (defect)
        else:
            clusters = sorted(set(labels.tolist()))
    else:
        stop = "max_iter"
    return {"labels": labels, "names": iterations[-1]["selected_names"], "iterations": iterations,
            "stop_reason": stop, "n_iterations": len(iterations), "config": asdict(cfg), "num_slots": num_slots,
            "ptsup": ptsup}


def initial_cluster_names(result: dict, labelled_cluster_names: dict | None = None) -> dict:
    """Names given to the *initial* clusters in iteration 1 (before any reassignment): cluster id -> name.
    Labelled clusters (partially supervised) keep their known name. Clusters without a slot get no entry."""
    first = result["iterations"][0]
    out = dict(labelled_cluster_names or {})
    names = first["voted_names"]
    named = [sl for sl in first["slots"] if sl["v"] is not None]
    for sl, n in zip(named, names):
        if sl["cluster"] is not None:
            out[sl["cluster"]] = n
    return out
