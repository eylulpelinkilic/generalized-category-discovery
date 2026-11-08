"""Semi-supervised k-means (upstream `faster_mix_k_means_pytorch.K_Means.fit_mix`), repaired.

Same algorithm: labelled anchors are fixed to the cluster of their class (classes in sorted
order → clusters 0..L-1); k-means++ starts from the labelled class means and adds K-L centres
from the unlabelled points; Lloyd iterations over labelled + unlabelled members; stop when the
squared total centre shift falls below `tol`; keep the initialisation with the lowest inertia.
Repairs (AUDIT.md §2): empty clusters keep their previous centre instead of becoming NaN;
singletons are handled by a vectorised update; the returned labels, centres and inertia come
from one consistent final state; iteration counts are real; the k-means++ draw cannot reuse a
stale index. Distances are squared Euclidean on the (L2-normalised) features.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


def sq_dist(x: torch.Tensor, c: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
    """||x - c||^2 for all pairs, computed in row chunks, clamped at 0."""
    c_norm = (c * c).sum(1)
    out = torch.empty(x.shape[0], c.shape[0], dtype=x.dtype, device=x.device)
    for s in range(0, x.shape[0], chunk):
        xb = x[s:s + chunk]
        out[s:s + chunk] = ((xb * xb).sum(1, keepdim=True) + c_norm[None, :] - 2.0 * xb @ c.T).clamp_min_(0)
    return out


def assign(x: torch.Tensor, centres: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    d = sq_dist(x, centres)
    dist, labels = d.min(1)
    return labels, dist


def kmeans_pp(x: torch.Tensor, k: int, rs: np.random.RandomState, pre_centres: torch.Tensor | None = None):
    """Upstream `kpp`: one `rs.rand()` per added centre, first index with cumulative prob >= r."""
    events = 0
    centres = pre_centres.clone() if pre_centres is not None else x[rs.randint(0, len(x))][None, :].clone()
    d2 = sq_dist(x, centres).min(1).values
    while centres.shape[0] < k:
        total = d2.sum()
        r = rs.rand()
        if not torch.isfinite(total) or total <= 0:      # every point already coincides with a centre
            ind = int(r * len(x)) % len(x)
            events += 1
        else:
            cum = torch.cumsum(d2 / total, 0)
            ind = int(torch.searchsorted(cum, torch.tensor([r], dtype=cum.dtype, device=cum.device)).clamp_max(len(x) - 1))
        centres = torch.cat([centres, x[ind][None, :]], 0)
        d2 = torch.minimum(d2, sq_dist(x, x[ind][None, :])[:, 0])
    return centres, events


@dataclass
class FitResult:
    labels_labelled: np.ndarray
    labels_unlabelled: np.ndarray
    centres: np.ndarray
    inertia: float
    n_iter: int
    converged: bool
    empty_events: int
    kpp_degenerate_events: int
    final_label_changes: int
    cluster_sizes: np.ndarray
    labelled_classes: np.ndarray
    init_inertias: list = field(default_factory=list)
    best_init: int = 0


def _update(centres, members_x, members_lab, k):
    sums = torch.zeros_like(centres).index_add_(0, members_lab, members_x)
    counts = torch.bincount(members_lab, minlength=k)
    nonempty = counts > 0
    new = centres.clone()
    new[nonempty] = sums[nonempty] / counts[nonempty, None].to(centres.dtype)
    return new, int((~nonempty).sum())


def fit_mix_once(u: torch.Tensor, l: torch.Tensor, l_ncid: torch.Tensor, n_l_classes: int, k: int,
                 max_iter: int, tol: float, rs: np.random.RandomState) -> dict:
    l_centres = torch.stack([l[l_ncid == c].mean(0) for c in range(n_l_classes)])
    centres, kpp_events = kmeans_pp(u, k, rs, pre_centres=l_centres)
    cat = torch.cat([l, u])
    empty_events, converged, it = 0, False, 0
    u_lab = None
    for it in range(1, max_iter + 1):
        old = centres
        u_lab, _ = assign(u, centres)
        centres, n_empty = _update(centres, cat, torch.cat([l_ncid, u_lab]), k)
        empty_events += n_empty
        shift = torch.sqrt(((centres - old) ** 2).sum(1)).sum()
        if float(shift) ** 2 < tol:
            converged = True
            break
    final_u, u_dist = assign(u, centres)                  # one consistent final state
    l_dist = ((l - centres[l_ncid]) ** 2).sum(1)
    return {"u_lab": final_u, "centres": centres, "inertia": float(u_dist.sum() + l_dist.sum()), "n_iter": it,
            "converged": converged, "empty_events": empty_events, "kpp_events": kpp_events,
            "changes": int((final_u != u_lab).sum()) if u_lab is not None else 0}


def fit_mix(u_feats, l_feats, l_targets, k: int, n_init: int = 100, max_iter: int = 200, tol: float = 1e-4,
            seed: int = 0, device: str = "cpu") -> FitResult:
    """u_feats [N_u, D], l_feats [N_l, D], l_targets [N_l] class ids (any integers)."""
    u = torch.as_tensor(np.asarray(u_feats), dtype=torch.float32, device=device)
    l = torch.as_tensor(np.asarray(l_feats), dtype=torch.float32, device=device)
    classes, l_ncid = np.unique(np.asarray(l_targets), return_inverse=True)   # sorted, as torch.unique upstream
    if len(classes) > k:
        raise ValueError(f"K={k} is smaller than the number of labelled classes ({len(classes)})")
    l_ncid_t = torch.as_tensor(l_ncid, device=device)
    rs = np.random.RandomState(seed)
    best, inertias, best_i = None, [], 0
    for i in range(n_init):
        res = fit_mix_once(u, l, l_ncid_t, len(classes), k, max_iter, tol, rs)
        inertias.append(res["inertia"])
        if best is None or res["inertia"] < best["inertia"]:
            best, best_i = res, i
    all_labels = torch.cat([l_ncid_t, best["u_lab"]])
    return FitResult(labels_labelled=l_ncid.astype(np.int64), labels_unlabelled=best["u_lab"].cpu().numpy(),
                     centres=best["centres"].cpu().numpy(), inertia=best["inertia"], n_iter=best["n_iter"],
                     converged=best["converged"], empty_events=best["empty_events"],
                     kpp_degenerate_events=best["kpp_events"], final_label_changes=best["changes"],
                     cluster_sizes=torch.bincount(all_labels, minlength=k).cpu().numpy(),
                     labelled_classes=classes, init_inertias=inertias, best_init=best_i)


def fit_plain(x, k: int, n_init: int = 10, max_iter: int = 300, tol: float = 1e-4, seed: int = 0,
              device: str = "cpu") -> tuple[np.ndarray, float]:
    """Unsupervised k-means with the same machinery (used by the K estimator)."""
    x = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)
    rs = np.random.RandomState(seed)
    best = None
    for _ in range(n_init):
        centres, _ = kmeans_pp(x, k, rs)
        for _it in range(max_iter):
            lab, _ = assign(x, centres)
            new, _ = _update(centres, x, lab, k)
            shift = torch.sqrt(((new - centres) ** 2).sum(1)).sum()
            centres = new
            if float(shift) ** 2 < tol:
                break
        lab, dist = assign(x, centres)
        inertia = float(dist.sum())
        if best is None or inertia < best[1]:
            best = (lab.cpu().numpy(), inertia)
    return best
