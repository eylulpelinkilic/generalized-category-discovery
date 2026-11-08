"""SS-k-means on GCD features, K estimation, held-out assignment and clustering evaluation.

Inputs that carry labels: only `labels_labelled.csv` (anchors, K estimation). The evaluation
labels are read in `evaluate_clustering`, after the assignments are written.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from e1 import upstream
from e1.config import section_hash
from e1.gcd_features import load_features
from e1.gcd_train import select_device
from e1.manifests import Manifest
from e1.provenance import StageRecord, stage_completed, write_json
from e1.sskmeans import assign, fit_mix, fit_plain


def estimate_k(cfg: dict, layout, train_seed: int) -> dict:
    """Upstream estimate_k objective: plain k-means on all training features (labelled +
    unlabelled), accuracy on the labelled subset only, Brent's bounded search over K."""
    run_dir = layout.gcd_run(train_seed)
    out = run_dir / "estimate_k.json"
    if out.exists():
        return json.loads(out.read_text())
    ck = cfg["clustering"]["estimate_k"]
    manifest = Manifest(layout.data)
    ids_l, f_l = load_features(run_dir, "labelled")
    ids_u, f_u = load_features(run_dir, "unlabelled")
    y_l = manifest.labels("labelled").loc[ids_l].to_numpy()
    x = np.concatenate([f_l, f_u])
    device = str(select_device(cfg.get("device", "auto")))
    evaluations = []

    def objective(k_float):
        k = int(k_float)
        pred, inertia = fit_plain(x, k, n_init=ck["n_init"], max_iter=ck["max_iter"], seed=ck["seed"], device=device)
        acc = float(upstream.cluster_acc(y_l.astype(int), pred[: len(f_l)].astype(int)))
        evaluations.append({"K": k, "labelled_acc": acc, "inertia": inertia})
        return -acc

    lo = len(manifest.known_class_ids)
    res = minimize_scalar(objective, bounds=(lo, ck["max_k"]), method="bounded")
    best = max(evaluations, key=lambda e: e["labelled_acc"])
    result = {"method": "upstream estimate_k objective (labelled accuracy), Brent bounded search",
              "bounds": [lo, ck["max_k"]], "brent_x": float(res.x), "K_estimate": int(res.x),
              "best_evaluated": best, "evaluations": evaluations,
              "labels_used": "labels_labelled.csv only"}
    write_json(out, result)
    return result


def run(cfg: dict, layout, train_seed: int, k_mode: str, cluster_seed: int) -> Path:
    ck = cfg["clustering"]
    run_dir = layout.gcd_run(train_seed)
    out = layout.clustering(train_seed, k_mode, cluster_seed)
    h = section_hash(cfg, "clustering")
    if stage_completed(out, h):
        return out
    manifest = Manifest(layout.data)
    k = len(manifest.classes) if k_mode == "known" else estimate_k(cfg, layout, train_seed)["K_estimate"]
    ids_l, f_l = load_features(run_dir, "labelled")
    ids_u, f_u = load_features(run_dir, "unlabelled")
    y_l = manifest.labels("labelled").loc[ids_l].to_numpy()
    record = StageRecord(out, "clustering", h, {"train_seed": train_seed, "k_mode": k_mode, "K": k,
                                                "cluster_seed": cluster_seed})
    device = str(select_device(cfg.get("device", "auto")))
    res = fit_mix(f_u, f_l, y_l, k=k, n_init=ck["n_init"], max_iter=ck["max_iter"], tol=ck["tol"],
                  seed=cluster_seed, device=device)

    centres = torch.as_tensor(res.centres)
    _, d_l = assign(torch.as_tensor(f_l), centres)
    _, d_u = assign(torch.as_tensor(f_u), centres)
    d_l_own = ((torch.as_tensor(f_l) - centres[res.labels_labelled]) ** 2).sum(1)
    assignments = pd.DataFrame({
        "sample_id": ids_l + ids_u,
        "cluster_id": np.concatenate([res.labels_labelled, res.labels_unlabelled]).astype(int),
        "is_anchor": [True] * len(ids_l) + [False] * len(ids_u),
        "sq_dist_to_centre": np.concatenate([d_l_own.numpy(), d_u.numpy()]),
    })
    assignments.to_csv(out / "assignments.csv", index=False)
    np.save(out / "centres.npy", res.centres)
    sizes = assignments.groupby("cluster_id").agg(n_total=("sample_id", "size"), n_anchor=("is_anchor", "sum"))
    sizes = sizes.reindex(range(k), fill_value=0)
    sizes["n_unlabelled"] = sizes["n_total"] - sizes["n_anchor"]
    sizes["empty"] = sizes["n_total"] == 0
    sizes["singleton"] = sizes["n_total"] == 1
    sizes["empty_unlabelled"] = sizes["n_unlabelled"] == 0
    sizes.index.name = "cluster_id"
    sizes.reset_index().to_csv(out / "cluster_sizes.csv", index=False)

    held = []
    for pool in ("heldout_dev", "heldout_calib"):
        ids_h, f_h = load_features(run_dir, pool)
        if not ids_h:
            continue
        lab_h, dist_h = assign(torch.as_tensor(f_h), centres)
        held.append(pd.DataFrame({"sample_id": ids_h, "pool": pool, "cluster_id": lab_h.numpy(),
                                  "sq_dist_to_centre": dist_h.numpy()}))
    pd.concat(held).to_csv(out / "heldout_assignments.csv", index=False)

    # consistency of the saved state: unlabelled labels are the nearest saved centres
    reassigned, _ = assign(torch.as_tensor(f_u), torch.as_tensor(np.load(out / "centres.npy")))
    consistent = bool((reassigned.numpy() == res.labels_unlabelled).all())
    if not consistent:
        raise RuntimeError("saved centres do not reproduce the saved unlabelled assignments")
    info = {"K": k, "k_mode": k_mode, "known_K_assumption": k_mode == "known", "cluster_seed": cluster_seed,
            "n_init": ck["n_init"], "max_iter": ck["max_iter"], "tol": ck["tol"], "inertia": res.inertia,
            "n_iter": res.n_iter, "converged": res.converged, "empty_events_during_fit": res.empty_events,
            "kpp_degenerate_events": res.kpp_degenerate_events, "final_label_changes": res.final_label_changes,
            "best_init": res.best_init, "init_inertias": res.init_inertias,
            "anchored_clusters": {int(i): int(c) for i, c in enumerate(res.labelled_classes)},
            "anchored_clusters_note": "cluster i < L holds the labelled anchors of class labelled_classes[i]; "
                                      "permitted (labelled) information, never passed to grounding",
            "empty_clusters": int(sizes["empty"].sum()), "singleton_clusters": int(sizes["singleton"].sum()),
            "clusters_without_unlabelled_members": int(sizes["empty_unlabelled"].sum()),
            "saved_state_consistent": consistent,
            "ordering": "fit_mix receives labelled and unlabelled arrays explicitly; results mapped by sample id"}
    write_json(out / "clustering.json", info)
    record.complete(**{k_: info[k_] for k_ in ("K", "inertia", "n_iter", "converged", "empty_clusters")})
    return out


def evaluate_clustering(layout, out: Path) -> dict:
    """Upstream v2 on the unlabelled population (one Hungarian assignment), NMI, ARI, sizes."""
    manifest = Manifest(layout.data)
    a = pd.read_csv(out / "assignments.csv")
    a = a[~a["is_anchor"]]
    y = manifest.labels("eval").loc[a["sample_id"]].to_numpy().astype(int)
    pred = a["cluster_id"].to_numpy().astype(int)
    known = set(manifest.known_class_ids.tolist())
    mask_old = np.array([c in known for c in y])
    all_acc, old_acc, new_acc = upstream.split_cluster_acc_v2(y, pred, mask_old)
    sizes = pd.read_csv(out / "cluster_sizes.csv")
    result = {"population": "unlabelled training images (GCD discovery population)", "n_images": int(len(y)),
              "v2_all": float(all_acc), "v2_old": float(old_acc), "v2_new": float(new_acc),
              "nmi": float(normalized_mutual_info_score(y, pred)), "ari": float(adjusted_rand_score(y, pred)),
              "cluster_size_unlabelled": sizes["n_unlabelled"].describe().round(2).to_dict(),
              "empty_clusters": int(sizes["empty"].sum()), "singleton_clusters": int(sizes["singleton"].sum()),
              "note": "The Hungarian map of v2 is used only here, never for semantic names."}
    write_json(out / "gcd_eval.json", result)
    return result
