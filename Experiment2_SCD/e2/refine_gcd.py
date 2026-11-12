"""Protocol R: the full SCD refinement started from Experiment 1's GCD clusters (CUB, membership-changing).

Not a paper setting. Differences from the paper's CUB run: a closed vocabulary of the 200 CUB names (the paper
used ~11k Wikipedia bird names, whose name list is not distributed), Experiment 1's GCD features and SS-k-means
(not the authors' GCD checkpoint and CSS-k-means), CLIP ViT-B/32 with Experiment 1's ensemble80 prompts (the paper
used ViT-B/16). With |V| = K = 200 the distinct-name constraint is feasible.

Modes: `unsupervised` (all 200 clusters are named from votes) and `partially_supervised` (the 100 known names are
permitted, from the labelled anchors' classes; the 100 clusters holding anchors keep them; 100 slots are voted).

Because refinement changes memberships, Experiment 1's cluster bootstrap does not apply. Uncertainty: (i) variation
over Experiment 1's three clustering seeds, (ii) a paired image bootstrap of final versus initial sACC, conditional
on each run (it ignores clustering randomness).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from e1.manifests import Manifest
from e1.provenance import StageRecord, save_npz_atomic, write_json
from e2.config import Layout, chash, e1_layout
from e2.embed import load_e1_inputs

from scd.metrics import cluster_acc_v2, name_iou, semantic_accuracy
from scd.naming import NamingConfig, image_topk, initial_cluster_names, refine
from scd.vocabulary import Vocabulary


def naming_config(cfg: dict, **over) -> NamingConfig:
    n = {k: v for k, v in cfg["naming"].items() if k in NamingConfig.__dataclass_fields__}
    n.update(over)
    return NamingConfig(**n)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    d = a[idx].mean(1) - b[idx].mean(1)
    return {"point": float(a.mean() - b.mean()), "ci95": [float(x) for x in np.percentile(d, [2.5, 97.5])],
            "unit": "image (conditional on the run)"}


def run(cfg: dict, clip_name: str, cluster_seed: int, mode: str) -> dict:
    rc = cfg["refine_on_gcd"]
    e1c, e1l = e1_layout(cfg)
    e = cfg["e1"]
    manifest = Manifest(e1l.data)
    inp = load_e1_inputs(cfg, clip_name)              # embeddings, text embeddings, columns (clustering seed 0 run)
    cdir = e1l.clustering(e["train_seed"], e["k_mode"], cluster_seed)
    a = pd.read_csv(cdir / "assignments.csv")
    K = json.loads((cdir / "clustering.json").read_text())["K"]
    seed_tag = f"seed{e['train_seed']}_{e['k_mode']}_c{cluster_seed}"
    out = Layout(cfg).refine_on_gcd(clip_name, seed_tag, mode)
    rec = StageRecord(out, "refine_on_gcd", chash(cfg["naming"], rc), {"clustering": str(cdir), "mode": mode})
    emb_all = dict(zip(inp["rows"]["sample_id"], inp["emb"]))
    unl = a[~a["is_anchor"]].reset_index(drop=True)
    X = np.stack([emb_all[s] for s in unl["sample_id"]])
    classes = manifest.classes.sort_values("class_id")
    if not np.array_equal(classes["class_id"].to_numpy(), np.asarray(inp["class_ids"])):
        raise RuntimeError("text columns are not in class-id order")
    vocab = Vocabulary(classes["canonical_name"].tolist(), source="CUB-200 canonical names (Experiment 1 columns)")
    if len(set(vocab.names)) != len(vocab):
        raise RuntimeError("CUB names collide after normalisation")
    T = inp["class_emb"][inp["selection"]["prompt_strategy"]].astype(np.float32)
    ncfg = naming_config(cfg, known_filter=rc["known_filter"])
    topk, _, tinfo = image_topk(X, T, cfg["naming"]["topk_store"], logit_dtype=ncfg_dtype(cfg))
    init = unl["cluster_id"].to_numpy()
    kw = {}
    lab_names = {}
    if mode == "partially_supervised":
        anchors = a[a["is_anchor"]]
        y_lab = manifest.labels("labelled")
        known_ids = sorted(set(y_lab.loc[anchors["sample_id"]].tolist()))
        known_names = [vocab.names[c] for c in known_ids]
        lab_clusters = sorted(set(anchors["cluster_id"].tolist()))
        # labelled cluster -> its anchors' class name (permitted: labelled table)
        for c in lab_clusters:
            ys = y_lab.loc[anchors.loc[anchors["cluster_id"] == c, "sample_id"]].unique()
            if len(ys) != 1:
                raise RuntimeError(f"cluster {c} holds anchors of several classes")
            lab_names[c] = vocab.names[int(ys[0])]
        kw = {"known_names": known_names, "labelled_clusters": lab_clusters, "cluster_ids": sorted(set(a["cluster_id"]))}
    res = refine(X, T, vocab, topk, init, ncfg, n_cluster=K, **kw)

    # ---- trace (no labels)
    its = res["iterations"]
    save_npz_atomic(out / "labels_by_iteration.npz", sample_ids=unl["sample_id"].to_numpy().astype(str), initial=init,
                    **{f"iter{r['iteration']}": r["labels_after"] for r in its})
    trace = [{k: v for k, v in r.items() if k not in ("labels_before", "labels_after")} for r in its]
    write_json(out / "trace.json", {"stop_reason": res["stop_reason"], "n_iterations": res["n_iterations"],
                                    "config": res["config"], "num_slots": res["num_slots"], "topk": tinfo,
                                    "iterations": trace})

    # ---- evaluation (labels_eval)
    y = manifest.labels("eval").loc[unl["sample_id"]].to_numpy()
    known = set(manifest.known_class_ids.tolist())
    old = np.isin(y, list(known))
    true_names = np.array([vocab.names[c] for c in y], dtype=object)
    init_names = initial_cluster_names(res, lab_names)
    pred_init = np.array([init_names.get(int(c), "") for c in init], dtype=object)

    def evaluate(pred_names, clusters):
        r = {"acc": cluster_acc_v2(y, clusters, old)}
        for tag, m in (("all", slice(None)), ("old", old), ("new", ~old)):
            r[f"sacc_{tag}"] = semantic_accuracy(pred_names[m], true_names[m])
        r["n_selected_names"] = int(len(set(pred_names) - {""}))
        r["iou_names"] = name_iou(set(pred_names) - {""}, set(true_names))
        return r

    metrics = {"initial_clustering_named_by_iteration1_votes": evaluate(pred_init, init)}
    per_iter = []
    correct_by_iter = []
    for r in its:
        names = np.array(r["selected_names"], dtype=object)[r["labels_after"]]
        m = evaluate(names, r["labels_after"])
        m["iteration"] = r["iteration"]
        per_iter.append(m)
        correct_by_iter.append(names == true_names)
    metrics["by_iteration"] = per_iter
    metrics["final"] = per_iter[-1]
    c0 = pred_init == true_names
    c1, cf = correct_by_iter[0], correct_by_iter[-1]
    metrics["P4_refinement_flips"] = {
        "initial_named_correct_to_final_wrong": int((c0 & ~cf).sum()),
        "initial_named_wrong_to_final_correct": int((~c0 & cf).sum()),
        "after_iteration1_correct_to_final_wrong": int((c1 & ~cf).sum()),
        "after_iteration1_wrong_to_final_correct": int((~c1 & cf).sum()), "n_images": int(len(y))}
    metrics["paired_final_minus_initial_sacc"] = paired_bootstrap(cf.astype(float), c0.astype(float),
                                                                  rc["n_bootstrap"], rc["bootstrap_seed"])
    metrics["stop_reason"] = res["stop_reason"]
    metrics["n_iterations"] = res["n_iterations"]
    metrics["fidelity"] = "non-paper setting (closed 200-name vocabulary, Experiment 1 GCD clusters, ViT-B/32)"
    write_json(out / "metrics.json", metrics)

    # ---- per-image predictions and every incorrect final decision
    names_final = np.array(its[-1]["selected_names"], dtype=object)[its[-1]["labels_after"]]
    rel = dict(zip(manifest.table["sample_id"], manifest.table["rel_path"]))
    df = pd.DataFrame({"sample_id": unl["sample_id"], "image_path": [rel[s] for s in unl["sample_id"]],
                       "initial_cluster": init, "initial_name": pred_init, "final_position": its[-1]["labels_after"],
                       "final_name": names_final, "true_class_id": y, "true_name": true_names, "true_old": old,
                       "initial_correct": c0, "iter1_correct": c1, "final_correct": cf})
    for t, ok in enumerate(correct_by_iter, 1):
        df[f"iter{t}_correct"] = ok
    df.to_csv(out / "image_predictions.csv.gz", index=False)
    df[~df["final_correct"]].to_csv(out / "incorrect_final_decisions.csv.gz", index=False)
    _flip_gallery(cfg, out, df, manifest)
    rec.complete(stop_reason=res["stop_reason"], n_iterations=res["n_iterations"],
                 final_sacc=metrics["final"]["sacc_all"]["image"], initial_sacc=
                 metrics["initial_clustering_named_by_iteration1_votes"]["sacc_all"]["image"])
    return {"dir": str(out), "stop_reason": res["stop_reason"], "n_iterations": res["n_iterations"],
            "initial_sacc": metrics["initial_clustering_named_by_iteration1_votes"]["sacc_all"]["image"],
            "final_sacc": metrics["final"]["sacc_all"]["image"], "final_acc": metrics["final"]["acc"]["all"]}


def ncfg_dtype(cfg: dict) -> str:
    return cfg["naming"].get("logit_dtype", "float32")


def _flip_gallery(cfg, out, df: pd.DataFrame, manifest):
    from e2.embed import image_root
    from e2.errors import cluster_gallery

    root = image_root(cfg, manifest)
    flips = df[df["initial_correct"] & ~df["final_correct"]]
    panels = []
    groups = sorted(flips.groupby(["initial_cluster", "initial_name"]), key=lambda kv: -len(kv[1]))[:8]
    for (c, n0), grp in groups:
        rows = grp.head(8)
        panels.append({"title": f"initial cluster {c} named '{n0}' (correct for these images) -> final names "
                                f"{', '.join(sorted(set(rows['final_name'])))[:90]} · {len(grp)} images flipped",
                       "files": [root / p for p in rows["image_path"]],
                       "captions": [f"{t} -> {f}" for t, f in zip(rows["true_name"], rows["final_name"])]})
    if panels:
        cluster_gallery(out / "galleries" / "P4_refinement_flips.png", panels,
                        "P4: images whose initial-cluster name was correct and whose final name is wrong")
