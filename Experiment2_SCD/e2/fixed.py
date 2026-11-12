"""Protocol B: naming on Experiment 1's fixed GCD clusters. Mean-based labels versus vote-based labels, forced
naming and UNMAPPED rejection, on Experiment 1's episodes, pools and evaluation population.

Everything except the label rule and its confidence scores is Experiment 1's: rows and memberships
(`rows.csv`), the dev-selected prompt strategy and T*, per-class text embeddings, episodes, the development and
calibration criteria, the image-level evaluation and the cluster bootstrap. Experiment 1's methods are recomputed
from the validated embeddings and checked against its saved metrics.

Label access (as Experiment 1): predictions for all populations are computed and written first (no labels);
the dev pool's labels choose k for the top-k vote; the calibration pool's labels set thresholds; evaluation labels
are read last. Distinct-name assignment is evaluated only on the complete vocabulary (forced naming); every
rejection experiment uses independent cluster decisions (see PROTOCOL.md §B4).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from e1 import grounding as G
from e1 import metrics as M
from e1.ground import (bootstrap, by_size, cluster_diagnostics, complete_vocabulary_accuracy, dev_criterion,
                       episode_counts, paired_summary, summarise)
from e1.manifests import Manifest
from e1.matrices import aggregate, select_members, similarity
from e1.provenance import StageRecord, write_json
from e1.vocab import episode_classes
from e2 import voting as V
from e2.config import Layout, chash, e1_layout
from e2.embed import load_e1_inputs

E1_METHODS = list(G.METHODS)
MEAN_LABEL_METHODS = [m for m in E1_METHODS if not m.startswith("mcm")]


def vote_name(k: int) -> str:
    return "plurality" if k == 1 else f"top{k}vote"


def new_methods(ks) -> list[str]:
    out = []
    for k in ks:
        out += [f"{vote_name(k)}_forced"] + [f"{vote_name(k)}@{s}" for s in V.SCORES]
    return out


def is_forced(method: str) -> bool:
    return method == "forced_top1" or method.endswith("_forced")


def predict_population(inp: dict, R: dict, population: str, strategy: str, T: float, episodes: pd.DataFrame,
                       ks, distinct: dict | None) -> pd.DataFrame:
    rows, K = inp["rows"], inp["K"]
    pop = rows[rows["population"] == population]
    members = select_members(pop, "cluster_id", list(range(K)), "all")
    agg = aggregate(R[strategy], inp["emb"], members)
    agg_mcm = aggregate(R["mcm"], inp["emb"], members)
    idx = np.concatenate(members)
    grp = np.concatenate([np.full(len(m), k) for k, m in enumerate(members)])
    n_pop = pop.groupby("cluster_id").size().reindex(range(K), fill_value=0).to_numpy()
    nonempty = int((n_pop > 0).sum())
    frames = []

    def add(ep, method, pred_col, score, cols, extra=None):
        pc = np.where(pred_col >= 0, cols[np.clip(pred_col, 0, None)], -1)
        f = {"episode_id": ep, "cluster_id": np.arange(K), "method": method, "pred_class_id": pc, "score": score,
             "n_used": agg["n"], "n_population": n_pop}
        f.update(extra or {})
        frames.append(pd.DataFrame(f))

    for e in episodes.itertuples():
        cols = episode_classes(e._asdict())
        for method, (pc, sc) in G.score_clusters(agg["A"], R[strategy][idx], grp, agg_mcm["A"], R["mcm"][idx], cols,
                                                 T).items():
            add(e.episode_id, method, pc, sc, cols)
        for k in ks:
            vr = V.score_votes(agg["A"], R[strategy][idx], grp, cols, k)
            ex = {"v1": vr["v1"], "v2": vr["v2"]}
            add(e.episode_id, f"{vote_name(k)}_forced", vr["pred"], np.where(vr["pred"] >= 0, 1.0, np.nan), cols, ex)
            for s in V.SCORES:
                add(e.episode_id, f"{vote_name(k)}@{s}", vr["pred"], vr["scores"][s], cols, ex)
        # distinct names: complete vocabulary only (the method's valid setting), and only if feasible
        if distinct and len(cols) == R[strategy].shape[1] and len(cols) >= nonempty and population == "discovery":
            pred, info = V.distinct_names(R[strategy][idx], grp, K, cols, distinct["k"], distinct["num_common_vote"],
                                          distinct["num_common_linear"])
            add(e.episode_id, f"distinct_top{distinct['k']}_forced", pred, np.where(pred >= 0, 1.0, np.nan), cols)
            distinct.setdefault("info", {})[e.episode_id] = info
    df = pd.concat(frames, ignore_index=True)
    df.insert(0, "representation", "all")
    df.insert(0, "population", population)
    df["strategy"] = np.where(df["method"].str.startswith("mcm"), "mcm", strategy)
    return df


def forced_paired(counts: pd.DataFrame, episodes: pd.DataFrame, pairs: list, W: np.ndarray) -> dict:
    """Paired cluster-bootstrap differences of forced accuracy (available images), per vocabulary size."""
    shared = episodes[episodes["kind"] == "shared"]
    out = {}
    for size, eps in shared.groupby("size"):
        ids = eps["episode_id"].tolist()
        acc = {}
        for method in {m for p in pairs for m in p}:
            t = counts[(counts["method"] == method) & counts["episode_id"].isin(ids)].sort_values(["episode_id", "cluster_id"])
            if len(t) != len(ids) * W.shape[1]:
                continue
            nc = t["n_correct"].to_numpy(float).reshape(len(ids), -1)
            na = t["n_avail"].to_numpy(float).reshape(len(ids), -1)
            acc[method] = (np.nanmean(nc.sum(1) / na.sum(1)), np.nanmean((W @ nc.T) / (W @ na.T), 1))
        for a, b in pairs:
            if a in acc and b in acc:
                d = acc[a][1] - acc[b][1]
                out[f"{a}-{b}|size={size}"] = {"point": float(acc[a][0] - acc[b][0]),
                                               "ci95": [float(x) for x in np.nanpercentile(d, [2.5, 97.5])],
                                               "share_of_replicates_above_0": float(np.mean(d > 0))}
    return out


def e1_consistency(summary: pd.DataFrame, g: Path) -> dict:
    saved = pd.read_csv(g / "metrics_by_episode.csv")
    saved = saved[saved["representation"] == "all"]
    m = summary.merge(saved, on=["episode_id", "method"], suffixes=("", "_e1"))
    out = {"rows_compared": int(len(m))}
    for col in ("forced_accuracy_available", "aurc", "auroc_label_available", "end_to_end_accuracy@tau_e2e"):
        if col in m and f"{col}_e1" in m:
            d = (m[col] - m[f"{col}_e1"]).abs()
            out[col] = {"max_abs_diff": float(d.max()), "n_diff_gt_1e-6": int((d > 1e-6).sum())}
    return out


def run(cfg: dict, clip_name: str) -> Path:
    fc = cfg["fixed"]
    e1c, e1l = e1_layout(cfg)
    manifest = Manifest(e1l.data)
    inp = load_e1_inputs(cfg, clip_name)
    seed_tag = f"seed{cfg['e1']['train_seed']}_{cfg['e1']['k_mode']}_c{cfg['e1']['cluster_seed']}"
    out = Layout(cfg).fixed(clip_name, seed_tag)
    rec = StageRecord(out, "fixed_cluster", chash(fc, cfg["e1"]),
                      {"e1_grounding": str(inp["dir"]), "embeddings": str(inp["embedding_dir"]), "clip": clip_name})
    strategy, T = inp["selection"]["prompt_strategy"], inp["selection"]["T_dev"]
    R = {s: similarity(inp["emb"], inp["class_emb"][s]) for s in ("single", "ensemble80", "mcm")}
    K = int(inp["rows"]["cluster_id"].max()) + 1
    inp["K"] = K
    episodes = inp["episodes"]
    shared = episodes[episodes["kind"] == "shared"]
    ks = fc["vote_topk_grid"]
    distinct = {"k": fc["official_topk"], **fc["distinct"]}

    # 1. predictions, every population (no labels)
    preds = [predict_population(inp, R, pop, strategy, T, episodes if pop == "discovery" else shared, ks,
                                distinct if pop == "discovery" else None) for pop in ("dev", "calib", "discovery")]
    pred = pd.concat(preds, ignore_index=True)
    pred.to_csv(out / "cluster_predictions.csv.gz", index=False)
    write_json(out / "distinct_assignment_info.json", distinct.get("info", {}))

    # 2. development choice: k of the top-k vote (dev labels only)
    y_held = manifest.labels("heldout")
    dev = pred[pred["population"] == "dev"]
    dev_counts = episode_counts(dev, inp["rows"], "dev", y_held, shared, K)
    crit = {k: dev_criterion(dev_counts, shared, f"{vote_name(k)}@win_fraction") for k in ks}
    k_star = min(crit, key=crit.get)
    selection = {"k_star": int(k_star), "criterion": "dev AURC of <vote>@win_fraction (mean over sizes of mean over "
                 "shared episodes), lower is better", "values": {str(k): v for k, v in crit.items()},
                 "official_k": fc["official_topk"], "labels_used": "labels_heldout.csv, heldout_dev rows only",
                 "inherited_from_experiment1": {"prompt_strategy": strategy, "T_dev": T}}
    write_json(out / "selection.json", selection)

    # 3. calibration thresholds (calibration labels only)
    cal = pred[pred["population"] == "calib"]
    cal_counts = episode_counts(cal, inp["rows"], "calib", y_held, shared, K)
    thresholds = {}
    for m, t in cal_counts.groupby("method"):
        if is_forced(m):
            thresholds[m] = {"tau_e2e": -np.inf, "tau_95": -np.inf, "note": "forced naming: always accepts"}
        else:
            thresholds[m] = M.choose_thresholds(t["score"].to_numpy(float), t["n"].to_numpy(float),
                                                t["n_avail"].to_numpy(float), t["n_correct"].to_numpy(float))
    for m in pred["method"].unique():
        thresholds.setdefault(m, {"tau_e2e": -np.inf, "tau_95": -np.inf, "note": "forced naming: always accepts"})
    write_json(out / "thresholds.json", {"rule": "Experiment 1 PROTOCOL.md §7", "labels_used":
                                         "labels_heldout.csv, heldout_calib rows only", "by_method": thresholds})

    # 4. evaluation on the discovery population (evaluation labels)
    y_eval = manifest.labels("eval")
    disc = pred[pred["population"] == "discovery"]
    known = set(manifest.known_class_ids.tolist())
    counts = episode_counts(disc, inp["rows"], "discovery", y_eval, episodes, K, known=known)
    counts.to_csv(out / "cluster_counts.csv.gz", index=False)
    summary = summarise(counts, thresholds)
    summary.to_csv(out / "metrics_by_episode.csv", index=False)
    by_size(summary, episodes).to_csv(out / "metrics_by_vocabulary_size.csv", index=False)
    full_id = episodes.iloc[0]["episode_id"]
    write_json(out / "complete_vocabulary_accuracy.json",
               complete_vocabulary_accuracy(counts[counts["episode_id"] == full_id], inp["rows"], y_eval, manifest))
    cluster_diagnostics(inp["rows"], y_eval, manifest, K).to_csv(out / "cluster_diagnostics.csv", index=False)
    paired_summary(counts, episodes, inp["rows"], y_eval, thresholds).to_csv(out / "paired_diagnostic.csv", index=False)
    curves = []
    for (ep, method), t in counts[counts["episode_id"].isin(shared["episode_id"])].groupby(["episode_id", "method"]):
        rc = M.risk_coverage(t["score"].to_numpy(float), t["n"].to_numpy(float), t["n_correct"].to_numpy(float))
        curves.append(pd.DataFrame({"episode_id": ep, "method": method, "threshold": rc["threshold"],
                                    "coverage": rc["coverage"], "risk": rc["risk"]}))
    pd.concat(curves, ignore_index=True).to_csv(out / "risk_coverage_curves.csv.gz", index=False)
    gcfg = {"n_bootstrap": fc["n_bootstrap"], "bootstrap_seed": fc["bootstrap_seed"],
            "reference_method": fc["reference_method"]}
    write_json(out / "bootstrap.json", bootstrap(counts, episodes, thresholds, gcfg))
    W = M.bootstrap_weights(K, fc["n_bootstrap"], fc["bootstrap_seed"])
    pairs = [(f"{vote_name(k)}_forced", "forced_top1") for k in ks] + \
            [(f"distinct_top{fc['official_topk']}_forced", "forced_top1"),
             (f"distinct_top{fc['official_topk']}_forced", f"{vote_name(fc['official_topk'])}_forced")]
    write_json(out / "forced_paired.json", forced_paired(counts, episodes, pairs, W))
    write_json(out / "e1_consistency.json", e1_consistency(summary[summary["method"].isin(E1_METHODS)], inp["dir"]))

    # 5. errors, failure patterns and galleries
    from e2 import errors

    errors.fixed_cluster_errors(cfg, out, inp, R[strategy], pred, counts, thresholds, episodes, y_eval, manifest,
                                k_star)
    rec.complete(k_star=int(k_star), n_methods=int(pred["method"].nunique()))
    return out
