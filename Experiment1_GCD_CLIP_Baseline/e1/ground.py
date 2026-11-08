"""Grounding stage: matrices, episodes, predictions, dev choices, calibration, evaluation.

Label access by step (PROTOCOL.md §1, §7):
  1-3 inputs, matrices, episodes and all predictions/scores ............ no labels
  4   prompt strategy and T* on the dev pool ........................... labels_heldout (dev rows)
  5   thresholds on the calibration pool ............................... labels_heldout (calib rows)
  6   evaluation on the discovery population ........................... labels_eval
Predictions are written to disk before step 6 reads any evaluation label.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from e1 import grounding as G
from e1 import metrics as M
from e1.clip_extract import encode_texts, load_images
from e1.clip_model import load_clip
from e1.config import section_hash
from e1.gcd_train import select_device
from e1.manifests import Manifest
from e1.matrices import aggregate, select_members, similarity
from e1.provenance import StageRecord, save_npz_atomic, write_json
from e1.vocab import class_embeddings, class_texts, episode_classes, paired_episodes, shared_episodes

STRATEGIES = ("single", "ensemble80", "mcm")


# ------------------------------------------------------------------ 1. inputs (no labels)
def build_inputs(cfg: dict, layout, train_seed: int, k_mode: str, cluster_seed: int, clip_name: str,
                 out: Path) -> dict:
    manifest = Manifest(layout.data)
    cdir = layout.clustering(train_seed, k_mode, cluster_seed)
    K = json.loads((cdir / "clustering.json").read_text())["K"]
    a = pd.read_csv(cdir / "assignments.csv")
    h = pd.read_csv(cdir / "heldout_assignments.csv")
    rows = pd.concat([
        a[~a["is_anchor"]].assign(population="discovery"),
        a[a["is_anchor"]].assign(population="anchor"),
        h[h["pool"] == "heldout_dev"].assign(population="dev", is_anchor=False),
        h[h["pool"] == "heldout_calib"].assign(population="calib", is_anchor=False),
    ], ignore_index=True)[["sample_id", "population", "cluster_id", "is_anchor", "sq_dist_to_centre"]]
    rows.insert(0, "row", np.arange(len(rows)))
    emb = load_images(layout.clip(clip_name), rows["sample_id"].tolist())
    classes = manifest.classes
    class_ids = classes["class_id"].to_numpy()
    device = select_device(cfg.get("device", "auto"))
    model, _, tokenizer = load_clip(cfg["clip"]["models"][clip_name], device)
    R, cemb = {}, {}
    for s in STRATEGIES:
        texts = class_texts(classes, s)
        temb = encode_texts(model, tokenizer, texts["text"].tolist(), device)
        cemb[s] = class_embeddings(texts, temb, class_ids)
        R[s] = similarity(emb, cemb[s])
        texts.to_csv(out / f"texts_{s}.csv", index=False)
        save_npz_atomic(out / f"text_embeddings_{s}.npz", per_text=temb, per_class=cemb[s], class_ids=class_ids)
        np.save(out / f"R_{s}.npy", R[s])
    rows.to_csv(out / "rows.csv", index=False)
    classes[["class_id", "canonical_name", "aliases", "gcd_status"]].to_csv(out / "columns.csv", index=False)
    return {"rows": rows, "emb": emb, "R": R, "class_emb": cemb, "K": K, "classes": classes,
            "class_ids": class_ids}


# ------------------------------------------------------------------ 2-3. episodes and predictions (no labels)
def make_episodes(cfg: dict, inputs: dict) -> pd.DataFrame:
    ep = cfg["episodes"]
    shared = shared_episodes(inputs["class_ids"], ep["sizes"], ep["n_vocab_seeds"])
    parts = [shared]
    if ep.get("paired", True):
        parts.append(paired_episodes(inputs["class_ids"], inputs["class_emb"]["single"],
                                     n_distractors=ep["paired_distractors"], seed=ep["paired_seed"]))
    return pd.concat(parts, ignore_index=True)


def predict(inputs: dict, population: str, representation: str, strategy: str, T_dev: float,
            episodes: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rows, K = inputs["rows"], inputs["K"]
    pop = rows[rows["population"] == population]
    members = select_members(pop, "cluster_id", list(range(K)), representation, seed)
    agg = aggregate(inputs["R"][strategy], inputs["emb"], members)
    agg_mcm = aggregate(inputs["R"]["mcm"], inputs["emb"], members)
    idx = np.concatenate(members) if members else np.zeros(0, int)
    grp = np.concatenate([np.full(len(m), k) for k, m in enumerate(members)]) if members else np.zeros(0, int)
    n_pop = pop.groupby("cluster_id").size().reindex(range(K), fill_value=0).to_numpy()
    out = []
    for e in episodes.itertuples():
        cols = episode_classes(e._asdict())
        sc = G.score_clusters(agg["A"], inputs["R"][strategy][idx], grp, agg_mcm["A"], inputs["R"]["mcm"][idx],
                              cols, T_dev)
        for method, (pred_col, score) in sc.items():
            pred_class = np.where(pred_col >= 0, cols[np.clip(pred_col, 0, None)], -1)
            out.append(pd.DataFrame({"episode_id": e.episode_id, "cluster_id": np.arange(K), "method": method,
                                     "pred_class_id": pred_class, "score": score, "n_used": agg["n"],
                                     "n_population": n_pop}))
    df = pd.concat(out, ignore_index=True)
    df.insert(0, "representation", representation)
    df.insert(0, "population", population)
    df["strategy"] = np.where(df["method"].str.startswith("mcm"), "mcm", strategy)
    return df


# ------------------------------------------------------------------ evaluation helpers (labels passed in)
def episode_counts(pred: pd.DataFrame, rows: pd.DataFrame, population: str, y: pd.Series, episodes: pd.DataFrame,
                   K: int, known: set | None = None) -> pd.DataFrame:
    """Per (episode, method, cluster) counts. `y`: sample_id -> class id for this population.
    With `known`, the counts are also split into Old (known-class) and New images."""
    pop = rows[rows["population"] == population]
    g = pop["cluster_id"].to_numpy()
    y_img = y.loc[pop["sample_id"]].to_numpy()
    old = np.array([v in known for v in y_img]) if known is not None else None
    vocab = {e.episode_id: set(episode_classes(e._asdict()).tolist()) for e in episodes.itertuples()}
    out = []
    for (ep_id, method), t in pred.groupby(["episode_id", "method"], sort=False):
        t = t.sort_values("cluster_id")
        pc = t["pred_class_id"].to_numpy()
        c = M.cluster_counts(g, y_img, pc, vocab[ep_id], K)
        frame = {"episode_id": ep_id, "method": method, "cluster_id": np.arange(K), "score": t["score"].to_numpy(),
                 "pred_class_id": pc, "n": c["n"], "n_avail": c["n_avail"], "n_correct": c["n_correct"]}
        if old is not None:
            for tag, mask in (("old", old), ("new", ~old)):
                cs = M.cluster_counts(g[mask], y_img[mask], pc, vocab[ep_id], K)
                frame.update({f"n_{tag}": cs["n"], f"n_avail_{tag}": cs["n_avail"], f"n_correct_{tag}": cs["n_correct"]})
        out.append(pd.DataFrame(frame))
    return pd.concat(out, ignore_index=True)


def summarise(counts: pd.DataFrame, thresholds: dict | None) -> pd.DataFrame:
    rows = []
    for (ep_id, method), t in counts.groupby(["episode_id", "method"], sort=False):
        s, n, na, nc = (t[c].to_numpy(dtype=float) for c in ("score", "n", "n_avail", "n_correct"))
        r = {"episode_id": ep_id, "method": method, "n_images": n.sum(), "n_available": na.sum(),
             "forced_accuracy_available": nc.sum() / na.sum() if na.sum() else np.nan,
             "aurc": M.risk_coverage(s, n, nc)["aurc"],
             "auroc_label_available": M.auroc_clusters(s, na, n - na),
             "auroc_available_and_correct": M.auroc_clusters(s, nc, n - nc)}
        for name in ("tau_e2e", "tau_95"):
            if thresholds and method in thresholds:
                for k, v in M.at_threshold(s, n, na, nc, thresholds[method][name]).items():
                    r[f"{k}@{name}"] = v
        for tag in ("old", "new"):
            if f"n_{tag}" in t:
                n_t, na_t, nc_t = (t[f"{c}_{tag}"].to_numpy(dtype=float) for c in ("n", "n_avail", "n_correct"))
                r[f"forced_accuracy_available_{tag}"] = nc_t.sum() / na_t.sum() if na_t.sum() else np.nan
                if thresholds and method in thresholds:
                    r[f"end_to_end_accuracy_{tag}@tau_e2e"] = M.at_threshold(
                        s, n_t, na_t, nc_t, thresholds[method]["tau_e2e"])["end_to_end_accuracy"]
        rows.append(r)
    return pd.DataFrame(rows)


def by_size(summary: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    s = summary.merge(episodes[["episode_id", "kind", "size"]], on="episode_id")
    s = s[s["kind"] == "shared"]
    num = s.drop(columns=["episode_id", "kind"]).groupby(["method", "size"])
    mean = num.mean(numeric_only=True).add_suffix("_mean")
    sd = num.std(numeric_only=True).add_suffix("_sd")
    return pd.concat([mean, sd, num.size().rename("n_episodes")], axis=1).reset_index()


def dev_criterion(counts: pd.DataFrame, episodes: pd.DataFrame, method: str) -> float:
    """Mean over vocabulary sizes of the mean AURC of that size's shared episodes."""
    t = counts[counts["method"] == method].merge(episodes[["episode_id", "kind", "size"]], on="episode_id")
    t = t[t["kind"] == "shared"]
    per_ep = {ep: M.risk_coverage(g["score"].to_numpy(float), g["n"].to_numpy(float), g["n_correct"].to_numpy(float))["aurc"]
              for ep, g in t.groupby("episode_id")}
    sizes = t.drop_duplicates("episode_id").set_index("episode_id")["size"]
    per_size = pd.Series(per_ep).groupby(sizes).mean()
    return float(per_size.mean())


# ------------------------------------------------------------------ the stage
def run(cfg: dict, layout, train_seed: int, k_mode: str, cluster_seed: int, clip_name: str) -> Path:
    gcfg = cfg["grounding"]
    out = layout.grounding(train_seed, k_mode, cluster_seed, clip_name)
    out.mkdir(parents=True, exist_ok=True)
    record = StageRecord(out, "grounding", section_hash(cfg, "vocabulary", "episodes", "grounding"),
                         {"train_seed": train_seed, "k_mode": k_mode, "cluster_seed": cluster_seed, "clip": clip_name})
    manifest = Manifest(layout.data)

    # 1-2. inputs, matrices, episodes (no labels)
    inputs = build_inputs(cfg, layout, train_seed, k_mode, cluster_seed, clip_name, out)
    episodes = make_episodes(cfg, inputs)
    episodes.to_csv(out / "episodes.csv", index=False)
    shared = episodes[episodes["kind"] == "shared"]
    reps = gcfg["representations"]
    K = inputs["K"]

    # 4. development choices (dev labels only)
    y_held = manifest.labels("heldout")
    dev_pred = {s: predict(inputs, "dev", "all", s, 1.0, shared) for s in gcfg["prompt_candidates"]}
    crit = {s: dev_criterion(episode_counts(p, inputs["rows"], "dev", y_held, shared, K), shared, "max_mean_sim")
            for s, p in dev_pred.items()}
    strategy = min(crit, key=crit.get)
    crit_T = {}
    for T in gcfg["temperature_grid"]:
        p = predict(inputs, "dev", "all", strategy, T, shared)
        crit_T[T] = dev_criterion(episode_counts(p, inputs["rows"], "dev", y_held, shared, K), shared, "msp_dev")
    T_dev = min(crit_T, key=crit_T.get)
    selection = {"criterion": "dev AURC (mean over vocabulary sizes of mean over shared episodes)",
                 "prompt_strategy": strategy, "prompt_criterion": crit, "method_for_prompt_choice": "max_mean_sim",
                 "T_dev": T_dev, "temperature_criterion": {str(k): v for k, v in crit_T.items()},
                 "pilot_T": G.PILOT_T, "mcm_T": G.MCM_T, "labels_used": "labels_heldout.csv, heldout_dev rows only"}
    write_json(out / "selection.json", selection)

    # 3. predictions for every population, representation and episode with the frozen choices (no labels)
    preds = []
    for population in ("dev", "calib", "discovery"):
        for rep in reps:
            eps = episodes if (population == "discovery" and rep == reps[0]) else shared
            preds.append(predict(inputs, population, rep, strategy, T_dev, eps, seed=gcfg["random_seed"]))
    pred = pd.concat(preds, ignore_index=True)
    pred.to_csv(out / "cluster_predictions.csv.gz", index=False)

    # 5. calibration thresholds (calibration labels only)
    thresholds = {}
    for rep in reps:
        c = episode_counts(pred[(pred["population"] == "calib") & (pred["representation"] == rep)], inputs["rows"],
                           "calib", y_held, shared, K)
        thresholds[rep] = {m: M.choose_thresholds(t["score"].to_numpy(float), t["n"].to_numpy(float),
                                                  t["n_avail"].to_numpy(float), t["n_correct"].to_numpy(float))
                           for m, t in c.groupby("method") if m != "forced_top1"}
        thresholds[rep]["forced_top1"] = {"tau_e2e": -np.inf, "tau_95": -np.inf,
                                          "note": "forced labelling: always accepts, never calibrated"}
    write_json(out / "thresholds.json", {"rule": "PROTOCOL.md §7", "labels_used": "labels_heldout.csv, heldout_calib rows only",
                                         "by_representation": thresholds})
    write_image_predictions(out, pred, inputs, thresholds, reps[0], episodes)
    record_inputs = {"selection": selection, "n_rows": len(inputs["rows"]), "K": K}

    # 6. evaluation on the discovery population (evaluation labels)
    y_eval = manifest.labels("eval")
    evaluate(out, pred, inputs, episodes, thresholds, y_eval, manifest, gcfg, reps)
    export_episode_matrices(out, inputs, episodes, strategy)
    record.complete(**record_inputs)
    return out


def write_image_predictions(out: Path, pred: pd.DataFrame, inputs: dict, thresholds: dict, rep: str,
                            episodes: pd.DataFrame) -> None:
    """Image-level predictions (complete vocabulary, primary representation); no labels inside."""
    full = episodes.iloc[0]["episode_id"]
    p = pred[(pred["population"] == "discovery") & (pred["representation"] == rep) & (pred["episode_id"] == full)]
    rows = inputs["rows"][inputs["rows"]["population"] == "discovery"][["sample_id", "cluster_id"]]
    img = rows.merge(p[["cluster_id", "method", "pred_class_id", "score"]], on="cluster_id")
    tau = {m: v for m, v in thresholds[rep].items()}
    img["accept_tau_e2e"] = [s >= tau[m]["tau_e2e"] for s, m in zip(img["score"], img["method"])]
    img["accept_tau_95"] = [s >= tau[m]["tau_95"] for s, m in zip(img["score"], img["method"])]
    img["episode_id"] = full
    img.to_csv(out / "image_predictions_complete_vocabulary.csv.gz", index=False)


def evaluate(out, pred, inputs, episodes, thresholds, y_eval, manifest: Manifest, gcfg, reps) -> None:
    K, rows = inputs["K"], inputs["rows"]
    disc = pred[pred["population"] == "discovery"]
    tables = []
    counts_by_rep = {}
    for rep in reps:
        eps = episodes if rep == reps[0] else episodes[episodes["kind"] == "shared"]
        c = episode_counts(disc[disc["representation"] == rep], rows, "discovery", y_eval, eps, K,
                           known=set(manifest.known_class_ids.tolist()))
        counts_by_rep[rep] = c
        s = summarise(c, thresholds[rep])
        s.insert(0, "representation", rep)
        tables.append(s)
    summary = pd.concat(tables, ignore_index=True)
    summary.to_csv(out / "metrics_by_episode.csv", index=False)
    sized = pd.concat([by_size(t.drop(columns="representation"), episodes).assign(representation=r)
                       for r, t in summary.groupby("representation")], ignore_index=True)
    sized.to_csv(out / "metrics_by_vocabulary_size.csv", index=False)

    primary = counts_by_rep[reps[0]]
    curves = []
    for (ep_id, method), t in primary[primary["episode_id"].isin(episodes.loc[episodes["kind"] == "shared",
                                                                              "episode_id"])].groupby(["episode_id", "method"]):
        rc = M.risk_coverage(t["score"].to_numpy(float), t["n"].to_numpy(float), t["n_correct"].to_numpy(float))
        curves.append(pd.DataFrame({"episode_id": ep_id, "method": method, "threshold": rc["threshold"],
                                    "coverage": rc["coverage"], "risk": rc["risk"]}))
    pd.concat(curves, ignore_index=True).to_csv(out / "risk_coverage_curves.csv", index=False)
    full_id = episodes.iloc[0]["episode_id"]
    complete = complete_vocabulary_accuracy(primary[primary["episode_id"] == full_id], rows, y_eval, manifest)
    write_json(out / "complete_vocabulary_accuracy.json", complete)
    cluster_diagnostics(rows, y_eval, manifest, K).to_csv(out / "cluster_diagnostics.csv", index=False)
    paired_summary(primary, episodes, rows, y_eval, thresholds[reps[0]]).to_csv(out / "paired_diagnostic.csv",
                                                                                index=False)
    write_json(out / "bootstrap.json", bootstrap(primary, episodes, thresholds[reps[0]], gcfg))
    write_json(out / "failure_cases.json", failure_cases(primary, episodes, rows, y_eval, manifest,
                                                         thresholds[reps[0]], K))


def complete_vocabulary_accuracy(c: pd.DataFrame, rows, y_eval, manifest: Manifest) -> dict:
    """Forced semantic accuracy with every class in V: image-weighted, class-macro, Old/New."""
    pop = rows[rows["population"] == "discovery"]
    y = y_eval.loc[pop["sample_id"]].to_numpy()
    known = set(manifest.known_class_ids.tolist())
    out = {}
    for method, t in c.groupby("method"):
        pred_by_cluster = t.set_index("cluster_id")["pred_class_id"]
        p = pred_by_cluster.loc[pop["cluster_id"]].to_numpy()
        ok = p == y
        old = np.array([v in known for v in y])
        per_class = pd.Series(ok).groupby(y).mean()
        out[method] = {"image_weighted": float(ok.mean()), "class_macro": float(per_class.mean()),
                       "old": float(ok[old].mean()) if old.any() else None,
                       "new": float(ok[~old].mean()) if (~old).any() else None,
                       "n_images": int(len(y)), "n_classes_present": int(len(per_class))}
    return out


def cluster_diagnostics(rows, y_eval, manifest: Manifest, K: int) -> pd.DataFrame:
    pop = rows[rows["population"] == "discovery"]
    y = y_eval.loc[pop["sample_id"]].to_numpy()
    names = manifest.classes.set_index("class_id")["canonical_name"]
    known = set(manifest.known_class_ids.tolist())
    out = []
    for k in range(K):
        yk = y[pop["cluster_id"].to_numpy() == k]
        if len(yk) == 0:
            out.append({"cluster_id": k, "n_unlabelled": 0, "purity": np.nan, "dominant_class_id": -1,
                        "dominant_name": "", "strict_majority": False, "mixed": False, "known_fraction": np.nan,
                        "composition_top3": ""})
            continue
        vc = pd.Series(yk).value_counts()
        top = vc.index[0]
        out.append({"cluster_id": k, "n_unlabelled": len(yk), "purity": vc.iloc[0] / len(yk),
                    "dominant_class_id": int(top), "dominant_name": names[top],
                    "strict_majority": bool(vc.iloc[0] > len(yk) / 2), "mixed": bool(vc.iloc[0] <= len(yk) / 2),
                    "known_fraction": float(np.mean([v in known for v in yk])),
                    "composition_top3": "; ".join(f"{names[c]}:{n}" for c, n in vc.head(3).items())})
    return pd.DataFrame(out)


def paired_summary(c, episodes, rows, y_eval, thresholds) -> pd.DataFrame:
    pe = episodes[episodes["kind"].str.startswith("paired")]
    if pe.empty:
        return pd.DataFrame()
    pop = rows[rows["population"] == "discovery"]
    y = y_eval.loc[pop["sample_id"]].to_numpy()
    g = pop["cluster_id"].to_numpy()
    out = []
    for e in pe.itertuples():
        mask = y == e.target_class_id
        if not mask.any():
            continue
        for method, t in c[c["episode_id"] == e.episode_id].groupby("method"):
            t = t.set_index("cluster_id")
            pc = t["pred_class_id"].to_numpy()[g[mask]]
            sc = t["score"].to_numpy()[g[mask]]
            acc = sc >= thresholds[method]["tau_e2e"]
            out.append({"episode_id": e.episode_id, "variant": e.kind, "target_class_id": e.target_class_id,
                        "method": method, "n_target_images": int(mask.sum()),
                        "named_target": float(np.mean(pc == e.target_class_id)),
                        "named_replacement": float(np.mean(pc == e.replacement_class_id)) if e.replacement_class_id >= 0 else np.nan,
                        "accepted@tau_e2e": float(acc.mean()),
                        "accepted_and_named_target@tau_e2e": float(np.mean(acc & (pc == e.target_class_id)))})
    df = pd.DataFrame(out)
    return df.groupby(["variant", "method"]).mean(numeric_only=True).reset_index() if len(df) else df


def bootstrap(c: pd.DataFrame, episodes: pd.DataFrame, thresholds: dict, gcfg: dict) -> dict:
    """Cluster bootstrap: clusters (with all images and episodes) are resampled; episodes of one
    vocabulary size are averaged within each replicate. Paired differences use the same replicates."""
    shared = episodes[episodes["kind"] == "shared"]
    K = c["cluster_id"].max() + 1
    W = M.bootstrap_weights(K, gcfg["n_bootstrap"], gcfg["bootstrap_seed"])
    result = {"n_bootstrap": gcfg["n_bootstrap"], "unit": "discovered cluster", "metrics": {}, "paired_differences": {}}
    reps = {}
    for size, eps in shared.groupby("size"):
        ids = eps["episode_id"].tolist()
        for method, t in c[c["episode_id"].isin(ids)].groupby("method"):
            t = t.sort_values(["episode_id", "cluster_id"])
            arr = lambda col: t[col].to_numpy(float).reshape(len(ids), K)  # noqa: E731
            s, n, na, nc = arr("score"), arr("n"), arr("n_avail"), arr("n_correct")
            tau = thresholds[method]["tau_e2e"]
            acc = np.isfinite(s) & (s >= tau)
            e2e_num = np.where(acc, nc, n - na)
            point_e2e, ci_e2e = M.ratio_ci(e2e_num, n, W)
            point_f, ci_f = M.ratio_ci(nc, na, W)
            au = np.array([M.auroc_clusters(s[e], na[e], n[e] - na[e], W) for e in range(len(ids))])
            au_point = np.nanmean([M.auroc_clusters(s[e], na[e], n[e] - na[e]) for e in range(len(ids))])
            key = f"{method}|size={size}"
            result["metrics"][key] = {"end_to_end@tau_e2e": [point_e2e, ci_e2e], "forced_accuracy_available": [point_f, ci_f],
                                      "auroc_label_available": [float(au_point),
                                                                [float(x) for x in np.nanpercentile(np.nanmean(au, 0), [2.5, 97.5])]]}
            reps[(method, size)] = {"e2e": np.nanmean((W @ e2e_num.T) / (W @ n.T), 1), "auroc": np.nanmean(au, 0)}
    base = gcfg.get("reference_method", "max_mean_sim")
    for (method, size), r in reps.items():
        if method == base or (base, size) not in reps:
            continue
        for metric in ("e2e", "auroc"):
            d = r[metric] - reps[(base, size)][metric]
            result["paired_differences"][f"{method}-{base}|size={size}|{metric}"] = \
                [float(np.nanmean(d)), [float(x) for x in np.nanpercentile(d, [2.5, 97.5])]]
    return result


def failure_cases(c, episodes, rows, y_eval, manifest: Manifest, thresholds, K, n_show: int = 8) -> dict:
    diag = cluster_diagnostics(rows, y_eval, manifest, K).set_index("cluster_id")
    names = manifest.classes.set_index("class_id")["canonical_name"]
    full = episodes.iloc[0]["episode_id"]
    m = "max_mean_sim"
    t = c[(c["episode_id"] == full) & (c["method"] == m)].set_index("cluster_id")
    t = t[t["n"] > 0].join(diag)
    impure = t.sort_values("purity").head(n_show)
    wrong = t[(t["score"] >= thresholds[m]["tau_e2e"]) & (t["pred_class_id"] != t["dominant_class_id"])]
    wrong = wrong.sort_values("score", ascending=False).head(n_show)
    replaced = []
    for e in episodes[episodes["kind"] == "shared"].itertuples():
        if e.episode_id == full:
            continue
        vocab = set(episode_classes(e._asdict()).tolist())
        te = c[(c["episode_id"] == e.episode_id) & (c["method"] == m)].set_index("cluster_id").join(diag)
        hit = te[(te["n"] > 0) & (~te["dominant_class_id"].isin(vocab)) & (te["score"] >= thresholds[m]["tau_e2e"])]
        for k, r in hit.iterrows():
            replaced.append({"episode_id": e.episode_id, "cluster_id": int(k), "score": float(r["score"]),
                             "true_dominant": r["dominant_name"], "predicted": names[int(r["pred_class_id"])],
                             "purity": float(r["purity"]), "n": int(r["n"])})
    fmt = lambda df: [{"cluster_id": int(k), "score": float(r["score"]), "predicted": names[int(r["pred_class_id"])],  # noqa: E731
                       "composition_top3": r["composition_top3"], "purity": float(r["purity"]), "n": int(r["n"])}
                      for k, r in df.iterrows()]
    return {"method": m, "threshold": "tau_e2e", "impure_clusters": fmt(impure), "confident_wrong_labels": fmt(wrong),
            "absent_label_replaced_and_accepted": sorted(replaced, key=lambda x: -x["score"])[:n_show]}


def export_episode_matrices(out: Path, inputs: dict, episodes: pd.DataFrame, strategy: str) -> None:
    """Cluster-by-label matrices of the discovery population for each episode (primary strategy,
    all members). Rows: cluster_id 0..K-1; columns: the episode's class ids in ascending order."""
    rows, K = inputs["rows"], inputs["K"]
    pop = rows[rows["population"] == "discovery"]
    members = select_members(pop, "cluster_id", list(range(K)), "all")
    agg = aggregate(inputs["R"][strategy], inputs["emb"], members)
    save_npz_atomic(out / f"discovery_aggregates_{strategy}.npz", A=agg["A"], B=agg["B"], mean_norm=agg["norm"],
                    n_members=agg["n"], class_ids=inputs["class_ids"], cluster_ids=np.arange(K))
    ep_dir = out / "episodes"
    ep_dir.mkdir(exist_ok=True)
    for e in episodes.itertuples():
        cols = episode_classes(e._asdict())
        save_npz_atomic(ep_dir / f"{e.episode_id}.npz", A=agg["A"][:, cols], B=agg["B"][:, cols],
                        class_ids=cols, cluster_ids=np.arange(K), strategy=np.array(strategy))
