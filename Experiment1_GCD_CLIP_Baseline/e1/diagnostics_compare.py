"""Compare the similarity matrices of incorrectly and correctly clustered examples (exploratory).

Reads a `diagnostics/clustering_examples/` export and writes `comparison/` inside it. Uses the
labels of the exported population only (dev labels for the default export); evaluation-only
composition fields are not used. Effect sizes are AUROC (incorrect vs correct) with bootstrap
intervals; same-cluster pairs give a within-cluster comparison; the same features are recomputed
for every population image in the exported clusters as a check that the pattern is not an
artefact of the spread-out selection.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

T_SOFTMAX = 0.02   # the run's dev-selected temperature (selection.json); used only to describe peakedness
USES_TRUE_LABEL = {"s_true", "rank_true_in_image_row", "true_minus_best_other", "clip_image_correct",
                   "rank_true_in_cluster_mean", "cluster_mean_score_true", "cluster_pred_minus_true_in_image_row",
                   "text_cos_true_vs_cluster_pred", "text_cos_true_vs_mapped"}
BY_CONSTRUCTION = {"text_cos_true_vs_mapped": "equals 1 for every correctly clustered image (true class = mapped class)"}


def _softmax_entropy(row: np.ndarray, T: float = T_SOFTMAX) -> float:
    z = (row - row.max()) / T
    p = np.exp(z) / np.exp(z).sum()
    return float(-(p * np.log(np.clip(p, 1e-300, None))).sum() / np.log(len(row)))


def _rank(row: np.ndarray, j: int) -> int:
    return int((row > row[j]).sum() + 1)


def image_features(row, y, cluster_mean, cluster_rows, own_index, pred_cluster, mapped, text, col_of) -> dict:
    """Features of one image's similarity row (row over all classes) and of its cluster context."""
    yi = col_of[y]
    srt = np.sort(row)
    others = np.delete(row, yi)
    corr = np.corrcoef(row, cluster_mean)[0, 1]
    member_corr = np.array([np.corrcoef(r, cluster_mean)[0, 1] for r in cluster_rows])
    f = {
        "s_true": row[yi],
        "rank_true_in_image_row": _rank(row, yi),
        "top1_score": srt[-1],
        "top1_minus_top2": srt[-1] - srt[-2],
        "true_minus_best_other": row[yi] - others.max(),
        "norm_entropy_T0.02": _softmax_entropy(row),
        "clip_image_correct": float(np.argmax(row) == yi),
        "profile_corr_with_cluster_mean": corr,
        "profile_corr_percentile_in_cluster": float((member_corr < corr).mean()),
        "rank_true_in_cluster_mean": _rank(cluster_mean, yi),
        "cluster_mean_score_true": cluster_mean[yi],
        "cluster_pred_minus_true_in_image_row": row[col_of[pred_cluster]] - row[yi],
        "text_cos_true_vs_cluster_pred": float(text[yi] @ text[col_of[pred_cluster]]),
    }
    if mapped in col_of:
        f["text_cos_true_vs_mapped"] = float(text[yi] @ text[col_of[mapped]])
    return f


def cluster_features(M: np.ndarray, cluster_mean: np.ndarray, mean_norm: float) -> dict:
    """Label-free cluster facts: dispersion of member profiles and peakedness of the mean profile."""
    C = np.corrcoef(M) if len(M) > 1 else np.ones((1, 1))
    srt = np.sort(cluster_mean)
    return {"cluster_mean_pairwise_profile_corr": float(C[np.triu_indices(len(M), 1)].mean()) if len(M) > 1 else np.nan,
            "cluster_mean_top1": srt[-1], "cluster_mean_top1_minus_top2": srt[-1] - srt[-2],
            "cluster_clip_mean_norm": mean_norm}


def auroc_ci(x_inc, x_cor, n_boot=2000, seed=0):
    """AUROC that a feature is higher for incorrect than for correct examples; percentile bootstrap."""
    x_inc, x_cor = np.asarray(x_inc, float), np.asarray(x_cor, float)
    x_inc, x_cor = x_inc[np.isfinite(x_inc)], x_cor[np.isfinite(x_cor)]
    y = np.r_[np.ones(len(x_inc)), np.zeros(len(x_cor))]
    point = roc_auc_score(y, np.r_[x_inc, x_cor])
    rng = np.random.default_rng(seed)
    reps = [roc_auc_score(y, np.r_[rng.choice(x_inc, len(x_inc)), rng.choice(x_cor, len(x_cor))]) for _ in range(n_boot)]
    return float(point), [float(v) for v in np.percentile(reps, [2.5, 97.5])]


def cluster_bootstrap_auroc(df: pd.DataFrame, feature: str, n_boot=1000, seed=0):
    """AUROC over all population images, resampling clusters (images of a cluster move together)."""
    d = df[np.isfinite(df[feature])]
    point = roc_auc_score(~d["correct"], d[feature])
    rng = np.random.default_rng(seed)
    groups = [g for _, g in d.groupby("cluster_id")]
    reps = []
    for _ in range(n_boot):
        s = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if s["correct"].nunique() == 2:
            reps.append(roc_auc_score(~s["correct"], s[feature]))
    return float(point), [float(v) for v in np.percentile(reps, [2.5, 97.5])]


def compare(export_dir) -> Path:
    export_dir = Path(export_dir)
    out = export_dir / "comparison"
    out.mkdir(exist_ok=True)
    summary = json.loads((export_dir / "export_summary.json").read_text())
    population = summary["population"]
    ex = pd.read_csv(export_dir / "examples.csv")
    gdir = export_dir.parent.parent
    text = np.load(gdir / f"text_embeddings_{summary['run']['prompt_strategy']}.npz")
    col_ids = text["class_ids"]
    col_of = {int(c): i for i, c in enumerate(col_ids)}
    T = text["per_class"]
    mapping = pd.read_csv(export_dir / f"hungarian_mapping_{population}.csv").set_index("cluster_id")["mapped_class_id"]

    per_image, pop_rows, clusters = [], [], {}
    for cdir in sorted((export_dir / "cluster_context").glob("cluster_*")):
        z = np.load(cdir / "similarity_matrix.npz")
        mem = pd.read_csv(cdir / "members.csv", keep_default_na=False)
        ctx = json.loads((cdir / "context.json").read_text())
        agg = pd.read_csv(cdir / "aggregate_scores.csv")
        k = ctx["cluster_id"]
        assert list(z["row_sample_ids"]) == mem["sample_id"].tolist() and np.array_equal(z["col_class_ids"], col_ids)
        M = z["R"]
        mean = agg["A_discovery_saved"].to_numpy()
        if not np.isfinite(mean).all():           # cluster without discovery members: use all members
            mean = M.mean(0)
        pred = ctx["semantic_prediction_discovery"]["class_id"]
        clusters[k] = cluster_features(M, mean, ctx["discovery_mean_norm"])
        for i, r in mem.iterrows():
            if r["population"] != population:
                continue
            f = image_features(M[i], int(r["class_id"]), mean, M, i, pred, int(mapping[k]), T, col_of)
            f.update(sample_id=r["sample_id"], cluster_id=k, y=int(r["class_id"]),
                     correct=bool(int(mapping[k]) == int(r["class_id"])), selected_group=r["selected_group"],
                     sq_dist_to_gcd_centre=float(r["sq_dist_to_centre"]), **clusters[k])
            pop_rows.append(f)
    pop = pd.DataFrame(pop_rows)
    sel = pop[pop["selected_group"] != ""].copy()
    sel["group"] = sel["selected_group"]
    if set(sel["sample_id"]) != set(ex["sample_id"]):
        raise RuntimeError("selected images in the cluster contexts do not match examples.csv")
    sel = sel.merge(ex[["sample_id", "paired_incorrect_sample_id", "selection_reason"]], on="sample_id")
    sel.to_csv(out / "per_image_features_selected.csv", index=False)
    pop.to_csv(out / f"per_image_features_all_{population}_in_exported_clusters.csv", index=False)

    features = [c for c in sel.columns if c not in {"sample_id", "cluster_id", "y", "correct", "selected_group", "group",
                                                    "paired_incorrect_sample_id", "selection_reason"}]
    inc, cor = sel[sel["group"] == "incorrect"], sel[sel["group"] == "correct"]
    rows = []
    for f in features:
        a, ci = auroc_ci(inc[f], cor[f])
        pa, pci = cluster_bootstrap_auroc(pop, f)
        rows.append({"feature": f, "median_incorrect": float(inc[f].median()), "median_correct": float(cor[f].median()),
                     "iqr_incorrect": [float(v) for v in inc[f].quantile([.25, .75])],
                     "iqr_correct": [float(v) for v in cor[f].quantile([.25, .75])],
                     "auroc_incorrect_higher_selected": a, "auroc_ci95_selected": ci,
                     "auroc_incorrect_higher_all_in_clusters": pa, "auroc_ci95_all_in_clusters_cluster_bootstrap": pci,
                     "uses_true_label": f in USES_TRUE_LABEL, "note": BY_CONSTRUCTION.get(f, ""),
                     "n_selected": [int(inc[f].notna().sum()), int(cor[f].notna().sum())],
                     "n_all_in_clusters": [int((~pop["correct"]).sum()), int(pop["correct"].sum())]})
    stats = pd.DataFrame(rows)
    stats.to_csv(out / "summary.csv", index=False)

    # within-cluster pairs: control minus its incorrect partner from the same cluster
    pairs = sel[sel["selection_reason"] == "same_cluster"].merge(
        sel[sel["group"] == "incorrect"], left_on="paired_incorrect_sample_id", right_on="sample_id", suffixes=("_cor", "_inc"))
    prow = []
    for f in features:
        d = (pairs[f + "_inc"] - pairs[f + "_cor"]).to_numpy(float)
        d = d[np.isfinite(d)]
        prow.append({"feature": f, "n_pairs": len(d), "median_diff_incorrect_minus_correct": float(np.median(d)) if len(d) else np.nan,
                     "share_incorrect_higher": float((d > 0).mean()) if len(d) else np.nan,
                     "share_tied": float((d == 0).mean()) if len(d) else np.nan})
    pd.DataFrame(prow).to_csv(out / "same_cluster_pairs.csv", index=False)
    figure(out / "comparison.png", sel, stats)
    meta = {"population": population, "labels": summary["label_permission"], "n_selected": [len(inc), len(cor)],
            "n_all_in_exported_clusters": [int((~pop["correct"]).sum()), int(pop["correct"].sum())],
            "n_same_cluster_pairs": len(pairs), "softmax_T_for_entropy": T_SOFTMAX,
            "note": "exploratory; known classes only (dev pool); the selection is spread by design, not random"}
    (out / "comparison_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out


def figure(path: Path, sel: pd.DataFrame, stats: pd.DataFrame, panels=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = panels or ["s_true", "rank_true_in_image_row", "true_minus_best_other", "profile_corr_with_cluster_mean",
                        "rank_true_in_cluster_mean", "cluster_pred_minus_true_in_image_row",
                        "profile_corr_percentile_in_cluster", "sq_dist_to_gcd_centre"]
    colours = {"incorrect": "#eb6834", "correct": "#2a78d6"}
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.6))
    rng = np.random.default_rng(0)
    for ax, f in zip(axes.ravel(), panels):
        for x, g in enumerate(("incorrect", "correct")):
            v = sel.loc[sel["group"] == g, f].dropna().to_numpy()
            ax.scatter(x + rng.uniform(-0.14, 0.14, len(v)), v, s=14, color=colours[g], alpha=0.75, edgecolor="white", linewidth=0.4)
            ax.hlines(np.median(v), x - 0.28, x + 0.28, color="#222222", linewidth=2)
        r = stats.set_index("feature").loc[f]
        lo, hi = r["auroc_ci95_selected"]
        ax.set_title(f"{f}\nAUROC(incorrect higher) {r['auroc_incorrect_higher_selected']:.2f} [{lo:.2f}, {hi:.2f}]", fontsize=8.5)
        ax.set_xticks([0, 1], ["incorrect (n=50)", "correct (n=50)"], fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if "rank" in f:
            ax.set_yscale("log")
    fig.suptitle("Selected dev-pool images: incorrectly vs correctly clustered (black bar = median; rank panels on log scale)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
