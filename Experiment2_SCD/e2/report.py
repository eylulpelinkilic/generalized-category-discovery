"""Generated tables for reports/: every number comes from files on disk; missing stages are listed as pending
with the command that produces them. The narrative (reports/REPORT.md) is written from these tables."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from e2.config import EXP2, Layout, load_config

PRIMARY_REJ = ["max_mean_sim", "vote_fraction", "plurality@win_fraction", "plurality@margin", "plurality@neg_entropy",
               "plurality@mean_sim"]


def pct(x, nd=1):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * x:.{nd}f}"


def table(rows: list[list], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _latest(paths):
    paths = [p for p in paths if (p / "status.json").exists() and json.loads((p / "status.json").read_text())["status"] == "completed"]
    return paths[-1] if paths else None


def vocabulary_section(cfg) -> str:
    d = Layout(cfg).vocabulary()
    if not (d / "alignment.json").exists():
        return "**Pending**: `python -m e2 --config configs/cub.yaml vocab`\n"
    a = json.loads((d / "alignment.json").read_text())
    return table([[a["vocabulary"]["n_columns"], a["vocabulary"]["n_unique_normalised"], a["vocabulary"]["n_duplicate_columns"],
                   a["exact_check_positions"], f"{a['exact_min_cosine']:.5f}", f"{a.get('retrieval_top1_rate', float('nan')):.4f}",
                   f"{a.get('retrieval_top5_rate', float('nan')):.5f}",
                   f"{a['negative_control_sorted_order']['exact_min_cosine']:.3f}",
                   f"{a['duplicate_columns']['min_cosine_to_first']:.4f}", "passed" if a["passed"] else "FAILED"]],
                 ["columns", "unique normalised names", "duplicate columns", "exact-check positions",
                  "min cosine (80 templates)", "1-template retrieval top-1", "top-5", "sorted-order control min cosine",
                  "duplicate pair min cosine", "result"]) + f"\n\nSource: `{d.relative_to(EXP2)}/alignment.json`.\n"


def fixed_section(cfg) -> str:
    base = Layout(cfg).base / "fixed_cluster"
    d = _latest(sorted(base.glob("*/*/*"))) if base.exists() else None
    if d is None:
        return "**Pending**: `python -m e2 --config configs/cub.yaml cub-embed --clip vitb32` then `... fixed --clip vitb32`\n"
    s = []
    emb = sorted((Layout(cfg).base / "inputs").glob("clip_vitb32_*/validation.json"))
    if emb:
        v = json.loads(emb[-1].read_text())
        s.append(f"Embedding validation: aggregate max |ΔA| = {v['aggregate_max_abs_diff']:.2e}, member counts equal = "
                 f"{v['n_members_equal']}, Experiment 1 prediction agreement = {v['prediction_agreement']:.4f} over "
                 f"{v['prediction_rows']:,} (episode, cluster, method) rows; max score difference {v['max_score_diff']:.2e}.\n")
    cons = json.loads((d / "e1_consistency.json").read_text())
    s.append("Experiment 1 methods recomputed vs its saved metrics: " + ", ".join(
        f"{k} max |Δ| {v['max_abs_diff']:.1e}" for k, v in cons.items() if isinstance(v, dict)) + ".\n")
    sel = json.loads((d / "selection.json").read_text())
    s.append(f"Dev selection: k* = {sel['k_star']} ({', '.join(f'k={k}: {v:.4f}' for k, v in sel['values'].items())}; "
             f"dev AURC, lower is better).\n")
    cva = json.loads((d / "complete_vocabulary_accuracy.json").read_text())
    boot = json.loads((d / "bootstrap.json").read_text())
    fp = json.loads((d / "forced_paired.json").read_text())
    n_img = next(iter(cva.values()))["n_images"]
    n_names = int(pd.read_csv(d / "metrics_by_vocabulary_size.csv")["size"].max())
    rows = []
    for m in [m for m in cva if m.endswith("_forced") or m == "forced_top1"]:
        c = cva[m]
        ci = boot["metrics"].get(f"{m}|size={n_names}", {}).get("forced_accuracy_available")
        diff = fp.get(f"{m}-forced_top1|size={n_names}")
        rows.append([m, pct(c["image_weighted"]), f"[{pct(ci[1][0])}, {pct(ci[1][1])}]" if ci else "–", pct(c["class_macro"]),
                     pct(c["old"]), pct(c["new"]),
                     f"{100 * diff['point']:+.1f} [{100 * diff['ci95'][0]:+.1f}, {100 * diff['ci95'][1]:+.1f}]" if diff else "–"])
    s.append(f"**Forced naming, complete vocabulary ({n_names} names), discovery population ({n_img:,} images)**\n\n" + table(
        rows, ["label rule", "accuracy %", "95% CI", "class-macro %", "Old %", "New %", "Δ vs mean (paired) pp"]))
    size = pd.read_csv(d / "metrics_by_vocabulary_size.csv")
    rows = []
    for m in ["forced_top1", "plurality_forced"] + PRIMARY_REJ + [x for x in size["method"].unique() if x.startswith("top") and "@" in x]:
        t = size[size["method"] == m].sort_values("size", ascending=False)
        for g in t.to_dict("records"):
            rows.append([m, int(g["size"]), pct(g["forced_accuracy_available_mean"]), f"{g['auroc_label_available_mean']:.3f}"
                         if np.isfinite(g["auroc_label_available_mean"]) else "–", pct(g.get("coverage@tau_e2e_mean")),
                         pct(g.get("accuracy_among_accepted@tau_e2e_mean")), pct(g.get("false_acceptance_absent@tau_e2e_mean")),
                         pct(g.get("end_to_end_accuracy@tau_e2e_mean"))])
    s.append("\n**By vocabulary size (mean over episodes; thresholds τ_e2e from the calibration pool)**\n\n" + table(
        rows, ["method", "vocab size", "forced acc (avail) %", "AUROC avail", "coverage %", "accepted acc %",
               "false accept absent %", "end-to-end %"]))
    pdiff = boot["paired_differences"]
    rows = []
    for k, v in sorted(pdiff.items()):
        m, ref, sz, metric = k.split("|")[0].split("-")[0], "max_mean_sim", k.split("size=")[1].split("|")[0], k.split("|")[-1]
        if (m in PRIMARY_REJ or m.startswith("top")) and np.isfinite(v[0]):
            rows.append([m, sz, metric, f"{v[0]:+.3f}", f"[{v[1][0]:+.3f}, {v[1][1]:+.3f}]"])
    s.append("\n**Paired cluster-bootstrap differences against max_mean_sim** (e2e = end-to-end accuracy at τ_e2e)\n\n" +
             table(rows, ["method", "vocab size", "metric", "difference", "95% CI"]))
    fpat = json.loads((d / "errors" / "failure_patterns.json").read_text())
    rows = []
    for m, p in fpat["by_method"].items():
        rows.append([m, p["clusters_evaluated"]["clusters"], p["wrong_label"]["clusters"],
                     f"{p['P1_weak_support']['clusters']} ({p['P1_weak_support']['images']})",
                     f"{p['P2_aggregation_override']['clusters']} ({p['P2a_dominant_strict_majority']['clusters']})",
                     f"{p['P3_competition']['clusters']} ({p['P3_label_owned_by_another_cluster']['clusters']})",
                     f"{p['P5_mixed_clusters']['clusters']}: {pct(p['P5_cluster_error_rate_mixed'])} vs {pct(p['P5_cluster_error_rate_pure'])}",
                     f"{p['P6_related_name']['clusters']} (head word {p['P6_same_head_word']['clusters']}, text-NN {p['P6_text_nearest5']['clusters']})"])
    s.append("\n**Failure patterns, pooled over the 21 shared episodes** (cluster decisions; images in parentheses for P1)\n\n" +
             table(rows, ["method", "cluster decisions", "wrong", "P1 weak support (images)", "P2 override (strict majority)",
                          "P3 competition (label owned elsewhere)", "P5 mixed: n, error mixed vs pure %", "P6 related name"]))
    rows = []
    for m, by in fpat.get("P7_P8_by_method_and_size", {}).items():
        for sz, v in sorted(by.items(), key=lambda kv: -int(kv[0])):
            rows.append([m, sz, pct(v["P7_absent_accepted_rate"]), v["P7_absent_images"], pct(v["P8_available_rejected_rate"]),
                         v["P8_rejected_but_label_correct"], v["P8_rejected_and_label_wrong"]])
    s.append("\n**P7/P8 (image level, pooled over episodes of a size)**\n\n" + table(
        rows, ["method", "vocab size", "absent accepted %", "absent images", "available rejected %", "rejected, label correct",
               "rejected, label wrong"]))
    dis = fpat["mean_vs_plurality_disagreement_images"]
    s.append("\n**Mean vs plurality label (forced), image level, pooled over episodes of a size**\n\n" + table(
        [[k, v["images_with_different_labels"], v["of"], v["mean_correct_vote_wrong"], v["vote_correct_mean_wrong"]]
         for k, v in sorted(dis.items(), key=lambda kv: -int(kv[0]))],
        ["vocab size", "images with different labels", "image decisions", "mean right, vote wrong", "vote right, mean wrong"]))
    s.append(f"\nSource: `{d.relative_to(EXP2)}`.\n")
    return "\n".join(s), d


def refine_section(cfg) -> str:
    base = Layout(cfg).base / "refine_on_gcd"
    runs = sorted(p for p in base.glob("*/*/*") if (p / "metrics.json").exists()) if base.exists() else []
    if not runs:
        return "**Pending**: `python -m e2 --config configs/cub.yaml refine-gcd --cluster-seed {0,1,2} --mode {unsupervised,partially_supervised}`\n"
    rows, agg = [], {}
    for p in runs:
        m = json.loads((p / "metrics.json").read_text())
        mode = p.name.rsplit("_", 1)[0]
        i, f = m["initial_clustering_named_by_iteration1_votes"], m["final"]
        flips = m["P4_refinement_flips"]
        pb = m["paired_final_minus_initial_sacc"]
        rows.append([p.parent.name, mode, pct(i["sacc_all"]["image"]), pct(f["sacc_all"]["image"]),
                     f"{100 * pb['point']:+.1f} [{100 * pb['ci95'][0]:+.1f}, {100 * pb['ci95'][1]:+.1f}]",
                     pct(i["acc"]["all"]), pct(f["acc"]["all"]), pct(f["sacc_old"]["image"]), pct(f["sacc_new"]["image"]),
                     f["n_selected_names"], m["n_iterations"], m["stop_reason"],
                     f"{flips['initial_named_correct_to_final_wrong']} / {flips['initial_named_wrong_to_final_correct']}"])
        agg.setdefault(mode, []).append((i["sacc_all"]["image"], f["sacc_all"]["image"], i["acc"]["all"], f["acc"]["all"]))
    s = table(rows, ["clustering", "mode", "initial sACC %", "final sACC %", "final − initial (paired image bootstrap)",
                     "initial ACC %", "final ACC %", "final sACC Old %", "final sACC New %", "names", "iterations", "stop",
                     "P4 flips correct→wrong / wrong→correct"])
    s += "\n\nMean ± SD over clustering seeds: " + "; ".join(
        f"{mode}: sACC {100 * np.mean([x[0] for x in v]):.1f}±{100 * np.std([x[0] for x in v], ddof=1) if len(v) > 1 else 0:.1f} → "
        f"{100 * np.mean([x[1] for x in v]):.1f}±{100 * np.std([x[1] for x in v], ddof=1) if len(v) > 1 else 0:.1f}, ACC "
        f"{100 * np.mean([x[2] for x in v]):.1f} → {100 * np.mean([x[3] for x in v]):.1f} (n = {len(v)})" for mode, v in agg.items())
    return s + "\n", runs


def repro_section() -> str:
    cfg = load_config("imagenet100")
    p = Layout(cfg).base / "repro_summary.json"
    if not p.exists():
        return ("**Pending (needs ImageNet and a GPU)**: run `kaggle/e2_imagenet100.ipynb`, or on a GPU machine\n\n"
                "```bash\nE2=\"python -m e2 --config configs/imagenet100.yaml --set e1.runs_root=<Experiment 1 runs>\"\n"
                "$E2 vocab && $E2 repro features && $E2 repro cluster && $E2 repro name && $E2 repro evaluate\n```\n")
    s = json.loads(p.read_text())
    rows = []
    for r in s["runs"]:
        rows.append([r["setting"], Path(r["seed_dir"]).parent.name, r["variant"], pct(r["initial"]["acc"]["all"]),
                     pct(r["final"]["sacc_all"]["image"]),
                     pct((r["final"].get("soft_sacc_all") or {}).get("official")),
                     pct(r["final"]["acc"]["all"]), r["final"]["n_selected_names"], r["n_iterations"], r["stop_reason"]])
    return table(rows, ["setting", "seed", "variant", "init ACC %", "sACC %", "Soft-sACC (official) %", "ACC %", "names",
                        "iterations", "stop"]) + "\n"


SERIES = {"max_mean_sim": ("#2a78d6", "mean label · max similarity"),          # palette slots 1-2 (validated)
          "plurality@win_fraction": ("#eb6834", "vote label · win fraction")}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def risk_coverage_figure(d: Path, path: Path) -> Path | None:
    """Small multiples by vocabulary size: risk (1 - accuracy among accepted, absent-label images count as errors)
    against coverage, mean over the size's episodes on a common coverage grid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = d / "risk_coverage_curves.csv.gz"
    if not f.exists():
        return None
    rc = pd.read_csv(f)
    eps = pd.read_csv(json.loads((d / "status.json").read_text())["inputs"]["e1_grounding"] + "/episodes.csv")
    rc = rc.merge(eps[["episode_id", "size"]], on="episode_id")
    sizes = sorted(rc["size"].unique(), reverse=True)
    grid = np.linspace(0.02, 1.0, 50)
    fig, axes = plt.subplots(1, len(sizes), figsize=(2.6 * len(sizes), 2.9), sharey=True, facecolor=SURFACE)
    for ax, size in zip(np.atleast_1d(axes), sizes):
        ax.set_facecolor(SURFACE)
        for method, (col, label) in SERIES.items():
            curves = []
            for _, t in rc[(rc["size"] == size) & (rc["method"] == method)].groupby("episode_id"):
                t = t.sort_values("coverage")
                idx = np.searchsorted(t["coverage"].to_numpy(), grid - 1e-12)
                ok = idx < len(t)
                y = np.full(len(grid), np.nan)
                y[ok] = t["risk"].to_numpy()[idx[ok]]
                curves.append(y)
            if not curves:
                continue
            m = np.nanmean(np.stack(curves), 0)
            ax.plot(grid, m, color=col, lw=2, solid_capstyle="round", label=label)
        ax.set_title(f"|V| = {size}", fontsize=9, color=INK, loc="left")
        ax.grid(True, color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=7.5)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1)
        ax.set_xlabel("coverage", fontsize=8, color=INK2)
    np.atleast_1d(axes)[0].set_ylabel("risk", fontsize=8, color=INK2)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.suptitle("Risk–coverage on the discovery clusters (CUB, ViT-B/32), mean over vocabulary seeds", fontsize=9.5,
                 color=INK, x=0.01, y=0.985, ha="left")
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="upper left", ncol=2, labelcolor=INK,
               bbox_to_anchor=(0.005, 0.925))
    fig.tight_layout(rect=(0, 0, 1, 0.83))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


REPORT_GALLERIES = {"forced_top1_P1_weak_support", "forced_top1_P3_competition", "forced_top1_P5_mixed",
                    "forced_top1_P6_related_name", "distinct_top3_forced_P2_aggregation_override"}


def to_jpeg(src: Path, dst: Path, width: int = 1400, quality: int = 80) -> Path:
    """Committed figures stay small: photo galleries as JPEG at a fixed width (originals remain in outputs/)."""
    from PIL import Image

    im = Image.open(src).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    im.save(dst, "JPEG", quality=quality, optimize=True)
    return dst


def write(cfg: dict) -> Path:
    out = EXP2 / "reports"
    out.mkdir(exist_ok=True)
    if (out / "figures").exists():
        shutil.rmtree(out / "figures")
    parts = ["# Experiment 2: generated tables", "",
             "Every number below is read from `outputs/`. Narrative and interpretation: `REPORT.md`.", "",
             "## Vocabulary verification", vocabulary_section(load_config("cub"))]
    fx = fixed_section(load_config("cub"))
    if isinstance(fx, tuple):
        text, d = fx
        figs = out / "figures"
        figs.mkdir(exist_ok=True)
        for g in (d / "errors" / "galleries").glob("*.png"):
            if g.stem in REPORT_GALLERIES:
                to_jpeg(g, figs / f"B_{g.stem}.jpg")
        risk_coverage_figure(d, figs / "B_risk_coverage.png")
        parts += ["## Protocol B: fixed clusters (CUB, ViT-B/32)", text]
    else:
        parts += ["## Protocol B: fixed clusters (CUB, ViT-B/32)", fx]
    rf = refine_section(load_config("cub"))
    if isinstance(rf, tuple):
        text, runs = rf
        for p in runs:
            g = p / "galleries" / "P4_refinement_flips.png"
            if g.exists() and p.parent.name.endswith("_c0"):
                to_jpeg(g, out / "figures" / f"R_{p.parent.name}_{p.name.rsplit('_', 1)[0]}_P4.jpg")
        parts += ["## Protocol R: refinement from GCD clusters (CUB, closed vocabulary; not a paper setting)", text]
    else:
        parts += ["## Protocol R: refinement from GCD clusters (CUB)", rf]
    parts += ["## Protocol A: reproduction (ImageNet-100)", repro_section()]
    path = out / "GENERATED_TABLES.md"
    path.write_text("\n".join(parts) + "\n")
    return path
