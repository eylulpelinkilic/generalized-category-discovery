"""Markdown reports. Synthetic (fixture) runs never enter the results report."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ovar.config import PROJECT_ROOT, project_path

DATASET_ORDER = ["ucf101", "hmdb51", "arid", "necdrone"]
DISPLAY = {"ucf101": "UCF101", "hmdb51": "HMDB51", "arid": "ARID", "necdrone": "NEC-Drone"}
REFERENCE = PROJECT_ROOT / "docs" / "reference" / "xov_reported_numbers.csv"
CONCEPTS = PROJECT_ROOT / "docs" / "reference" / "cross_domain_concepts.csv"


def _pct(block: dict | None) -> str:
    if not block or block.get("acc") is None:
        return "—"
    text = f"{100 * block['acc']:.2f}"
    if block.get("ci95"):
        lo, hi = block["ci95"]
        text += f" [{100 * lo:.1f}, {100 * hi:.1f}]"
    if "n" in block:
        text += f" (n={block['n']})"
    return text


def collect_runs(runs_root) -> list[tuple[Path, dict, dict]]:
    runs = []
    for summary_path in sorted(Path(runs_root).glob("*/summary.json")):
        run_dir = summary_path.parent
        runs.append((run_dir, json.loads((run_dir / "run.json").read_text()),
                     json.loads(summary_path.read_text())))
    return runs


def _run_table(runs) -> list[str]:
    lines = ["| Run | Model | Protocol | Subset | Dataset | Overall | Familiar | Unfamiliar | Failed |",
             "|---|---|---|---|---|---|---|---|---|"]
    for run_dir, record, summary in runs:
        for name in [d for d in DATASET_ORDER if d in summary["per_dataset"]] + \
                [d for d in summary["per_dataset"] if d not in DATASET_ORDER]:
            acc = summary["per_dataset"][name]
            info = record["datasets"][name]
            subset = info["subset"]
            subset_text = "full list" if not (subset["per_class"] or subset["limit"]) else \
                f"subset per_class={subset['per_class']} limit={subset['limit']}"
            lines.append(f"| {run_dir.name} | {record['model'].get('name')} | {record['protocol'].get('name')} "
                         f"| {subset_text} | {DISPLAY.get(name, name)} | {_pct(acc['overall'])} | "
                         f"{_pct(acc['familiar'])} | {_pct(acc['unfamiliar'])} | {info['failed']} |")
        if "domain_average_unweighted" in summary:
            avg = summary["domain_average_unweighted"]
            pooled = summary["pooled_over_videos"]
            lines.append(f"| {run_dir.name} | | | | **AVG (unweighted over domains)** | {_pct(avg['overall'])} | "
                         f"{_pct(avg['familiar'])} | {_pct(avg['unfamiliar'])} | |")
            lines.append(f"| {run_dir.name} | | | | pooled over videos | {_pct(pooled['overall'])} | "
                         f"{_pct(pooled['familiar'])} | {_pct(pooled['unfamiliar'])} | |")
    return lines


def results_report(runs_root=None, out_path=None) -> Path:
    runs_root = Path(runs_root or project_path("runs"))
    out_path = Path(out_path or PROJECT_ROOT / "reports" / "results.md")
    real = [r for r in collect_runs(runs_root) if not r[1]["synthetic"]] if runs_root.exists() else []
    lines = [
        "# Results report", "",
        "Only experiments actually run on benchmark videos appear here. Accuracies are top-1 (%) over the "
        "full target vocabulary; brackets are 95% cluster-bootstrap intervals over source groups. "
        "Familiar/unfamiliar = XOV closed-set/open-set categories (relative to Kinetics400). "
        "Scores are cosine similarities, not calibrated probabilities.", "",
    ]
    if not real:
        lines += ["**No experiment on real benchmark videos has been run in this workspace yet.** "
                  "See STATUS.md for the blockers and the commands that produce these results.", ""]
    for kind, title in (("native", "Native inference configurations"),
                        ("common", "Common comparison protocol")):
        group = [r for r in real if r[1].get("comparison") == kind]
        if group:
            lines += [f"## {title}", ""] + _run_table(group) + [""]
            lines += ["| Run | Device | Model load (s) | Total (s) | Encode (s) | Decode (s) | Memory |",
                      "|---|---|---|---|---|---|---|"]
            for run_dir, record, _ in group:
                enc = sum(d["encode_s"] for d in record["datasets"].values())
                dec = sum(d["decode_s"] for d in record["datasets"].values())
                lines.append(f"| {run_dir.name} | {record['environment']['device']} | "
                             f"{record['timing']['model_load_s']} | {record['timing']['total_s']} | "
                             f"{enc:.1f} | {dec:.1f} | {json.dumps(record['memory'])} |")
            lines.append("")
    other = [r for r in real if r[1].get("comparison") not in ("native", "common")]
    if other:
        lines += ["## Other configurations", ""] + _run_table(other) + [""]

    sensitivity = []
    for run_dir, record, _ in real:
        for name in record["datasets"]:
            metrics = json.loads((run_dir / name / "metrics.json").read_text())
            extra = metrics.get("secondary_unfamiliar_excluding_near_familiar")
            if extra:
                sensitivity.append(f"| {run_dir.name} | {DISPLAY.get(name, name)} | "
                                   f"{_pct(metrics['accuracy']['unfamiliar'])} | {_pct(extra)} | "
                                   f"{len(extra['excluded_classes'])} |")
    if sensitivity:
        lines += ["## Sensitivity: unfamiliar accuracy without near-familiar classes", "",
                  "Secondary analysis only. Excluded classes are officially unfamiliar but have a close "
                  "Kinetics400 counterpart according to an unverified review (data/audit/category_review.csv).", "",
                  "| Run | Dataset | Unfamiliar (official) | Unfamiliar excluding near-familiar | Excluded classes |",
                  "|---|---|---|---|---|"] + sensitivity + [""]

    ref = pd.read_csv(REFERENCE)
    lines += [
        "## Reference numbers reported by XOV-Action (not reproduced here)", "",
        "Transcribed from arXiv:2403.01560v3. Their Table III models were trained on Kinetics400 by the XOV "
        "authors with T=16 frames, one centre-cropped clip, and the checkpoint with the best average overall "
        "accuracy on the four test domains (test-set model selection). They are therefore not directly "
        "comparable with the official checkpoints evaluated here.", "",
        "| Source | Training | Backbone | Model | Dataset | Closed | Open | All |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in ref.itertuples():
        lines.append(f"| {r.source} | {r.training} | {r.backbone} | {r.model} | {DISPLAY.get(r.dataset, r.dataset)} "
                     f"| {r.closed:.2f} | {getattr(r, 'open'):.2f} | {getattr(r, 'all'):.2f} |")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def error_analysis(run_dir, diagnostic_dirs=(), out_path=None, min_n: int = 5) -> Path:
    """Compact error analysis. Observations are computed; hypotheses are left to the researcher."""
    run_dir = Path(run_dir)
    record = json.loads((run_dir / "run.json").read_text())
    out_path = Path(out_path or run_dir / "error_analysis.md")
    banner = ["> **SYNTHETIC FIXTURE RUN — software verification only, not research evidence.**", ""] \
        if record["synthetic"] else []
    lines = [f"# Error analysis: {run_dir.name}", ""] + banner + ["## Observations", ""]

    per_class = {}
    for name in record["datasets"]:
        pc = pd.read_csv(run_dir / name / "per_class.csv", keep_default_na=False)
        per_class[name] = pc
        metrics = json.loads((run_dir / name / "metrics.json").read_text())
        worst = pc[pc["n"] >= min_n].sort_values("acc").head(5)
        lines += [f"### {DISPLAY.get(name, name)}", "",
                  f"Overall {_pct(metrics['accuracy']['overall'])}; familiar "
                  f"{_pct(metrics['accuracy']['familiar'])}; unfamiliar {_pct(metrics['accuracy']['unfamiliar'])}.", "",
                  f"Lowest-accuracy classes (n ≥ {min_n}):", ""]
        lines += [f"- {r.action_name} ({r.category_status}): {100 * r.acc:.1f}% of {r.n}; most often predicted "
                  f"as {r.most_confused_with or '—'} ({r.most_confused_count})" for r in worst.itertuples()] \
            or [f"- no class has n ≥ {min_n}"]
        lines += ["", "Most frequent confusions:", ""]
        lines += [f"- {p['true']} → {p['predicted']}: {p['count']} ({100 * p['rate_of_true_class']:.1f}% of the "
                  f"true class)" for p in metrics["confused_pairs"]] or ["- none with count ≥ 2"]
        lines.append("")

    if CONCEPTS.exists() and len(per_class) > 1:
        concepts = pd.read_csv(CONCEPTS, keep_default_na=False)
        lines += ["### Shared concepts across domains (curated mapping, docs/reference/cross_domain_concepts.csv)", "",
                  "| Concept | " + " | ".join(DISPLAY.get(d, d) for d in per_class) + " | Match quality |",
                  "|---|" + "---|" * len(per_class) + "---|"]
        for c in concepts.itertuples():
            cells = []
            for d in per_class:
                cls = getattr(c, d, "")
                row = per_class[d][per_class[d]["action_name"] == cls]
                cells.append(f"{100 * row['acc'].iloc[0]:.1f} (n={row['n'].iloc[0]})" if cls and len(row) else "—")
            lines.append(f"| {c.concept} | " + " | ".join(cells) + f" | {c.match_quality} |")
        lines.append("")

    for diag_dir in map(Path, diagnostic_dirs):
        summary = pd.read_csv(diag_dir / "summary.csv", keep_default_na=False)
        lines += [f"### Diagnostics: {diag_dir.name} (development subset)", "",
                  "| Variant | Family | Dataset | Acc ref | Acc variant | Δ acc [95% CI] | Prediction change | "
                  "Variant failures | Note |", "|---|---|---|---|---|---|---|---|---|"]
        for r in summary.itertuples():
            if r.status != "run":
                lines.append(f"| {r.variant} | {r.family} | | | | {r.status} | | | |")
                continue
            ci = json.loads(r.delta_acc_ci95.replace("'", '"')) if isinstance(r.delta_acc_ci95, str) and \
                r.delta_acc_ci95 else None
            ci_text = f" [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]" if ci else ""
            lines.append(f"| {r.variant} | {r.family} | {r.dataset} | {100 * float(r.acc_reference):.1f} | "
                         f"{100 * float(r.acc_variant):.1f} | {100 * float(r.delta_acc):+.1f}{ci_text} | "
                         f"{100 * float(r.prediction_change_rate):.1f}% of both-decoded | "
                         f"{r.variant_failed} | {r.note} |")
        lines += ["", "Degradation variants are synthetic stress tests and are not evidence about real dark or "
                      "drone domains. Temporal and crop effects are diagnostics: an action can be recognisable "
                      "from static cues, and crops can remove evidence or add artefacts.", ""]
        per_cls = diag_dir / "per_class.csv"
        if per_cls.exists():
            pcd = pd.read_csv(per_cls)
            top = pcd[pcd["n_changed"] > 0].sort_values("prediction_change_rate", ascending=False).head(8)
            lines += ["Classes with the highest prediction-change rate (videos decoded in both runs):", ""]
            lines += [f"- {r.variant} / {r.action_name} ({r.category_status}): {r.n_changed} of "
                      f"{r.n_both_decoded} changed, Δ acc {100 * r.delta_acc:+.0f} pp (n={r.n}, variant "
                      f"failures {r.variant_failed})" for r in top.itertuples()] or ["- no prediction changed"]
            lines.append("")

    lines += ["## Hypotheses", "",
              "_To be written by the researcher. Each hypothesis must cite the observation above that "
              "motivates it and name the experiment that would refute it. This report does not infer that any "
              "particular architecture is required._", ""]
    out_path.write_text("\n".join(lines) + "\n")
    return out_path
