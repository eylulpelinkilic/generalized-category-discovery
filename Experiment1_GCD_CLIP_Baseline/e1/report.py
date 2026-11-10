"""Baseline report: what ran, what is pending, and the results with links to the files."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from e1.config import Layout, section_hash


def _read(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _f(x, pct=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{100 * x:.2f}" if pct else f"{x:.4f}"


def _ci(v):
    return f"{100 * v[0]:.2f} [{100 * v[1][0]:.1f}, {100 * v[1][1]:.1f}]" if v and v[1][0] == v[1][0] else "—"


def _link(path: Path, base: Path) -> str:
    return f"[{path.name}]({os.path.relpath(path, base)})"


def write_report(cfg: dict, layout: Layout) -> Path:
    out_dir = layout.report
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "BASELINE_REPORT.md"
    smoke = cfg["dataset"]["name"] == "smoke"
    L = [f"# Experiment 1 baseline report: {cfg['dataset']['name']}", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} from `{cfg['_config_file']}`. "
         "Protocol: PROTOCOL.md. Every number below is read from the linked files.", ""]
    if smoke:
        L += ["> **SMOKE RUN — synthetic images, random weights. Software check only; not a research result.**", ""]

    # ---------------- data
    meta = _read(layout.data / "manifest_meta.json")
    L += ["## Data", ""]
    if meta is None:
        L += ["Pending: `prepare-data` has not run.", ""]
    else:
        c = meta["counts"]
        up = _read(layout.data / "upstream_partition_check.json")
        L += [f"Manifest sha256 `{meta['manifest_sha256'][:16]}…` ({_link(layout.data / 'manifest.csv', out_dir)}). "
              f"Labelled {c['labelled']}, unlabelled {c['unlabelled']} (known-class {c['unlabelled_known']}, "
              f"novel-class {c['unlabelled_novel']}), held-out dev {c['heldout_dev']}, calibration {c['heldout_calib']}. "
              f"Files missing: {meta['checks']['files']['missing']}; corrupt: {meta['checks']['files'].get('corrupt', '—')}. "
              f"Upstream partition replay: {'passed' if up and up['passed'] else ('FAILED' if up else 'not run')}.", ""]

    # ---------------- status
    L += ["## Run status", "", "| Stage | Run | Status |", "|---|---|---|"]
    pending = []
    seeds, cseeds = cfg["seeds"]["train"], cfg["seeds"]["cluster"]
    clusterings, groundings = [], []
    for s in seeds:
        rd = layout.gcd_run(s)
        log = rd / "train_log.jsonl"
        n_ep = len(log.read_text().splitlines()) if log.exists() else 0
        run = _read(rd / "run.json")
        status = "completed" if run else (f"in progress: {n_ep}/{cfg['gcd_train']['epochs']} epochs" if n_ep else "pending")
        L.append(f"| GCD training | seed {s} | {status} |")
        if not run:
            pending.append(f"GCD training seed {s}")
        feats = (rd / "features" / "status.json").exists()
        L.append(f"| GCD features | seed {s} | {'completed' if feats else 'pending'} |")
        for k_mode in ("known", "estimated"):
            for c in (cseeds if k_mode == "known" else cseeds[:1]):     # estimated K: primary clustering seed
                cd = layout.clustering(s, k_mode, c)
                ev = _read(cd / "gcd_eval.json")
                L.append(f"| SS-k-means ({k_mode} K) | seed {s}, clustering seed {c} | {'completed' if ev else 'pending'} |")
                if ev:
                    clusterings.append((s, k_mode, c, _read(cd / "clustering.json"), ev, cd))
                else:
                    pending.append(f"clustering seed {s}/{k_mode}/c{c}")
    for name in cfg["clip"]["models"]:
        cd = layout.clip(name)
        idx, ver = _read(cd / "index.json"), _read(cd / "verification.json")
        v = "passed" if ver and ver["passed"] else ("FAILED" if ver else "not run")
        L.append(f"| CLIP {name} | {cfg['clip']['models'][name]['open_clip_name']} | "
                 f"{'extracted' if idx else 'pending'}; reference check {v} |")
        if not idx:
            pending.append(f"CLIP extraction {name}")
        for s in seeds:
            for c in cseeds[:1]:                                         # grounding: primary clustering seed
                gd = layout.grounding(s, "known", c, name)
                done = _read(gd / "status.json")
                ok = done and done.get("status") == "completed"
                L.append(f"| Grounding | seed {s}, known K, c{c}, {name} | {'completed' if ok else 'pending'} |")
                if ok:
                    groundings.append((s, c, name, gd))
                else:
                    pending.append(f"grounding seed {s}/c{c}/{name}")
    L += ["", f"Pending ({len(pending)}): " + (", ".join(pending) if pending else "none"), ""]

    # ---------------- clustering results
    L += ["## Discovery (GCD) on the unlabelled population", "",
          "v2: one Hungarian assignment over the whole population; All/Old/New from it (used only here).", "",
          "| Training seed | K | Clustering seed | All | Old | New | NMI | ARI | Empty | Singleton | Converged | Files |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s, k_mode, c, info, ev, cd in clusterings:
        L.append(f"| {s} | {info['K']} ({k_mode}) | {c} | {_f(ev['v2_all'])} | {_f(ev['v2_old'])} | {_f(ev['v2_new'])} | "
                 f"{_f(ev['nmi'], False)} | {_f(ev['ari'], False)} | {ev['empty_clusters']} | {ev['singleton_clusters']} | "
                 f"{info['converged']} | {_link(cd / 'assignments.csv', out_dir)} |")
    prim = [ev for s, k, c, _, ev, _ in clusterings if k == "known" and c == cseeds[0]]
    if len(prim) > 1:
        arr = np.array([[e["v2_all"], e["v2_old"], e["v2_new"]] for e in prim])
        L += ["", f"Across {len(prim)} completed training seeds (known K, clustering seed {cseeds[0]}): "
              f"All {100 * arr[:, 0].mean():.2f} ± {100 * arr[:, 0].std(ddof=1):.2f}, "
              f"Old {100 * arr[:, 1].mean():.2f} ± {100 * arr[:, 1].std(ddof=1):.2f}, "
              f"New {100 * arr[:, 2].mean():.2f} ± {100 * arr[:, 2].std(ddof=1):.2f} (mean ± SD)."]
    L.append("")

    # ---------------- grounding results
    for s, c, name, gd in groundings:
        sel = _read(gd / "selection.json")
        L += [f"## Grounding: training seed {s}, known K, clustering seed {c}, CLIP {name}", "",
              f"Dev choices: prompt `{sel['prompt_strategy']}`, T* = {sel['T_dev']} (pilot T = {sel['pilot_T']}, "
              f"MCM T = {sel['mcm_T']}). Thresholds: {_link(gd / 'thresholds.json', out_dir)}. "
              f"Predictions: {_link(gd / 'cluster_predictions.csv.gz', out_dir)}, "
              f"{_link(gd / 'image_predictions_complete_vocabulary.csv.gz', out_dir)}.", ""]
        comp = _read(gd / "complete_vocabulary_accuracy.json")
        L += ["Complete vocabulary, forced naming (image level, all unlabelled members represent each cluster):", "",
              "| Method | Image-weighted | Class-macro | Old | New |", "|---|---|---|---|---|"]
        for m in ("forced_top1", "mcm_aggregate"):
            r = comp[m]
            L.append(f"| {m} (label = argmax {'MCM prompts' if m.startswith('mcm') else 'selected prompts'}) | "
                     f"{_f(r['image_weighted'])} | {_f(r['class_macro'])} | {_f(r['old'])} | {_f(r['new'])} |")
        sized = pd.read_csv(gd / "metrics_by_vocabulary_size.csv")
        sized = sized[sized["representation"] == cfg["grounding"]["representations"][0]]
        L += ["", "By vocabulary size (mean over vocabulary seeds; AUROC positive class = true label in V; "
              "end-to-end = correct label accepted, or absent label rejected; thresholds from calibration):", "",
              "| Size | Method | AUROC avail. | AURC | Forced acc. (avail.) | Coverage @τe2e | Acc. accepted @τe2e | "
              "False accept. absent @τe2e | End-to-end @τe2e | End-to-end @τ95 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in sized.sort_values(["size", "method"], ascending=[False, True]).to_dict("records"):
            g = lambda k: r.get(k, np.nan)  # noqa: E731
            L.append(f"| {r['size']} | {r['method']} | {_f(g('auroc_label_available_mean'), False)} | "
                     f"{_f(g('aurc_mean'), False)} | {_f(g('forced_accuracy_available_mean'))} | "
                     f"{_f(g('coverage@tau_e2e_mean'))} | {_f(g('accuracy_among_accepted@tau_e2e_mean'))} | "
                     f"{_f(g('false_acceptance_absent@tau_e2e_mean'))} | {_f(g('end_to_end_accuracy@tau_e2e_mean'))} | "
                     f"{_f(g('end_to_end_accuracy@tau_95_mean'))} |")
        boot = _read(gd / "bootstrap.json")
        L += ["", f"Cluster-bootstrap 95% intervals ({boot['n_bootstrap']} resamples; clusters carry their images and "
              "episodes): see " + _link(gd / "bootstrap.json", out_dir) + ".", "",
              "| Method, size | End-to-end @τe2e | AUROC avail. |", "|---|---|---|"]
        for key, v in sorted(boot["metrics"].items()):
            L.append(f"| {key} | {_ci(v['end_to_end@tau_e2e'])} | {_ci(v['auroc_label_available'])} |")
        fc = _read(gd / "failure_cases.json")
        L += ["", f"Failure cases ({fc['method']}, {fc['threshold']}; full list {_link(gd / 'failure_cases.json', out_dir)}):", ""]
        for title, key in (("Impure clusters", "impure_clusters"), ("Confident wrong labels", "confident_wrong_labels")):
            L.append(f"* {title}: " + "; ".join(f"cluster {x['cluster_id']} → '{x['predicted']}' (score {x['score']:.3f}, "
                                              f"purity {x['purity']:.2f}; {x['composition_top3']})" for x in fc[key][:4]))
        L.append("* Absent label replaced and accepted: " + "; ".join(
            f"{x['episode_id']}: cluster of '{x['true_dominant']}' named '{x['predicted']}' (score {x['score']:.3f})"
            for x in fc["absent_label_replaced_and_accepted"][:4]))
        L += ["", f"Other files: {_link(gd / 'metrics_by_episode.csv', out_dir)}, {_link(gd / 'risk_coverage_curves.csv', out_dir)}, "
              f"{_link(gd / 'cluster_diagnostics.csv', out_dir)}, {_link(gd / 'paired_diagnostic.csv', out_dir)}, "
              f"{_link(gd / 'episodes.csv', out_dir)}.", ""]
    out.write_text("\n".join(L) + "\n")
    return out
