"""Diagnostic suite on a designated development subset (not the evaluation lists).

Each variant is a normal inference run that differs from the reference run in
one factor (temporal order, actor crop, or a synthetic degradation). Comparisons
are paired per video. Observations only: a prediction change does not by itself
show that a model fails, and a crop can remove action evidence or add artefacts.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ovar import inference
from ovar.config import load_experiment, load_yaml, project_path, CONFIG_DIR
from ovar.metrics import cluster_bootstrap, interval


def load_diagnostic(path) -> dict:
    path = Path(path)
    if not path.exists():
        path = CONFIG_DIR / "diagnostics" / f"{path}.yaml"
    return load_yaml(path)


def variants(diag: dict) -> list[dict]:
    """Variant specs: name, family, temporal, pixel."""
    out = [{"name": f"temporal_{kind}", "family": "temporal", "temporal": kind, "pixel": None}
           for kind in diag.get("temporal", [])]
    for spec in diag.get("degradations", []):
        spec = dict(spec)
        out.append({"name": spec.pop("name"), "family": "degradation", "temporal": "ordered", "pixel": spec})
    boxes = diag.get("actor_boxes")
    for spec in diag.get("spatial", []):
        spec = dict(spec)
        name = spec.pop("name")
        if not boxes:
            out.append({"name": name, "family": "spatial", "skipped": "no actor boxes configured"})
        else:
            out.append({"name": name, "family": "spatial", "temporal": "ordered",
                        "pixel": {"kind": "actor_crop", "boxes": boxes, **spec}})
    return out


def compare(reference: pd.DataFrame, variant: pd.DataFrame, n_boot: int, seed: int) -> tuple[dict, pd.DataFrame]:
    merged = reference.merge(variant, on="video_id", suffixes=("_ref", "_var"), validate="one_to_one")
    if len(merged) != len(reference):
        raise ValueError("variant run does not cover the reference videos")
    both_ok = (merged["status_ref"] == "ok") & (merged["status_var"] == "ok")
    changed = both_ok & (merged["pred_id_ref"] != merged["pred_id_var"])
    diff = (merged["correct_var"] - merged["correct_ref"]).to_numpy().astype(float)
    groups = merged["source_group_id_ref"].to_numpy()
    boot = cluster_bootstrap(diff, groups, {"all": np.ones(len(merged), bool)}, n_boot, seed)["all"]
    summary = {
        "n_videos": len(merged), "n_both_decoded": int(both_ok.sum()),
        "acc_reference": float(merged["correct_ref"].mean()), "acc_variant": float(merged["correct_var"].mean()),
        "delta_acc": float(diff.mean()), "delta_acc_ci95": interval(boot),
        "prediction_change_rate": float(changed.sum() / max(both_ok.sum(), 1)),
        "variant_failed": int((merged["status_var"] != "ok").sum()),
        "reference_failed": int((merged["status_ref"] != "ok").sum()),
        "correct_to_wrong": int(((merged["correct_ref"] == 1) & (merged["correct_var"] == 0)).sum()),
        "wrong_to_correct": int(((merged["correct_ref"] == 0) & (merged["correct_var"] == 1)).sum()),
    }
    per_video = merged.loc[changed | (merged["correct_ref"] != merged["correct_var"]), [
        "video_id", "action_name_ref", "category_status_ref", "pred_name_ref", "pred_name_var",
        "correct_ref", "correct_var"]].rename(columns={"action_name_ref": "action_name",
                                                         "category_status_ref": "category_status"})
    return summary, per_video


def per_class_delta(reference: pd.DataFrame, variant: pd.DataFrame) -> pd.DataFrame:
    """Per-class accuracy change; prediction changes counted only where both runs decoded."""
    merged = reference.merge(variant, on="video_id", suffixes=("_ref", "_var"))
    merged["both_ok"] = (merged["status_ref"] == "ok") & (merged["status_var"] == "ok")
    merged["changed"] = merged["both_ok"] & (merged["pred_id_ref"] != merged["pred_id_var"])
    merged["variant_failed"] = merged["status_var"] != "ok"
    out = merged.groupby(["action_name_ref", "category_status_ref"]).agg(
        n=("video_id", "size"), n_both_decoded=("both_ok", "sum"), variant_failed=("variant_failed", "sum"),
        acc_reference=("correct_ref", "mean"), acc_variant=("correct_var", "mean"),
        n_changed=("changed", "sum")).reset_index()
    out = out.rename(columns={"action_name_ref": "action_name", "category_status_ref": "category_status"})
    out["prediction_change_rate"] = out["n_changed"] / out["n_both_decoded"].where(out["n_both_decoded"] > 0)
    out["delta_acc"] = out["acc_variant"] - out["acc_reference"]
    return out.sort_values("delta_acc")


def run_suite(diag_path, overrides=(), n_boot: int = 1000, backbone=None, experiment_cfg=None) -> Path:
    diag = load_diagnostic(diag_path)
    cfg = experiment_cfg or load_experiment(diag["experiment"], overrides)
    cfg = dict(cfg)
    for key in ("datasets", "manifest_split"):
        if key in diag:
            cfg[key] = diag[key]
    cfg["run_name"] = f"diag_{diag['name']}"
    out_dir = project_path(cfg["paths"].get("runs_dir", "runs")) / "diagnostics" / diag["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    per_class = diag.get("per_class")

    if backbone is None:
        from ovar.backbones import build_backbone

        device = inference.select_device(cfg.get("device", "auto"))
        cache_root = project_path(cfg["paths"].get("cache_dir", "cache"))
        backbone = build_backbone(cfg["model"], cfg["protocol"], device, cache_root)
    inference.run(cfg, run_dir=out_dir / "reference", per_class=per_class, backbone=backbone)
    for spec in variants(diag):
        if not spec.get("skipped"):
            inference.run(cfg, run_dir=out_dir / spec["name"], per_class=per_class, temporal=spec["temporal"],
                          pixel=spec["pixel"], variant_name=spec["name"], backbone=backbone)
    (out_dir / "diagnostic.json").write_text(json.dumps({
        "config": diag, "backbone": backbone.provenance(), "frame_independent": backbone.frame_independent,
        "synthetic": json.loads((out_dir / "reference" / "run.json").read_text())["synthetic"],
    }, indent=2, default=str) + "\n")
    summarise(out_dir, n_boot)
    return out_dir


def summarise(out_dir, n_boot: int = 1000) -> Path:
    """Paired comparison of every variant run in `out_dir` against `out_dir/reference`."""
    out_dir = Path(out_dir)
    meta = json.loads((out_dir / "diagnostic.json").read_text())
    reference_dir = out_dir / "reference"
    rows, class_rows, changed_rows = [], [], []
    for spec in variants(meta["config"]):
        if spec.get("skipped"):
            rows.append({"variant": spec["name"], "family": spec["family"], "status": f"not run: {spec['skipped']}"})
            continue
        var_dir = out_dir / spec["name"]
        for dataset in json.loads((reference_dir / "run.json").read_text())["datasets"]:
            ref = pd.read_csv(reference_dir / dataset / "predictions.csv", keep_default_na=False)
            var = pd.read_csv(var_dir / dataset / "predictions.csv", keep_default_na=False)
            summary, changed = compare(ref, var, n_boot, seed=len(rows))
            note = ""
            if spec["family"] == "temporal" and meta["frame_independent"] and spec["temporal"] in (
                    "shuffled", "reversed"):
                note = "invariant by construction: frame-independent encoder with mean pooling"
            rows.append({"variant": spec["name"], "family": spec["family"], "dataset": dataset,
                         "status": "run", "note": note, **summary})
            class_rows.append(per_class_delta(ref, var).assign(variant=spec["name"], dataset=dataset))
            changed_rows.append(changed.assign(variant=spec["name"], dataset=dataset))

    pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)
    if class_rows:
        pd.concat(class_rows).to_csv(out_dir / "per_class.csv", index=False)
        pd.concat(changed_rows).to_csv(out_dir / "changed_videos.csv", index=False)
    return out_dir
