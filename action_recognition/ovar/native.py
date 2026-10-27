"""Bridges to the official evaluation code for native-configuration checks.

`export_*` writes our manifests (XOV labels and class texts) in each official
repository's format, so the official test scripts score exactly the same videos
and class names. Paths follow the folder layout in the paths config (HMDB51
single or double class folders), as `ovar infer` resolves them.
`compare_openvclip` checks the official saved predictions against this
repository's Open-VCLIP adapter run, video by video. `import_tcclip` turns the
per-video scores saved by envs/tcclip_eval.py into a regular run directory.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ovar.manifest import load_manifest, load_vocab, on_disk_path


def _manifest(name: str, split: str, per_class: int | None, seed: int) -> pd.DataFrame:
    """Same selection and order as `ovar infer --per-class`, so outputs align row by row."""
    from ovar.inference import subset_per_class

    manifest = load_manifest(name, split)
    return subset_per_class(manifest, per_class, seed) if per_class else manifest


def export_openvclip(datasets: dict, out_dir, per_class: int | None = None, seed: int = 0,
                     paths: dict | None = None) -> list[Path]:
    """datasets: name -> split. Writes <out>/<name>/test.csv and <name>-index2cls.json."""
    written = []
    for name, split in datasets.items():
        manifest, vocab = _manifest(name, split, per_class, seed), load_vocab(name)
        target = Path(out_dir) / name
        target.mkdir(parents=True, exist_ok=True)
        lines = [f"{on_disk_path(name, p, paths or {})},{a}"
                 for p, a in zip(manifest["video_path"], manifest["action_id"])]
        (target / "test.csv").write_text("\n".join(lines) + "\n")
        mapping = {str(i): t for i, t in zip(vocab["action_id"], vocab["text_name"])}
        (target / f"{name}-index2cls.json").write_text(json.dumps(mapping, indent=1) + "\n")
        written.append(target)
    return written


def export_tcclip(datasets: dict, out_dir, per_class: int | None = None, seed: int = 0,
                  paths: dict | None = None) -> list[Path]:
    """Writes <out>/<name>/test.txt ("<path> <label>") and <name>_labels.csv ("id,name")."""
    written = []
    for name, split in datasets.items():
        manifest, vocab = _manifest(name, split, per_class, seed), load_vocab(name)
        target = Path(out_dir) / name
        target.mkdir(parents=True, exist_ok=True)
        lines = [f"{on_disk_path(name, p, paths or {})} {a}"
                 for p, a in zip(manifest["video_path"], manifest["action_id"])]
        (target / "test.txt").write_text("\n".join(lines) + "\n")
        pd.DataFrame({"id": vocab["action_id"], "name": vocab["text_name"]}).to_csv(
            target / f"{name}_labels.csv", index=False)
        written.append(target)
    return written


def compare_openvclip(official_results, run_dir, dataset: str, trust_pickle: bool = False) -> dict:
    """Agreement between the official `TEST.SAVE_RESULTS_PATH` pickle and an adapter run."""
    if not trust_pickle:
        raise PermissionError("the official results file is a pickle; pass --trust-pickle for a file you "
                              "produced yourself")
    with open(official_results, "rb") as f:
        preds, labels = pickle.load(f)  # noqa: S301 - user-produced file, explicit opt-in
    preds, labels = np.asarray(preds), np.asarray(labels)
    ours = pd.read_csv(Path(run_dir) / dataset / "predictions.csv", keep_default_na=False)
    manifest_ids = ours["action_id"].to_numpy()
    if len(ours) != len(preds) or not np.array_equal(manifest_ids, labels):
        raise ValueError("official results and adapter run cover different videos or label order")
    official_top1 = preds.argmax(1)
    ok = ours["status"] == "ok"
    differ = (ok & (official_top1 != ours["pred_id"].to_numpy())).to_numpy()
    return {
        "videos": int(len(ours)), "adapter_failed": int((~ok).sum()),
        "top1_agreement": float(np.mean(official_top1[ok] == ours.loc[ok, "pred_id"].to_numpy())),
        "official_accuracy": float(np.mean(official_top1 == labels)),
        "adapter_accuracy_failures_as_errors": float(ours["correct"].mean()),
        "disagreements": [{"video_id": v, "action_id": int(a), "official_pred": int(o), "adapter_pred": int(p)}
                          for v, a, o, p in zip(ours.loc[differ, "video_id"], labels[differ],
                                                official_top1[differ], ours.loc[differ, "pred_id"])],
    }


def import_tcclip(cfg: dict, scores_dir, run_dir, per_class: int | None = None,
                  provenance: dict | None = None) -> Path:
    """Run directory (predictions, scores, run.json) from TC-CLIP's saved per-video scores.

    `cfg` is a loaded experiment; each dataset's `<scores_dir>/<name>.npz` must cover exactly the
    exported list (same subset and order as `export-native --backbone tcclip`), with labels equal
    to the manifest's. Nothing is reordered silently: rows are placed by `file_id`, and any
    missing, duplicated or mislabelled row is an error.
    """
    from datetime import datetime, timezone

    from ovar.config import stable_hash
    from ovar.inference import write_dataset_outputs

    run_dir = Path(run_dir)
    datasets, sources = {}, {}
    for name in cfg["datasets"]:
        split = load_benchmark_split(name)
        manifest, vocab = _manifest(name, split, per_class, cfg.get("seed", 0)), load_vocab(name)
        manifest = manifest.reset_index(drop=True)
        data = np.load(Path(scores_dir) / f"{name}.npz", allow_pickle=False)
        file_id, label, scores = data["file_id"].astype(int), data["label"].astype(int), data["scores"]
        meta = json.loads(str(data["meta"]))
        if sorted(file_id.tolist()) != list(range(len(manifest))):
            raise ValueError(f"{name}: saved file ids do not cover the {len(manifest)} exported videos once each")
        if scores.shape != (len(manifest), len(vocab)):
            raise ValueError(f"{name}: scores have shape {scores.shape}, expected {(len(manifest), len(vocab))}")
        order = np.argsort(file_id)
        if not np.array_equal(label[order], manifest["action_id"].to_numpy()):
            raise ValueError(f"{name}: saved labels differ from the manifest labels")
        pred = manifest[["dataset_name", "video_id", "source_group_id", "source_group_basis", "action_id",
                         "action_name", "category_status"]].copy()
        pred = pred.assign(status="ok", error="", cached=False, decode_s=0.0, encode_s=0.0)
        config_id = stable_hash({"model": cfg["model"], "protocol": cfg["protocol"], "official": {
            k: meta[k] for k in ("num_frames", "num_clip", "num_crop", "seed", "wise_ft", "opt_level")}})[:12]
        write_dataset_outputs(run_dir / name, pred, scores[order].astype(np.float32), vocab,
                              model_name=cfg["model"].get("name", "tcclip"),
                              protocol_name=cfg["protocol"].get("name", ""), config_id=config_id)
        datasets[name] = {"manifest": f"data/manifests/{name}_{split}.csv", "videos": len(manifest),
                          "failed": 0, "cached": 0, "subset": {"per_class": per_class, "limit": None},
                          "decode_s": 0.0, "encode_s": 0.0}
        # The official Acc@1 should equal this up to exact score ties (topk vs argsort order).
        meta["recomputed_acc1"] = float(100 * np.mean(scores.argmax(1) == label))
        sources[name] = meta
    record = {
        "run_name": run_dir.name, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "synthetic": False, "config_id": config_id,
        "comparison": cfg["protocol"].get("comparison", "unspecified"), "variant_name": None,
        "model": cfg["model"], "protocol": cfg["protocol"], "temporal_transform": "ordered",
        "pixel_transform": None, "backbone": provenance or {}, "official_scores": sources,
        "environment": (provenance or {}).get("environment", {"device": "cuda"}),
        "timing": {"model_load_s": None, "total_s": None}, "memory": {}, "datasets": datasets,
        "experiment_file": cfg.get("experiment_file"),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
    return run_dir


def load_benchmark_split(name: str) -> str:
    from ovar.benchmark import load_benchmark

    return load_benchmark()["datasets"][name]["split"]
