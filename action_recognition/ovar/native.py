"""Bridges to the official evaluation code for native-configuration checks.

`export_*` writes our manifests (XOV labels and class texts) in each official
repository's format, so the official test scripts score exactly the same videos
and class names. `compare_openvclip` checks the official saved predictions
against this repository's Open-VCLIP adapter run, video by video.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ovar.manifest import load_manifest, load_vocab


def _manifest(name: str, split: str, per_class: int | None, seed: int) -> pd.DataFrame:
    """Same selection and order as `ovar infer --per-class`, so outputs align row by row."""
    from ovar.inference import subset_per_class

    manifest = load_manifest(name, split)
    return subset_per_class(manifest, per_class, seed) if per_class else manifest


def export_openvclip(datasets: dict, out_dir, per_class: int | None = None, seed: int = 0) -> list[Path]:
    """datasets: name -> split. Writes <out>/<name>/test.csv and <name>-index2cls.json."""
    written = []
    for name, split in datasets.items():
        manifest, vocab = _manifest(name, split, per_class, seed), load_vocab(name)
        target = Path(out_dir) / name
        target.mkdir(parents=True, exist_ok=True)
        lines = [f"{p},{a}" for p, a in zip(manifest["video_path"], manifest["action_id"])]
        (target / "test.csv").write_text("\n".join(lines) + "\n")
        mapping = {str(i): t for i, t in zip(vocab["action_id"], vocab["text_name"])}
        (target / f"{name}-index2cls.json").write_text(json.dumps(mapping, indent=1) + "\n")
        written.append(target)
    return written


def export_tcclip(datasets: dict, out_dir, per_class: int | None = None, seed: int = 0) -> list[Path]:
    """Writes <out>/<name>/test.txt ("<path> <label>") and <name>_labels.csv ("id,name")."""
    written = []
    for name, split in datasets.items():
        manifest, vocab = _manifest(name, split, per_class, seed), load_vocab(name)
        target = Path(out_dir) / name
        target.mkdir(parents=True, exist_ok=True)
        lines = [f"{p} {a}" for p, a in zip(manifest["video_path"], manifest["action_id"])]
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
