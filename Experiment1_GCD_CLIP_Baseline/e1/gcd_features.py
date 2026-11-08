"""Backbone (CLS) features of the trained GCD model, keyed by sample id."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from e1 import upstream
from e1.datasets import ImageList
from e1.gcd_train import build_backbone, select_device
from e1.manifests import Manifest
from e1.provenance import StageRecord, save_npz_atomic, sha256_file, stage_completed
from e1.config import stable_hash

GROUPS = ("labelled", "unlabelled", "heldout_dev", "heldout_calib")


def feature_path(run_dir: Path, group: str) -> Path:
    return run_dir / "features" / f"{group}.npz"


@torch.no_grad()
def extract(cfg: dict, layout, train_seed: int, checkpoint: str = "final_backbone.pt") -> dict:
    gcd = cfg["gcd_train"]
    run_dir = layout.gcd_run(train_seed)
    ckpt = run_dir / "checkpoints" / checkpoint
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} missing: training has not finished (see train_log.jsonl)")
    ckpt_sha = sha256_file(ckpt)
    out_dir = run_dir / "features"
    stage_hash = stable_hash({"checkpoint_sha256": ckpt_sha, "transform": "upstream test transform"})
    if stage_completed(out_dir, stage_hash):
        return {"status": "cached", "dir": str(out_dir)}
    manifest = Manifest(layout.data)
    record = StageRecord(out_dir, "gcd_features", stage_hash, {"checkpoint": str(ckpt), "checkpoint_sha256": ckpt_sha,
                                                               "manifest_sha256": manifest.sha256})
    device = select_device(cfg.get("device", "auto"))
    model, _ = build_backbone(gcd, None, require_weights=False)   # architecture only; weights loaded below
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.to(device).eval()
    _, test_tf = upstream.get_transform(gcd["transform"], image_size=gcd["image_size"],
                                        interpolation=gcd["interpolation"], crop_pct=gcd["crop_pct"])
    counts = {}
    for group in GROUPS:
        ids = manifest.ids(group)
        loader = DataLoader(ImageList(manifest, ids, test_tf), batch_size=gcd["feature_batch_size"],
                            num_workers=gcd["num_workers"], shuffle=False)
        feats = np.zeros((len(ids), model.embed_dim), dtype=np.float32)
        for images, pos in loader:
            feats[pos.numpy()] = model(images.to(device)).float().cpu().numpy()
        save_npz_atomic(feature_path(run_dir, group), sample_ids=np.array(ids), features=feats,
                        checkpoint_sha256=np.array(ckpt_sha))
        counts[group] = len(ids)
    record.complete(counts=counts)
    return {"status": "extracted", "counts": counts}


def load_features(run_dir: Path, group: str, normalise: bool = True) -> tuple[list[str], np.ndarray]:
    data = np.load(feature_path(run_dir, group))
    feats = data["features"].astype(np.float32)
    if normalise:
        feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    return data["sample_ids"].tolist(), feats
