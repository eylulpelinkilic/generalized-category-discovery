"""Normalised CLIP image/text embeddings, cached in resumable shards keyed by sample id."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from e1.clip_model import describe, load_clip
from e1.config import stable_hash
from e1.datasets import ImageList
from e1.gcd_train import select_device
from e1.manifests import Manifest
from e1.provenance import save_npz_atomic, write_json

IMAGE_GROUPS = ("labelled", "unlabelled", "heldout_dev", "heldout_calib")


def cache_identity(model_info: dict, model_cfg: dict) -> dict:
    return {"model": model_cfg, "activations": model_info["activations"], "preprocess": model_info["preprocess"],
            "open_clip_version": model_info["open_clip_version"], "dtype": "float32", "normalised": True,
            "checkpoint": model_cfg.get("checkpoint_sha256", model_info.get("pretrained_source"))}


@torch.no_grad()
def extract_images(cfg: dict, layout, clip_name: str, shard_size: int = 4096) -> dict:
    model_cfg = cfg["clip"]["models"][clip_name]
    out = layout.clip(clip_name)
    device = select_device(cfg.get("device", "auto"))
    model, preprocess, tokenizer = load_clip(model_cfg, device)
    info = describe(model, preprocess, tokenizer, model_cfg)
    ident = cache_identity(info, model_cfg)
    ident_hash = stable_hash(ident)
    manifest = Manifest(layout.data)
    ids = manifest.ids(*IMAGE_GROUPS)
    shard_dir = out / "image_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    written = reused = 0
    for s, start in enumerate(range(0, len(ids), shard_size)):
        shard_ids = ids[start:start + shard_size]
        path = shard_dir / f"shard_{s:05d}.npz"
        if path.exists():
            data = np.load(path)
            if str(data["identity_hash"]) == ident_hash and data["sample_ids"].tolist() == shard_ids:
                reused += 1
                continue
        loader = DataLoader(ImageList(manifest, shard_ids, preprocess), batch_size=cfg["clip"]["batch_size"],
                            num_workers=cfg["clip"]["num_workers"], shuffle=False)
        emb = np.zeros((len(shard_ids), info["embed_dim"]), dtype=np.float32)
        for images, pos in loader:
            z = model.encode_image(images.to(device)).float()
            emb[pos.numpy()] = torch.nn.functional.normalize(z, dim=-1).cpu().numpy()
        save_npz_atomic(path, sample_ids=np.array(shard_ids), embeddings=emb, identity_hash=np.array(ident_hash))
        written += 1
    index = {"identity": ident, "identity_hash": ident_hash, "model_description": info, "n_images": len(ids),
             "shard_size": shard_size, "n_shards": -(-len(ids) // shard_size), "manifest_sha256": manifest.sha256}
    write_json(out / "index.json", index)
    load_images(out, ids)   # coverage check
    return {"dir": str(out), "shards_written": written, "shards_reused": reused, "n_images": len(ids)}


def load_images(out_dir, sample_ids: list[str]) -> np.ndarray:
    """Embeddings for `sample_ids` in that order; errors on missing, duplicate or foreign shards."""
    out_dir = Path(out_dir)
    index = json.loads((out_dir / "index.json").read_text())
    rows, chunks = {}, []
    for k, path in enumerate(sorted((out_dir / "image_shards").glob("shard_*.npz"))):
        data = np.load(path)
        if str(data["identity_hash"]) != index["identity_hash"]:
            raise RuntimeError(f"{path} was written by a different model/preprocessing configuration")
        chunks.append(data["embeddings"])
        for j, sid in enumerate(data["sample_ids"].tolist()):
            if sid in rows:
                raise RuntimeError(f"duplicate sample id {sid} in CLIP cache")
            rows[sid] = (k, j)
    missing = [s for s in sample_ids if s not in rows]
    if missing:
        raise RuntimeError(f"{len(missing)} sample ids missing from the CLIP cache, e.g. {missing[:3]}")
    return np.stack([chunks[rows[s][0]][rows[s][1]] for s in sample_ids]) if sample_ids else np.zeros((0, 0))


@torch.no_grad()
def encode_texts(model, tokenizer, texts: list[str], device, batch: int = 256) -> np.ndarray:
    out = []
    for s in range(0, len(texts), batch):
        z = model.encode_text(tokenizer(texts[s:s + batch]).to(device)).float()
        out.append(torch.nn.functional.normalize(z, dim=-1).cpu().numpy())
    return np.concatenate(out)
