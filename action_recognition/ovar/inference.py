"""Run a backbone over dataset manifests and save per-video predictions and scores.

Every manifest row produces exactly one prediction row. Videos that cannot be
read are kept with status `failed` (counted as errors by the evaluation), so
denominators never change silently.
"""
from __future__ import annotations

import functools
import hashlib
import json
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ovar import frames as frame_ops
from ovar.backbones import build_backbone
from ovar.backbones.base import normalise, video_scores
from ovar.benchmark import load_benchmark
from ovar.cache import ArrayCache, file_identity
from ovar.config import PROJECT_ROOT, project_path, stable_hash
from ovar.manifest import load_manifest, load_vocab, resolve_video_path
from ovar.preprocess import preprocess
from ovar.sampling import sample_views
from ovar.text import build_prompts
from ovar.video import VideoDecodeError, probe, read_frames


def select_device(name: str = "auto") -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class MemoryMeter:
    """Peak accelerator memory where measurable, plus process peak RSS."""

    def __init__(self, device: str):
        self.device = device
        self.peak_accel = 0
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def update(self):
        if self.device == "mps":
            self.peak_accel = max(self.peak_accel, torch.mps.driver_allocated_memory())

    def summary(self) -> dict:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_bytes = rss if sys.platform == "darwin" else rss * 1024
        out = {"process_peak_rss_mb": round(rss_bytes / 2**20, 1)}
        if self.device == "cuda":
            out["peak_gpu_memory_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        elif self.device == "mps":
            out["peak_mps_driver_memory_mb_sampled"] = round(self.peak_accel / 2**20, 1)
        return out


def subset_per_class(manifest: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    """Deterministic small subset: `per_class` videos per class, ordered by a seeded hash."""
    key = manifest["video_id"].map(lambda v: hashlib.sha256(f"{seed}:{v}".encode()).hexdigest())
    ranked = manifest.assign(_key=key).sort_values(["action_id", "_key"])
    return ranked.groupby("action_id").head(per_class).drop(columns="_key").sort_index()


CLASS_NAME_COLUMNS = ("text_name", "official_name")


def class_texts(vocab: pd.DataFrame, protocol: dict) -> list[str]:
    """Class texts for the prompts: `text_name` (default) or the raw XOV label `official_name`."""
    column = protocol["prompts"].get("class_names", "text_name")
    if column not in CLASS_NAME_COLUMNS:
        raise ValueError(f"prompts.class_names must be one of {CLASS_NAME_COLUMNS}, got {column!r}")
    return vocab[column].tolist()


def dataset_entries(cfg: dict) -> list[dict]:
    """Normalise `datasets` (names or dicts) into dicts with manifest/vocab/root."""
    bench = load_benchmark()
    entries = []
    for item in cfg["datasets"]:
        entry = {"name": item} if isinstance(item, str) else dict(item)
        name = entry["name"]
        if "manifest" not in entry:
            split = (cfg.get("manifest_split") or {}).get(name) or bench["datasets"][name]["split"]
            entry["split"] = split
            entry["manifest"] = str(PROJECT_ROOT / "data" / "manifests" / f"{name}_{split}.csv")
        entry.setdefault("vocab", str(PROJECT_ROOT / "data" / "vocab" / f"{name}.csv"))
        entries.append(entry)
    return entries


@functools.lru_cache(maxsize=4)
def _box_table(path: str) -> dict:
    """Actor boxes CSV: dataset_name, video_id, frame_index, x1, y1, x2, y2 (pixels)."""
    table = pd.read_csv(project_path(path))
    out = {}
    for (dataset, vid), group in table.groupby(["dataset_name", "video_id"]):
        group = group.sort_values("frame_index")
        out[(dataset, vid)] = (group["frame_index"].to_numpy(),
                               group[["x1", "y1", "x2", "y2"]].to_numpy(float))
    return out


def actor_boxes(path: str, dataset: str, vid: str, indices: np.ndarray) -> np.ndarray:
    """Box of the nearest annotated frame for each sampled frame index."""
    entry = _box_table(path).get((dataset, vid))
    if entry is None:
        raise ValueError(f"no actor boxes for {dataset}/{vid}")
    frames, boxes = entry
    nearest = np.abs(frames[None, :] - np.asarray(indices)[:, None]).argmin(1)
    return boxes[nearest]


def _video_path(entry: dict, rel: str, paths: dict) -> Path:
    if entry.get("root"):
        return project_path(entry["root"]) / rel
    return resolve_video_path(entry["name"], rel, paths)


def _environment(device: str) -> dict:
    import open_clip

    return {
        "python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
        "open_clip": open_clip.__version__, "numpy": np.__version__, "device": device,
        "processor": platform.processor(),
    }


def run(cfg: dict, *, run_dir=None, per_class: int | None = None, limit: int | None = None,
        temporal: str = "ordered", pixel: dict | None = None, variant_name: str | None = None,
        allow_download: bool = False, backbone=None) -> Path:
    """Run inference for every dataset in `cfg`; returns the run directory."""
    protocol, model_cfg = cfg["protocol"], cfg["model"]
    device = select_device(cfg.get("device", "auto"))
    cache_root = project_path(cfg["paths"].get("cache_dir", "cache"))
    suffix = f"__{variant_name}" if variant_name else ""
    run_dir = Path(run_dir) if run_dir else \
        project_path(cfg["paths"].get("runs_dir", "runs")) / f"{cfg['run_name']}{suffix}"
    entries = dataset_entries(cfg)
    roots = cfg["paths"].get("dataset_roots") or {}
    unconfigured = [e["name"] for e in entries if not e.get("root") and not roots.get(e["name"])]
    if unconfigured:
        raise RuntimeError(f"dataset_roots not set for {unconfigured} in {cfg['paths'].get('_source')}")
    absent = [e["name"] for e in entries
              if not project_path(e.get("root") or roots[e["name"]]).is_dir()]
    if absent:
        raise RuntimeError(f"dataset root directories do not exist for {absent} "
                           f"(configured in {cfg['paths'].get('_source')}); see docs/data_preparation.md")
    run_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    if backbone is None:
        backbone = build_backbone(model_cfg, protocol, device, cache_root, allow_download=allow_download)
    load_s = time.time() - started
    identity = backbone.identity()
    synthetic = bool(cfg.get("synthetic")) or identity["backbone"] == "fake_for_tests"
    embeddings = ArrayCache(cache_root / "embeddings")
    config_id = stable_hash({"model": identity, "protocol": protocol, "temporal": temporal,
                             "pixel": pixel})[:12]
    memory = MemoryMeter(device)
    summary = {}

    for entry in entries:
        name = entry["name"]
        manifest = load_manifest(name, entry.get("split", ""), path=entry["manifest"])
        vocab = load_vocab(name, path=entry["vocab"])
        if per_class:
            manifest = subset_per_class(manifest, per_class, cfg.get("seed", 0))
        if limit:
            manifest = manifest.head(limit)
        manifest = manifest.reset_index(drop=True)

        prompts = build_prompts(class_texts(vocab, protocol), protocol["prompts"]["template_set"])
        text_key = stable_hash({"model": identity, "prompts": prompts})
        class_emb = embeddings.get(text_key)
        if class_emb is None:
            class_emb = backbone.encode_text(prompts).numpy().astype(np.float32)
            embeddings.put(text_key, class_emb, {"kind": "text", "model": identity, "dataset": name,
                                                 "template_set": protocol["prompts"]["template_set"]})
        class_emb = normalise(class_emb.astype(np.float64))

        rows, scores = [], np.full((len(manifest), len(vocab)), np.nan, dtype=np.float32)
        decode_total = encode_total = 0.0
        for i, row in manifest.iterrows():
            record = {k: row[k] for k in ("dataset_name", "video_id", "source_group_id",
                                          "source_group_basis", "action_id", "action_name",
                                          "category_status")}
            record.update(status="ok", error="", cached=False, decode_s=0.0, encode_s=0.0)
            try:
                path = _video_path(entry, row["video_path"], cfg["paths"])
                if not path.is_file():
                    raise FileNotFoundError(f"missing file {path}")
                temporal_in_key = None if backbone.frame_independent else temporal
                key = stable_hash({
                    "model": identity, "sampling": protocol["sampling"], "spatial": protocol["spatial"],
                    "temporal": temporal_in_key, "pixel": pixel,
                    "video": {"dataset": name, "path": row["video_path"], **file_identity(path)},
                })
                emb = embeddings.get(key)
                if emb is None:
                    t0 = time.time()
                    meta = probe(path)
                    views = sample_views(meta.num_frames, meta.fps, protocol["sampling"])
                    if temporal_in_key:
                        views = [v[frame_ops.temporal_positions(temporal, len(v), f"{cfg.get('seed', 0)}:"
                                                                f"{name}:{row['video_id']}:{j}")]
                                 for j, v in enumerate(views)]
                    pixels = read_frames(path, np.concatenate(views), meta)
                    if pixel and pixel["kind"] == "actor_crop":
                        boxes = actor_boxes(pixel["boxes"], name, row["video_id"], np.concatenate(views))
                        pixels = frame_ops.crop_to_box(pixels, boxes, pixel.get("context", 1.0))
                    elif pixel:
                        pixels = frame_ops.degrade(pixels, pixel)
                    x = preprocess(pixels, protocol["spatial"])
                    x = x.view(len(views), len(views[0]), *x.shape[1:])
                    t1 = time.time()
                    emb = backbone.encode_frames(x).cpu().numpy().astype(np.float32)
                    memory.update()
                    record.update(decode_s=round(t1 - t0, 4), encode_s=round(time.time() - t1, 4))
                    embeddings.put(key, emb, {"kind": "video", "dataset": name, "video_id": row["video_id"],
                                              "video_meta": meta.to_dict(),
                                              "frame_indices": [v.tolist() for v in views]})
                else:
                    record["cached"] = True
                if backbone.frame_independent and temporal != "ordered":
                    # Valid only because each frame embedding ignores the other frames.
                    emb = np.stack([e[frame_ops.temporal_positions(
                        temporal, len(e), f"{cfg.get('seed', 0)}:{name}:{row['video_id']}:{j}")]
                        for j, e in enumerate(emb)])
                scores[i] = video_scores(emb, class_emb)
                decode_total += record["decode_s"]
                encode_total += record["encode_s"]
            except (VideoDecodeError, FileNotFoundError, ValueError) as exc:
                record.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:300])
            rows.append(record)

        pred = pd.DataFrame(rows)
        ok = pred["status"] == "ok"
        top = np.argsort(-np.nan_to_num(scores, nan=-np.inf), axis=1)
        names = vocab["official_name"].tolist()
        pred["pred_id"] = np.where(ok, top[:, 0], -1)
        pred["pred_name"] = [names[p] if p >= 0 else "" for p in pred["pred_id"]]
        pred["correct"] = (ok & (pred["pred_id"] == pred["action_id"])).astype(int)
        gt = np.clip(pred["action_id"].to_numpy(), 0, len(vocab) - 1)
        pred["gt_score"] = np.where(ok, scores[np.arange(len(pred)), gt], np.nan)
        pred["pred_score"] = np.where(ok, scores[np.arange(len(pred)), top[:, 0]], np.nan)
        second = scores[np.arange(len(pred)), top[:, 1]] if len(vocab) > 1 else np.nan
        pred["top1_minus_top2"] = np.where(ok, pred["pred_score"] - second, np.nan)
        pred["top5_ids"] = [" ".join(map(str, t[:5])) if o else "" for t, o in zip(top, ok)]
        pred["model_name"] = model_cfg.get("name", identity["backbone"])
        pred["protocol_name"] = protocol.get("name", "")
        pred["temporal_transform"] = temporal
        pred["pixel_transform"] = json.dumps(pixel) if pixel else ""
        pred["config_id"] = config_id

        out = run_dir / name
        out.mkdir(parents=True, exist_ok=True)
        pred.to_csv(out / "predictions.csv", index=False)
        np.save(out / "scores.npy", scores)
        vocab[["action_id", "official_name", "text_name", "category_status"]].to_csv(
            out / "classes.csv", index=False)
        pred.loc[~ok, ["video_id", "error"]].to_csv(out / "failures.csv", index=False)
        summary[name] = {
            "manifest": str(entry["manifest"]), "videos": len(pred), "failed": int((~ok).sum()),
            "cached": int(pred["cached"].sum()), "subset": {"per_class": per_class, "limit": limit},
            "decode_s": round(decode_total, 2), "encode_s": round(encode_total, 2),
        }

    record = {
        "run_name": cfg["run_name"], "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "synthetic": synthetic, "config_id": config_id,
        "comparison": protocol.get("comparison", "unspecified"), "variant_name": variant_name,
        "model": model_cfg, "protocol": protocol, "temporal_transform": temporal, "pixel_transform": pixel,
        "backbone": backbone.provenance(), "environment": _environment(device),
        "timing": {"model_load_s": round(load_s, 2), "total_s": round(time.time() - started, 2)},
        "memory": memory.summary(), "datasets": summary, "experiment_file": cfg.get("experiment_file"),
    }
    (run_dir / "run.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
    return run_dir
