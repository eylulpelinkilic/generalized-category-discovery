"""Image embeddings for both protocols.

CUB (protocol B/R): Experiment 1 shipped its similarity aggregates but not its image embeddings. They are
recomputed here with Experiment 1's own model loader and preprocessing, for exactly Experiment 1's rows, and
accepted only if (a) the cache identity (model, activations, preprocessing, open_clip version, dtype) equals the
one recorded by Experiment 1, (b) the discovery aggregates match Experiment 1's saved `A`, and (c) Experiment 1's
saved cluster predictions are reproduced on every population.

ImageNet-100 (protocol A): one pass over the GCD train split computes CLIP ViT-B/16 (official OpenAI package, fp16
on CUDA), DINO ViT-B/16 and the authors' GCD ViT-B/16, all with CLIP's preprocessing as in the official code
(`test_transform = preprocess`). Features are L2-normalised and stored by sample id.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from e1.provenance import StageRecord, save_npz_atomic, write_json
from e2.config import Layout, chash, e1_layout, resolve


# ----------------------------------------------------------------------------- CUB
def image_root(cfg: dict, manifest) -> Path:
    """Where the dataset's images are: the extracted official archive (CUB) or the manifest's own root."""
    if cfg["dataset"].get("image_source") == "manifest":
        return Path(manifest.meta["image_root"])
    return resolve(cfg["dataset"]["archive_cache"]) / "CUB_200_2011"


def e1_grounding_dir(cfg: dict, clip_name: str, cluster_seed: int | None = None) -> Path:
    e1c, e1l = e1_layout(cfg)
    e = cfg["e1"]
    g = e1l.grounding(e["train_seed"], e["k_mode"], e["cluster_seed"] if cluster_seed is None else cluster_seed, clip_name)
    st = g / "status.json"
    if not st.exists() or json.loads(st.read_text()).get("status") != "completed":
        raise FileNotFoundError(f"Experiment 1 grounding run not completed: {g}")
    return g


def cub_embeddings(cfg: dict, clip_name: str) -> dict:
    import torch
    from torch.utils.data import DataLoader

    from e1.clip_extract import cache_identity
    from e1.clip_model import describe, load_clip
    from e1.config import stable_hash
    from e1.datasets import ImageList

    from e1.gcd_train import select_device
    from e1.manifests import Manifest

    e1c, e1l = e1_layout(cfg)
    g = e1_grounding_dir(cfg, clip_name)
    rows = pd.read_csv(g / "rows.csv")
    manifest = Manifest(e1l.data)
    ids = list(dict.fromkeys(rows["sample_id"]))
    model_cfg = cfg["clip"]["models"][clip_name]
    if model_cfg != e1c["clip"]["models"][clip_name]:
        raise RuntimeError("CLIP definition differs from Experiment 1's")
    e1_index = json.loads((e1l.clip(clip_name) / "index.json").read_text())
    device = select_device(cfg.get("device", "auto"))
    model, preprocess, tok = load_clip(model_cfg, device)
    ident = cache_identity(describe(model, preprocess, tok, model_cfg), model_cfg)
    ident_hash = stable_hash(ident)
    out = Layout(cfg).embeddings(clip_name, ident_hash)
    path = out / "embeddings.npz"
    rec = StageRecord(out, "cub_embeddings", ident_hash, {"e1_grounding": str(g), "n_ids": len(ids)})
    if ident_hash != e1_index["identity_hash"]:
        raise RuntimeError(f"cache identity {ident_hash} != Experiment 1's {e1_index['identity_hash']}: "
                           f"{json.dumps(ident)} vs {json.dumps(e1_index['identity'])}")
    if not (path.exists() and np.load(path)["sample_ids"].tolist() == ids):
        if cfg["dataset"].get("image_source") != "manifest":
            from e2.assets import fetch_cub_files as fetch_parallel

            cache = resolve(cfg["dataset"]["archive_cache"])
            lookup = dict(zip(manifest.table["sample_id"], manifest.table["rel_path"]))
            fetch = fetch_parallel([lookup[s] for s in ids], cache, Layout(cfg).cache / "archives")
            write_json(out / "image_fetch.json", fetch)
        os.environ["E1_IMAGE_ROOT"] = str(image_root(cfg, manifest))
        m2 = Manifest(e1l.data)
        loader = DataLoader(ImageList(m2, ids, preprocess), batch_size=cfg["clip"]["batch_size"],
                            num_workers=cfg["clip"]["num_workers"], shuffle=False)
        emb = np.zeros((len(ids), int(model.visual.output_dim)), np.float32)
        with torch.no_grad():
            for images, pos in loader:
                z = model.encode_image(images.to(device)).float()
                emb[pos.numpy()] = torch.nn.functional.normalize(z, dim=-1).cpu().numpy()
        save_npz_atomic(path, sample_ids=np.array(ids).astype(str), embeddings=emb, identity_hash=np.array(ident_hash))
    val = validate_against_e1(cfg, clip_name, g, path)
    write_json(out / "validation.json", val)
    rec.complete(**{k: val[k] for k in ("passed", "aggregate_max_abs_diff", "prediction_agreement")})
    if not val["passed"]:
        raise RuntimeError(f"recomputed embeddings do not reproduce Experiment 1: {val}")
    return {"dir": str(out), **{k: val[k] for k in ("passed", "aggregate_max_abs_diff", "prediction_agreement")}}


def load_e1_inputs(cfg: dict, clip_name: str, cluster_seed: int | None = None) -> dict:
    """Experiment 1 rows, episodes, columns, per-class text embeddings and our validated image embeddings."""
    g = e1_grounding_dir(cfg, clip_name, cluster_seed)
    g0 = e1_grounding_dir(cfg, clip_name)
    emb_dirs = sorted((Layout(cfg).base / "inputs").glob(f"clip_{clip_name}_*"))
    emb_dirs = [d for d in emb_dirs if (d / "validation.json").exists() and json.loads((d / "validation.json").read_text())["passed"]]
    if not emb_dirs:
        raise FileNotFoundError(f"run `python -m e2 --config ... cub-embed --clip {clip_name}` first")
    data = np.load(emb_dirs[-1] / "embeddings.npz")
    pos = {s: i for i, s in enumerate(data["sample_ids"].tolist())}
    rows = pd.read_csv(g / "rows.csv")
    missing = [s for s in rows["sample_id"] if s not in pos]
    if missing:
        raise RuntimeError(f"{len(missing)} rows of {g.name} have no embedding")
    emb = data["embeddings"][[pos[s] for s in rows["sample_id"]]]
    sel = json.loads((g0 / "selection.json").read_text())
    text = {s: np.load(g0 / f"text_embeddings_{s}.npz") for s in ("single", "ensemble80", "mcm")}
    return {"dir": g, "selection_dir": g0, "rows": rows, "emb": emb, "episodes": pd.read_csv(g0 / "episodes.csv"),
            "columns": pd.read_csv(g0 / "columns.csv", keep_default_na=False), "selection": sel,
            "class_emb": {s: t["per_class"] for s, t in text.items()},
            "class_ids": text["ensemble80"]["class_ids"], "embedding_dir": emb_dirs[-1]}


def validate_against_e1(cfg: dict, clip_name: str, g: Path, emb_path: Path) -> dict:
    from e1 import ground
    from e1.matrices import aggregate, select_members, similarity

    data = np.load(emb_path)
    pos = {s: i for i, s in enumerate(data["sample_ids"].tolist())}
    rows = pd.read_csv(g / "rows.csv")
    emb = data["embeddings"][[pos[s] for s in rows["sample_id"]]]
    sel = json.loads((g / "selection.json").read_text())
    strategy, T = sel["prompt_strategy"], sel["T_dev"]
    text = {s: np.load(g / f"text_embeddings_{s}.npz")["per_class"] for s in ("single", "ensemble80", "mcm")}
    R = {s: similarity(emb, t) for s, t in text.items()}
    saved = np.load(g / f"discovery_aggregates_{strategy}.npz")
    K = len(saved["n_members"])
    disc = rows[rows["population"] == "discovery"]
    agg = aggregate(R[strategy], emb, select_members(disc, "cluster_id", list(range(K)), "all"))
    ok = ~np.isnan(saved["A"][:, 0])
    diff = float(np.nanmax(np.abs(agg["A"][ok] - saved["A"][ok])))
    norm_diff = float(np.nanmax(np.abs(agg["norm"][ok] - saved["mean_norm"][ok])))
    inputs = {"rows": rows, "emb": emb, "R": R, "K": K}
    episodes = pd.read_csv(g / "episodes.csv")
    shared = episodes[episodes["kind"] == "shared"]
    saved_pred = pd.read_csv(g / "cluster_predictions.csv.gz")
    saved_pred = saved_pred[saved_pred["representation"] == "all"]
    agree, n, score_diff = 0, 0, 0.0
    per_pop = {}
    for population in ("dev", "calib", "discovery"):
        ours = ground.predict(inputs, population, "all", strategy, T, shared)
        m = ours.merge(saved_pred[saved_pred["population"] == population], on=["episode_id", "cluster_id", "method"],
                       suffixes=("", "_e1"))
        m = m[m["n_used"] > 0]
        a = (m["pred_class_id"] == m["pred_class_id_e1"]).to_numpy()
        d = np.abs(m["score"] - m["score_e1"]).to_numpy()
        d = d[np.isfinite(d)]
        per_pop[population] = {"rows": int(len(m)), "agreement": float(a.mean()), "max_score_diff": float(d.max()) if len(d) else 0.0}
        agree += int(a.sum()); n += len(m); score_diff = max(score_diff, per_pop[population]["max_score_diff"])
    res = {"e1_grounding": str(g), "strategy": strategy, "T_dev": T, "aggregate_max_abs_diff": diff,
           "mean_norm_max_abs_diff": norm_diff, "n_members_equal": bool(np.array_equal(agg["n"], saved["n_members"])),
           "prediction_agreement": agree / n, "prediction_rows": n, "max_score_diff": score_diff, "by_population": per_pop,
           "criteria": "aggregate max abs diff <= 1e-3, member counts equal, prediction agreement >= 0.995"}
    res["passed"] = bool(diff <= 1e-3 and res["n_members_equal"] and res["prediction_agreement"] >= 0.995)
    return res


# ----------------------------------------------------------------------------- ImageNet-100 (protocol A)
class _ImagesWithIds:
    def __init__(self, paths, transform):
        self.paths, self.transform = list(paths), transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from e1.datasets import load_rgb

        return self.transform(load_rgb(self.paths[i])), i


def load_dino_vitb16(state_dict_path: Path, device):
    """DINO ViT-B/16 (the repository's upstream `vision_transformer`, the same definition torch.hub uses)."""
    import torch

    from e1 import upstream

    vits = upstream.vision_transformer()
    m = vits.__dict__["vit_base"]()
    sd = torch.load(str(state_dict_path), map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd and not any(k.startswith("blocks.") for k in sd):
        sd = sd["state_dict"]
    sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or [u for u in unexpected if not u.startswith("head")]:
        raise RuntimeError(f"state dict mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}")
    return m.to(device).eval()


def imagenet_features(cfg: dict, models=("clip", "dino", "gcd")) -> dict:
    """Protocol A features for the GCD train split (labelled first, then unlabelled, as upstream MergedDataset)."""
    import torch
    from torch.utils.data import DataLoader

    from e1.manifests import Manifest
    from e2 import assets

    rc = cfg["repro"]
    e1c, e1l = e1_layout(cfg)
    manifest = Manifest(e1l.data)
    ids = manifest.ids("labelled") + manifest.ids("unlabelled")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import clip  # official OpenAI package

    clip_model, preprocess = clip.load(rc["clip"]["arch"], device=device)      # fp16 on CUDA, as official
    nets = {"clip": clip_model}
    ckpts = {}
    if "dino" in models:
        ckpts["dino"] = assets.fetch_dino(cfg)
        nets["dino"] = load_dino_vitb16(ckpts["dino"]["path"], device)
    if "gcd" in models:
        ckpts["gcd"] = assets.fetch_gcd_checkpoint(cfg)
        nets["gcd"] = load_dino_vitb16(ckpts["gcd"]["path"], device)
    ident = {m: {"model": rc["clip"] if m == "clip" else ckpts[m], "preprocess": str(preprocess),
                 "dtype": "float16 (official clip.load on CUDA)" if m == "clip" else "float32",
                 "torch": torch.__version__, "manifest_sha256": manifest.sha256, "order": "labelled then unlabelled"}
             for m in nets}
    outs = {m: Layout(cfg).features(m, chash(ident[m])) for m in nets}
    todo = [m for m in nets if not (outs[m] / "features.npz").exists()]
    if todo:
        loader = DataLoader(_ImagesWithIds(manifest.paths(ids), preprocess), batch_size=rc["feature_batch_size"],
                            num_workers=rc["num_workers"], shuffle=False)
        feats = {m: None for m in todo}
        with torch.no_grad():
            for x, pos in loader:
                x = x.to(device)
                for m in todo:
                    z = nets[m].encode_image(x.half() if device.type == "cuda" else x) if m == "clip" else nets[m](x)
                    z = torch.nn.functional.normalize(z.float(), dim=-1).cpu().numpy()
                    if feats[m] is None:
                        feats[m] = np.zeros((len(ids), z.shape[1]), np.float32)
                    feats[m][pos.numpy()] = z
        for m in todo:
            rec = StageRecord(outs[m], f"features_{m}", chash(ident[m]), ident[m])
            save_npz_atomic(outs[m] / "features.npz", sample_ids=np.array(ids).astype(str), features=feats[m])
            rec.complete(n=len(ids), dim=int(feats[m].shape[1]))
    return {m: str(outs[m]) for m in nets}


def load_features(cfg: dict, model: str) -> tuple[list[str], np.ndarray]:
    dirs = [d for d in sorted((Layout(cfg).base / "features").glob(f"{model}_*")) if (d / "features.npz").exists()]
    if not dirs:
        raise FileNotFoundError(f"no {model} features; run `repro features`")
    if len(dirs) > 1:
        raise RuntimeError(f"several {model} feature caches {[d.name for d in dirs]}; remove the stale ones")
    z = np.load(dirs[0] / "features.npz")
    return z["sample_ids"].tolist(), z["features"]
