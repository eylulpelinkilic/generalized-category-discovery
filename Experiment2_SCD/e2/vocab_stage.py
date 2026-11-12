"""Stage V: rebuild the official WordNet vocabulary and verify it against the authors' text embeddings."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np

from e1.provenance import StageRecord, sha256_file, write_json
from e2.config import Layout, chash, resolve


def ensure_wordnet(nltk_dir: Path):
    import nltk

    nltk_dir.mkdir(parents=True, exist_ok=True)
    if not (nltk_dir / "corpora" / "wordnet.zip").exists() and not (nltk_dir / "corpora" / "wordnet").exists():
        nltk.download("wordnet", download_dir=str(nltk_dir), quiet=True)
    from scd.vocabulary import load_wordnet

    return load_wordnet(nltk_dir)


def ensure_official_weights(cfg: dict) -> Path:
    """Download (if needed) and checksum the authors' zip; extract the noun weights; return their path."""
    from e1.sources import fetch

    vc = cfg["vocabulary"]
    zpath = resolve(vc["official_zip"])
    if not zpath.exists():
        fetch(vc["official_zip_url"], zpath)
    digest = sha256_file(zpath)
    if digest != vc["official_zip_sha256"]:
        raise RuntimeError(f"{zpath.name}: sha256 {digest} != {vc['official_zip_sha256']}")
    out = zpath.parent / Path(vc["official_member"]).name
    if not out.exists():
        with zipfile.ZipFile(zpath) as z:
            out.write_bytes(z.read(vc["official_member"]))
    return out


def run(cfg: dict) -> Path:
    import torch

    from e1.clip_model import load_clip
    from e1.gcd_train import select_device
    from scd.text import check_alignment, load_official_weights
    from scd.vocabulary import official_wordnet_vocabulary

    layout = Layout(cfg)
    vc = cfg["vocabulary"]
    out = layout.vocabulary()
    rec = StageRecord(out, "vocabulary", chash(vc))
    wn = ensure_wordnet(resolve(vc["nltk_dir"]))
    vocab = official_wordnet_vocabulary(wn)
    wpath = ensure_official_weights(cfg)
    W = load_official_weights(wpath)                       # float32 [67,893, 512]
    device = select_device(cfg.get("device", "auto"))
    model, _, tok = load_clip(vc["alignment_clip"], device)
    model.visual = torch.nn.Identity()          # only the text tower is used; frees ~340 MB
    if device.type == "mps":
        torch.mps.empty_cache()
    dup = vocab.duplicate_columns
    extra = sorted(set(dup[:40] + [vocab.first_index[vocab.names[v]] for v in dup[:40]]))
    res = check_alignment(W, vocab.raw, model, tok, device, n_sample=vc["alignment_sample"],
                          seed=vc["alignment_seed"], extra_positions=extra)
    # duplicate columns: same normalised name -> are the stored vectors the same?
    pair_cos = np.array([float(W[v] @ W[vocab.first_index[vocab.names[v]]]) for v in dup])
    res["duplicate_columns"] = {"n": len(dup), "min_cosine_to_first": float(pair_cos.min()),
                                "n_cosine_below_0.999": int((pair_cos < 0.999).sum()),
                                "examples": [[vocab.raw[vocab.first_index[vocab.names[v]]], vocab.raw[v]] for v in dup[:8]]}
    # sorted order as a negative control: the same exact check must fail
    order = np.argsort(np.array(vocab.raw, dtype=object))
    ctrl = check_alignment(W, [vocab.raw[i] for i in order], model, tok, device, n_sample=40,
                           seed=vc["alignment_seed"], retrieval=False)
    res["negative_control_sorted_order"] = {"exact_min_cosine": ctrl["exact_min_cosine"],
                                            "passed": ctrl["passed"]}
    res["weights_file"] = {"path": str(wpath), "sha256": sha256_file(wpath), "shape_stored": [int(W.shape[1]), int(W.shape[0])],
                           "dtype_stored": "float16"}
    res["vocabulary"] = vocab.describe()
    res["alignment_model"] = {**vc["alignment_clip"], "note": "open_clip fp32; the authors used the OpenAI package in fp16"}
    write_json(out / "alignment.json", res)
    (out / "vocabulary_raw.txt").write_text("\n".join(vocab.raw) + "\n")
    rec.complete(passed=res["passed"], n_columns=len(vocab), sha256_raw=vocab.sha256())
    if not res["passed"]:
        raise RuntimeError(f"vocabulary/embedding alignment failed: {json.dumps(res)[:400]}")
    return out


def load(cfg: dict):
    """The verified vocabulary and the official weights ([N, D] float32). Refuses an unverified vocabulary."""
    from scd.text import load_official_weights
    from scd.vocabulary import official_wordnet_vocabulary

    out = Layout(cfg).vocabulary()
    a = json.loads((out / "alignment.json").read_text()) if (out / "alignment.json").exists() else None
    if not a or not a["passed"]:
        raise RuntimeError("run `python -m e2 vocab` first: the vocabulary must be verified before use")
    wn = ensure_wordnet(resolve(cfg["vocabulary"]["nltk_dir"]))
    vocab = official_wordnet_vocabulary(wn)
    if vocab.sha256() != a["vocabulary"]["sha256_raw"]:
        raise RuntimeError("rebuilt vocabulary differs from the verified one")
    return wn, vocab, load_official_weights(a["weights_file"]["path"])
