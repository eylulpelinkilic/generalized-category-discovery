"""Prompt templates, zero-shot text weights and alignment checks for large vocabularies."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

# sha256 of "\n".join(t.format("{}") for t in imagenet_templates) in the official local_utils/clip_lang_util.py
OFFICIAL_TEMPLATES_SHA256 = "4f976686fb651bb5803d4689dcd849f75efe0ef19a3c801355bc9a6f03c8a5af"


def imagenet_templates() -> list:
    """OpenAI's 80 ImageNet templates as callables, in the official SCD order."""
    from open_clip.zero_shot_metadata import OPENAI_IMAGENET_TEMPLATES

    templates = list(OPENAI_IMAGENET_TEMPLATES)
    digest = hashlib.sha256("\n".join(t("{}") for t in templates).encode()).hexdigest()
    if digest != OFFICIAL_TEMPLATES_SHA256:
        raise RuntimeError("open_clip's ImageNet templates differ from the official SCD list")
    return templates


def load_official_weights(path: str | Path) -> np.ndarray:
    """The authors' fp16 weight file ([D, N] on CUDA) as float32 [N, D]; tensors only (weights_only)."""
    import torch

    w = torch.load(str(path), map_location="cpu", weights_only=True)
    if w.ndim != 2:
        raise ValueError(f"expected a 2-D tensor, got {tuple(w.shape)}")
    return w.float().T.contiguous().numpy()


def encode_texts(model, tokenizer, texts: list[str], device, batch: int = 1024) -> np.ndarray:
    import torch

    out = []
    with torch.no_grad():
        for s in range(0, len(texts), batch):
            z = model.encode_text(tokenizer(texts[s:s + batch]).to(device)).float()
            out.append(torch.nn.functional.normalize(z, dim=-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 0), np.float32)


def zeroshot_weights(model, tokenizer, raw_names: list[str], templates, device, batch_names: int = 64) -> np.ndarray:
    """Official `zeroshot_classifier`: per name, mean of the normalised template embeddings, renormalised.
    Returns float32 [N, D]."""
    rows = []
    for s in range(0, len(raw_names), batch_names):
        chunk = raw_names[s:s + batch_names]
        texts = [t(n) for n in chunk for t in templates]
        z = encode_texts(model, tokenizer, texts, device, batch=256).reshape(len(chunk), len(templates), -1).mean(1)
        rows.append(z / np.linalg.norm(z, axis=1, keepdims=True))
    return np.concatenate(rows).astype(np.float32)


def check_alignment(weights: np.ndarray, raw_names: list[str], model, tokenizer, device, n_sample: int = 300,
                    seed: int = 0, retrieval_template: str = "a photo of a {}.", retrieval: bool = True,
                    extra_positions: list[int] | None = None) -> dict:
    """Do `raw_names` describe the columns of `weights` ([N, D]) in this order?

    1. size check;
    2. exact check on a seeded sample of positions (plus `extra_positions`): the 80-template weight recomputed from
       the name has cosine >= 0.999 with the stored column (fp16 storage and a different CLIP build give ~1e-4);
    3. retrieval check on every position (cheap, one template): the stored column of position p is the nearest
       column to the single-template text of name p, counting columns with an identical stored vector as tied.
    """
    n = weights.shape[0]
    out = {"n_weights": int(n), "n_names": len(raw_names), "size_match": n == len(raw_names)}
    if not out["size_match"]:
        out["passed"] = False
        return out
    rng = np.random.RandomState(seed)
    pos = sorted(set(rng.choice(n, size=min(n_sample, n), replace=False).tolist()) | {0, n - 1} |
                 set(extra_positions or []))
    w_rec = zeroshot_weights(model, tokenizer, [raw_names[p] for p in pos], imagenet_templates(), device)
    cos = (w_rec * weights[pos]).sum(1)
    out.update({"exact_check_positions": len(pos), "exact_min_cosine": float(cos.min()),
                "exact_mean_cosine": float(cos.mean()), "exact_worst_positions": [int(pos[i]) for i in np.argsort(cos)[:5]]})
    if retrieval:
        q = encode_texts(model, tokenizer, [retrieval_template.format(r) for r in raw_names], device, batch=256)
        hits, rank_ok = 0, 0
        for s in range(0, n, 512):                                   # 512 x N float32 blocks (~140 MB)
            sims = q[s:s + 512] @ weights.T                          # [b, N]; transposed view, no copy
            own = sims[np.arange(len(sims)), np.arange(s, s + len(sims))]
            best = sims.max(1)
            hits += int((own >= best - 1e-6).sum())
            rank_ok += int((np.sum(sims > own[:, None] + 1e-6, 1) < 5).sum())
        out.update({"retrieval_template": retrieval_template, "retrieval_top1_rate": hits / n,
                    "retrieval_top5_rate": rank_ok / n})
    # Pass/fail is the exact check only. Retrieval with one template is supporting evidence: near-synonyms can
    # legitimately beat a name's own 80-template column, so its rate has no fixed threshold.
    out["passed"] = bool(out["exact_min_cosine"] >= 0.999)
    out["criterion"] = "exact 80-template recomputation, min cosine >= 0.999 on every checked position"
    return out
