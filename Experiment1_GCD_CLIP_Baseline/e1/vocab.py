"""Class vocabulary, prompt strategies and vocabulary episodes.

Every text keeps its class id, so dropping a class from an episode drops all of its texts. Episode
construction reads class ids only (no images, no labels of images).
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

PILOT_TEMPLATE = "a photo of a {}."          # pilot (extract_text_embeddings.py)
MCM_TEMPLATE = "a photo of a {}"             # deeplearning-wisc/MCM utils/detection_util.py


def openai_templates() -> list:
    from open_clip.zero_shot_metadata import OPENAI_IMAGENET_TEMPLATES

    return list(OPENAI_IMAGENET_TEMPLATES)


def class_texts(classes: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Rows (text_id, class_id, name, template, text) for one prompt strategy."""
    rows = []
    for c in classes.itertuples():
        name = c.canonical_name
        if strategy == "single":
            rows.append((c.class_id, name, PILOT_TEMPLATE, PILOT_TEMPLATE.format(name)))
        elif strategy == "mcm":
            rows.append((c.class_id, name, MCM_TEMPLATE, MCM_TEMPLATE.format(name)))
        elif strategy == "ensemble80":
            for k, template in enumerate(openai_templates()):
                rows.append((c.class_id, name, f"openai_imagenet_template_{k:02d}", template(name)))
        else:
            raise ValueError(f"unknown prompt strategy {strategy!r}")
    out = pd.DataFrame(rows, columns=["class_id", "name", "template", "text"])
    out.insert(0, "text_id", range(len(out)))
    return out


def class_embeddings(texts: pd.DataFrame, text_emb: np.ndarray, class_ids: np.ndarray) -> np.ndarray:
    """Per class: mean of its normalised text embeddings, renormalised (standard CLIP ensembling)."""
    out = np.zeros((len(class_ids), text_emb.shape[1]), dtype=np.float32)
    for j, cid in enumerate(class_ids):
        m = text_emb[(texts["class_id"] == cid).to_numpy()].mean(0)
        out[j] = m / np.linalg.norm(m)
    return out


def _seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:4], "little")


def shared_episodes(class_ids: np.ndarray, sizes: list[int], n_seeds: int) -> pd.DataFrame:
    """One shared vocabulary per episode: the complete one, then `n_seeds` draws per smaller size."""
    class_ids = np.sort(np.asarray(class_ids))
    rows = [{"episode_id": f"shared_full_{len(class_ids)}", "kind": "shared", "size": len(class_ids),
             "vocab_seed": -1, "class_ids": " ".join(map(str, class_ids))}]
    for size in sizes:
        if size >= len(class_ids):
            continue
        for s in range(n_seeds):
            rng = np.random.RandomState(_seed("shared", size, s))
            ids = np.sort(rng.choice(class_ids, size=size, replace=False))
            rows.append({"episode_id": f"shared_{size}_s{s}", "kind": "shared", "size": size, "vocab_seed": s,
                         "class_ids": " ".join(map(str, ids))})
    return pd.DataFrame(rows)


def paired_episodes(class_ids: np.ndarray, single_text_emb: np.ndarray, n_distractors: int = 9,
                    seed: int = 0) -> pd.DataFrame:
    """Per target class: fixed distractors; target present / replaced by a random class /
    replaced by the remaining class whose `single`-prompt text embedding is most similar."""
    class_ids = np.asarray(class_ids)
    sim = single_text_emb @ single_text_emb.T
    rows = []
    for j, target in enumerate(class_ids):
        rng = np.random.RandomState(_seed("paired", seed, int(target)))
        others = np.array([c for c in class_ids if c != target])
        distractors = np.sort(rng.choice(others, size=n_distractors, replace=False))
        pool = np.array([c for c in others if c not in set(distractors)])
        random_alt = int(rng.choice(pool))
        pool_idx = np.array([int(np.flatnonzero(class_ids == c)[0]) for c in pool])
        hard_alt = int(pool[np.argmax(sim[j, pool_idx])])
        for variant, extra in (("present", int(target)), ("random", random_alt), ("hard", hard_alt)):
            ids = np.sort(np.append(distractors, extra))
            rows.append({"episode_id": f"pair_t{int(target)}_{variant}", "kind": f"paired_{variant}",
                         "size": len(ids), "vocab_seed": seed, "target_class_id": int(target),
                         "replacement_class_id": extra if variant != "present" else -1,
                         "class_ids": " ".join(map(str, ids))})
    return pd.DataFrame(rows)


def episode_classes(row) -> np.ndarray:
    return np.array([int(x) for x in str(row["class_ids"]).split()])
