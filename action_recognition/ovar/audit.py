"""Audit of the official familiar/unfamiliar status against the Kinetics400 categories.

The XOV-Action closed/open lists stay authoritative. This audit only ranks
target classes by text similarity to their closest Kinetics400 categories, so
a human can review the cases where the official status looks surprising:

* familiar classes whose best Kinetics400 match is weaker than the median best
  match of unfamiliar classes, and
* unfamiliar classes whose best match is stronger than the median best match
  of familiar classes.

Text similarity is a screening signal, not evidence of semantic equivalence.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ovar.benchmark import DATA_DIR
from ovar.manifest import load_vocab
from ovar.text import build_prompts

DATASETS = ("ucf101", "hmdb51", "arid", "necdrone")


def _stem(word: str) -> str:
    for suffix in ("ing", "es", "s", "e"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    if len(word) > 3 and word[-1] == word[-2]:  # hugg -> hug, swimm -> swim
        word = word[:-1]
    return word


def _tokens(text: str) -> set[str]:
    """Crude word stems plus the whole name without spaces (pull ups ~ pullup, shot put ~ shotput)."""
    words = re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", text).lower())
    stems = {_stem(w) for w in words if len(w) > 2}
    if len(words) > 1:
        stems.add(_stem("".join(words)))
    return stems


def run_lexical_audit(top_k: int = 3) -> pd.DataFrame:
    """Model-free screen: Kinetics400 classes ranked by shared (crudely stemmed) words."""
    k400 = pd.read_csv(DATA_DIR / "vocab" / "kinetics400.csv")["official_name"].tolist()
    k400_tokens = [_tokens(n) for n in k400]
    rows = []
    for dataset in DATASETS:
        for _, r in load_vocab(dataset).iterrows():
            mine = _tokens(r["text_name"])
            scored = sorted(((len(mine & t) / len(mine | t), n) for t, n in zip(k400_tokens, k400) if mine & t),
                            reverse=True)
            best = scored[0][0] if scored else 0.0
            flag = ""
            if r["category_status"] == "familiar" and not scored:
                flag = "review: familiar but no Kinetics400 class shares a word"
            elif r["category_status"] == "unfamiliar" and scored:
                flag = "review: unfamiliar but a Kinetics400 class shares a word"
            rows.append({
                "dataset": dataset, "action_id": r["action_id"], "action_name": r["official_name"],
                "text_name": r["text_name"], "official_status": r["category_status"],
                "best_word_jaccard": round(best, 3),
                **{f"k400_match_{n + 1}": (f"{scored[n][1]} ({scored[n][0]:.2f})" if n < len(scored) else "")
                   for n in range(top_k)},
                "review_flag": flag, "reviewer_decision": "",
            })
    return pd.DataFrame(rows)


def run_audit(backbone, top_k: int = 3) -> pd.DataFrame:
    k400 = pd.read_csv(DATA_DIR / "vocab" / "kinetics400.csv")
    k400_names = k400["official_name"].tolist()
    k400_emb = backbone.encode_text(build_prompts(k400_names, "video_person")).numpy()
    rows = []
    for dataset in DATASETS:
        vocab = load_vocab(dataset)
        emb = backbone.encode_text(build_prompts(vocab["text_name"].tolist(), "video_person")).numpy()
        sims = emb @ k400_emb.T
        for i, r in vocab.iterrows():
            order = np.argsort(-sims[i])[:top_k]
            overlap = [k400_names[j] for j in range(len(k400_names))
                       if _tokens(r["text_name"]) & _tokens(k400_names[j])]
            rows.append({
                "dataset": dataset, "action_id": r["action_id"], "action_name": r["official_name"],
                "text_name": r["text_name"], "official_status": r["category_status"],
                "best_similarity": float(sims[i, order[0]]),
                **{f"k400_match_{n + 1}": f"{k400_names[j]} ({sims[i, j]:.3f})" for n, j in enumerate(order)},
                "k400_shared_word_classes": "; ".join(overlap[:6]),
            })
    audit = pd.DataFrame(rows)
    flags = []
    for dataset, group in audit.groupby("dataset"):
        fam = group.loc[group["official_status"] == "familiar", "best_similarity"]
        unf = group.loc[group["official_status"] == "unfamiliar", "best_similarity"]
        for idx, r in group.iterrows():
            flag = ""
            if r["official_status"] == "familiar" and len(unf) and r["best_similarity"] < unf.median():
                flag = "review: familiar but weak Kinetics400 match"
            elif r["official_status"] == "unfamiliar" and len(fam) and r["best_similarity"] > fam.median():
                flag = "review: unfamiliar but strong Kinetics400 match"
            flags.append((idx, flag))
    audit["review_flag"] = pd.Series(dict(flags))
    audit["reviewer_decision"] = ""
    return audit
