"""The WordNet noun vocabulary used by the official SCD code, rebuilt and verified.

The official code reads `wordnet_all_noun.txt` from a machine-specific path that is not distributed. Its text
embeddings are distributed (`zeroshot_weights_all_nouns_vit_b_16.pt`, fp16 [512, 67,893]). The list is rebuilt
here as:

    the unique *raw, case-sensitive* first lemma of every noun synset of WordNet 3.0, in the order in which
    `nltk.corpus.wordnet.all_synsets('n')` first yields it.

This rule gives exactly 67,893 names (other rules give 67,181-67,889), and the embeddings of the names under this
order agree with the distributed columns (see `scd.text.check_alignment` and Experiment2_SCD/docs/INSPECTION.md).
The text that was embedded is the raw lemma (underscores kept, e.g. "a photo of a physical_entity.").

The official code then normalises every entry with `lower().replace('-', '_')`; 712 normalised names occur twice
(case variants such as 'Cardigan'/'cardigan'). The official code keeps both columns and maps a name back to a
column with `list.index`, i.e. its first occurrence.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path

WORDNET_VERSION = "3.0"
OFFICIAL_SIZE = 67893
NOUN_MAX_DEPTH = 19                     # WordNet 3.0 nouns (nltk), so lch(x, x) = log(2 * 19)
LCH_MAX = math.log(2 * NOUN_MAX_DEPTH)


def load_wordnet(nltk_dir: str | Path | None = None):
    import nltk

    if nltk_dir is not None:
        nltk_dir = str(nltk_dir)
        if nltk_dir not in nltk.data.path:
            nltk.data.path.insert(0, nltk_dir)
    from nltk.corpus import wordnet as wn

    try:
        version = wn.get_version()
    except LookupError as exc:
        raise RuntimeError("WordNet corpus missing: nltk.download('wordnet', download_dir=...)") from exc
    if version != WORDNET_VERSION:
        raise RuntimeError(f"WordNet {version} loaded; the official vocabulary is WordNet {WORDNET_VERSION}")
    return wn


def normalise(name: str) -> str:
    """The official normalisation (main_unsup.py / main_ptsup.py): lower case, '-' -> '_'."""
    return name.lower().replace("-", "_")


def wnid(synset) -> str:
    return "n{:08d}".format(synset.offset())


@dataclass
class Vocabulary:
    """Columns of a text-embedding matrix. `raw[v]` is the embedded string, `names[v]` its normalised form."""

    raw: list[str]
    source: str
    names: list[str] = field(init=False)
    first_index: dict = field(init=False)      # normalised name -> first column (official `nouns.index`)
    columns_of: dict = field(init=False)       # normalised name -> every column with that name

    def __post_init__(self):
        self.names = [normalise(r) for r in self.raw]
        self.first_index, self.columns_of = {}, {}
        for v, n in enumerate(self.names):
            self.first_index.setdefault(n, v)
            self.columns_of.setdefault(n, []).append(v)

    def __len__(self):
        return len(self.raw)

    @property
    def duplicate_columns(self) -> list[int]:
        """Columns that are not the first occurrence of their normalised name."""
        return [v for v, n in enumerate(self.names) if self.first_index[n] != v]

    def unique_first_columns(self) -> list[int]:
        return sorted(self.first_index.values())

    def sha256(self) -> str:
        return hashlib.sha256("\n".join(self.raw).encode()).hexdigest()

    def describe(self) -> dict:
        return {"source": self.source, "n_columns": len(self.raw), "n_unique_normalised": len(self.first_index),
                "n_duplicate_columns": len(self.duplicate_columns), "sha256_raw": self.sha256(),
                "first": self.raw[:5], "last": self.raw[-3:]}


def official_wordnet_vocabulary(wn) -> Vocabulary:
    raw = list(dict.fromkeys(s.lemma_names()[0] for s in wn.all_synsets("n")))
    if len(raw) != OFFICIAL_SIZE:
        raise RuntimeError(f"rebuilt {len(raw)} names, the official file has {OFFICIAL_SIZE}")
    return Vocabulary(raw, source=f"WordNet {WORDNET_VERSION}: unique raw first lemmas of all noun synsets, "
                                  "first-occurrence order of wn.all_synsets('n')")


def wordnet_dicts(wn):
    """The official `get_wordnet_dict`: wnid -> synset, wnid -> normalised first lemma, name -> wnids."""
    wnid_to_synset, wnid_to_name, name_to_wnids = {}, {}, {}
    for s in wn.all_synsets("n"):
        w = wnid(s)
        name = normalise(s.lemma_names()[0])
        wnid_to_synset[w] = s
        wnid_to_name[w] = name
        name_to_wnids.setdefault(name, []).append(w)
    return wnid_to_synset, wnid_to_name, name_to_wnids


def imagenet_target_name(wn, wnid_str: str) -> str:
    """The official ImageNet class name: normalised first lemma of the class synset."""
    s = wn.synset_from_pos_and_offset("n", int(wnid_str[1:]))
    return normalise(s.lemma_names()[0])
