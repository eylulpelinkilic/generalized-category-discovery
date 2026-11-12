"""Software check of protocol A's naming and evaluation code on synthetic embeddings (NOT a research result).

Real WordNet nouns form the vocabulary (so Soft-sACC runs end to end); text and image embeddings are synthetic.
"""
from __future__ import annotations

import numpy as np

from e1.provenance import write_json
from e2.config import Layout
from e2.repro import evaluate_run, run_naming, zero_shot_rows
from e2.vocab_stage import ensure_wordnet
from e2.config import resolve

from scd.clustering import constrained_ss_kmeans, kmeans_official
from scd.metrics import SoftScorer
from scd.naming import NamingConfig, image_topk, initial_cluster_names
from scd.vocabulary import Vocabulary, wordnet_dicts

NAMES = ["dog", "Dog", "cat", "car", "tree", "fish", "bird", "house", "boat", "puppy", "truck", "oak"]
TARGETS = {0: "dog", 1: "cat", 2: "car", 3: "tree", 4: "fish", 5: "bird"}       # 0-2 known, 3-5 novel


def run(cfg: dict) -> dict:
    rng = np.random.RandomState(0)
    wn = ensure_wordnet(resolve(cfg["vocabulary"]["nltk_dir"]))
    w2s, _, n2w = wordnet_dicts(wn)
    vocab = Vocabulary(NAMES, source="smoke")
    D = 16
    T = rng.randn(len(NAMES), D)
    T[1] = T[0]                                    # 'Dog' duplicates 'dog' after normalisation (as in the official list)
    T[9] = T[0] + 0.6 * rng.randn(D)               # 'puppy' near 'dog'
    T = T / np.linalg.norm(T, axis=1, keepdims=True)
    y = np.repeat(np.arange(6), 40)
    X = np.stack([T[vocab.first_index[TARGETS[c]]] for c in y]) + 0.35 * rng.randn(len(y), D)
    X = (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)
    old = y < 3
    scorer = SoftScorer(w2s, n2w)
    topk, _, _ = image_topk(X, T, 5)
    out = {"NOTE": "synthetic smoke run: software check only", "zero_shot": zero_shot_rows(X, T, vocab, TARGETS, y, old, scorer)}
    init_u, _ = kmeans_official(X, 6, seed=0, n_init=4)
    lab = rng.rand(len(y)) < 0.3
    lab &= old
    css = constrained_ss_kmeans(X[~lab], X[lab], y[lab], 6, size_min=10, size_max=80, n_init=2, max_iter=5, seed=0)
    for setting, init, Xs, ys, olds, kw, lab_names in (
            ("unsupervised", init_u, X, y, old, {}, {}),
            ("partially_supervised", css["labels_unlabelled"], X[~lab], y[~lab], old[~lab],
             {"known_names": ["dog", "cat", "car"], "labelled_clusters": sorted(set(css["labels_labelled"].tolist())),
              "cluster_ids": list(range(6))},
             {int(c): TARGETS[int(np.unique(y[lab][css["labels_labelled"] == c])[0])]
              for c in np.unique(css["labels_labelled"])})):
        tk = topk if setting == "unsupervised" else topk[~lab]
        for variant, ncfg in (("official", NamingConfig(topk_vote=3, num_common_vote=4, num_common_linear=2)),
                              ("voting_once", NamingConfig(topk_vote=3, num_common_vote=4, num_common_linear=2,
                                                           assignment="plurality", iterate=False)),
                              ("known_filter_fixed", NamingConfig(topk_vote=3, num_common_vote=4, num_common_linear=2,
                                                                  known_filter="fixed"))):
            if variant == "known_filter_fixed" and setting == "unsupervised":
                continue
            res = run_naming(vocab, T, Xs, init, ncfg, setting, 6, tk, **kw)
            labs = [r["labels_after"] for r in res["iterations"]]
            sels = [r["selected_names"] for r in res["iterations"]]
            ev = evaluate_run(labs, sels, init, initial_cluster_names(res, lab_names), ys, olds, TARGETS, scorer)
            out[f"{setting}/{variant}"] = {"stop_reason": res["stop_reason"], "iterations": res["n_iterations"],
                                           "initial_sacc": ev["initial"]["sacc_all"]["image"],
                                           "final_sacc": ev["final"]["sacc_all"]["image"],
                                           "final_acc": ev["final"]["acc"]["all"],
                                           "final_soft_official": ev["final"]["soft_sacc_all"]["official"],
                                           "names": sels[-1], "P4": ev["P4_refinement_flips"]}
    path = Layout(cfg).out / "smoke" / "smoke_protocol_A.json"
    write_json(path, out)
    return {"written": str(path), **{k: v for k, v in out.items() if "/" in k}}
