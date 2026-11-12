"""Protocol A: SCD reproduction on ImageNet-100 (GCD split), no textual enhancement.

Stages (`python -m e2 --config configs/imagenet100.yaml repro <stage>`):
  features  CLIP ViT-B/16 (official package, fp16), DINO ViT-B/16, the authors' GCD ViT-B/16; CLIP preprocessing
  cluster   unsupervised: k-means on DINO features of the unlabelled images (official random_state=0, plus seeds);
            partially supervised: CSS-k-means on GCD features with the labelled anchors (seeded re-implementation)
  name      per-image top-5 over the 67,893-name vocabulary (official weights, verified order), then every variant
            in `repro.variants` (official loop, Table 5 ablations, repairs, sensitivity runs)
  evaluate  sACC (exact), Soft-sACC (both normalisations), clustering ACC (Hungarian, clustering only), All/Old/New,
            initial vs final, number of names, convergence; zero-shot UB / Baseline rows; errors and patterns
Fidelity labels: `official-assets` (official weights and GCD checkpoint, official parameters, our features),
`deviation:<what>` for every variant that changes behaviour (see PROTOCOL.md §A).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from e1.provenance import StageRecord, save_npz_atomic, write_json
from e2.config import Layout, chash, e1_layout

from scd.metrics import SoftScorer, cluster_acc_v2, name_iou, semantic_accuracy
from scd.naming import NamingConfig, image_topk, initial_cluster_names, refine
from scd.vocabulary import imagenet_target_name, wordnet_dicts

PUBLISHED = {  # paper Tables 1-2 (arXiv v2), ImageNet-100, All
    "zero_shot_ub": {"sacc": 85.0, "soft_sacc": 92.0, "acc": 85.1},
    "zero_shot_baseline": {"sacc": 22.7, "soft_sacc": 57.7},
    "unsupervised": {"init_acc": 73.2, "sacc": 41.2, "soft_sacc": 71.3, "acc": 78.2},
    "unsupervised_table5": {"voting_once": {"sacc": 40.7, "acc": 77.5}, "voting_iterated": {"sacc": 40.9, "acc": 78.0},
                            "official": {"sacc": 41.2, "acc": 78.2}},
    "partially_supervised": {"init_acc": 78.7, "sacc": 54.8, "soft_sacc": 77.5, "acc": 83.0},
}
MODEL_TAG = {"unsupervised": "clipvitb16_dino_kmeans", "partially_supervised": "clipvitb16_gcd_csskmeans"}


def _settings(args) -> list[str]:
    return ["unsupervised", "partially_supervised"] if args.setting == "both" else [args.setting]


def _data(cfg):
    from e1.manifests import Manifest

    e1c, e1l = e1_layout(cfg)
    m = Manifest(e1l.data)
    return m, m.ids("labelled"), m.ids("unlabelled")


# ----------------------------------------------------------------------------- stages
def features(cfg, args):
    from e2.embed import imagenet_features

    return imagenet_features(cfg)


def cluster(cfg, args):
    from e2.embed import load_features
    from scd.clustering import constrained_ss_kmeans, kmeans_official

    rc = cfg["repro"]
    m, lab_ids, unl_ids = _data(cfg)
    out = {}
    for setting in _settings(args):
        model = "dino" if setting == "unsupervised" else "gcd"
        ids, F = load_features(cfg, model)
        pos = {s: i for i, s in enumerate(ids)}
        Fu = F[[pos[s] for s in unl_ids]]
        seeds = [args.seed] if args.seed is not None else (rc["kmeans"] if setting == "unsupervised" else rc["css_kmeans"])["seeds"]
        for seed in seeds:
            sec = rc["kmeans"] if setting == "unsupervised" else rc["css_kmeans"]
            d = Layout(cfg).repro(setting, MODEL_TAG[setting], seed, {"init": sec, "n": rc["n_cluster"], "model": model}) / "init"
            if (d / "clustering.npz").exists():
                out[f"{setting}/seed{seed}"] = str(d)
                continue
            rec = StageRecord(d, f"init_{setting}", chash(sec), {"seed": seed, "features": model})
            if setting == "unsupervised":
                labels, info = kmeans_official(Fu, rc["n_cluster"], seed=seed, n_init=sec["n_init"])
                save_npz_atomic(d / "clustering.npz", sample_ids=np.array(unl_ids).astype(str), labels=labels,
                                labelled_ids=np.array([], dtype=str), labelled_labels=np.array([], dtype=int))
            else:
                y_lab = m.labels("labelled")
                Fl = F[[pos[s] for s in lab_ids]]
                res = constrained_ss_kmeans(Fu, Fl, y_lab.loc[lab_ids].to_numpy(), rc["n_cluster"],
                                            size_min=sec["size_min"], size_max=sec["size_max"], n_init=sec["n_init"],
                                            max_iter=sec["max_iter"], tol=sec["tol"], seed=seed)
                labels = res["labels_unlabelled"]
                info = {k: v for k, v in res.items() if k not in ("labels_labelled", "labels_unlabelled", "centres")}
                save_npz_atomic(d / "clustering.npz", sample_ids=np.array(unl_ids).astype(str), labels=labels,
                                labelled_ids=np.array(lab_ids).astype(str), labelled_labels=res["labels_labelled"])
            write_json(d / "clustering.json", info)
            rec.complete(**{k: v for k, v in info.items() if k in ("inertia", "n_iter", "best_iteration")})
            out[f"{setting}/seed{seed}"] = str(d)
    return out


def variant_config(cfg: dict, name: str) -> tuple[NamingConfig, dict]:
    v = dict(cfg["repro"]["variants"][name])
    extra = {"merge_duplicate_names": v.pop("merge_duplicate_names", False)}
    base = {k: val for k, val in cfg["naming"].items() if k in NamingConfig.__dataclass_fields__}
    base.update(v)
    return NamingConfig(**base), extra


def topk_cache(cfg, vocab, W, X, ids, logit_dtype: str, merge: bool) -> np.ndarray:
    cols = vocab.unique_first_columns() if merge else None
    key = chash({"sha": vocab.sha256(), "dtype": logit_dtype, "merge": merge, "k": cfg["naming"]["topk_store"],
                 "n": len(ids), "first": ids[:3]})
    path = Layout(cfg).base / "topk" / f"top{cfg['naming']['topk_store']}_{logit_dtype}_{'merged' if merge else 'all'}_{key}.npz"
    if path.exists():
        z = np.load(path)
        if z["sample_ids"].tolist() == list(ids):
            return z["topk"]
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    topk, val, info = image_topk(X, W, cfg["naming"]["topk_store"], columns=cols, logit_dtype=logit_dtype, device=dev)
    save_npz_atomic(path, sample_ids=np.array(ids).astype(str), topk=topk, scores=val)
    write_json(path.with_suffix(".json"), info)
    return topk


def run_naming(vocab, W, X, init_labels, ncfg: NamingConfig, setting: str, n_cluster: int, topk: np.ndarray,
               known_names=None, labelled_clusters=None, cluster_ids=None) -> dict:
    kw = {}
    if setting == "partially_supervised":
        kw = {"known_names": known_names, "labelled_clusters": labelled_clusters, "cluster_ids": cluster_ids}
    return refine(X, W, vocab, topk, init_labels, ncfg, n_cluster=n_cluster, **kw)


def save_run(d: Path, res: dict, ids):
    d.mkdir(parents=True, exist_ok=True)
    its = res["iterations"]
    save_npz_atomic(d / "labels_by_iteration.npz", sample_ids=np.array(ids).astype(str),
                    **{f"iter{r['iteration']}": r["labels_after"] for r in its})
    write_json(d / "trace.json", {k: res[k] for k in ("stop_reason", "n_iterations", "config", "num_slots", "ptsup")} |
               {"iterations": [{k: v for k, v in r.items() if k not in ("labels_before", "labels_after")} for r in its]})


def name(cfg, args):
    from e2 import vocab_stage
    from e2.embed import load_features

    rc = cfg["repro"]
    wn, vocab, W = vocab_stage.load(cfg)
    m, lab_ids, unl_ids = _data(cfg)
    cids, C = load_features(cfg, "clip")
    pos = {s: i for i, s in enumerate(cids)}
    X = C[[pos[s] for s in unl_ids]]
    known_ids = sorted(m.known_class_ids.tolist())
    targets = class_targets(wn, m)
    known_names = [targets[c] for c in known_ids]
    variants = args.variants.split(",") if args.variants else list(rc["variants"])
    done = {}
    for setting in _settings(args):
        for init_dir in sorted((Layout(cfg).base / f"repro_{setting}" / MODEL_TAG[setting]).glob("seed*/*/init")):
            z = np.load(init_dir / "clustering.npz")
            if z["sample_ids"].tolist() != unl_ids:
                raise RuntimeError(f"{init_dir}: sample order differs from the manifest")
            init = z["labels"]
            kw = {}
            if setting == "partially_supervised":
                lab_clusters = sorted(set(z["labelled_labels"].tolist()))
                kw = {"known_names": known_names, "labelled_clusters": lab_clusters,
                      "cluster_ids": sorted(set(init.tolist()) | set(lab_clusters))}
            for v in variants:
                if v == "known_filter_fixed" and setting != "partially_supervised":
                    continue
                ncfg, extra = variant_config(cfg, v)
                d = init_dir.parent / v
                if (d / "trace.json").exists():
                    done[str(d)] = "exists"
                    continue
                topk = topk_cache(cfg, vocab, W, X, unl_ids, ncfg_dtype(ncfg, cfg, v), extra["merge_duplicate_names"])
                res = run_naming(vocab, W, X, init, ncfg, setting, rc["n_cluster"], topk, **kw)
                save_run(d, res, unl_ids)
                done[str(d)] = res["stop_reason"]
    return done


def ncfg_dtype(ncfg, cfg, variant) -> str:
    return cfg["repro"]["variants"][variant].get("logit_dtype", cfg["naming"]["logit_dtype"])


def class_targets(wn, manifest) -> dict:
    """class id -> declared semantic target: the normalised first lemma of the class synset (official)."""
    cl = manifest.classes
    return {int(c): imagenet_target_name(wn, k) for c, k in zip(cl["class_id"], cl["class_key"])}


# ----------------------------------------------------------------------------- evaluation
def evaluate_run(res_labels_by_iter: list, selected_by_iter: list, init_labels, init_names: dict, y, old,
                 targets: dict, scorer: SoftScorer | None) -> dict:
    true_names = np.array([targets[int(c)] for c in y], dtype=object)

    def ev(pred_names, clusters):
        r = {"acc": cluster_acc_v2(y, clusters, old)}
        for tag, msk in (("all", np.ones(len(y), bool)), ("old", old), ("new", ~old)):
            r[f"sacc_{tag}"] = semantic_accuracy(pred_names[msk], true_names[msk])
            if scorer is not None:
                ok = pred_names[msk] != ""
                r[f"soft_sacc_{tag}"] = scorer.soft_sacc(pred_names[msk][ok], true_names[msk][ok]) if ok.any() else None
        r["n_selected_names"] = int(len(set(pred_names) - {""}))
        r["iou_names"] = name_iou(set(pred_names) - {""}, set(targets.values()))
        return r

    pred_init = np.array([init_names.get(int(c), "") for c in init_labels], dtype=object)
    out = {"initial": ev(pred_init, init_labels), "by_iteration": []}
    correct = []
    for lab, sel in zip(res_labels_by_iter, selected_by_iter):
        names = np.array(sel, dtype=object)[lab]
        out["by_iteration"].append(ev(names, lab))
        correct.append(names == true_names)
    out["final"] = out["by_iteration"][-1]
    c0, c1, cf = pred_init == true_names, correct[0], correct[-1]
    out["P4_refinement_flips"] = {"initial_correct_to_final_wrong": int((c0 & ~cf).sum()),
                                  "initial_wrong_to_final_correct": int((~c0 & cf).sum()),
                                  "iter1_correct_to_final_wrong": int((c1 & ~cf).sum()),
                                  "iter1_wrong_to_final_correct": int((~c1 & cf).sum()), "n_images": int(len(y))}
    return out


def wordnet_relation(wn, pred: str, true: str, name_to_wnids: dict, wnid_to_synset: dict) -> str:
    """Relation between a predicted and a true name: exact / synonym (shares a synset) / hypernym / hyponym /
    sibling (share a direct hypernym) / other. Measurable; whether a confusion is 'visual' is a hypothesis."""
    if pred == true:
        return "exact"
    P = [wnid_to_synset[w] for w in name_to_wnids.get(pred, [])]
    T = [wnid_to_synset[w] for w in name_to_wnids.get(true, [])]
    true_lemmas = {l.lower() for t in T for l in t.lemma_names()}
    if any(pred in {l.lower() for l in p.lemma_names()} for p in T) or any(true in {l.lower() for l in p.lemma_names()} for p in P) \
            or pred in true_lemmas:
        return "synonym"
    for t in T:
        hyper = set(t.closure(lambda s: s.hypernyms() + s.instance_hypernyms()))
        if any(p in hyper for p in P):
            return "hypernym"     # the prediction is broader
    for p in P:
        hyper = set(p.closure(lambda s: s.hypernyms() + s.instance_hypernyms()))
        if any(t in hyper for t in T):
            return "hyponym"      # the prediction is narrower
    th = {h for t in T for h in t.hypernyms()}
    if any(h in th for p in P for h in p.hypernyms()):
        return "sibling"
    return "other"


def evaluate(cfg, args):
    from e2 import vocab_stage
    from e2.embed import load_features

    rc = cfg["repro"]
    wn, vocab, W = vocab_stage.load(cfg)
    w2s, _, n2w = wordnet_dicts(wn)
    scorer = SoftScorer(w2s, n2w)
    m, lab_ids, unl_ids = _data(cfg)
    y = m.labels("eval").loc[unl_ids].to_numpy()
    old = np.isin(y, m.known_class_ids)
    targets = class_targets(wn, m)
    missing = [t for t in targets.values() if t not in vocab.first_index]
    if missing:
        raise RuntimeError(f"targets missing from the vocabulary: {missing}")
    cids, C = load_features(cfg, "clip")
    pos = {s: i for i, s in enumerate(cids)}
    X = C[[pos[s] for s in unl_ids]]
    summary = {"published": PUBLISHED, "zero_shot": zero_shot_rows(X, W, vocab, targets, y, old, scorer), "runs": []}
    known_ids = sorted(m.known_class_ids.tolist())
    for setting in _settings(args):
        for init_dir in sorted((Layout(cfg).base / f"repro_{setting}" / MODEL_TAG[setting]).glob("seed*/*/init")):
            z = np.load(init_dir / "clustering.npz")
            init = z["labels"]
            lab_names = {}
            if setting == "partially_supervised":
                # labelled cluster -> its anchors' class (permitted labels only)
                yl = m.labels("labelled").loc[z["labelled_ids"].tolist()].to_numpy()
                for c in np.unique(z["labelled_labels"]):
                    cls = np.unique(yl[z["labelled_labels"] == c])
                    lab_names[int(c)] = targets[int(cls[0])]
            for run_dir in sorted(p for p in init_dir.parent.iterdir() if (p / "trace.json").exists()):
                tr = json.loads((run_dir / "trace.json").read_text())
                lz = np.load(run_dir / "labels_by_iteration.npz")
                labs = [lz[f"iter{r['iteration']}"] for r in tr["iterations"]]
                sels = [r["selected_names"] for r in tr["iterations"]]
                fake = {"iterations": [{"voted_names": tr["iterations"][0]["voted_names"],
                                        "slots": tr["iterations"][0]["slots"]}]}
                init_names = initial_cluster_names(fake, lab_names)
                ev = evaluate_run(labs, sels, init, init_names, y, old, targets, scorer)
                ev.update({"setting": setting, "variant": run_dir.name, "seed_dir": str(init_dir.parent),
                           "stop_reason": tr["stop_reason"], "n_iterations": tr["n_iterations"],
                           "fidelity": fidelity_label(run_dir.name, setting),
                           "repairs": sum((r["repairs"] for r in tr["iterations"]), [])})
                write_json(run_dir / "metrics.json", ev)
                export_errors(run_dir, ev, labs[-1], sels[-1], y, unl_ids, targets, m, wn, n2w, w2s, init, init_names)
                summary["runs"].append({k: ev[k] for k in ("setting", "variant", "seed_dir", "stop_reason", "n_iterations",
                                                          "fidelity")} | {"initial": ev["initial"], "final": ev["final"],
                                                                          "P4": ev["P4_refinement_flips"]})
    out = Layout(cfg).base / "repro_summary.json"
    write_json(out, summary)
    return {"summary": str(out), "n_runs": len(summary["runs"])}


def fidelity_label(variant: str, setting: str) -> str:
    base = ("official-assets: official zero-shot weights (order verified), official parameters; our CLIP/DINO "
            "features and clusterings; logits in fp32 with lowest-index ties")
    if setting == "partially_supervised":
        base += "; CSS-k-means re-implemented with explicit seeds (official: unseeded)"
    return base if variant == "official" else f"deviation:{variant}"


def zero_shot_rows(X, W, vocab, targets, y, old, scorer) -> dict:
    """Paper rows 'Zero-shot transfer (Baseline)' (argmax over the whole vocabulary) and '(UB)' (argmax over the
    100 target names), per image; ACC only for UB (the Baseline predicts thousands of names)."""
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}
    true = np.array([targets[int(c)] for c in y], dtype=object)
    for tag, cols in (("baseline", None), ("ub", sorted({vocab.first_index[t] for t in targets.values()}))):
        idx, _, _ = image_topk(X, W, 1, columns=cols, device=dev)
        pred = np.array([vocab.names[v] for v in idx[:, 0]], dtype=object)
        r = {f"sacc_{t}": semantic_accuracy(pred[mk], true[mk]) for t, mk in
             (("all", np.ones(len(y), bool)), ("old", old), ("new", ~old))}
        r["soft_sacc_all"] = scorer.soft_sacc(pred, true)
        r["n_distinct_predicted"] = int(len(set(pred)))
        if tag == "ub":
            r["acc"] = cluster_acc_v2(y, idx[:, 0], old)
        out[tag] = r
    return out


def export_errors(run_dir: Path, ev, labels, selected, y, ids, targets, manifest, wn, n2w, w2s, init, init_names):
    true = np.array([targets[int(c)] for c in y], dtype=object)
    pred = np.array(selected, dtype=object)[labels]
    pred_init = np.array([init_names.get(int(c), "") for c in init], dtype=object)
    rel = dict(zip(manifest.table["sample_id"], manifest.table["rel_path"]))
    df = pd.DataFrame({"sample_id": ids, "image_path": [rel[s] for s in ids], "initial_cluster": init,
                       "initial_name": pred_init, "final_position": labels, "final_name": pred, "true_class_id": y,
                       "true_name": true, "true_old": np.isin(y, manifest.known_class_ids)})
    df["correct"] = df["final_name"] == df["true_name"]
    pairs = df.loc[~df["correct"], ["final_name", "true_name"]].drop_duplicates()
    rel_of = {(p, t): wordnet_relation(wn, p, t, n2w, w2s) for p, t in pairs.itertuples(index=False)}
    df["relation"] = [("exact" if c else rel_of[(p, t)]) for p, t, c in zip(df["final_name"], df["true_name"], df["correct"])]
    df.to_csv(run_dir / "image_predictions.csv.gz", index=False)
    df[~df["correct"]].to_csv(run_dir / "incorrect_decisions.csv.gz", index=False)
    # cluster-level view of the final assignment
    cl = df.groupby("final_position").agg(name=("final_name", "first"), n=("sample_id", "size"),
                                          dominant=("true_name", lambda s: s.value_counts().index[0]),
                                          purity=("true_name", lambda s: s.value_counts().iloc[0] / len(s)))
    cl["name_correct_for_dominant"] = cl["name"] == cl["dominant"]
    cl["dominant_also_named_elsewhere"] = cl["dominant"].map(cl["dominant"].value_counts()) > 1
    cl.to_csv(run_dir / "final_clusters.csv")
    write_json(run_dir / "error_patterns.json", {
        "relations_of_wrong_images": df.loc[~df["correct"], "relation"].value_counts().to_dict(),
        "P5_mixed_final_clusters": int((cl["purity"] <= 0.5).sum()),
        "P3_dominant_class_split_over_clusters": int(cl["dominant_also_named_elsewhere"].sum()),
        "P1_P2_note": "per-image vote support is in trace.json (top_votes per slot); see the report",
        "P4": ev["P4_refinement_flips"]})


def all(cfg, args):  # noqa: A001 - CLI stage name
    return {"features": features(cfg, args), "cluster": cluster(cfg, args), "name": name(cfg, args),
            "evaluate": evaluate(cfg, args)}
