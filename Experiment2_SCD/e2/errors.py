"""Error export, failure patterns and galleries.

Patterns (README / PROTOCOL.md §E). Measurable definitions, per (episode, cluster, method); `d` = the cluster's
dominant true class, `V` = the episode vocabulary, "votes" = each member's top-1 within V:
  P1 weak_support        d in V, label != d, and d is not the most-voted concept (images do not back d)
  P2 aggregation_override d in V, label != d, but d is the most-voted concept (P2a: d has a strict majority)
  P3 competition         label != d and another cluster of the same episode received the same label
  P4 refinement_flip     (protocol R/A) an image correct after iteration 1 is wrong at the end
  P5 mixed_cluster       purity <= 1/2 (no strict majority); members of other classes are wrong under any label
  P6 related_name        d in V, label != d and the label shares its head word with d (CUB, e.g. "... Sparrow")
                         or is among d's 5 nearest names by text embedding; "visually similar" is a hypothesis
  P7 absent_accepted     image class not in V, decision accepted (at tau_e2e)
  P8 available_rejected  image class in V, decision rejected (at tau_e2e); split by whether the label was correct
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from e1.ground import cluster_diagnostics
from e1.provenance import write_json
from e1.vocab import episode_classes

PATTERN_DOC = __doc__


# ----------------------------------------------------------------------------- galleries
def cluster_gallery(path: Path, panels: list[dict], title: str, per_row: int = 8):
    """Rows of member thumbnails. Each panel: {"title": str, "files": [paths], "captions": [str]}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    if not panels:
        return None
    from matplotlib.gridspec import GridSpec

    side = 200
    fig = plt.figure(figsize=(per_row * 1.55, 0.35 + len(panels) * 2.05), facecolor="white")
    gs = GridSpec(2 * len(panels), per_row, figure=fig, height_ratios=[0.22, 1.0] * len(panels),
                  hspace=0.25, wspace=0.05, left=0.01, right=0.99, top=1 - 0.3 / (0.35 + len(panels) * 2.05), bottom=0.01)
    for r, p in enumerate(panels):
        head = fig.add_subplot(gs[2 * r, :])
        head.axis("off")
        head.text(0, 0.1, p["title"], fontsize=6.8, va="bottom", ha="left", transform=head.transAxes)
        for c in range(per_row):
            ax = fig.add_subplot(gs[2 * r + 1, c])
            ax.axis("off")
            if c < len(p["files"]) and Path(p["files"][c]).exists():
                im = Image.open(p["files"][c]).convert("RGB")
                im.thumbnail((side, side))
                canvas = Image.new("RGB", (side, side), (255, 255, 255))       # letterbox: every cell is square
                canvas.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
                ax.imshow(canvas)
                if p.get("captions"):
                    ax.set_title(p["captions"][c][:30], fontsize=5.3, pad=2)
    fig.suptitle(title, fontsize=9.5, x=0.01, y=0.998, ha="left", va="top")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor="white")
    plt.close(fig)
    return path


def head_word(name: str) -> str:
    return name.strip().split()[-1].lower() if name.strip() else ""


# ----------------------------------------------------------------------------- protocol B
def _episode_cluster_table(Rd, g, y, K, cols, pred_ep: dict, diag: pd.DataFrame, text_nn: dict, names) -> pd.DataFrame:
    """One row per (method, cluster) of one episode with the quantities the patterns need."""
    vset = set(cols.tolist())
    top1 = cols[np.argmax(Rd[:, cols], 1)]
    n = np.bincount(g, minlength=K)
    rows = []
    for method, (pc, sc) in pred_ep.items():
        label_count = pd.Series(pc[pc >= 0]).value_counts()
        for k in np.flatnonzero(n > 0):
            m = g == k
            d = int(diag.at[k, "dominant_class_id"])
            vc = pd.Series(top1[m]).value_counts()
            winner_votes = int(vc.iloc[0])
            s_d = int(vc.get(d, 0))
            lab = int(pc[k])
            yk = y[m]
            rows.append({
                "method": method, "cluster_id": int(k), "n": int(m.sum()), "purity": float(diag.at[k, "purity"]),
                "dominant_class_id": d, "dominant_available": d in vset, "label": lab, "score": float(sc[k]),
                "label_correct": lab == d, "n_avail": int(np.isin(yk, cols).sum()),
                "n_correct": int(((yk == lab) & np.isin(yk, cols)).sum()),
                "votes_for_dominant": s_d, "votes_for_label": int(vc.get(lab, 0)), "votes_winner": winner_votes,
                "dominant_is_vote_winner": s_d == winner_votes and d in vset,
                "label_shared_with_other_clusters": int(label_count.get(lab, 0)) - 1 if lab >= 0 else 0,
                "label_is_dominant_elsewhere": bool(((diag["dominant_class_id"] == lab) & (diag.index != k)).any()),
                "same_head_word": lab >= 0 and lab != d and head_word(names[lab]) == head_word(names[d]),
                "label_in_text_nn5_of_dominant": lab in text_nn.get(d, ()),
            })
    return pd.DataFrame(rows)


def _patterns(t: pd.DataFrame) -> dict:
    """Counts of P1-P3, P5, P6 over a cluster table (clusters and their images)."""
    wrong = t[~t["label_correct"]]
    avail_wrong = wrong[wrong["dominant_available"]]

    def cnt(df):
        return {"clusters": int(len(df)), "images": int(df["n"].sum())}

    p1 = avail_wrong[~avail_wrong["dominant_is_vote_winner"]]
    p2 = avail_wrong[avail_wrong["dominant_is_vote_winner"]]
    p2a = p2[p2["votes_for_dominant"] > p2["n"] / 2]
    p3 = wrong[wrong["label_shared_with_other_clusters"] > 0]
    p6 = avail_wrong[avail_wrong["same_head_word"] | avail_wrong["label_in_text_nn5_of_dominant"]]
    mixed = t[t["purity"] <= 0.5]
    pure = t[t["purity"] > 0.5]
    return {
        "clusters_evaluated": cnt(t), "wrong_label": cnt(wrong), "wrong_with_dominant_available": cnt(avail_wrong),
        "P1_weak_support": cnt(p1),
        "P1_support_of_dominant_median": float((p1["votes_for_dominant"] / p1["n"]).median()) if len(p1) else None,
        "P2_aggregation_override": cnt(p2), "P2a_dominant_strict_majority": cnt(p2a),
        "P3_competition": cnt(p3), "P3_label_owned_by_another_cluster": cnt(p3[p3["label_is_dominant_elsewhere"]]),
        "P5_mixed_clusters": cnt(mixed),
        "P5_cluster_error_rate_mixed": float(1 - mixed["label_correct"].mean()) if len(mixed) else None,
        "P5_cluster_error_rate_pure": float(1 - pure["label_correct"].mean()) if len(pure) else None,
        "P5_membership_errors_images": int((t["n"] - (t["purity"] * t["n"]).round()).sum()),
        "P6_related_name": cnt(p6), "P6_same_head_word": cnt(avail_wrong[avail_wrong["same_head_word"]]),
        "P6_text_nearest5": cnt(avail_wrong[avail_wrong["label_in_text_nn5_of_dominant"]]),
        "image_accuracy_available": float(t["n_correct"].sum() / t["n_avail"].sum()) if t["n_avail"].sum() else None,
    }


def _matched_correct(t: pd.DataFrame, err: pd.DataFrame, n: int) -> list[tuple]:
    """For each error cluster, the correct cluster (same method/episode table) nearest in (log size, purity)."""
    ok = t[t["label_correct"]].copy()
    pairs, used = [], set()
    for r in err.head(n).itertuples():
        if ok.empty:
            break
        dist = (np.log(ok["n"]) - np.log(r.n)) ** 2 + (ok["purity"] - r.purity) ** 2
        dist[ok["cluster_id"].isin(used)] = np.inf
        j = dist.idxmin()
        used.add(int(ok.at[j, "cluster_id"]))
        pairs.append((r, ok.loc[j]))
    return pairs


def fixed_cluster_errors(cfg, out: Path, inp: dict, R: np.ndarray, pred: pd.DataFrame, counts: pd.DataFrame,
                         thresholds: dict, episodes: pd.DataFrame, y_eval: pd.Series, manifest, k_star: int) -> dict:
    from e2.fixed import vote_name

    ed = out / "errors"
    ed.mkdir(parents=True, exist_ok=True)
    rows, K = inp["rows"], inp["K"]
    disc = rows[rows["population"] == "discovery"].reset_index(drop=True)
    Rd = R[disc["row"].to_numpy()]
    g = disc["cluster_id"].to_numpy()
    y = y_eval.loc[disc["sample_id"]].to_numpy()
    class_ids = np.asarray(inp["class_ids"])
    names = manifest.classes.set_index("class_id")["canonical_name"]
    known = set(manifest.known_class_ids.tolist())
    rel = dict(zip(manifest.table["sample_id"], manifest.table["rel_path"]))
    from e2.embed import image_root

    img_root = image_root(cfg, manifest)
    np.savez_compressed(ed / "discovery_similarity.npz", R=Rd.astype(np.float32), sample_ids=disc["sample_id"].to_numpy().astype(str),
                        cluster_ids=g, class_ids=class_ids)
    diag = cluster_diagnostics(rows, y_eval, manifest, K).set_index("cluster_id")
    single = inp["class_emb"]["single"]
    sim = single @ single.T
    np.fill_diagonal(sim, -np.inf)
    text_nn = {int(c): set(class_ids[np.argsort(-sim[j])[:5]].tolist()) for j, c in enumerate(class_ids)}

    vk = vote_name(k_star)
    forced = ["forced_top1", "plurality_forced"] + ([f"{vk}_forced"] if k_star > 1 else [])
    rejecting = ["max_mean_sim", "vote_fraction", "plurality@win_fraction", "plurality@margin", "plurality@neg_entropy",
                 "plurality@mean_sim"] + ([f"{vk}@win_fraction"] if k_star > 1 else [])
    distinct = [m for m in pred["method"].unique() if m.startswith("distinct_")]
    primary = forced + ["max_mean_sim", "plurality@win_fraction"] + distinct
    disc_pred = pred[pred["population"] == "discovery"]
    by_ep = {k: t for k, t in disc_pred[disc_pred["method"].isin(set(forced + rejecting + distinct))]
             .groupby(["episode_id", "method"])}

    # --- image-level decisions (shared episodes, primary methods) and every incorrect decision
    img_frames, err_frames, tables = [], [], []
    shared = episodes[episodes["kind"] == "shared"]
    for e in shared.itertuples():
        cols = episode_classes(e._asdict())
        vmask = np.isin(y, cols)
        Rv = Rd[:, cols]
        own = cols[np.argmax(Rv, 1)]
        true_rank = np.where(vmask, (Rv > Rd[np.arange(len(y)), np.searchsorted(class_ids, y)][:, None]).sum(1) + 1, -1)
        pred_ep = {}
        for method in forced + rejecting + distinct:
            t = by_ep.get((e.episode_id, method))
            if t is None:
                continue
            t = t.sort_values("cluster_id")
            pc, sc = t["pred_class_id"].to_numpy(), t["score"].to_numpy(float)
            pred_ep[method] = (pc, sc)
            if method not in primary:
                continue
            lab, s = pc[g], sc[g]
            tau = thresholds[method]["tau_e2e"]
            acc = np.isfinite(s) & (s >= tau)
            correct = vmask & (lab == y)
            outcome = np.select([acc & correct, acc & vmask & ~correct, acc & ~vmask, ~acc & correct, ~acc & vmask],
                                ["correct_accepted", "wrong_accepted", "absent_accepted", "correct_rejected",
                                 "wrong_rejected"], "absent_rejected")
            f = pd.DataFrame({"method": method, "episode_id": e.episode_id, "vocab_size": e.size,
                              "sample_id": disc["sample_id"], "cluster_id": g, "true_class_id": y,
                              "true_old": np.isin(y, list(known)), "true_available": vmask, "pred_class_id": lab,
                              "score": s, "tau_e2e": tau, "accepted": acc, "correct": correct, "outcome": outcome,
                              "image_top1_class_id": own, "image_top1_cos": Rv.max(1),
                              "cos_pred": np.where(lab >= 0, Rd[np.arange(len(y)), np.searchsorted(class_ids, np.clip(lab, 0, None))], np.nan),
                              "true_rank_in_V": true_rank})
            img_frames.append(f)
            err_frames.append(f[~f["outcome"].isin(["correct_accepted", "absent_rejected"])])
        tb = _episode_cluster_table(Rd, g, y, K, cols, {m: pred_ep[m] for m in primary if m in pred_ep}, diag, text_nn, names)
        tb.insert(0, "episode_id", e.episode_id)
        tb.insert(1, "vocab_size", e.size)
        tables.append(tb)
    imgs = pd.concat(img_frames, ignore_index=True)
    for df in (imgs,):
        df["true_name"] = names.reindex(df["true_class_id"]).to_numpy()
        df["pred_name"] = np.where(df["pred_class_id"] >= 0, names.reindex(df["pred_class_id"].clip(lower=0)).to_numpy(), "")
        df["image_path"] = [rel[s] for s in df["sample_id"]]
    comp = diag["composition_top3"]
    imgs["cluster_purity"] = diag["purity"].reindex(imgs["cluster_id"]).to_numpy()
    imgs["cluster_composition_top3"] = comp.reindex(imgs["cluster_id"]).to_numpy()
    full_id = episodes.iloc[0]["episode_id"]
    imgs[imgs["episode_id"] == full_id].to_csv(ed / "image_decisions_complete_vocabulary.csv.gz", index=False)
    imgs.to_csv(ed / "image_decisions_shared_episodes.csv.gz", index=False)
    errs = imgs[~imgs["outcome"].isin(["correct_accepted", "absent_rejected"])]
    errs.to_csv(ed / "incorrect_decisions_shared_episodes.csv.gz", index=False)
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(ed / "cluster_decisions_shared_episodes.csv.gz", index=False)

    # --- paired episodes: target-class images only
    pe = episodes[episodes["kind"].str.startswith("paired")]
    pf = []
    for e in pe.itertuples():
        m_t = y == e.target_class_id
        if not m_t.any():
            continue
        cols = episode_classes(e._asdict())
        for method in ("max_mean_sim", "plurality@win_fraction", "forced_top1", "plurality_forced"):
            t = by_ep.get((e.episode_id, method))
            if t is None:
                continue
            t = t.sort_values("cluster_id")
            lab, s = t["pred_class_id"].to_numpy()[g[m_t]], t["score"].to_numpy(float)[g[m_t]]
            acc = np.isfinite(s) & (s >= thresholds[method]["tau_e2e"])
            pf.append(pd.DataFrame({"method": method, "episode_id": e.episode_id, "variant": e.kind,
                                    "target_class_id": e.target_class_id, "replacement_class_id": e.replacement_class_id,
                                    "sample_id": disc["sample_id"][m_t].to_numpy(), "cluster_id": g[m_t],
                                    "pred_class_id": lab, "score": s, "accepted": acc,
                                    "named_replacement": lab == e.replacement_class_id,
                                    "target_in_V": e.kind == "paired_present"}))
    paired = pd.concat(pf, ignore_index=True) if pf else pd.DataFrame()
    paired.to_csv(ed / "paired_target_images.csv.gz", index=False)

    # --- pattern counts
    patterns = {"definitions": PATTERN_DOC, "by_method": {}, "by_method_and_size": {}}
    for method, t in table.groupby("method"):
        patterns["by_method"][method] = _patterns(t)
        patterns["by_method_and_size"][method] = {int(s): _patterns(ts) for s, ts in t.groupby("vocab_size")}
    for method in ("max_mean_sim", "plurality@win_fraction") + ((f"{vk}@win_fraction",) if k_star > 1 else ()):
        if method not in thresholds:
            continue
        mm = imgs[imgs["method"] == method] if method in imgs["method"].unique() else None
        if mm is None:
            continue
        res = {}
        for s, ts in mm.groupby("vocab_size"):
            absent = ts[~ts["true_available"]]
            avail = ts[ts["true_available"]]
            res[int(s)] = {"P7_absent_accepted_rate": float(absent["accepted"].mean()) if len(absent) else None,
                           "P7_absent_images": int(len(absent)),
                           "P8_available_rejected_rate": float((~avail["accepted"]).mean()) if len(avail) else None,
                           "P8_rejected_but_label_correct": int((~avail["accepted"] & avail["correct"]).sum()),
                           "P8_rejected_and_label_wrong": int((~avail["accepted"] & ~avail["correct"]).sum()),
                           "accepted_score_median_absent": float(absent.loc[absent["accepted"], "score"].median())
                           if absent["accepted"].any() else None,
                           "accepted_score_median_correct": float(avail.loc[avail["accepted"] & avail["correct"], "score"].median())
                           if (avail["accepted"] & avail["correct"]).any() else None}
        patterns.setdefault("P7_P8_by_method_and_size", {})[method] = res
    # mean vs vote disagreements (forced, image level)
    a = imgs[imgs["method"] == "forced_top1"].reset_index(drop=True)
    b = imgs[imgs["method"] == "plurality_forced"].reset_index(drop=True)
    dis = {}
    for s in sorted(a["vocab_size"].unique()):
        ma, mb = a[a["vocab_size"] == s], b[b["vocab_size"] == s]
        differ = ma["pred_class_id"].to_numpy() != mb["pred_class_id"].to_numpy()
        dis[int(s)] = {"images_with_different_labels": int(differ.sum()), "of": int(len(ma)),
                       "mean_correct_vote_wrong": int((differ & ma["correct"].to_numpy() & ~mb["correct"].to_numpy()).sum()),
                       "vote_correct_mean_wrong": int((differ & ~ma["correct"].to_numpy() & mb["correct"].to_numpy()).sum())}
    patterns["mean_vs_plurality_disagreement_images"] = dis
    write_json(ed / "failure_patterns.json", patterns)

    # --- galleries: complete vocabulary, failure clusters with matched correct clusters
    def files_of(k, nmax=8):
        sids = disc.loc[disc["cluster_id"] == k].sort_values("sq_dist_to_centre")["sample_id"].tolist()[:nmax]
        return [img_root / rel[s] for s in sids], [f"{names[y_eval[s]]}" for s in sids]

    def panel(r, tag):
        f, cap = files_of(int(r.cluster_id))
        lab = names[int(r.label)] if int(r.label) >= 0 else "(none)"
        return {"title": f"{tag} cluster {int(r.cluster_id)} · n={int(r.n)} · purity {r.purity:.2f} · dominant "
                         f"'{names[int(r.dominant_class_id)]}' · label '{lab}' · votes dominant/label/winner "
                         f"{int(r.votes_for_dominant)}/{int(r.votes_for_label)}/{int(r.votes_winner)}",
                "files": f, "captions": cap}

    ft = table[table["episode_id"] == full_id]
    gallery = {}
    for method in ("forced_top1", "plurality_forced") + tuple(distinct):
        t = ft[ft["method"] == method]
        wrong = t[~t["label_correct"]]
        aw = wrong[wrong["dominant_available"]]
        for key, err in (("P1_weak_support", aw[~aw["dominant_is_vote_winner"]]),
                         ("P2_aggregation_override", aw[aw["dominant_is_vote_winner"]]),
                         ("P3_competition", wrong[wrong["label_shared_with_other_clusters"] > 0]),
                         ("P5_mixed", wrong[wrong["purity"] <= 0.5]),
                         ("P6_related_name", aw[aw["same_head_word"] | aw["label_in_text_nn5_of_dominant"]])):
            err = err.sort_values("n", ascending=False)
            panels = []
            for e_r, ok_r in _matched_correct(t, err, 4):
                panels += [panel(e_r, "ERROR"), panel(ok_r, "matched correct")]
            p = cluster_gallery(ed / "galleries" / f"{method}_{key}.png", panels,
                                f"{method} · {key} · complete vocabulary ({len(err)} clusters) · each error with the "
                                "correct cluster nearest in size and purity")
            if p:
                gallery[f"{method}_{key}"] = str(p.relative_to(out))
    write_json(ed / "galleries.json", gallery)
    write_json(ed / "README.json", {"files": {
        "image_decisions_complete_vocabulary.csv.gz": "every discovery image x primary method, complete vocabulary",
        "image_decisions_shared_episodes.csv.gz": "every discovery image x primary method x shared episode",
        "incorrect_decisions_shared_episodes.csv.gz": "rows of the above whose outcome is not correct_accepted/absent_rejected",
        "cluster_decisions_shared_episodes.csv.gz": "per (episode, method, cluster): votes, dominant class, pattern flags",
        "paired_target_images.csv.gz": "paired episodes: decisions on the target class's images",
        "discovery_similarity.npz": "R (images x classes) of the discovery population, to recompute any decision",
        "failure_patterns.json": "quantified patterns P1-P3, P5-P8 (P4 is in protocol R/A runs)"}})
    return patterns
