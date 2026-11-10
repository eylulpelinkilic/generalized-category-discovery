"""Diagnostic export: correctly and incorrectly clustered images with their similarity context.

Post-processing of a completed run; nothing here changes a result. The selection unit is an
image. Clustering correctness uses the fixed global Hungarian mapping of the upstream `v2`
evaluation, computed once over the full chosen population (the development pool by default):
an image is correctly clustered when its cluster's mapped class equals its class. CLIP grounding
correctness is a separate field.

Similarity rows come from the run's saved `R_<strategy>.npy` when present. If the downloaded
results lack it, rows are recomputed with the run's CLIP configuration and accepted only if they
reproduce the run's saved cluster aggregates and saved dev/calibration scores.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from e1 import upstream
from e1.config import Layout
from e1.manifests import CUB_MD5, CUB_URL, Manifest
from e1.provenance import sha256_file, write_json

POPULATION_POOL = {"dev": "heldout_dev", "calib": "heldout_calib"}


# ------------------------------------------------------------------ mapping and selection
def hungarian_mapping(y: np.ndarray, pred: np.ndarray) -> tuple[dict, np.ndarray, float]:
    """The upstream v2 construction: one assignment on the full population (cluster -> class)."""
    y, pred = y.astype(int), pred.astype(int)
    D = max(pred.max(), y.max()) + 1
    w = np.zeros((D, D), dtype=int)
    np.add.at(w, (pred, y), 1)
    rows, cols = linear_sum_assignment(w.max() - w)
    mapping = {int(r): int(c) for r, c in zip(rows, cols)}
    acc = sum(w[r, c] for r, c in mapping.items()) / len(y)
    # upstream cluster_acc = the same global assignment as v2's "All" (v2 itself needs Old and New images)
    if abs(acc - upstream.cluster_acc(y, pred)) > 1e-12:
        raise RuntimeError("mapping does not reproduce the upstream global-assignment accuracy")
    return mapping, w, float(acc)


def _spread_pick(cands: pd.DataFrame, n: int, rng: np.random.RandomState, used_cls=None, used_k=None) -> list:
    """Greedy: repeatedly take the candidate whose class and cluster are least used (random ties)."""
    used_cls, used_k = dict(used_cls or {}), dict(used_k or {})
    pool = cands.assign(_r=rng.permutation(len(cands))).copy()
    picked = []
    while len(picked) < n and len(pool):
        key = pool["y"].map(lambda c: used_cls.get(c, 0)) * 1000 + pool["cluster_id"].map(lambda k: used_k.get(k, 0))
        i = pool.index[np.lexsort((pool["_r"].to_numpy(), key.to_numpy()))[0]]
        row = pool.loc[i]
        picked.append(i)
        used_cls[row["y"]] = used_cls.get(row["y"], 0) + 1
        used_k[row["cluster_id"]] = used_k.get(row["cluster_id"], 0) + 1
        pool = pool.drop(index=i)
    return picked


def select(pop: pd.DataFrame, n: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`pop`: sample_id, y, cluster_id, correct, cluster_size. Returns (incorrect, correct)."""
    rng = np.random.RandomState(seed)
    wrong = pop[~pop["correct"]]
    inc = pop.loc[_spread_pick(wrong, min(n, len(wrong)), rng)].copy()
    inc["selection_reason"] = "incorrectly clustered; spread over classes and clusters"
    right = pop[pop["correct"]]
    chosen, reasons, paired = [], [], []
    used_cls, used_k = {}, {}
    for _, r in inc.iterrows():                          # one control per incorrect example
        if len(chosen) >= n:
            break
        avail = right[~right.index.isin(chosen)]
        for reason, mask in (("same_cluster", avail["cluster_id"] == r["cluster_id"]),
                             ("same_true_class", avail["y"] == r["y"])):
            if mask.any():
                break
        else:
            reason = "nearest_cluster_size"
            gap = (avail["cluster_size"] - r["cluster_size"]).abs()
            mask = gap == gap.min()
        if not mask.any():
            continue
        i = _spread_pick(avail[mask], 1, rng, used_cls, used_k)[0]
        chosen.append(i)
        reasons.append(reason)
        paired.append(r["sample_id"])
        used_cls[right.loc[i, "y"]] = used_cls.get(right.loc[i, "y"], 0) + 1
        used_k[right.loc[i, "cluster_id"]] = used_k.get(right.loc[i, "cluster_id"], 0) + 1
    extra = n - len(chosen)                              # if there are fewer incorrect examples than n
    if extra > 0:
        rest = right[~right.index.isin(chosen)]
        more = _spread_pick(rest, min(extra, len(rest)), rng, used_cls, used_k)
        chosen += more
        reasons += ["spread_fill"] * len(more)
        paired += [""] * len(more)
    cor = pop.loc[chosen].copy()
    cor["selection_reason"] = reasons
    cor["paired_incorrect_sample_id"] = paired
    inc["paired_incorrect_sample_id"] = ""
    return inc, cor


# ------------------------------------------------------------------ images
class _HashingReader:
    def __init__(self, raw):
        self.raw, self.md5, self.n = raw, hashlib.md5(), 0  # noqa: S324 - matches the published checksum

    def read(self, size=-1):
        data = self.raw.read(size)
        self.md5.update(data)
        self.n += len(data)
        return data


def fetch_cub_files(rel_paths: list[str], dest: Path) -> dict:
    """Stream the official CUB archive once and extract only `rel_paths` (paths under
    CUB_200_2011/). The whole stream is MD5-checked against the upstream constant."""
    dest = Path(dest)
    record_path = dest / "fetch_record.json"
    todo = {f"CUB_200_2011/{p}" for p in rel_paths if not (dest / "CUB_200_2011" / p).exists()}
    if not todo:
        prior = json.loads(record_path.read_text()) if record_path.exists() else None
        return {"source": "existing extraction", "files": len(rel_paths), "extraction_record": prior}
    req = urllib.request.Request(CUB_URL, headers={"User-Agent": "e1-diagnostics"})
    with urllib.request.urlopen(req) as resp:  # noqa: S310 - fixed official URL
        reader = _HashingReader(resp)
        with tarfile.open(fileobj=reader, mode="r|gz") as tar:
            for member in tar:
                if member.name in todo:
                    tar.extract(member, dest, filter="data")
        while reader.read(1 << 20):                       # hash the remainder of the stream
            pass
    md5 = reader.md5.hexdigest()
    if md5 != CUB_MD5:
        raise RuntimeError(f"CUB archive MD5 {md5} != {CUB_MD5}; extracted files must not be used")
    missing = [p for p in todo if not (dest / p).exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} files not found in the archive, e.g. {missing[:3]}")
    record = {"source": CUB_URL, "archive_md5": md5, "md5_check": "passed", "archive_bytes": reader.n,
              "extracted": len(todo)}
    write_json(record_path, record)
    return record


# ------------------------------------------------------------------ similarity rows
def similarity_rows(cfg, gdir: Path, strategy: str, rows: pd.DataFrame, need_rows: np.ndarray,
                    image_paths: dict, clip_name: str) -> tuple[dict, dict]:
    """{row: R[row]} for `need_rows`; saved R if present, otherwise recomputed and validated."""
    saved = gdir / f"R_{strategy}.npy"
    if saved.exists():
        R = np.load(saved, mmap_mode="r")
        return {int(r): np.asarray(R[r]) for r in need_rows}, {"source": f"saved {saved.name}",
                                                              "sha256": sha256_file(saved)}
    import torch

    from e1.clip_extract import load_clip
    from e1.datasets import load_rgb

    T = np.load(gdir / f"text_embeddings_{strategy}.npz")["per_class"]
    model, preprocess, _ = load_clip(cfg["clip"]["models"][clip_name], "cpu")
    out = {}
    ids = rows.set_index("row").loc[need_rows, "sample_id"].tolist()
    with torch.no_grad():
        for s in range(0, len(ids), 64):
            batch = torch.stack([preprocess(load_rgb(image_paths[i])) for i in ids[s:s + 64]])
            z = torch.nn.functional.normalize(model.encode_image(batch).float(), dim=-1).numpy()
            for r, v in zip(need_rows[s:s + 64], z @ T.T):
                out[int(r)] = v.astype(np.float32)
    return out, {"source": f"recomputed: {saved.name} is not in the downloaded results",
                 "model": cfg["clip"]["models"][clip_name], "device": "cpu",
                 "text_embeddings": f"saved text_embeddings_{strategy}.npz (sha256 "
                                    f"{sha256_file(gdir / f'text_embeddings_{strategy}.npz')[:16]}…)"}


def validate_rows(R: dict, rows: pd.DataFrame, gdir: Path, strategy: str, clusters: list[int], full_ep: str) -> dict:
    """Recomputed rows must reproduce the run's saved aggregates over complete clusters."""
    agg = np.load(gdir / f"discovery_aggregates_{strategy}.npz")
    pred = pd.read_csv(gdir / "cluster_predictions.csv.gz")
    pred = pred[(pred["representation"] == "all") & (pred["episode_id"] == full_ep)]
    diffs, score_diffs, label_mismatch = [], [], 0
    for k in clusters:
        disc = rows[(rows["population"] == "discovery") & (rows["cluster_id"] == k)]["row"]
        if len(disc):
            diffs.append(np.abs(np.mean([R[r] for r in disc], 0) - agg["A"][k]).max())
        for pop in ("dev", "calib"):
            m = rows[(rows["population"] == pop) & (rows["cluster_id"] == k)]["row"]
            if not len(m):
                continue
            a = np.mean([R[r] for r in m], 0)
            p = pred[(pred["population"] == pop) & (pred["cluster_id"] == k)].set_index("method")
            score_diffs.append(abs(a.max() - p.loc["max_mean_sim", "score"]))
            label_mismatch += int(np.argmax(a) != p.loc["forced_top1", "pred_class_id"])
    res = {"clusters_checked": len(clusters), "max_abs_diff_discovery_aggregate_A": float(max(diffs, default=0)),
           "max_abs_diff_dev_calib_max_mean_sim": float(max(score_diffs, default=0)),
           "dev_calib_group_label_mismatches": label_mismatch}
    res["passed"] = res["max_abs_diff_discovery_aggregate_A"] < 1e-4 and \
        res["max_abs_diff_dev_calib_max_mean_sim"] < 1e-4 and label_mismatch == 0
    return res


# ------------------------------------------------------------------ figures
def heatmap(path: Path, M: np.ndarray, row_labels, col_labels, marks, vmin, vmax, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = max(3.0, 0.16 * len(row_labels) + 1.6)
    w = max(6.0, 0.32 * len(col_labels) + 3.2)
    fig, (ax_m, ax) = plt.subplots(1, 2, figsize=(w, h), gridspec_kw={"width_ratios": [0.35, len(col_labels)], "wspace": 0.02})
    colours = {"incorrect": "#c0392b", "correct": "#2e7d32", "": "#ffffff"}
    ax_m.imshow(np.array([[matplotlib.colors.to_rgb(colours[m])] for m in marks]), aspect="auto")
    ax_m.set_xticks([])
    ax_m.set_yticks(range(len(row_labels)))
    ax_m.set_yticklabels(row_labels, fontsize=6)
    im = ax.imshow(M, aspect="auto", cmap="Blues", vmin=vmin, vmax=vmax)
    ax.set_yticks([])
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=70, ha="right", fontsize=6)
    ax.set_title(title, fontsize=8, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label("image-text cosine similarity (fixed scale for all heatmaps)", fontsize=6)
    cb.ax.tick_params(labelsize=6)
    handles = [matplotlib.patches.Patch(color=colours[k], label=f"selected: {k}ly clustered") for k in ("incorrect", "correct")]
    fig.legend(handles=handles, loc="lower left", fontsize=6, frameon=False)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def contact_sheet(path: Path, items: list[dict], title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    cols = 10
    rows = max(1, -(-len(items) // cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.1, rows * 2.75))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, it in zip(np.atleast_1d(axes).ravel(), items):
        im = Image.open(it["file"]).convert("RGB")
        im.thumbnail((240, 240))
        ax.imshow(im)
        ax.set_title(f"{it['sample_id']} · cluster {it['cluster_id']}\ntrue: {it['true'][:26]}\nmapped: {it['mapped'][:26]}",
                     fontsize=6.5, loc="left")
    fig.suptitle(title, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ the export
def export(cfg: dict, train_seed: int, k_mode: str, cluster_seed: int, clip_name: str, population: str = "dev",
           n: int = 50, seed: int = 42, image_root: str | None = None, image_cache: str | None = None,
           allow_synthetic: bool = False) -> Path:
    layout = Layout(cfg)
    manifest = Manifest(layout.data)
    gdir = layout.grounding(train_seed, k_mode, cluster_seed, clip_name)
    cdir = layout.clustering(train_seed, k_mode, cluster_seed)
    run = json.loads((layout.gcd_run(train_seed) / "run.json").read_text())
    if (manifest.meta.get("SYNTHETIC") or cfg["dataset"]["name"] == "smoke") and not allow_synthetic:
        raise RuntimeError("smoke/synthetic run: diagnostics are only exported for real data")
    out = gdir / "diagnostics" / "clustering_examples"
    if out.exists():
        shutil.rmtree(out)
    for sub in ("incorrect", "correct", "cluster_context"):
        (out / sub).mkdir(parents=True)

    strategy = json.loads((gdir / "selection.json").read_text())["prompt_strategy"]
    episodes = pd.read_csv(gdir / "episodes.csv")
    full_ep = episodes.iloc[0]["episode_id"]
    class_ids = np.load(gdir / f"text_embeddings_{strategy}.npz")["class_ids"]
    names = manifest.classes.set_index("class_id")["canonical_name"]
    status = manifest.classes.set_index("class_id")["gcd_status"]
    rows = pd.read_csv(gdir / "rows.csv")
    sizes = rows.groupby(["cluster_id", "population"]).size().unstack(fill_value=0)

    # 1. population, labels (permission of that population) and the fixed mapping
    if population == "discovery":
        y_all, permission = manifest.labels("eval"), "labels_eval.csv (evaluation only)"
    else:
        y_all, permission = manifest.labels("heldout"), f"labels_heldout.csv ({POPULATION_POOL[population]} rows)"
    pop = rows[rows["population"] == population].copy()
    pop["y"] = y_all.loc[pop["sample_id"]].to_numpy()
    mapping, w, acc = hungarian_mapping(pop["y"].to_numpy(), pop["cluster_id"].to_numpy())
    pop["mapped_class_id"] = pop["cluster_id"].map(mapping)
    pop["correct"] = pop["mapped_class_id"] == pop["y"]
    pop["cluster_size"] = pop["cluster_id"].map(sizes.sum(1))
    pd.DataFrame([{"cluster_id": k, "mapped_class_id": c, "mapped_class_name": names.get(c, f"(class {c}: none in population)"),
                   "support_in_population": int(w[k, c])} for k, c in sorted(mapping.items())]
                 ).to_csv(out / f"hungarian_mapping_{population}.csv", index=False)

    # 2. selection
    inc, cor = select(pop.reset_index(drop=True), n, seed)
    inc["group"], cor["group"] = "incorrect", "correct"
    sel = pd.concat([inc, cor], ignore_index=True)
    clusters = sorted(sel["cluster_id"].unique().tolist())

    # 3. images (original files) and similarity rows
    members = rows[rows["cluster_id"].isin(clusters)]
    rel = dict(zip(manifest.table["sample_id"], manifest.table["rel_path"]))
    if image_root:
        root = Path(image_root)
        image_info = {"source": f"local image root {root}"}
    elif cfg["dataset"]["name"] == "cub":
        cache = Path(image_cache or Path.home() / ".cache" / "e1_cub_images")
        image_info = fetch_cub_files(sorted({rel[s] for s in members["sample_id"]}), cache)
        root = cache / "CUB_200_2011"
    else:
        raise RuntimeError("set --image-root to the dataset's image root")
    paths = {s: root / rel[s] for s in members["sample_id"]}
    missing = [s for s, p in paths.items() if not p.exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} images missing under {root}, e.g. {missing[:3]}")
    R, sim_info = similarity_rows(cfg, gdir, strategy, rows, members["row"].to_numpy(), paths, clip_name)
    if sim_info["source"].startswith("recomputed"):
        sim_info["validation"] = validate_rows(R, rows, gdir, strategy, clusters, full_ep)
        if not sim_info["validation"]["passed"]:
            raise RuntimeError(f"recomputed similarities do not reproduce the saved run: {sim_info['validation']}")

    # 4. cluster-level facts (saved predictions; evaluation-only composition marked as such)
    pred = pd.read_csv(gdir / "cluster_predictions.csv.gz")
    pred = pred[(pred["representation"] == "all") & (pred["episode_id"] == full_ep) & (pred["method"] == "forced_top1")]
    cpred = {(p, k): c for p, k, c in zip(pred["population"], pred["cluster_id"], pred["pred_class_id"])}
    y_eval, y_held, y_lab = manifest.labels("eval"), manifest.labels("heldout"), manifest.labels("labelled")
    label_of = {**y_eval.to_dict(), **y_held.to_dict(), **y_lab.to_dict()}
    permission_of = {**{s: "evaluation_only" for s in y_eval.index}, **{s: "heldout_dev_calib" for s in y_held.index},
                     **{s: "labelled_anchor" for s in y_lab.index}}

    def composition(k, which):
        m = rows[(rows["cluster_id"] == k) & (rows["population"] == which)]["sample_id"]
        vc = pd.Series([label_of[s] for s in m]).value_counts()
        return vc

    agg = np.load(gdir / f"discovery_aggregates_{strategy}.npz")
    all_vals = np.concatenate([R[r] for r in members["row"]])
    vmin, vmax = (float(x) for x in np.percentile(all_vals, [1, 99]))
    context_rows = []
    for k in clusters:
        cd = out / "cluster_context" / f"cluster_{k:03d}"
        cd.mkdir()
        m = members[members["cluster_id"] == k].copy()
        order = {"discovery": 0, "anchor": 1, "dev": 2, "calib": 3}
        m = m.assign(_o=m["population"].map(order)).sort_values(["_o", "sample_id"]).drop(columns="_o")
        M = np.stack([R[r] for r in m["row"]])
        np.savez(cd / "similarity_matrix.npz", R=M, row_sample_ids=m["sample_id"].to_numpy().astype(str),
                 row_population=m["population"].to_numpy().astype(str), col_class_ids=class_ids,
                 col_class_names=np.array([names[c] for c in class_ids]), strategy=np.array(strategy),
                 episode_id=np.array(full_ep))
        mem = m[["sample_id", "population", "is_anchor", "sq_dist_to_centre"]].copy()
        mem["class_id"] = [label_of[s] for s in mem["sample_id"]]
        mem["class_name"] = [names[c] for c in mem["class_id"]]
        mem["label_permission"] = [permission_of[s] for s in mem["sample_id"]]
        mem["selected_group"] = mem["sample_id"].map(dict(zip(sel["sample_id"], sel["group"]))).fillna("")
        mem.to_csv(cd / "members.csv", index=False)
        comp = []
        for which in ("discovery", "anchor", "dev", "calib"):
            for c, cnt in composition(k, which).items():
                comp.append({"population": which, "class_id": int(c), "class_name": names[c], "count": int(cnt),
                             "label_permission": {"discovery": "evaluation_only", "anchor": "labelled_anchor"}.get(which, "heldout_dev_calib")})
        pd.DataFrame(comp).to_csv(cd / "class_composition.csv", index=False)
        dev_rows = m[m["population"] == population]["row"]
        agg_pop = np.mean([R[r] for r in dev_rows], 0) if len(dev_rows) else np.full(len(class_ids), np.nan)
        pd.DataFrame({"class_id": class_ids, "class_name": [names[c] for c in class_ids],
                      "A_discovery_saved": agg["A"][k], "B_discovery_saved": agg["B"][k],
                      f"A_{population}_group": agg_pop}).to_csv(cd / "aggregate_scores.csv", index=False)
        disc_comp, pop_comp = composition(k, "discovery"), composition(k, population)
        info = {"cluster_id": k, "n_members": {p: int((m["population"] == p).sum()) for p in order},
                "mapped_class_id_" + population: mapping.get(k), "discovery_mean_norm": float(agg["mean_norm"][k]),
                "semantic_prediction_discovery": {"class_id": int(cpred[("discovery", k)]), "name": names[cpred[("discovery", k)]],
                                                  "rule": "forced_top1 = argmax of saved A (complete vocabulary)"},
                f"semantic_prediction_{population}_group": ({"class_id": int(cpred[(population, k)]), "name": names[cpred[(population, k)]]}
                                                             if (population, k) in cpred and cpred[(population, k)] >= 0 else None),
                "dominant_class_discovery_evaluation_only": (int(disc_comp.index[0]) if len(disc_comp) else None),
                "purity_discovery_evaluation_only": (float(disc_comp.iloc[0] / disc_comp.sum()) if len(disc_comp) else None),
                f"dominant_class_{population}": int(pop_comp.index[0]) if len(pop_comp) else None,
                f"purity_{population}": float(pop_comp.iloc[0] / pop_comp.sum()) if len(pop_comp) else None}
        # heatmap: all member rows; columns = top-12 by saved discovery aggregate + true, mapped and predicted classes
        top = list(np.argsort(-agg["A"][k])[:12]) if not np.isnan(agg["A"][k]).all() else []
        extra = [label_of[s] for s in sel.loc[sel["cluster_id"] == k, "sample_id"]] + \
                [c for c in (mapping.get(k), info["dominant_class_discovery_evaluation_only"]) if c is not None and c < len(class_ids)]
        cols = list(dict.fromkeys(int(c) for c in top + extra))
        col_idx = [int(np.flatnonzero(class_ids == c)[0]) for c in cols]
        marks = mem["selected_group"].tolist()
        heatmap(cd / "heatmap.png", M[:, col_idx], [f"{s} [{p[:4]}] {names[label_of[s]][:18]}" for s, p in zip(m["sample_id"], m["population"])],
                [f"{c}:{names[c][:20]}" for c in cols], marks, vmin, vmax,
                f"cluster {k}: rows = all members (discovery, anchor, dev, calib); columns = displayed subset")
        info["heatmap_displayed_columns"] = cols
        info["heatmap_displayed_rows"] = "all members"
        info["heatmap_colour_scale"] = [vmin, vmax]
        write_json(cd / "context.json", info)
        context_rows.append(info)

    # 5. per-image export
    run_id = f"{layout.gcd_run(train_seed).name}/{cdir.name}/{gdir.name}"
    out_rows, sim_rows, items = [], [], {"incorrect": [], "correct": []}
    for i, r in sel.reset_index(drop=True).iterrows():
        s, k = r["sample_id"], int(r["cluster_id"])
        fname = f"{r['group']}_{i % n:02d}__{s.replace(':', '-')}__cluster{k:03d}{Path(rel[s]).suffix}"
        dst = out / r["group"] / fname
        shutil.copyfile(paths[s], dst)
        info = next(c for c in context_rows if c["cluster_id"] == k)
        row_vec = R[int(r["row"])]
        img_pred = int(class_ids[np.argmax(row_vec)])
        group_pred = info[f"semantic_prediction_{population}_group"]
        mapped = int(r["mapped_class_id"])
        out_rows.append({
            "group": r["group"], "sample_id": s, "exported_filename": f"{r['group']}/{fname}",
            "original_path": str(paths[s]), "original_rel_path": rel[s], "dataset": cfg["dataset"]["name"],
            "source_split": manifest.table.set_index("sample_id").loc[s, "source_split"], "population": population,
            "run_id": run_id, "train_seed": train_seed, "cluster_seed": cluster_seed, "k_mode": k_mode,
            "checkpoint_sha256": run["summary"].get("final_checkpoint_sha256"),
            "true_class_id": int(r["y"]), "true_class_name": names[int(r["y"])], "gcd_status": status[int(r["y"])],
            "cluster_id": k, "hungarian_mapped_class_id": mapped,
            "hungarian_mapped_class_name": names.get(mapped, f"(class {mapped}: none in population)"),
            "hungarian_mapping_support": int(w[k, mapped]),
            "clustering_correct": bool(mapped == int(r["y"])),
            "cluster_size_total": int(sum(info["n_members"].values())),
            **{f"cluster_size_{p}": v for p, v in info["n_members"].items()},
            "cluster_dominant_class_discovery_evalonly": info["dominant_class_discovery_evaluation_only"],
            "cluster_dominant_name_discovery_evalonly": names.get(info["dominant_class_discovery_evaluation_only"], ""),
            "cluster_purity_discovery_evalonly": info["purity_discovery_evaluation_only"],
            "agrees_with_dominant_discovery_evalonly": int(r["y"]) == info["dominant_class_discovery_evaluation_only"],
            f"cluster_dominant_class_{population}": info[f"dominant_class_{population}"],
            f"cluster_purity_{population}": info[f"purity_{population}"],
            f"agrees_with_dominant_{population}": int(r["y"]) == info[f"dominant_class_{population}"],
            "clip_image_pred_class_id": img_pred, "clip_image_pred_name": names[img_pred],
            "clip_image_pred_score": float(row_vec.max()), "clip_image_correct": img_pred == int(r["y"]),
            f"clip_cluster_pred_{population}_group_id": group_pred["class_id"] if group_pred else None,
            f"clip_cluster_pred_{population}_group_name": group_pred["name"] if group_pred else "",
            "clip_cluster_pred_discovery_id": info["semantic_prediction_discovery"]["class_id"],
            "clip_cluster_pred_discovery_name": info["semantic_prediction_discovery"]["name"],
            "grounding_correct_cluster_aggregate": bool(group_pred and group_pred["class_id"] == int(r["y"])),
            "candidate_vocabulary": f"{full_ep} (all {len(class_ids)} classes), prompts {strategy}, CLIP {clip_name}",
            "selection_reason": r["selection_reason"], "paired_incorrect_sample_id": r["paired_incorrect_sample_id"],
            "cluster_context_dir": f"cluster_context/cluster_{k:03d}", "similarity_row_index": i,
            "label_permission": permission})
        sim_rows.append(row_vec)
        items[r["group"]].append({"file": dst, "sample_id": s, "cluster_id": k, "true": names[int(r["y"])],
                                  "mapped": names.get(mapped, "none")})
    ex = pd.DataFrame(out_rows)
    ex.to_csv(out / "examples.csv", index=False)
    np.savez(out / "image_similarity_rows.npz", R=np.stack(sim_rows), row_sample_ids=ex["sample_id"].to_numpy().astype(str),
             col_class_ids=class_ids, col_class_names=np.array([names[c] for c in class_ids]),
             strategy=np.array(strategy), episode_id=np.array(full_ep))
    wide = pd.DataFrame(np.stack(sim_rows), columns=[f"class_{c}" for c in class_ids])
    wide.insert(0, "sample_id", ex["sample_id"])
    wide.to_csv(out / "image_similarity_rows.csv", index=False, float_format="%.6f")
    for g in ("incorrect", "correct"):
        contact_sheet(out / f"contact_sheet_{g}.png", items[g],
                      f"{g} clustering ({population} population, run {run_id}); title: sample · cluster / true / Hungarian-mapped")

    # 6. validation
    checks = validate_export(out, ex, sel, mapping, paths, R, rows, class_ids, label_of, rel, cfg)
    summary = {"run": {"dataset": cfg["dataset"]["name"], "run_id": run_id, "train_seed": train_seed, "k_mode": k_mode,
                       "cluster_seed": cluster_seed, "clip": clip_name, "prompt_strategy": strategy,
                       "discovery_method": "GCD (upstream contrastive training, final epoch) + SS-k-means",
                       "checkpoint_sha256": run["summary"].get("final_checkpoint_sha256"),
                       "manifest_sha256": manifest.sha256, "real_data": not manifest.meta.get("SYNTHETIC", False)},
               "population": population, "label_permission": permission,
               "hungarian": {"population_size": int(len(pop)), "accuracy": acc, "n_clusters": len(mapping),
                             "rule": "upstream v2 construction, one assignment over the full population"},
               "eligible": {"incorrect": int((~pop['correct']).sum()), "correct": int(pop["correct"].sum())},
               "selected": {"incorrect": int(len(inc)), "correct": int(len(cor)), "requested_each": n,
                            "shortfall": {"incorrect": max(0, n - len(inc)), "correct": max(0, n - len(cor))}},
               "control_reasons": cor["selection_reason"].value_counts().to_dict(), "seed": seed,
               "clusters_with_context": len(clusters), "images": image_info, "similarities": sim_info,
               "heatmap_colour_scale": [vmin, vmax], "validation": checks}
    write_json(out / "export_summary.json", summary)
    write_readme(out, summary)
    archive = shutil.make_archive(str(out.parent / "clustering_examples"), "zip", out.parent, "clustering_examples")
    summary["archive"] = archive
    write_json(out / "export_summary.json", summary)
    if not checks["passed"]:
        raise RuntimeError(f"export validation failed: {checks}")
    return out


def validate_export(out, ex, sel, mapping, paths, R, rows, class_ids, label_of, rel, cfg) -> dict:
    c = {}
    c["unique_samples"] = bool(ex["sample_id"].is_unique)
    c["groups_disjoint"] = set(ex.loc[ex.group == "incorrect", "sample_id"]).isdisjoint(ex.loc[ex.group == "correct", "sample_id"])
    c["files_match_source_bytes"] = all(sha256_file(out / f) == sha256_file(p) for f, p in zip(ex["exported_filename"], ex["original_path"]))
    c["correctness_matches_mapping"] = all((mapping[k] == y) == flag for k, y, flag in
                                           zip(ex["cluster_id"], ex["true_class_id"], ex["clustering_correct"]))
    c["group_matches_correctness"] = bool(((ex["group"] == "correct") == ex["clustering_correct"]).all())
    sim = np.load(out / "image_similarity_rows.npz")
    rowmap = dict(zip(rows["sample_id"], rows["row"]))
    c["similarity_rows_match"] = all(np.array_equal(sim["R"][i], R[rowmap[s]]) for i, s in enumerate(sim["row_sample_ids"]))
    ok_ctx, ok_ids = True, True
    for s, k in zip(ex["sample_id"], ex["cluster_id"]):
        z = np.load(out / "cluster_context" / f"cluster_{k:03d}" / "similarity_matrix.npz")
        ids = z["row_sample_ids"].tolist()
        ok_ids &= len(ids) == z["R"].shape[0] and len(z["col_class_ids"]) == z["R"].shape[1] and len(set(ids)) == len(ids)
        ok_ctx &= s in ids and np.array_equal(z["R"][ids.index(s)], R[rowmap[s]])
    c["every_example_in_its_cluster_context"] = bool(ok_ctx)
    c["matrices_have_row_and_column_ids"] = bool(ok_ids and np.array_equal(sim["col_class_ids"], class_ids))
    if cfg["dataset"]["name"] == "cub":       # CUB folder "NNN.Name" encodes the class: check labels against paths
        from e1.manifests import cub_class_split
        known, novel = cub_class_split()
        gcd_of = {cl: i for i, cl in enumerate(known + novel)}
        c["labels_agree_with_image_folders"] = all(gcd_of[int(rel[s].split("/")[1].split(".")[0]) - 1] == y
                                                   for s, y in zip(ex["sample_id"], ex["true_class_id"]))
    c["passed"] = all(v for v in c.values() if isinstance(v, bool))
    return c


def write_readme(out: Path, s: dict) -> None:
    v = s["validation"]
    text = f"""# Clustering diagnostic examples

Run: `{s['run']['run_id']}` ({s['run']['dataset']}, training seed {s['run']['train_seed']}, {s['run']['k_mode']} K,
clustering seed {s['run']['cluster_seed']}, CLIP {s['run']['clip']} with `{s['run']['prompt_strategy']}` prompts).
Discovery: {s['run']['discovery_method']}; checkpoint sha256 `{(s['run']['checkpoint_sha256'] or '')[:16]}…`.
Real data: {s['run']['real_data']}. Generated by `python -m e1 export-diagnostics` (seed {s['seed']}).

## Source population

`{s['population']}`: {'held-out official test images of the GCD-known classes, assigned to the fitted GCD centres without refitting (the development pool of PROTOCOL.md). It contains no novel classes, so only known-class clustering errors are visible here. The unlabelled discovery population is the final evaluation population and was not used.' if s['population'] == 'dev' else 'see PROTOCOL.md'}
Labels used for selection: {s['label_permission']}.

## Correctness

The cluster-to-class map is the upstream `v2` Hungarian assignment, computed once over the full
population ({s['hungarian']['population_size']} images; accuracy {100 * s['hungarian']['accuracy']:.2f}%) and saved in
`hungarian_mapping_{s['population']}.csv`. An image is **correctly clustered** when its cluster's mapped
class equals its class. A cluster mapped to a class absent from this population cannot be correct.
The dominant-class fields (`*_dominant_*`, `*_purity_*`, `agrees_with_dominant_*`) are separate and can disagree with the
Hungarian flag. CLIP grounding correctness is separate again (`clip_image_correct`, `grounding_correct_cluster_aggregate`).

## Sampling

Seed {s['seed']}. Incorrect examples: greedy spread, each pick from the least-used class and cluster, random tie-break.
Correct controls: one per incorrect example, preferring the same cluster, then the same true class, then the cluster
of nearest size; the reason is in `selection_reason` and the paired example in `paired_incorrect_sample_id`.
Eligible: {s['eligible']}; selected: {s['selected']}.

## Files

* `incorrect/`, `correct/`: original images, named `<group>_<nn>__<sample id>__cluster<id>.jpg`; source path in `examples.csv`.
* `examples.csv`: one row per image (identities, labels, mapping, cluster facts, CLIP predictions, vocabulary).
* `image_similarity_rows.npz` / `.csv`: each image's full row of cosine similarities over the complete vocabulary
  (`col_class_ids`, `col_class_names` give the column order).
* `cluster_context/cluster_<id>/`: `similarity_matrix.npz` (every member's row; `row_sample_ids`, `row_population`,
  `col_class_ids`), `members.csv` (with label permission per row), `class_composition.csv` (discovery composition is
  evaluation-only), `aggregate_scores.csv` (saved A and B of the discovery members, the {s['population']} group mean),
  `context.json` (sizes, mapped class, semantic predictions, displayed heatmap subset), `heatmap.png`.
* `contact_sheet_incorrect.png`, `contact_sheet_correct.png`; `hungarian_mapping_{s['population']}.csv`; `export_summary.json`.

To connect an image to its context: take `cluster_context_dir` from `examples.csv`, open `similarity_matrix.npz` and
find the row whose `row_sample_ids` equals the image's `sample_id`.

Heatmaps show every member row and a column subset (top 12 classes by the saved discovery aggregate plus the selected
images' true classes, the mapped class and the dominant class; listed in `context.json`). All heatmaps share one colour
scale, [{s['heatmap_colour_scale'][0]:.3f}, {s['heatmap_colour_scale'][1]:.3f}] (1st to 99th percentile of all exported similarities).

Similarities: {s['similarities']['source']}.{' Validation against the saved run: ' + json.dumps(s['similarities'].get('validation')) if s['similarities'].get('validation') else ''}
Images: {json.dumps(s['images'])}.

## Validation

{json.dumps(v, indent=1)}
"""
    (out / "README.md").write_text(text)
