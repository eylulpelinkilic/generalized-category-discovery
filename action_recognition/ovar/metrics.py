"""Evaluation of saved predictions.

* Denominators are all listed videos; failed videos count as errors.
* Familiar/unfamiliar accuracies are computed on predictions made over the
  full target vocabulary (the candidate set is never restricted per subgroup).
* Confidence intervals: percentile bootstrap resampling source groups
  (`source_group_id`) with replacement, so clips cut from one recording stay
  together.
* Cross-domain AVG is the unweighted mean of domain accuracies (as in
  XOV-Action); pooled accuracy over all videos is reported separately.
* Scores are cosine similarities, not probabilities.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SUBSETS = ("overall", "familiar", "unfamiliar")


def subset_masks(pred: pd.DataFrame) -> dict[str, np.ndarray]:
    status = pred["category_status"].to_numpy()
    return {"overall": np.ones(len(pred), bool), "familiar": status == "familiar",
            "unfamiliar": status == "unfamiliar"}


def cluster_bootstrap(correct: np.ndarray, groups: np.ndarray, masks: dict, n_boot: int,
                      seed: int) -> dict[str, np.ndarray]:
    """Bootstrap accuracy samples per subset, resampling whole groups."""
    _, inverse = np.unique(groups, return_inverse=True)
    n_groups = inverse.max() + 1 if len(inverse) else 0
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_groups, size=(n_boot, n_groups))
    weights = np.zeros((n_boot, n_groups))
    np.add.at(weights, (np.repeat(np.arange(n_boot), n_groups), draws.ravel()), 1)
    samples = {}
    for name, mask in masks.items():
        hits = np.bincount(inverse, weights=correct * mask, minlength=n_groups)
        count = np.bincount(inverse, weights=mask.astype(float), minlength=n_groups)
        num, den = weights @ hits, weights @ count
        samples[name] = np.divide(num, den, out=np.full(n_boot, np.nan), where=den > 0)
    return samples


def interval(samples: np.ndarray, level: float = 0.95) -> list[float] | None:
    valid = samples[~np.isnan(samples)]
    if len(valid) == 0:
        return None
    tail = (1 - level) / 2 * 100
    return [float(np.percentile(valid, tail)), float(np.percentile(valid, 100 - tail))]


def _accuracy_block(correct, masks, boot) -> dict:
    out = {}
    for name in SUBSETS:
        mask = masks[name]
        n = int(mask.sum())
        out[name] = {"n": n, "n_correct": int(correct[mask].sum()),
                     "acc": float(correct[mask].mean()) if n else None,
                     "ci95": interval(boot[name]) if n else None}
    return out


REVIEW_FILE = Path(__file__).resolve().parents[1] / "data" / "audit" / "category_review.csv"


def near_familiar_classes(dataset: str) -> set[str]:
    """Officially unfamiliar classes with a close Kinetics400 counterpart (unverified review)."""
    if not REVIEW_FILE.exists():
        return set()
    review = pd.read_csv(REVIEW_FILE, keep_default_na=False)
    rows = review[(review["dataset"] == dataset) & (review["sensitivity_tag"] == "near_familiar")]
    return set(rows["action_name"])


def dataset_metrics(pred: pd.DataFrame, classes: pd.DataFrame, n_boot: int = 1000,
                    seed: int = 0) -> tuple[dict, pd.DataFrame, pd.DataFrame, dict]:
    correct = pred["correct"].to_numpy().astype(float)
    masks = subset_masks(pred)
    groups = pred["source_group_id"].to_numpy()
    near = near_familiar_classes(pred["dataset_name"].iloc[0]) if "dataset_name" in pred and len(pred) else set()
    strict = masks["unfamiliar"] & ~pred["action_name"].isin(near).to_numpy() if near else None
    boot = cluster_bootstrap(correct, groups, {**masks, **({"strict": strict} if near else {})}, n_boot, seed)
    failed = pred["status"] != "ok"
    names = classes["official_name"].tolist()
    n_classes = len(names)

    confusion = np.zeros((n_classes, n_classes + 1), dtype=int)
    for gt, p, f in zip(pred["action_id"], pred["pred_id"], failed):
        confusion[gt, n_classes if f else p] += 1
    confusion_df = pd.DataFrame(confusion, index=names, columns=names + ["FAILED"])

    per_class = []
    for cid, name in enumerate(names):
        row = confusion[cid]
        n = int(row.sum())
        errors = row[:n_classes].copy()
        errors[cid] = 0
        top_err = int(errors.argmax()) if errors.any() else None
        per_class.append({
            "action_id": cid, "action_name": name, "category_status": classes["category_status"][cid],
            "n": n, "n_correct": int(row[cid]), "acc": row[cid] / n if n else np.nan,
            "n_failed": int(row[n_classes]),
            "most_confused_with": names[top_err] if top_err is not None else "",
            "most_confused_count": int(errors[top_err]) if top_err is not None else 0,
        })
    per_class_df = pd.DataFrame(per_class)

    top5 = [str(gt) in ids.split() for gt, ids in zip(pred["action_id"], pred["top5_ids"].fillna(""))]
    with_videos = per_class_df[per_class_df["n"] > 0]
    metrics = {
        "n_videos": len(pred), "n_failed": int(failed.sum()), "n_classes": n_classes,
        "accuracy": _accuracy_block(correct, masks, boot),
        "top5_overall": float(np.mean(top5)) if len(pred) else None,
        "mean_class_accuracy": float(with_videos["acc"].mean()) if len(with_videos) else None,
        "ci_method": {"type": "cluster percentile bootstrap", "unit": "source_group_id",
                      "n_groups": int(len(np.unique(groups))), "n_boot": n_boot, "seed": seed,
                      "group_bases": sorted(pred.get("source_group_basis", pd.Series(dtype=str)).unique())},
        "confused_pairs": confused_pairs(confusion_df),
    }
    if near:
        n = int(strict.sum())
        metrics["secondary_unfamiliar_excluding_near_familiar"] = {
            "n": n, "acc": float(correct[strict].mean()) if n else None,
            "ci95": interval(boot["strict"]) if n else None, "excluded_classes": sorted(near),
            "note": "sensitivity analysis only; official XOV status is the primary definition; "
                    "exclusions come from an unverified review (data/audit/category_review.csv)",
        }
    return metrics, per_class_df, confusion_df, boot


def confused_pairs(confusion_df: pd.DataFrame, top: int = 10, min_count: int = 2) -> list[dict]:
    names = list(confusion_df.index)
    values = confusion_df[names].to_numpy()
    totals = confusion_df.to_numpy().sum(1)
    pairs = []
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j and values[i, j] >= min_count:
                pairs.append({"true": names[i], "predicted": names[j], "count": int(values[i, j]),
                              "rate_of_true_class": round(values[i, j] / totals[i], 4)})
    return sorted(pairs, key=lambda p: (-p["count"], -p["rate_of_true_class"]))[:top]


def evaluate_run(run_dir, n_boot: int = 1000, seed: int = 0) -> dict:
    """Write metrics.json/per_class.csv/confusion.csv per dataset and summary.json for the run."""
    run_dir = Path(run_dir)
    record = json.loads((run_dir / "run.json").read_text())
    per_dataset, boots, preds = {}, {}, []
    for k, name in enumerate(record["datasets"]):
        pred = pd.read_csv(run_dir / name / "predictions.csv", keep_default_na=False,
                           dtype={"top5_ids": str})
        classes = pd.read_csv(run_dir / name / "classes.csv", keep_default_na=False)
        metrics, per_class, confusion, boot = dataset_metrics(pred, classes, n_boot, seed + k)
        (run_dir / name / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        per_class.to_csv(run_dir / name / "per_class.csv", index=False)
        confusion.to_csv(run_dir / name / "confusion.csv")
        per_dataset[name], boots[name] = metrics, boot
        preds.append(pred)

    summary = {"run_name": record["run_name"], "synthetic": record["synthetic"],
               "comparison": record.get("comparison"), "datasets": list(per_dataset),
               "per_dataset": {n: m["accuracy"] for n, m in per_dataset.items()}}
    if len(per_dataset) > 1:
        avg = {}
        for subset in SUBSETS:
            accs = [m["accuracy"][subset]["acc"] for m in per_dataset.values()]
            if all(a is not None for a in accs):
                stacked = np.vstack([boots[n][subset] for n in per_dataset])
                avg[subset] = {"acc": float(np.mean(accs)), "ci95": interval(stacked.mean(0))}
            else:
                avg[subset] = {"acc": None, "ci95": None, "note": "a domain lacks this subset"}
        pooled_pred = pd.concat(preds, ignore_index=True)
        correct = pooled_pred["correct"].to_numpy().astype(float)
        masks = subset_masks(pooled_pred)
        pooled_boot = cluster_bootstrap(correct, pooled_pred["source_group_id"].to_numpy(), masks,
                                        n_boot, seed + 1000)
        summary["domain_average_unweighted"] = avg
        summary["pooled_over_videos"] = _accuracy_block(correct, masks, pooled_boot)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
