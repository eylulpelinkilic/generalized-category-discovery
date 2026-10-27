"""Manifest loading and validation. Problems are reported, never silently fixed."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ovar.benchmark import (
    MANIFEST_COLUMNS, STATUSES, VOCAB_COLUMNS, label_matches_path, manifest_path,
    read_split_list, vocab_path,
)
from ovar.config import project_path
from ovar.video import VideoDecodeError, probe


def load_manifest(dataset: str, split: str, path=None) -> pd.DataFrame:
    df = pd.read_csv(path or manifest_path(dataset, split), dtype=str, keep_default_na=False)
    df["action_id"] = df["action_id"].astype(int)
    return df


def load_vocab(dataset: str, path=None) -> pd.DataFrame:
    df = pd.read_csv(path or vocab_path(dataset), dtype=str, keep_default_na=False)
    df["action_id"] = df["action_id"].astype(int)
    df["n_videos"] = df["n_videos"].astype(int)
    return df.sort_values("action_id").reset_index(drop=True)


def on_disk_path(dataset: str, rel_path: str, paths: dict) -> str:
    """Manifest path -> path below the dataset root, for the configured folder layout."""
    parts = rel_path.split("/")
    if dataset == "hmdb51" and paths.get("hmdb51_single_class_folder") and len(parts) >= 3 \
            and parts[0] == parts[1]:
        return "/".join(parts[1:])
    return rel_path


def resolve_video_path(dataset: str, rel_path: str, paths: dict) -> Path:
    root = (paths.get("dataset_roots") or {}).get(dataset)
    if not root:
        raise KeyError(f"no dataset_roots.{dataset} in {paths.get('_source', 'paths config')}")
    return project_path(root) / on_disk_path(dataset, rel_path, paths)


def validate(manifest: pd.DataFrame, vocab: pd.DataFrame, *, expected: dict | None = None,
             official_lists: dict | None = None, paths: dict | None = None,
             check_files: bool = False, check_decode: bool = False,
             decode_limit: int | None = None) -> dict:
    errors, warnings, failures = [], [], []

    def fail(row, check, detail):
        failures.append({"video_id": row["video_id"], "video_path": row["video_path"],
                         "check": check, "detail": detail})

    missing_cols = [c for c in MANIFEST_COLUMNS if c not in manifest.columns]
    missing_vocab = [c for c in VOCAB_COLUMNS if c not in vocab.columns]
    if missing_cols or missing_vocab:
        return {"errors": [f"missing manifest columns {missing_cols}, vocab columns {missing_vocab}"],
                "warnings": [], "failures": []}

    dataset_names = sorted(manifest["dataset_name"].unique())
    if len(dataset_names) != 1:
        errors.append(f"expected one dataset_name, found {dataset_names}")
    dataset = dataset_names[0] if dataset_names else "?"

    for col in MANIFEST_COLUMNS:
        empty = int((manifest[col].astype(str).str.strip() == "").sum())
        if empty:
            errors.append(f"{empty} rows with empty {col}")

    for col in ("video_id", "video_path"):
        dup = manifest[manifest[col].duplicated(keep=False)]
        if len(dup):
            errors.append(f"{dup[col].nunique()} duplicated {col} values ({len(dup)} rows)")

    # Candidate vocabulary: contiguous ids, because score columns are indexed by id.
    ids = vocab["action_id"].tolist()
    if ids != list(range(len(ids))):
        errors.append("vocabulary ids are not contiguous 0..C-1")
    if vocab["text_name"].str.strip().eq("").any():
        errors.append("vocabulary has empty text names")
    names = dict(zip(vocab["action_id"], vocab["official_name"]))
    vocab_status = dict(zip(vocab["action_id"], vocab["category_status"]))

    bad_label = 0
    for _, row in manifest.iterrows():
        aid = row["action_id"]
        if aid not in names:
            bad_label += 1
            fail(row, "label_mapping", f"action_id {aid} not in vocabulary")
        elif row["action_name"] != names[aid]:
            bad_label += 1
            fail(row, "label_mapping", f"action_name {row['action_name']!r} != {names[aid]!r}")
        elif label_matches_path(dataset, row["video_path"], aid, names[aid]) is False:
            bad_label += 1
            fail(row, "label_mapping", "class encoded in the path disagrees with action_id")
    if bad_label:
        errors.append(f"{bad_label} rows with inconsistent labels")

    statuses = set(manifest["category_status"])
    if not statuses <= set(STATUSES):
        errors.append(f"unknown category_status values {sorted(statuses - set(STATUSES))}")
    per_class = manifest.groupby("action_id")["category_status"].nunique()
    if (per_class > 1).any():
        errors.append(f"classes with mixed status: {per_class[per_class > 1].index.tolist()}")
    mismatch = [a for a, s in manifest.groupby("action_id")["category_status"].first().items()
                if vocab_status.get(a) != s]
    if mismatch:
        errors.append(f"manifest status disagrees with vocabulary for classes {mismatch}")

    used = set(manifest["action_id"])
    empty_classes = [names[i] for i in ids if i not in used]
    if empty_classes:
        warnings.append(f"{len(empty_classes)} vocabulary classes have no videos: {empty_classes}")

    class_status = manifest.groupby("action_id")["category_status"].first()
    counts = {
        "videos": len(manifest),
        "classes_in_vocabulary": len(vocab),
        "classes_with_videos": len(used),
        "familiar_classes": int((class_status == "familiar").sum()),
        "unfamiliar_classes": int((class_status == "unfamiliar").sum()),
        "familiar_videos": int((manifest["category_status"] == "familiar").sum()),
        "unfamiliar_videos": int((manifest["category_status"] == "unfamiliar").sum()),
        "source_groups": int(manifest["source_group_id"].nunique()),
        "source_group_bases": sorted(manifest["source_group_basis"].unique()),
    }
    for key, value in (expected or {}).items():
        actual = counts["classes_in_vocabulary"] if key == "classes" else counts.get(key)
        if actual != value:
            errors.append(f"{key}: expected {value}, found {actual}")

    if official_lists:
        listed = {rel: label for rel, label in read_split_list(official_lists["all"])}
        ours = dict(zip(manifest["video_path"], manifest["action_id"]))
        if set(listed) != set(ours):
            errors.append(f"split membership differs from official list: "
                          f"{len(set(listed) - set(ours))} missing, {len(set(ours) - set(listed))} extra")
        relabelled = [p for p in set(listed) & set(ours) if listed[p] != ours[p]]
        if relabelled:
            errors.append(f"{len(relabelled)} videos carry a different label than the official list")
        for key, status in (("closed", "familiar"), ("open", "unfamiliar")):
            official = {rel for rel, _ in read_split_list(official_lists[key])}
            ours_status = set(manifest.loc[manifest["category_status"] == status, "video_path"])
            if official != ours_status:
                errors.append(f"{status} videos differ from the official {key} list")

    files = {"checked": 0, "missing": 0, "decode_checked": 0, "decode_failed": 0}
    if check_files or check_decode:
        try:
            resolved = [resolve_video_path(dataset, p, paths or {}) for p in manifest["video_path"]]
        except KeyError as exc:
            errors.append(str(exc))
            resolved = []
        for (_, row), path in zip(manifest.iterrows(), resolved):
            files["checked"] += 1
            if not path.is_file():
                files["missing"] += 1
                fail(row, "file_missing", str(path))
                continue
            if check_decode and (decode_limit is None or files["decode_checked"] < decode_limit):
                files["decode_checked"] += 1
                try:
                    probe(path)
                except VideoDecodeError as exc:
                    files["decode_failed"] += 1
                    fail(row, "decode", str(exc))
        if files["missing"]:
            errors.append(f"{files['missing']} of {files['checked']} videos are missing")
        if files["decode_failed"]:
            errors.append(f"{files['decode_failed']} of {files['decode_checked']} videos failed to decode")

    return {"dataset": dataset, "errors": errors, "warnings": warnings, "counts": counts,
            "files": files, "failures": failures}
