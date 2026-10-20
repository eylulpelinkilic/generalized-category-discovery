"""Build manifests and candidate vocabularies from the official XOV-Action benchmark files.

Labels are always read through XOV's own label files: Open-VCLIP's HMDB51 index
order differs from XOV's for ten classes (climb/climb stairs, kick/kick ball,
push/pushup, sit/situp, sword/sword exercise).
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from ovar.config import CONFIG_DIR, PROJECT_ROOT, load_yaml, project_path

MANIFEST_COLUMNS = [
    "dataset_name", "video_id", "source_group_id", "source_group_basis", "video_path",
    "action_id", "action_name", "category_status", "split", "official_list",
]
VOCAB_COLUMNS = [
    "action_id", "official_name", "text_name", "text_name_source", "category_status", "n_videos",
]
STATUSES = ("familiar", "unfamiliar")
DATA_DIR = PROJECT_ROOT / "data"


def norm_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def sha256_text_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_labels(path) -> dict[int, str]:
    with open(path, newline="") as f:
        return {int(r["id"]): r["name"] for r in csv.DictReader(f)}


def read_split_list(path) -> list[tuple[str, int]]:
    """Lines of `<relative/path> <label>`; paths contain no spaces in these lists."""
    entries = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            rel, label = line.rsplit(maxsplit=1)
            entries.append((rel, int(label)))
    return entries


# ---------------------------------------------------------------------------
# Dataset-specific filename conventions
# ---------------------------------------------------------------------------

def nec_action_id(rel_path: str) -> int:
    """NEC-Drone codes: A0001-A0010 -> 0-9 (single person), B0001-B0006 -> 10-15."""
    m = re.search(r"_([AB])(\d{4})_", Path(rel_path).name)
    if not m:
        raise ValueError(f"no NEC-Drone action code in {rel_path!r}")
    return int(m.group(2)) - 1 + (0 if m.group(1) == "A" else 10)


def label_matches_path(dataset: str, rel_path: str, action_id: int, action_name: str) -> bool | None:
    """Whether the class encoded in the file path agrees with the label; None if no convention."""
    if dataset == "necdrone":
        return nec_action_id(rel_path) == action_id
    if dataset in ("ucf101", "hmdb51", "arid"):
        return norm_name(rel_path.split("/")[0]) == norm_name(action_name)
    return None


def source_group(dataset: str, rel_path: str) -> tuple[str, str]:
    """Group of clips cut from the same source recording (for clustered bootstrap).

    Returns (group_id, basis). Bases ending in `_assumed_*` are inferred from file
    naming and are not documented by the dataset authors; they are coarse on
    purpose, which can only widen confidence intervals.
    """
    name = Path(rel_path).name
    if dataset == "ucf101":
        m = re.match(r"v_(?P<cls>[A-Za-z]+)_g(?P<g>\d+)_c\d+\.avi$", name)
        if m:
            return f"ucf101:{m['cls']}_g{m['g']}", "ucf101_class_group"
    elif dataset == "hmdb51":
        folder = rel_path.split("/")[0]
        stem = Path(rel_path).stem
        # <title>_<action>_<body part>_<camera motion>_np<n>_<viewpoint>_<quality>_<clip>
        m = re.match(rf"^(?P<title>.+)_{re.escape(folder)}_[a-z]+_[a-z]+_np\d+_[a-z]+_[a-z]+_\d+$", stem)
        if m:
            return f"hmdb51:{m['title']}", "hmdb51_source_title"
        return f"hmdb51:{stem}", "hmdb51_unparsed_unique"
    elif dataset == "arid":
        m = re.match(r"^(?P<a>[^/]+)/(?P=a)_(?P<s>\d+)_\d+\.mp4$", rel_path)
        if m:
            return f"arid:s{m['s']}", "arid_field2_assumed_session"
    elif dataset == "necdrone":
        m = re.match(r"^NEC-(?P<d>Phantom-\d+)_[AB]\d{4}_(?:P\d{4})+_(?P<v>V\d{4})\.mp4$", name)
        if m:
            return f"necdrone:{m['d']}_{m['v']}", "necdrone_drone_and_v_assumed_session"
    return f"{dataset}:{rel_path}", "unparsed_unique"


def video_id(rel_path: str) -> str:
    return str(Path(rel_path).with_suffix(""))


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def load_benchmark(path=None) -> dict:
    return load_yaml(path or CONFIG_DIR / "benchmark" / "xovabench.yaml")


def _text_names(spec: dict, labels: dict[int, str]) -> tuple[dict[int, str], str]:
    if not spec.get("text_names"):
        return dict(labels), "xov_label_file"
    mapping = {int(k): v for k, v in json.loads(project_path(spec["text_names"]).read_text()).items()}
    bad = [i for i in labels if norm_name(mapping.get(i, "")) != norm_name(labels[i])]
    if sorted(mapping) != sorted(labels) or bad:
        raise ValueError(f"text-name file does not align with XOV label ids: {bad[:5]}")
    return mapping, Path(spec["text_names"]).name


def build_dataset(name: str, spec: dict, xov_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    labels = read_labels(xov_root / spec["label_file"])
    lists = {key: read_split_list(xov_root / rel) for key, rel in spec["lists"].items()}
    status_of_path = {}
    for key, status in (("closed", "familiar"), ("open", "unfamiliar")):
        for rel, _ in lists[key]:
            if rel in status_of_path:
                raise ValueError(f"{name}: {rel} is in both closed and open lists")
            status_of_path[rel] = status
    if set(status_of_path) != {rel for rel, _ in lists["all"]}:
        raise ValueError(f"{name}: closed + open lists do not equal the full list")

    rows, class_status = [], {}
    for rel, action_id in lists["all"]:
        status = status_of_path[rel]
        if class_status.setdefault(action_id, status) != status:
            raise ValueError(f"{name}: class {action_id} appears in both closed and open lists")
        group, basis = source_group(name, rel)
        rows.append({
            "dataset_name": name, "video_id": video_id(rel), "source_group_id": group,
            "source_group_basis": basis, "video_path": rel, "action_id": action_id,
            "action_name": labels[action_id], "category_status": status,
            "split": spec["split"], "official_list": spec["lists"]["all"],
        })
    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)

    texts, text_source = _text_names(spec, labels)
    counts = manifest["action_id"].value_counts()
    vocab = pd.DataFrame([{
        "action_id": i, "official_name": labels[i], "text_name": texts[i],
        "text_name_source": text_source, "category_status": class_status.get(i, ""),
        "n_videos": int(counts.get(i, 0)),
    } for i in sorted(labels)], columns=VOCAB_COLUMNS)

    meta = {
        "dataset_name": name,
        "source": "XOV-Action official benchmark files",
        "sources": {
            str(Path(spec["label_file"])): sha256_text_file(xov_root / spec["label_file"]),
            **{str(Path(rel)): sha256_text_file(xov_root / rel) for rel in spec["lists"].values()},
        },
        "n_videos": len(manifest),
        "category_status_definition": "familiar = XOV closed-set (shares meaning with a Kinetics400 "
                                      "training category); unfamiliar = XOV open-set",
    }
    return manifest, vocab, meta


def build_ucf_dev_pool(spec: dict, vocab: pd.DataFrame) -> pd.DataFrame:
    """UCF101 official trainlist01 (1-based labels, same class order as XOV ids)."""
    names = dict(zip(vocab["action_id"], vocab["official_name"]))
    status = dict(zip(vocab["action_id"], vocab["category_status"]))
    rows = []
    for rel, one_based in read_split_list(project_path(spec["list"])):
        action_id = one_based - 1
        if not label_matches_path("ucf101", rel, action_id, names[action_id]):
            raise ValueError(f"UCF101 trainlist label/path mismatch: {rel} {one_based}")
        group, basis = source_group("ucf101", rel)
        rows.append({
            "dataset_name": "ucf101", "video_id": video_id(rel), "source_group_id": group,
            "source_group_basis": basis, "video_path": rel, "action_id": action_id,
            "action_name": names[action_id], "category_status": status[action_id],
            "split": spec["split"], "official_list": spec["list"],
        })
    if len(rows) != spec["expected_videos"]:
        raise ValueError(f"UCF101 dev pool has {len(rows)} videos, expected {spec['expected_videos']}")
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def manifest_path(dataset: str, split: str) -> Path:
    return DATA_DIR / "manifests" / f"{dataset}_{split}.csv"


def vocab_path(dataset: str) -> Path:
    return DATA_DIR / "vocab" / f"{dataset}.csv"


def build_all(bench_path=None) -> dict:
    bench = load_benchmark(bench_path)
    xov_root = project_path(bench["xov_repo"])
    (DATA_DIR / "manifests").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "vocab").mkdir(parents=True, exist_ok=True)
    summary = {}
    vocabs = {}
    for name, spec in bench["datasets"].items():
        manifest, vocab, meta = build_dataset(name, spec, xov_root)
        meta["xov_commit"] = bench["xov_commit"]
        out = manifest_path(name, spec["split"])
        manifest.to_csv(out, index=False)
        out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        vocab.to_csv(vocab_path(name), index=False)
        vocabs[name] = vocab
        summary[name] = {"manifest": str(out.relative_to(PROJECT_ROOT)), "videos": len(manifest),
                         "classes": len(vocab)}

    for name, spec in bench.get("dev_pools", {}).items():
        if spec["format"] != "ucf_trainlist":
            raise ValueError(f"unsupported dev-pool format {spec['format']}")
        dev = build_ucf_dev_pool(spec, vocabs[name])
        out = manifest_path(name, spec["split"])
        dev.to_csv(out, index=False)
        out.with_suffix(".meta.json").write_text(json.dumps({
            "dataset_name": name, "source": "official UCF101 trainlist01.txt (copy in Open-VCLIP repo)",
            "sources": {spec["list"]: sha256_text_file(project_path(spec["list"]))},
            "n_videos": len(dev),
            "note": "Development pool for diagnostics; disjoint from the XOV evaluation list.",
        }, indent=2) + "\n")
        summary[f"{name}_dev"] = {"manifest": str(out.relative_to(PROJECT_ROOT)), "videos": len(dev)}

    k400 = read_labels(xov_root / bench["source_categories"]["label_file"])
    pd.DataFrame({"action_id": list(k400), "official_name": list(k400.values())}).to_csv(
        vocab_path("kinetics400"), index=False)
    summary["kinetics400_categories"] = len(k400)
    return summary
