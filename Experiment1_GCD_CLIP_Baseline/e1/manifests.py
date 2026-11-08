"""Immutable sample manifests and permission-separated label tables.

`manifest.csv` has no labels. Labels live in three tables whose names state who may read them
(PROTOCOL.md §1). Partitions reproduce the upstream GCD rules (`data/imagenet.py`, `data/cub.py`,
`data/data_utils.subsample_instances`) and are cross-checked against those functions when the
full data are available.
"""
from __future__ import annotations

import io
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from e1.config import REPO_ROOT
from e1.provenance import sha256_bytes, write_immutable, write_json

MANIFEST_COLUMNS = ["sample_id", "rel_path", "source_split", "gcd_membership", "dataset_version"]
MEMBERSHIPS = ("labelled", "unlabelled", "heldout_dev", "heldout_calib")
CLASS_COLUMNS = ["class_id", "class_key", "source_index", "canonical_name", "aliases", "gcd_status",
                 "name_source"]


# ---------------------------------------------------------------- upstream partition rules
def imagenet100_source_indices(seed: int = 0) -> np.ndarray:
    """`get_imagenet_100_datasets`: seed, choose 100 of 1000 ImageFolder indices, sort."""
    np.random.seed(seed)
    return np.sort(np.random.choice(range(1000), size=(100,), replace=False))


def upstream_labelled_subsample(n_rows: int, prop: float = 0.5) -> np.ndarray:
    """`data.data_utils.subsample_instances`: always reseeds 0; positions into the known-class rows."""
    np.random.seed(0)
    return np.random.choice(range(n_rows), replace=False, size=(int(prop * n_rows),))


def split_heldout(per_class: dict, seed: int) -> dict:
    """Per known class (in class-id order): shuffle with one seeded generator, first ⌊n/2⌋ → dev."""
    rng = np.random.RandomState(seed)
    out = {}
    for cls in sorted(per_class):
        ids = sorted(per_class[cls])
        order = rng.permutation(len(ids))
        half = len(ids) // 2
        for rank, j in enumerate(order):
            out[ids[j]] = "heldout_dev" if rank < half else "heldout_calib"
    return out


# ---------------------------------------------------------------- ImageNet-100
def _imagefolder_files(class_dir: Path) -> list[str]:
    """Files in `make_dataset` order: sorted os.walk, sorted names, torchvision's extension rule."""
    from torchvision.datasets.folder import IMG_EXTENSIONS, has_file_allowed_extension

    files = []
    for root, _, fnames in sorted(os.walk(class_dir, followlinks=True)):
        for fname in sorted(fnames):
            path = Path(root) / fname
            if has_file_allowed_extension(str(path), IMG_EXTENSIONS):
                files.append(str(path.relative_to(class_dir.parent)))
    return files


def locate_imagenet(root) -> dict:
    """Kaggle competition layout (ILSVRC/Data/CLS-LOC + LOC_*.txt/csv) or an ImageFolder root."""
    root = Path(root)
    found = {}

    def shallow(pattern, max_depth=6):           # never descends into the 1,000 class folders
        return [p for d in range(max_depth + 1) for p in sorted(root.glob("*/" * d + pattern))]

    for cand in [root, *shallow("CLS-LOC")]:
        if (cand / "train").is_dir():
            found["train_dir"], found["data_root"] = cand / "train", cand
            break
    if not found:
        raise FileNotFoundError(f"no train/ directory with ImageNet class folders under {root}")
    for name in ("LOC_synset_mapping.txt", "LOC_val_solution.csv"):
        hits = shallow(name, 4)
        found[name] = hits[0] if hits else None
    found["val_dir"] = found["data_root"] / "val" if (found["data_root"] / "val").is_dir() else None
    return found


def read_synset_mapping(path) -> dict:
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            wnid, names = line.split(" ", 1)
            out[wnid] = [n.strip() for n in names.split(",")]
    return out


def build_imagenet100(root, out_dir, *, heldout_seed: int = 0, verify_images: str = "all",
                      workers: int = 4, version: str = "ILSVRC2012-CLS-LOC") -> dict:
    loc = locate_imagenet(root)
    train_dir = loc["train_dir"]
    wnids = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
    if len(wnids) != 1000:
        raise RuntimeError(f"{train_dir} has {len(wnids)} class folders, expected 1000")
    synsets = read_synset_mapping(loc["LOC_synset_mapping.txt"]) if loc["LOC_synset_mapping.txt"] else None
    if synsets is not None and list(synsets) != wnids:
        raise RuntimeError("LOC_synset_mapping.txt order differs from the sorted train folders")
    selected = imagenet100_source_indices()

    classes, rows = [], []
    for gcd_id, src in enumerate(selected):
        wnid = wnids[src]
        names = synsets[wnid] if synsets else [wnid]
        classes.append({"class_id": gcd_id, "class_key": wnid, "source_index": int(src),
                        "canonical_name": names[0], "aliases": "|".join(names[1:]),
                        "gcd_status": "known" if gcd_id < 50 else "novel",
                        "name_source": "LOC_synset_mapping.txt (WordNet lemmas)" if synsets else "wnid"})
        for rel in _imagefolder_files(train_dir / wnid):
            rows.append({"sample_id": f"imagenet:train:{Path(rel).stem}", "rel_path": f"train/{rel}",
                         "source_split": "train", "class_id": gcd_id})
    train = pd.DataFrame(rows)

    known_rows = np.flatnonzero(train["class_id"].to_numpy() < 50)
    labelled_pos = known_rows[upstream_labelled_subsample(len(known_rows))]
    membership = np.full(len(train), "unlabelled", dtype=object)
    membership[labelled_pos] = "labelled"
    train["gcd_membership"] = membership

    held_all = _imagenet_heldout(loc, wnids, selected)
    held = held_all[held_all["class_id"] < 50].copy()    # novel-class held-out images are not used
    groups = {}
    for sid, cls in zip(held["sample_id"], held["class_id"]):
        groups.setdefault(int(cls), []).append(sid)
    pools = split_heldout(groups, heldout_seed)
    held["gcd_membership"] = held["sample_id"].map(pools)
    table = pd.concat([train, held], ignore_index=True)
    table["dataset_version"] = version
    return _finish(table, pd.DataFrame(classes, columns=CLASS_COLUMNS), out_dir, image_root=loc["data_root"],
                   dataset="imagenet100", verify_images=verify_images, workers=workers,
                   extra={"source": {k: str(v) for k, v in loc.items()},
                          "selected_source_indices": selected.tolist(), "heldout_seed": heldout_seed,
                          "novel_heldout_images_excluded": int(len(held_all) - len(held))})


def _imagenet_heldout(loc: dict, wnids: list[str], selected: np.ndarray) -> pd.DataFrame:
    """Validation images of the 100 classes, from LOC_val_solution.csv (flat val/) or val/<wnid>/."""
    gcd_of = {wnids[s]: i for i, s in enumerate(selected)}
    rows = []
    if loc.get("LOC_val_solution.csv") is not None and loc["val_dir"] is not None:
        sol = pd.read_csv(loc["LOC_val_solution.csv"])
        for image_id, pred in zip(sol["ImageId"], sol["PredictionString"]):
            wnid = pred.split()[0]
            if wnid in gcd_of:
                rows.append({"sample_id": f"imagenet:val:{image_id}", "rel_path": f"val/{image_id}.JPEG",
                             "source_split": "val", "class_id": gcd_of[wnid]})
    elif loc["val_dir"] is not None:
        for wnid, gcd_id in gcd_of.items():
            for rel in _imagefolder_files(loc["val_dir"] / wnid):
                rows.append({"sample_id": f"imagenet:val:{Path(rel).stem}", "rel_path": f"val/{rel}",
                             "source_split": "val", "class_id": gcd_id})
    else:
        raise FileNotFoundError("no ImageNet validation images (val/ with LOC_val_solution.csv or val/<wnid>/)")
    return pd.DataFrame(rows).sort_values("sample_id", kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------- CUB-200-2011
CUB_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
CUB_MD5 = "97eceeb196236b17998738112f37df78"      # data/cub.py (upstream constant)


def cub_class_split() -> tuple[list[int], list[int]]:
    """`get_class_splits` with use_ssb_splits=True: known, then Hard + Medium + Easy."""
    with open(REPO_ROOT / "data" / "ssb_splits" / "cub_osr_splits.pkl", "rb") as f:
        info = pickle.load(f)
    unknown = info["unknown_classes"]
    return list(info["known_classes"]), list(unknown["Hard"]) + list(unknown["Medium"]) + list(unknown["Easy"])


def build_cub(root, out_dir, *, heldout_seed: int = 0, verify_images: str = "all", workers: int = 4,
              version: str = "CUB_200_2011") -> dict:
    root = Path(root)
    base = root / "CUB_200_2011" if (root / "CUB_200_2011").is_dir() else root
    read = lambda name, cols: pd.read_csv(base / name, sep=" ", names=cols)  # noqa: E731
    images = read("images.txt", ["img_id", "filepath"])
    data = images.merge(read("image_class_labels.txt", ["img_id", "target"]), on="img_id")
    data = data.merge(read("train_test_split.txt", ["img_id", "is_training_img"]), on="img_id")
    names = {}
    for line in (base / "classes.txt").read_text().splitlines():
        num, raw = line.split(" ", 1)
        names[int(num) - 1] = raw.split(".", 1)[1].replace("_", " ")
    known, novel = cub_class_split()
    gcd_of = {cls: i for i, cls in enumerate(known + novel)}
    classes = pd.DataFrame([{"class_id": gcd_of[c], "class_key": f"{c + 1:03d}", "source_index": c,
                             "canonical_name": names[c], "aliases": "",
                             "gcd_status": "known" if gcd_of[c] < len(known) else "novel",
                             "name_source": "classes.txt"} for c in known + novel],
                           columns=CLASS_COLUMNS).sort_values("class_id").reset_index(drop=True)
    data["class_id"] = (data["target"] - 1).map(gcd_of)
    data["sample_id"] = "cub:" + data["img_id"].astype(str)
    data["rel_path"] = "images/" + data["filepath"]
    data["source_split"] = np.where(data["is_training_img"] == 1, "train", "test")

    train = data[data["is_training_img"] == 1].reset_index(drop=True)
    known_set = set(known)
    known_rows = np.flatnonzero((train["target"] - 1).isin(known_set).to_numpy())
    labelled_pos = known_rows[upstream_labelled_subsample(len(known_rows))]
    membership = np.full(len(train), "unlabelled", dtype=object)
    membership[labelled_pos] = "labelled"
    train["gcd_membership"] = membership

    held = data[(data["is_training_img"] == 0) & (data["class_id"] < len(known))].copy()
    groups = {}
    for sid, cls in zip(held["sample_id"], held["class_id"]):
        groups.setdefault(int(cls), []).append(sid)
    held["gcd_membership"] = held["sample_id"].map(split_heldout(groups, heldout_seed))
    cols = ["sample_id", "rel_path", "source_split", "class_id", "gcd_membership"]
    table = pd.concat([train[cols], held[cols]], ignore_index=True)
    table["dataset_version"] = version
    excluded = int(((data["is_training_img"] == 0) & (data["class_id"] >= len(known))).sum())
    return _finish(table, classes, out_dir, image_root=base, dataset="cub", verify_images=verify_images,
                   workers=workers, extra={"source": str(base), "heldout_seed": heldout_seed,
                                           "novel_heldout_images_excluded": excluded})


# ---------------------------------------------------------------- shared checks and writing
def _check_image(path: str) -> str | None:
    from PIL import Image

    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.convert("RGB").load()
        return None
    except Exception as exc:  # noqa: BLE001 - reported, never ignored
        return f"{type(exc).__name__}: {exc}"[:200]


def verify_files(table: pd.DataFrame, image_root: Path, mode: str = "all", workers: int = 4) -> dict:
    paths = [str(Path(image_root) / p) for p in table["rel_path"]]
    missing = [p for p in paths if not os.path.isfile(p)]
    report = {"missing": len(missing), "missing_examples": missing[:20]}
    if mode == "none" or missing:
        report["image_check"] = "not run" if mode == "none" else "skipped: files missing"
        return report
    todo = paths if mode == "all" else paths[:: max(1, len(paths) // int(mode.split(":")[1]))]
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            errors = list(pool.map(_check_image, todo, chunksize=256))
    else:
        errors = [_check_image(p) for p in todo]
    bad = [(p, e) for p, e in zip(todo, errors) if e]
    report.update(image_check=f"{len(todo)} images decoded", corrupt=len(bad), corrupt_examples=bad[:20])
    return report


def _finish(table: pd.DataFrame, classes: pd.DataFrame, out_dir, *, image_root, dataset: str,
            verify_images: str, workers: int, extra: dict) -> dict:
    out_dir = Path(out_dir)
    checks = {
        "duplicate_sample_ids": int(table["sample_id"].duplicated().sum()),
        "duplicate_paths": int(table["rel_path"].duplicated().sum()),
        "membership_values": sorted(table["gcd_membership"].dropna().unique().tolist()),
        "unassigned_heldout": int(table["gcd_membership"].isna().sum()),
        "labelled_rows_in_novel_classes": int(((table["gcd_membership"] == "labelled")
                                                & (table["class_id"].map(classes.set_index("class_id")["gcd_status"])
                                                   != "known")).sum()),
        "heldout_rows_from_train_split": int((table["gcd_membership"].str.startswith("heldout")
                                              & (table["source_split"] == "train")).sum()),
        "classes": int(len(classes)), "class_ids_contiguous": classes["class_id"].tolist() == list(range(len(classes))),
    }
    checks["files"] = verify_files(table, Path(image_root), verify_images, workers)
    problems = [k for k in ("duplicate_sample_ids", "duplicate_paths", "unassigned_heldout",
                            "labelled_rows_in_novel_classes", "heldout_rows_from_train_split") if checks[k]]
    problems += ["missing files"] if checks["files"]["missing"] else []
    problems += ["corrupt images"] if checks["files"].get("corrupt") else []
    problems += [] if checks["class_ids_contiguous"] else ["class ids not contiguous"]
    if problems:
        write_json(out_dir / "manifest_checks_FAILED.json", checks)
        raise RuntimeError(f"{dataset} manifest checks failed: {problems} (details in manifest_checks_FAILED.json)")

    status = table["class_id"].map(classes.set_index("class_id")["gcd_status"])
    m = table["gcd_membership"]
    counts = {"total_rows": len(table),
              **{k: int((m == k).sum()) for k in MEMBERSHIPS},
              "unlabelled_known": int(((m == "unlabelled") & (status == "known")).sum()),
              "unlabelled_novel": int(((m == "unlabelled") & (status == "novel")).sum()),
              "per_class_train": table[table["source_split"] == "train"].groupby("class_id").size().describe()
              .round(2).to_dict()}

    manifest = table[MANIFEST_COLUMNS].to_csv(index=False)
    write_immutable(out_dir / "manifest.csv", manifest)
    write_immutable(out_dir / "classes.csv", classes.to_csv(index=False))
    tables = {"labels_labelled.csv": m == "labelled",
              "labels_heldout.csv": m.isin(["heldout_dev", "heldout_calib"]),
              "labels_eval.csv": m == "unlabelled"}
    for name, mask in tables.items():
        write_immutable(out_dir / name, table.loc[mask, ["sample_id", "class_id"]].to_csv(index=False))
    meta = {"dataset": dataset, "image_root": str(image_root), "counts": counts, "checks": checks,
            "manifest_sha256": sha256_bytes(manifest.encode()),
            "label_permissions": {"labels_labelled.csv": "GCD training, SS-k-means anchors, K estimation",
                                  "labels_heldout.csv": "grounding development (dev) and calibration (calib)",
                                  "labels_eval.csv": "evaluation only"},
            **extra}
    write_json(out_dir / "manifest_meta.json", meta)
    return meta


# ---------------------------------------------------------------- loading
class Manifest:
    """Read-only view of a prepared dataset directory. Labels are loaded only on request."""

    def __init__(self, data_dir):
        self.dir = Path(data_dir)
        self.table = pd.read_csv(self.dir / "manifest.csv", keep_default_na=False)
        self.meta = json.loads((self.dir / "manifest_meta.json").read_text())
        self.classes = pd.read_csv(self.dir / "classes.csv", keep_default_na=False)
        # Mount points differ between machines/sessions; the relative paths do not.
        self.image_root = Path(os.environ.get("E1_IMAGE_ROOT") or self.meta["image_root"])
        self.sha256 = self.meta["manifest_sha256"]

    def ids(self, *memberships: str) -> list[str]:
        return self.table.loc[self.table["gcd_membership"].isin(memberships), "sample_id"].tolist()

    def paths(self, sample_ids) -> list[Path]:
        lookup = dict(zip(self.table["sample_id"], self.table["rel_path"]))
        return [self.image_root / lookup[s] for s in sample_ids]

    def labels(self, permission: str) -> pd.Series:
        """`labelled`, `heldout` or `eval`; returns sample_id → class_id."""
        name = {"labelled": "labels_labelled.csv", "heldout": "labels_heldout.csv", "eval": "labels_eval.csv"}[permission]
        t = pd.read_csv(self.dir / name)
        return pd.Series(t["class_id"].to_numpy(), index=t["sample_id"].to_numpy())

    @property
    def known_class_ids(self) -> np.ndarray:
        return self.classes.loc[self.classes["gcd_status"] == "known", "class_id"].to_numpy()


def write_image_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def synthetic_image(rng: np.random.RandomState, colour: tuple, size: int = 48) -> bytes:
    """Small PNG with a class colour plus noise (smoke data only)."""
    from PIL import Image

    arr = np.clip(np.array(colour)[None, None, :] + rng.randint(-40, 40, (size, size, 3)), 0, 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def build_smoke(out_dir, image_root, *, n_classes: int = 6, train_per_class: int = 12, heldout_per_class: int = 6,
                heldout_seed: int = 0) -> dict:
    """Synthetic dataset with the real schema and the real partition rules (software checks only)."""
    image_root = Path(image_root)
    palette = [("red", (220, 30, 30)), ("green", (30, 200, 40)), ("blue", (30, 40, 220)),
               ("yellow", (230, 220, 30)), ("purple", (150, 40, 180)), ("orange", (240, 140, 20)),
               ("cyan", (30, 210, 210)), ("grey", (128, 128, 128))][:n_classes]
    n_known = n_classes // 2
    rng = np.random.RandomState(0)
    classes, rows = [], []
    for cid, (name, colour) in enumerate(palette):
        classes.append({"class_id": cid, "class_key": name, "source_index": cid, "canonical_name": f"{name} square",
                        "aliases": f"{name} patch", "gcd_status": "known" if cid < n_known else "novel",
                        "name_source": "synthetic"})
        for split, n in (("train", train_per_class), ("test", heldout_per_class)):
            for j in range(n):
                rel = f"{split}/{name}/{name}_{j:03d}.png"
                write_image_bytes(image_root / rel, synthetic_image(rng, colour))
                rows.append({"sample_id": f"smoke:{split}:{name}_{j:03d}", "rel_path": rel, "source_split": split,
                             "class_id": cid})
    data = pd.DataFrame(rows)
    train = data[data["source_split"] == "train"].reset_index(drop=True)
    known_rows = np.flatnonzero(train["class_id"].to_numpy() < n_known)
    membership = np.full(len(train), "unlabelled", dtype=object)
    membership[known_rows[upstream_labelled_subsample(len(known_rows))]] = "labelled"
    train["gcd_membership"] = membership
    held = data[(data["source_split"] == "test") & (data["class_id"] < n_known)].copy()
    groups = {}
    for sid, cls in zip(held["sample_id"], held["class_id"]):
        groups.setdefault(int(cls), []).append(sid)
    held["gcd_membership"] = held["sample_id"].map(split_heldout(groups, heldout_seed))
    table = pd.concat([train, held], ignore_index=True)
    table["dataset_version"] = "synthetic-smoke-v1"
    return _finish(table, pd.DataFrame(classes, columns=CLASS_COLUMNS), out_dir, image_root=image_root,
                   dataset="smoke", verify_images="all", workers=1,
                   extra={"source": "synthetic", "SYNTHETIC": "software checks only, never research results"})
