"""Cross-check a manifest against the upstream dataset implementation, run on the real files.

ImageNet: the training portion of `data.imagenet.get_imagenet_100_datasets` is replayed with the
upstream functions (`ImageNetBase`, `subsample_classes`, `subsample_dataset`,
`data.data_utils.subsample_instances`); the upstream test-set branch needs class folders under
val/, which the Kaggle copy does not have, and is not needed here. CUB: `get_cub_datasets` is
called as is. Labelled and unlabelled path sets and labels must match exactly.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from e1 import upstream  # noqa: F401  (puts the repository root on sys.path)
from e1.manifests import Manifest, cub_class_split, imagenet100_source_indices, locate_imagenet


def _compare(manifest: Manifest, labelled: dict, unlabelled: dict) -> dict:
    t = manifest.table
    by_path = dict(zip(t["rel_path"], t["sample_id"]))
    ours_l, ours_u = manifest.labels("labelled"), manifest.labels("eval")
    up_l = {by_path.get(p): c for p, c in labelled.items()}
    up_u = {by_path.get(p): c for p, c in unlabelled.items()}
    result = {
        "upstream_labelled": len(up_l), "upstream_unlabelled": len(up_u),
        "paths_unknown_to_manifest": int(None in up_l) + int(None in up_u),
        "labelled_set_equal": set(up_l) == set(ours_l.index),
        "unlabelled_set_equal": set(up_u) == set(ours_u.index),
        "labelled_labels_equal": all(ours_l.get(s) == c for s, c in up_l.items()),
        "unlabelled_labels_equal": all(ours_u.get(s) == c for s, c in up_u.items()),
    }
    result["passed"] = all(v for k, v in result.items() if k.endswith("_equal")) and not result["paths_unknown_to_manifest"]
    return result


def check_imagenet(root, manifest: Manifest) -> dict:
    import data.imagenet as up
    from data.data_utils import subsample_instances

    train_dir = locate_imagenet(root)["train_dir"]
    full = up.ImageNetBase(root=str(train_dir), transform=None)
    selected = imagenet100_source_indices()
    cls_map = {i: j for i, j in zip(selected, range(100))}
    whole = up.subsample_classes(full, include_classes=selected)
    whole.samples = [(s[0], cls_map[s[1]]) for s in whole.samples]
    whole.targets = [s[1] for s in whole.samples]
    whole.uq_idxs = np.array(range(len(whole)))
    whole.target_transform = None
    lab = up.subsample_classes(deepcopy(whole), include_classes=range(50))
    lab = up.subsample_dataset(lab, subsample_instances(lab, prop_indices_to_subsample=0.5))
    unl = up.subsample_dataset(deepcopy(whole), np.array(list(set(whole.uq_idxs) - set(lab.uq_idxs))))
    rel = lambda p: "train/" + str(Path(p).relative_to(train_dir))  # noqa: E731
    return _compare(manifest, {rel(p): int(t) for p, t in lab.samples}, {rel(p): int(t) for p, t in unl.samples})


def check_cub(root, manifest: Manifest) -> dict:
    import data.cub as up

    root = Path(root)
    up.cub_root = str(root.parent if root.name == "CUB_200_2011" else root)
    known, novel = cub_class_split()
    gcd_of = {c: i for i, c in enumerate(known + novel)}
    ds = up.get_cub_datasets(None, None, train_classes=known, prop_train_labels=0.5, split_train_val=False)
    pairs = lambda d: {f"images/{fp}": gcd_of[int(t) - 1] for fp, t in zip(d.data["filepath"], d.data["target"])}  # noqa: E731
    return _compare(manifest, pairs(ds["train_labelled"]), pairs(ds["train_unlabelled"]))
