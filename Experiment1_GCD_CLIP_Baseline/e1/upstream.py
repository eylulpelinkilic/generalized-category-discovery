"""Thin adapter to the upstream GCD implementation in this repository.

The upstream modules import tensorboard and read machine-specific paths at import time. This
adapter executes only the named definitions from the upstream source files (the code itself is
unchanged), so the experiment uses the original losses, metric and model definitions without
those side effects. `models.vision_transformer` and `data.augmentations` import cleanly and are
imported normally.
"""
from __future__ import annotations

import ast
import math
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from e1.config import REPO_ROOT

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONTRASTIVE = REPO_ROOT / "methods" / "contrastive_training" / "contrastive_training.py"
METRICS = REPO_ROOT / "project_utils" / "cluster_and_log_utils.py"
CLUSTER_UTILS = REPO_ROOT / "project_utils" / "cluster_utils.py"


def load_definitions(path: Path, names: list[str], namespace: dict | None = None) -> dict:
    """Execute only the top-level functions/classes `names` of an upstream file in `namespace`."""
    tree = ast.parse(Path(path).read_text())
    wanted = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    missing = set(names) - {n.name for n in wanted}
    if missing:
        raise RuntimeError(f"{path} no longer defines {sorted(missing)}")
    ns = {"np": np, "torch": torch, "F": F, "math": math, **(namespace or {})}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(path), "exec"), ns)  # noqa: S102
    return ns


_contrastive = load_definitions(CONTRASTIVE, ["SupConLoss", "ContrastiveLearningViewGenerator", "info_nce_logits"],
                                {"device": torch.device("cpu")})
SupConLoss = _contrastive["SupConLoss"]
ContrastiveLearningViewGenerator = _contrastive["ContrastiveLearningViewGenerator"]


def info_nce_logits(features: torch.Tensor, n_views: int, temperature: float):
    """Upstream `info_nce_logits`; its module-level `device` is set to the features' device."""
    _contrastive["device"] = features.device
    return _contrastive["info_nce_logits"](features, types.SimpleNamespace(n_views=n_views,
                                                                           temperature=temperature))


_cluster = load_definitions(CLUSTER_UTILS, ["cluster_acc"], {"linear_assignment": linear_sum_assignment})
cluster_acc = _cluster["cluster_acc"]
_metrics = load_definitions(METRICS, ["split_cluster_acc_v1", "split_cluster_acc_v2"],
                            {"linear_assignment": linear_sum_assignment, "cluster_acc": cluster_acc})
split_cluster_acc_v2 = _metrics["split_cluster_acc_v2"]


def vision_transformer():
    from models import vision_transformer as vits

    return vits


def get_transform(transform_type: str = "imagenet", image_size: int = 224, interpolation: int = 3,
                  crop_pct: float = 0.875):
    """Upstream `data.augmentations.get_transform` with the DINO settings of the training script."""
    from data.augmentations import get_transform as upstream_get_transform

    args = types.SimpleNamespace(interpolation=interpolation, crop_pct=crop_pct)
    return upstream_get_transform(transform_type, image_size=image_size, args=args)
