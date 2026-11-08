"""Configuration loading, command-line overrides, hashing and run-directory layout."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

EXP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXP_ROOT.parent
CONFIG_DIR = EXP_ROOT / "configs"


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def parse_override(text: str) -> tuple[list[str], object]:
    if "=" not in text:
        raise ValueError(f"override must look like a.b=value, got {text!r}")
    key, value = text.split("=", 1)
    return key.split("."), yaml.safe_load(value)


def load_config(name_or_path, overrides=()) -> dict:
    """Load a YAML config (with optional `base:` inheritance) and apply `a.b=value` overrides."""
    path = Path(name_or_path)
    if not path.exists():
        path = CONFIG_DIR / f"{name_or_path}.yaml"
    cfg = yaml.safe_load(path.read_text())
    if "base" in cfg:
        cfg = deep_merge(load_config(cfg.pop("base")), cfg)
    for text in overrides:
        keys, value = parse_override(text)
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    cfg["_config_file"] = str(path)
    return cfg


def stable_hash(obj, length: int = 12) -> str:
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:length]


# Keys that locate files or tune loaders in one environment; they never change a result, and
# excluding them keeps run ids stable across machines and Kaggle sessions.
HASH_EXCLUDE = {"root", "image_root", "dino_checkpoint", "num_workers", "workers", "feature_batch_size",
                "verify_images", "batch_size_text"}


def _strip(node):
    if isinstance(node, dict):
        return {k: _strip(v) for k, v in node.items() if k not in HASH_EXCLUDE}
    return node


def section_hash(cfg: dict, *sections: str) -> str:
    """Hash of the config sections that determine one stage's output."""
    return stable_hash({s: _strip(cfg.get(s)) for s in sections})


class Layout:
    """Run-specific directories. Names carry seeds and the hash of the config that produced them."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.root = Path(cfg["paths"]["runs_root"]).expanduser()
        if not self.root.is_absolute():
            self.root = EXP_ROOT / self.root
        self.dataset = cfg["dataset"]["name"]
        self.base = self.root / self.dataset

    @property
    def data(self) -> Path:
        return self.base / "data"

    def gcd_run(self, train_seed: int) -> Path:
        h = section_hash(self.cfg, "dataset", "gcd_train")
        return self.base / "gcd" / f"seed{train_seed}_{h}"

    def clustering(self, train_seed: int, k_mode: str, cluster_seed: int) -> Path:
        h = section_hash(self.cfg, "clustering")
        return self.gcd_run(train_seed) / "clustering" / f"{k_mode}_c{cluster_seed}_{h}"

    def clip(self, clip_name: str) -> Path:
        h = stable_hash({"dataset": _strip(self.cfg["dataset"]), "clip": self.cfg["clip"]["models"][clip_name]})
        return self.base / "clip" / f"{clip_name}_{h}"

    def grounding(self, train_seed: int, k_mode: str, cluster_seed: int, clip_name: str) -> Path:
        h = section_hash(self.cfg, "vocabulary", "episodes", "grounding")
        return (self.base / "grounding" / f"seed{train_seed}_{k_mode}_c{cluster_seed}" /
                f"{clip_name}_{h}")

    @property
    def report(self) -> Path:
        return self.base / "report"
