"""Configuration and output layout. Runs are organised by dataset, protocol, model, seed and config hash."""
from __future__ import annotations

from pathlib import Path

import yaml

from e1.config import _strip, deep_merge, parse_override, stable_hash

from e2 import EXP2, REPO

CONFIG_DIR = EXP2 / "configs"


def load_config(name_or_path, overrides=()) -> dict:
    """Experiment 1's YAML conventions (`base:` inheritance, a.b=value overrides), resolved in this folder."""
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


def resolve(path_str: str | None, base: Path = EXP2) -> Path | None:
    if path_str is None:
        return None
    p = Path(path_str).expanduser()
    return p if p.is_absolute() else (base / p).resolve()


def chash(*parts) -> str:
    return stable_hash([_strip(p) for p in parts])


class Layout:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.out = resolve(cfg["paths"]["outputs"])
        self.dataset = cfg["dataset"]["name"]
        self.base = self.out / self.dataset

    @property
    def cache(self) -> Path:
        return self.out / "cache"

    @property
    def assets(self) -> Path:
        return self.out / "assets"

    def vocabulary(self) -> Path:
        return self.out / "vocabulary" / f"wordnet{self.cfg['vocabulary']['wordnet_version'].replace('.', '')}_official_" \
            f"{chash(self.cfg['vocabulary'])}"

    def embeddings(self, clip_name: str, identity_hash: str) -> Path:
        return self.base / "inputs" / f"clip_{clip_name}_{identity_hash}"

    def fixed(self, clip_name: str, seed_tag: str) -> Path:
        return self.base / "fixed_cluster" / clip_name / seed_tag / chash(self.cfg["fixed"], self.cfg["e1"])

    def refine_on_gcd(self, clip_name: str, seed_tag: str, mode: str) -> Path:
        return (self.base / "refine_on_gcd" / clip_name / seed_tag /
                f"{mode}_{chash(self.cfg['naming'], self.cfg['refine_on_gcd'], self.cfg['e1'])}")

    def features(self, model: str, identity_hash: str) -> Path:
        return self.base / "features" / f"{model}_{identity_hash}"

    def repro(self, setting: str, model_tag: str, seed: int, *sections) -> Path:
        return self.base / f"repro_{setting}" / model_tag / f"seed{seed}" / chash(*sections)

    @property
    def report_dir(self) -> Path:
        return EXP2 / "reports"


def e1_layout(cfg: dict):
    """Experiment 1's own layout pointed at the Experiment 1 outputs (by reference, never copied)."""
    from e1.config import Layout as E1Layout, load_config as e1_load

    e1c = e1_load(REPO / "Experiment1_GCD_CLIP_Baseline" / "configs" / f"{cfg['e1']['config']}.yaml",
                  [f"paths.runs_root={resolve(cfg['e1']['runs_root'])}"])
    return e1c, E1Layout(e1c)
