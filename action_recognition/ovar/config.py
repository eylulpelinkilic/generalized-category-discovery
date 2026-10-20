"""YAML configuration: `model`/`protocol` composition, `${...}` interpolation, stable hashing."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"

_REF = re.compile(r"\$\{([^}]+)\}")


def load_yaml(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def project_path(path) -> Path:
    """Expand `~` and resolve relative paths against the project root."""
    p = Path(str(path)).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_paths(paths_file=None) -> dict:
    """Machine-specific paths. Falls back to the example file (placeholders only)."""
    candidate = Path(paths_file) if paths_file else CONFIG_DIR / "paths.yaml"
    if not candidate.exists():
        candidate = CONFIG_DIR / "paths.example.yaml"
    paths = load_yaml(candidate)
    paths["_source"] = str(candidate)
    return paths


def _lookup(cfg: dict, dotted: str):
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"unresolved config reference ${{{dotted}}}")
        node = node[part]
    return node


def interpolate(node, root):
    if isinstance(node, dict):
        return {k: interpolate(v, root) for k, v in node.items()}
    if isinstance(node, list):
        return [interpolate(v, root) for v in node]
    if isinstance(node, str):
        whole = _REF.fullmatch(node)
        if whole:
            return _lookup(root, whole.group(1))
        return _REF.sub(lambda m: str(_lookup(root, m.group(1))), node)
    return node


def _component(kind: str, value) -> dict:
    if isinstance(value, str):
        loaded = load_yaml(CONFIG_DIR / kind / f"{value}.yaml")
        loaded.setdefault("name", value)
        return loaded
    return dict(value)


def parse_override(text: str) -> tuple[list[str], object]:
    key, _, raw = text.partition("=")
    if not key or not _:
        raise ValueError(f"override must look like a.b=value, got {text!r}")
    return key.split("."), yaml.safe_load(raw)


def load_experiment(path, overrides=(), paths_file=None) -> dict:
    """Load an experiment file and resolve its model/protocol references."""
    path = Path(path)
    if not path.exists():
        path = CONFIG_DIR / "experiments" / f"{path}.yaml"
    cfg = load_yaml(path)
    cfg["model"] = _component("models", cfg["model"])
    cfg["protocol"] = _component("protocols", cfg["protocol"])
    for text in overrides:
        keys, value = parse_override(text)
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    cfg["paths"] = load_paths(paths_file)
    cfg["experiment_file"] = str(path)
    return interpolate(cfg, cfg)


def stable_hash(obj) -> str:
    """SHA-256 of canonical JSON; used for cache keys and configuration ids."""
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()
