"""Extract pure functions from the pinned official repositories (no heavy imports)."""
from __future__ import annotations

import ast
import math
import types
import unittest
from pathlib import Path

import numpy as np
import torch

THIRD_PARTY = Path(__file__).resolve().parents[1] / "third_party"


def require(path: Path) -> Path:
    if not path.exists():
        raise unittest.SkipTest(f"{path} not present (run `python -m ovar fetch-repos`)")
    return path


def functions(path: Path, names: list[str], extra_globals: dict | None = None,
              class_name: str | None = None) -> dict:
    tree = ast.parse(require(path).read_text())
    scope = tree.body
    if class_name:
        scope = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name).body
    wanted = [n for n in scope if isinstance(n, ast.FunctionDef) and n.name in names]
    module = ast.Module(body=wanted, type_ignores=[])
    namespace = {"math": math, "np": np, "torch": torch, "random": __import__("random"), **(extra_globals or {})}
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102 - pinned official source
    return {name: namespace[name] for name in names}


class NumpyIntShim(types.SimpleNamespace):
    """mmaction code uses the removed alias `np.int`."""

    def __getattr__(self, name):
        if name == "int":
            return int
        return getattr(np, name)
