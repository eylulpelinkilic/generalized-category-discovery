"""Embedding cache. A key is the SHA-256 of every input that can change the array.

Video entries hash: backbone identity (including the checkpoint SHA-256),
sampling, preprocessing, pixel transform, and the video file identity
(dataset, relative path, size, mtime). Editing any of them yields a new key;
stale entries are never reused.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from ovar.config import stable_hash


def file_identity(path) -> dict:
    st = os.stat(path)
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def sha256_file(path, memo_dir=None) -> str:
    """SHA-256 of a (possibly large) file, memoised on (path, size, mtime)."""
    path = Path(path).expanduser().resolve()
    memo = None
    if memo_dir is not None:
        Path(memo_dir).mkdir(parents=True, exist_ok=True)
        memo = Path(memo_dir) / (stable_hash({"path": str(path), **file_identity(path)}) + ".sha256")
        if memo.exists():
            return memo.read_text().strip()
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    if memo is not None:
        memo.write_text(value + "\n")
    return value


class ArrayCache:
    def __init__(self, root):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.npy"

    def get(self, key: str):
        path = self._path(key)
        return np.load(path) if path.exists() else None

    def put(self, key: str, array: np.ndarray, meta: dict) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, array)
        os.replace(tmp, path)
        path.with_suffix(".json").write_text(json.dumps(meta, sort_keys=True, default=str) + "\n")
