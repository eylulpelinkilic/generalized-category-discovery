"""Environment, code version, checksums and safe file writing."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from e1.config import REPO_ROOT


def sha256_file(path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_state() -> dict:
    def run(*args):
        r = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None

    status = run("status", "--porcelain", "--", "Experiment1_GCD_CLIP_Baseline", "models", "methods",
                 "project_utils", "data")
    if status:   # bytecode caches are not code changes
        status = "\n".join(l for l in status.splitlines() if "__pycache__" not in l and not l.endswith(".pyc"))
    return {"commit": run("rev-parse", "HEAD"), "dirty_experiment_files": bool(status),
            "dirty_paths": status.splitlines()[:50] if status else []}


def environment() -> dict:
    import numpy
    import torch

    env = {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": numpy.__version__,
           "torch": torch.__version__, "cuda_available": torch.cuda.is_available()}
    for name in ("torchvision", "open_clip", "sklearn", "scipy", "pandas", "PIL"):
        try:
            module = __import__(name)
            env[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            env[name] = None
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
        env["cuda"] = torch.version.cuda
        env["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    return env


def write_atomic(path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, path)
    return path


def write_json(path, obj) -> Path:
    return write_atomic(path, json.dumps(obj, indent=2, default=str) + "\n")


def write_immutable(path, text: str) -> Path:
    """Write once. Rewriting identical content is allowed; different content is an error."""
    path = Path(path)
    if path.exists():
        if path.read_text() != text:
            raise RuntimeError(f"{path} already exists with different content; immutable files are never "
                               "overwritten (delete the dataset's data/ directory deliberately to rebuild)")
        return path
    return write_atomic(path, text)


def save_npz_atomic(path, **arrays) -> Path:
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    return path


class StageRecord:
    """`status.json` for one stage: started/completed, configuration hash, inputs and outputs."""

    def __init__(self, directory, stage: str, cfg_hash: str, inputs: dict | None = None):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "status.json"
        self.record = {"stage": stage, "config_hash": cfg_hash, "status": "started",
                       "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "inputs": inputs or {}, "environment": environment(), "git": git_state()}
        write_json(self.path, self.record)

    def complete(self, **outputs) -> dict:
        self.record.update(status="completed", outputs=outputs,
                           completed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        write_json(self.path, self.record)
        return self.record


def stage_completed(directory, cfg_hash: str | None = None) -> bool:
    path = Path(directory) / "status.json"
    if not path.exists():
        return False
    record = json.loads(path.read_text())
    return record.get("status") == "completed" and (cfg_hash is None or record.get("config_hash") == cfg_hash)
