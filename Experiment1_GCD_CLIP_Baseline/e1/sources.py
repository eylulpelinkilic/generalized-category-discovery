"""Downloads of legitimate sources with size and checksum verification (explicit calls only)."""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import tarfile
from pathlib import Path

from e1.provenance import sha256_file

DINO_VITB16 = {"url": "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth",
               "bytes": 343_242_485, "file": "dino_vitbase16_pretrain.pth"}
OPENAI_CLIP_JIT = {
    "ViT-B-32": {"sha256": "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"},
    "ViT-B-16": {"sha256": "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"},
}


def openai_jit_url(arch: str) -> str:
    return f"https://openaipublic.azureedge.net/clip/models/{OPENAI_CLIP_JIT[arch]['sha256']}/{arch}.pt"


def fetch(url: str, dest, expected_bytes: int | None = None) -> dict:
    """curl with resume; refuses a file whose size differs from `expected_bytes`."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not (dest.exists() and expected_bytes and dest.stat().st_size == expected_bytes):
        if shutil.which("curl") is None:
            raise RuntimeError("curl is needed for downloads")
        subprocess.run(["curl", "-sS", "-L", "--fail", "--retry", "5", "-C", "-", "-o", str(dest), url], check=True)
    size = dest.stat().st_size
    if expected_bytes and size != expected_bytes:
        raise RuntimeError(f"{dest.name}: {size} bytes, expected {expected_bytes}")
    return {"url": url, "path": str(dest), "bytes": size, "sha256": sha256_file(dest)}


def md5_file(path, chunk: int = 1 << 24) -> str:
    h = hashlib.md5()  # noqa: S324 - matches the published checksum
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def fetch_cub(dest_dir) -> dict:
    """Official CUB-200-2011 archive, MD5-checked against the upstream constant, then extracted."""
    from e1.manifests import CUB_MD5, CUB_URL

    dest_dir = Path(dest_dir)
    root = dest_dir / "CUB_200_2011"
    if (root / "images.txt").exists():
        return {"source": "existing extraction", "path": str(root)}
    archive = dest_dir / "CUB_200_2011.tgz"
    info = fetch(CUB_URL, archive)
    info["md5"] = md5_file(archive)
    if info["md5"] != CUB_MD5:
        raise RuntimeError(f"CUB archive MD5 {info['md5']} != {CUB_MD5}")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(dest_dir, filter="data")
    archive.unlink()
    info["path"] = str(root)
    return info
