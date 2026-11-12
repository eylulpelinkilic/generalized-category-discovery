"""Checkpoints used by protocol A, fetched explicitly and verified (size, CRC32 or sha256)."""
from __future__ import annotations

import struct
import urllib.request
import zlib
from pathlib import Path

from e1.provenance import sha256_file, write_json
from e2.config import Layout


def _get(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "e2-assets"})
    with urllib.request.urlopen(req, timeout=120) as r:      # noqa: S310 - fixed URLs from the config
        return r.read()


def zip_members(url: str, size: int) -> dict:
    """Central directory of a remote zip (zip64 aware): name -> (method, csize, usize, crc32, header offset)."""
    tail = _get(url, max(0, size - 200_000), size - 1)
    i = tail.rfind(b"PK\x05\x06")
    n, cd_size, cd_off = struct.unpack("<HII", tail[i + 10:i + 20])
    if cd_off == 0xFFFFFFFF or n == 0xFFFF:
        j = tail.rfind(b"PK\x06\x06")
        n, cd_size, cd_off = struct.unpack("<QQQ", tail[j + 32:j + 56])
    cd = _get(url, cd_off, cd_off + cd_size - 1)
    out, p = {}, 0
    while cd[p:p + 4] == b"PK\x01\x02":
        method, = struct.unpack("<H", cd[p + 10:p + 12])
        crc, csize, usize = struct.unpack("<III", cd[p + 16:p + 28])
        fl, el, cl = struct.unpack("<HHH", cd[p + 28:p + 34])
        off, = struct.unpack("<I", cd[p + 42:p + 46])
        name = cd[p + 46:p + 46 + fl].decode()
        extra = cd[p + 46 + fl:p + 46 + fl + el]
        q = 0
        while q + 4 <= len(extra):
            hid, hl = struct.unpack("<HH", extra[q:q + 4])
            if hid == 1:
                vals = list(struct.unpack("<" + "Q" * (hl // 8), extra[q + 4:q + 4 + hl]))
                if usize == 0xFFFFFFFF:
                    usize = vals.pop(0)
                if csize == 0xFFFFFFFF:
                    csize = vals.pop(0)
                if off == 0xFFFFFFFF:
                    off = vals.pop(0)
            q += 4 + hl
        out[name] = (method, csize, usize, crc, off)
        p += 46 + fl + el + cl
    return out


def extract_remote_member(url: str, zip_bytes: int, member: str, dest: Path, expect_bytes: int,
                          expect_crc32: str) -> dict:
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size == expect_bytes:
        crc = format(zlib.crc32(dest.read_bytes()) & 0xFFFFFFFF, "08x")
        if crc == expect_crc32:
            return {"path": str(dest), "bytes": expect_bytes, "crc32": crc, "sha256": sha256_file(dest), "source": "existing"}
    members = zip_members(url, zip_bytes)
    method, csize, usize, crc, off = members[member]
    head = _get(url, off, off + 29)
    fl, el = struct.unpack("<HH", head[26:30])
    start = off + 30 + fl + el
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    dec = zlib.decompressobj(-15) if method == 8 else None
    if method not in (0, 8):
        raise RuntimeError(f"unsupported zip method {method}")
    crc_run, step = 0, 64 << 20
    with open(tmp, "wb") as f:
        for s in range(start, start + csize, step):
            chunk = _get(url, s, min(start + csize, s + step) - 1)
            data = dec.decompress(chunk) if dec else chunk
            crc_run = zlib.crc32(data, crc_run)
            f.write(data)
        if dec:
            rest = dec.flush()
            crc_run = zlib.crc32(rest, crc_run)
            f.write(rest)
    got = format(crc_run & 0xFFFFFFFF, "08x")
    if tmp.stat().st_size != expect_bytes or got != expect_crc32 or got != format(crc, "08x"):
        raise RuntimeError(f"{member}: {tmp.stat().st_size} bytes crc {got}; expected {expect_bytes} / {expect_crc32}")
    tmp.replace(dest)
    return {"path": str(dest), "bytes": expect_bytes, "crc32": got, "sha256": sha256_file(dest),
            "source": f"{url} :: {member} (range requests)"}


def parallel_download(url: str, dest: Path, chunk: int = 16 << 20, workers: int = 12, retries: int = 8) -> dict:
    """Resumable download in parallel byte ranges (each range is one file under dest.parts/, kept until the
    whole file is assembled), for servers that are slow per connection. Returns size and timing."""
    import time
    from concurrent.futures import ThreadPoolExecutor

    dest = Path(dest)
    if dest.exists():
        return {"path": str(dest), "bytes": dest.stat().st_size, "source": "existing"}
    head = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": "e2-assets"})
    with urllib.request.urlopen(head, timeout=60) as r:          # noqa: S310 - fixed URLs from the config
        total = int(r.headers["Content-Range"].split("/")[-1])
    parts = dest.parent / (dest.name + ".parts")
    parts.mkdir(parents=True, exist_ok=True)
    ranges = [(s, min(total, s + chunk) - 1) for s in range(0, total, chunk)]

    def one(i_rng):
        i, (s, e) = i_rng
        p = parts / f"{i:05d}"
        if p.exists() and p.stat().st_size == e - s + 1:
            return 0
        for attempt in range(retries):
            try:
                data = _get(url, s, e)
                if len(data) == e - s + 1:
                    p.write_bytes(data)
                    return 1
            except Exception:                                     # noqa: BLE001 - retried
                time.sleep(2 + 3 * attempt)
        raise RuntimeError(f"range {s}-{e} failed {retries} times")

    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        fetched = sum(ex.map(one, enumerate(ranges)))
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with open(tmp, "wb") as f:
        for i in range(len(ranges)):
            f.write((parts / f"{i:05d}").read_bytes())
    if tmp.stat().st_size != total:
        raise RuntimeError(f"assembled {tmp.stat().st_size} bytes, expected {total}")
    tmp.replace(dest)
    import shutil

    shutil.rmtree(parts)
    return {"path": str(dest), "bytes": total, "ranges": len(ranges), "ranges_fetched_now": fetched,
            "seconds": round(time.time() - t0, 1), "source": url}


def fetch_cub_files(rel_paths: list[str], dest: Path, archive_dir: Path) -> dict:
    """Official CUB archive (parallel, resumable, MD5-checked against the upstream constant); extract only
    `rel_paths` (relative to CUB_200_2011/) into dest/CUB_200_2011/."""
    import tarfile

    from e1.manifests import CUB_MD5, CUB_URL
    from e1.sources import md5_file

    dest = Path(dest)
    todo = {f"CUB_200_2011/{p}" for p in rel_paths if not (dest / "CUB_200_2011" / p).exists()}
    record_path = dest / "fetch_record.json"
    if not todo:
        prior = __import__("json").loads(record_path.read_text()) if record_path.exists() else None
        return {"source": "existing extraction", "files": len(rel_paths), "extraction_record": prior}
    archive = Path(archive_dir) / "CUB_200_2011.tgz"
    info = parallel_download(CUB_URL, archive)
    md5 = md5_file(archive)
    if md5 != CUB_MD5:
        raise RuntimeError(f"CUB archive MD5 {md5} != {CUB_MD5}; delete {archive} and retry")
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if member.name in todo:
                tar.extract(member, dest, filter="data")
    missing = [p for p in todo if not (dest / p).exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} files not in the archive, e.g. {missing[:3]}")
    record = {"source": CUB_URL, "archive_md5": md5, "md5_check": "passed", "archive_bytes": info["bytes"],
              "extracted": len(todo), "download": info}
    write_json(record_path, record)
    return record


def fetch_dino(cfg: dict) -> dict:
    from e1.sources import fetch

    d = cfg["repro"]["dino"]
    info = fetch(d["url"], Layout(cfg).assets / "dino" / Path(d["url"]).name, d["bytes"])
    write_json(Layout(cfg).assets / "dino" / "fetch.json", info)
    return info


def fetch_gcd_checkpoint(cfg: dict) -> dict:
    g = cfg["repro"]["gcd"]
    dest = Layout(cfg).assets / "scd_official" / Path(g["member"]).name
    info = extract_remote_member(g["zip_url"], g["zip_bytes"], g["member"], dest, g["member_bytes"], g["member_crc32"])
    write_json(dest.with_suffix(".fetch.json"), info)
    return info
