"""Frame-accurate decoding through the ffmpeg/ffprobe command-line tools.

Frame index `n` is the n-th frame the decoder outputs (presentation order),
counted by `ffprobe -count_frames`. Frames are returned as RGB uint8 at the
stored resolution; rotation metadata is ignored (`-noautorotate`), matching
PyAV/decord defaults. Colour conversion uses swscale defaults, so values can
differ slightly from other decoders.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

FALLBACK_DIRS = ("/opt/homebrew/bin", "/usr/local/bin")


class VideoDecodeError(RuntimeError):
    pass


def tool(name: str) -> str:
    """Path to ffmpeg/ffprobe: $OVAR_FFMPEG_DIR, then PATH, then common Homebrew locations."""
    custom = os.environ.get("OVAR_FFMPEG_DIR")
    candidates = [Path(custom) / name] if custom else []
    found = shutil.which(name)
    candidates += [Path(found)] if found else []
    candidates += [Path(d) / name for d in FALLBACK_DIRS]
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    raise VideoDecodeError(f"{name} not found (set OVAR_FFMPEG_DIR or add it to PATH)")


@dataclass(frozen=True)
class VideoMeta:
    num_frames: int
    fps: float
    width: int
    height: int
    duration: float | None
    codec: str

    def to_dict(self) -> dict:
        return asdict(self)


def _run(cmd: list[str], timeout: float) -> bytes:
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
    except FileNotFoundError as exc:
        raise VideoDecodeError(f"{cmd[0]} not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise VideoDecodeError(f"{cmd[0]} timed out after {timeout}s") from exc
    if out.returncode != 0:
        message = out.stderr.decode(errors="replace").strip()
        raise VideoDecodeError(message[-500:] or f"{cmd[0]} exited with {out.returncode}")
    return out.stdout


def _rate(text) -> float | None:
    try:
        rate = Fraction(str(text))
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return float(rate) if rate > 0 else None


def probe(path, timeout: float = 120) -> VideoMeta:
    cmd = [
        tool("ffprobe"), "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries",
        "stream=nb_read_frames,avg_frame_rate,r_frame_rate,width,height,duration,codec_name",
        "-of", "json", str(path),
    ]
    streams = json.loads(_run(cmd, timeout) or b"{}").get("streams") or []
    if not streams:
        raise VideoDecodeError("no video stream")
    s = streams[0]
    num_frames = int(s.get("nb_read_frames") or 0)
    if num_frames <= 0:
        raise VideoDecodeError("no decodable frames")
    fps = _rate(s.get("avg_frame_rate")) or _rate(s.get("r_frame_rate"))
    if not fps:
        raise VideoDecodeError("unknown frame rate")
    duration = s.get("duration")
    return VideoMeta(
        num_frames=num_frames, fps=fps, width=int(s["width"]), height=int(s["height"]),
        duration=float(duration) if duration not in (None, "N/A") else None,
        codec=s.get("codec_name", ""),
    )


def read_frames(path, indices, meta: VideoMeta, timeout: float = 300) -> np.ndarray:
    """Return frames at `indices` (any order, repeats allowed) as [N, H, W, 3] uint8."""
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size == 0:
        raise ValueError("no frame indices requested")
    if idx.min() < 0 or idx.max() >= meta.num_frames:
        raise ValueError(f"frame index out of range [0, {meta.num_frames})")
    unique = np.unique(idx)
    select = "+".join(f"eq(n\\,{int(i)})" for i in unique)
    cmd = [
        tool("ffmpeg"), "-nostdin", "-v", "error", "-noautorotate", "-i", str(path),
        "-map", "0:v:0", "-vf", f"select={select}", "-fps_mode", "passthrough",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    raw = _run(cmd, timeout)
    frame_bytes = meta.width * meta.height * 3
    if len(raw) != frame_bytes * len(unique):
        raise VideoDecodeError(
            f"expected {len(unique)} frames of {meta.width}x{meta.height}, got {len(raw)} bytes")
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(len(unique), meta.height, meta.width, 3)
    return frames[np.searchsorted(unique, idx)]
