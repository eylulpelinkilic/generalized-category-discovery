"""Diagnostic transforms applied to sampled uint8 frames before model preprocessing.

Temporal transforms reorder or repeat positions within each sampled view.
Pixel transforms (degradations, actor crops) change frame content and are
supplementary stress tests, not substitutes for real target domains.
"""
from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image, ImageFilter

TEMPORAL_KINDS = ("ordered", "shuffled", "single_frame", "reversed")


def temporal_positions(kind: str, length: int, seed_key: str) -> np.ndarray:
    """Positions into a view of `length` frames. `seed_key` makes shuffles reproducible."""
    if kind == "ordered":
        return np.arange(length)
    if kind == "reversed":
        return np.arange(length)[::-1].copy()
    if kind == "single_frame":
        return np.full(length, length // 2)
    if kind == "shuffled":
        seed = int.from_bytes(hashlib.sha256(seed_key.encode()).digest()[:8], "little")
        return np.random.default_rng(seed).permutation(length)
    raise ValueError(f"unknown temporal transform {kind!r}")


def _resize(frame: np.ndarray, width: int, height: int, resample) -> np.ndarray:
    return np.asarray(Image.fromarray(frame).resize((width, height), resample=resample))


def degrade(frames: np.ndarray, spec: dict) -> np.ndarray:
    """Deterministic, frame-wise degradation of RGB uint8 frames."""
    kind = spec["kind"]
    if kind == "brightness":
        # Linear intensity scaling in sRGB space (factor < 1 darkens).
        return np.clip(np.rint(frames.astype(np.float32) * spec["factor"]), 0, 255).astype(np.uint8)
    if kind == "downsample":
        height, width = frames.shape[1:3]
        small = (max(1, round(width * spec["factor"])), max(1, round(height * spec["factor"])))
        return np.stack([
            _resize(_resize(f, *small, Image.Resampling.BOX), width, height, Image.Resampling.BILINEAR)
            for f in frames])
    if kind == "blur":
        return np.stack([np.asarray(Image.fromarray(f).filter(ImageFilter.GaussianBlur(spec["sigma"])))
                         for f in frames])
    raise ValueError(f"unknown degradation {kind!r}")


def crop_to_box(frames: np.ndarray, boxes: np.ndarray, context: float) -> np.ndarray:
    """Square crop around each frame's actor box, enlarged by `context` (1.0 = tight).

    `boxes` is [N, 4] (x1, y1, x2, y2) in pixels. The square is shifted to stay
    inside the frame where possible; any remainder is zero-padded, which is an
    artefact the diagnostic report must mention.
    """
    out = []
    for frame, (x1, y1, x2, y2) in zip(frames, boxes):
        height, width = frame.shape[:2]
        side = max(x2 - x1, y2 - y1) * context
        side = int(round(max(side, 2)))
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        left = int(round(min(max(cx - side / 2, 0), max(width - side, 0))))
        top = int(round(min(max(cy - side / 2, 0), max(height - side, 0))))
        canvas = np.zeros((side, side, 3), dtype=np.uint8)
        patch = frame[top:top + side, left:left + side]
        canvas[:patch.shape[0], :patch.shape[1]] = patch
        out.append(canvas)
    return np.stack(out)
