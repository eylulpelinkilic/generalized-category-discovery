"""Spatial preprocessing presets: uint8 RGB frames [N, H, W, 3] -> float tensor [N, 3, S, S].

clip_center      CLIP/open_clip test transform: shorter side to S (PIL bicubic,
                 antialiased), torchvision centre crop.
slowfast_center  Open-VCLIP/PySlowFast test transform: shorter side to S with
                 bilinear interpolation (no antialiasing), crop offset ceil((L-S)/2).
vifi_center      ViFi-CLIP/TC-CLIP test transform (approximation): shorter side to
                 round(256/224*S) bilinear, crop offset floor((L-S)/2). The
                 originals resize with OpenCV, which is not bit-identical.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _normalise(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD).view(1, 3, 1, 1)
    return (x - mean) / std


def _clip_center(frames: np.ndarray, size: int) -> torch.Tensor:
    pipeline = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])
    return torch.stack([pipeline(Image.fromarray(f)) for f in frames])


def _short_side_size(height: int, width: int, size: int) -> tuple[int, int]:
    if width < height:
        return int(math.floor(height / width * size)), size
    return size, int(math.floor(width / height * size))


def _bilinear_center(frames: np.ndarray, short: int, crop: int, ceil_offset: bool) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float() / 255.0
    x = _normalise(x)
    height, width = x.shape[-2:]
    if min(height, width) != short:
        x = F.interpolate(x, size=_short_side_size(height, width, short), mode="bilinear",
                          align_corners=False)
    height, width = x.shape[-2:]
    offset = (lambda d: int(math.ceil(d / 2))) if ceil_offset else (lambda d: d // 2)
    top, left = offset(height - crop), offset(width - crop)
    return x[:, :, top:top + crop, left:left + crop].contiguous()


def preprocess(frames: np.ndarray, spec: dict) -> torch.Tensor:
    preset, size = spec["preset"], spec.get("size", 224)
    if preset == "clip_center":
        return _clip_center(frames, size)
    if preset == "slowfast_center":
        return _bilinear_center(frames, size, size, ceil_offset=True)
    if preset == "vifi_center":
        return _bilinear_center(frames, int(256 / 224 * size), size, ceil_offset=False)
    raise ValueError(f"unknown preprocessing preset {preset!r}")
