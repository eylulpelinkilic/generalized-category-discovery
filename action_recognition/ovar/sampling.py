"""Deterministic temporal sampling. Each function returns one index array per view."""
from __future__ import annotations

import math

import numpy as np
import torch


def tsn_midpoints(num_frames: int, num_segments: int) -> list[np.ndarray]:
    """Midpoint of each of `num_segments` equal segments over the whole video.

    Same as mmaction2 `SampleFrames(clip_len=1, frame_interval=1, num_clips=T,
    test_mode=True)`, used by ViFi-CLIP and TC-CLIP at test time.
    """
    if num_frames <= 0:
        raise ValueError("video has no frames")
    interval = num_frames / float(num_segments)
    offsets = (np.arange(num_segments) * interval + interval / 2.0).astype(np.int64)
    return [offsets]


def slowfast_dense(num_frames: int, fps: float, num_samples: int, sampling_rate: int,
                   num_views: int, target_fps: float = 30.0, use_offset: bool = True) -> list[np.ndarray]:
    """Open-VCLIP/PySlowFast test-time sampling, decode-the-whole-video path.

    Each view is a window of `sampling_rate * num_samples` frames at `target_fps`,
    placed by `get_multiple_start_end_idx(..., use_offset)`; frames inside the
    window come from `temporal_sampling` (float32 linspace, clamp, truncate).
    PySlowFast's selective-decoding path computes windows from timestamps
    instead, so a few indices can differ from the official decoder.
    """
    if num_frames <= 0:
        raise ValueError("video has no frames")
    clip_size = max(1.0, sampling_rate * num_samples / target_fps * fps)
    max_start = max(num_frames - clip_size, 0)
    views = []
    for view in range(num_views):
        if use_offset:
            start = (math.floor(max_start / 2) if num_views == 1
                     else view * math.floor(max_start / (num_views - 1)))
        else:
            start = max_start * view / num_views
        end = start + clip_size - 1
        index = torch.linspace(start, end, num_samples)
        index = torch.clamp(index, 0, num_frames - 1).long()
        views.append(index.numpy().astype(np.int64))
    return views


def sample_views(num_frames: int, fps: float, spec: dict) -> list[np.ndarray]:
    method = spec["method"]
    if method == "tsn_midpoint":
        if spec.get("num_views", 1) != 1:
            raise ValueError("tsn_midpoint supports a single view")
        return tsn_midpoints(num_frames, spec["num_frames"])
    if method == "slowfast_dense":
        return slowfast_dense(num_frames, fps, spec["num_frames"], spec["sampling_rate"],
                              spec["num_views"], spec.get("target_fps", 30.0),
                              spec.get("use_offset", True))
    raise ValueError(f"unknown sampling method {method!r}")
