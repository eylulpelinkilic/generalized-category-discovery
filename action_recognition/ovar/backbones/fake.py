"""Deterministic stand-in backbone for software tests. Never used for results."""
from __future__ import annotations

import hashlib

import numpy as np
import torch

from ovar.backbones.base import Backbone, template_average

DIM = 32


class FakeBackbone(Backbone):
    kind = "fake_for_tests"
    frame_independent = True

    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.projection = torch.from_numpy(rng.standard_normal((3 * 8 * 8, DIM)).astype(np.float32))
        self.seed = seed

    def identity(self) -> dict:
        return {"backbone": self.kind, "seed": self.seed}

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        views, length = frames.shape[:2]
        pooled = torch.nn.functional.adaptive_avg_pool2d(frames.flatten(0, 1).float(), 8)
        return (pooled.flatten(1) @ self.projection).view(views, length, DIM)

    def encode_text(self, prompts: list[list[str]]) -> torch.Tensor:
        def vec(text):
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
            return np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
        per_template = [torch.from_numpy(np.stack([vec(p[k]) for p in prompts]))
                        for k in range(len(prompts[0]))]
        return template_average(per_template)
