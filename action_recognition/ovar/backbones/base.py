"""Backbone interface.

A backbone turns preprocessed frames [V, T, 3, S, S] into frame embeddings
[V, T, D] and class prompts [C][K] into L2-normalised class embeddings [C, D]
(each template embedding is normalised, averaged over the K templates and
renormalised). Video scoring is shared by all backbones, see `video_scores`.
"""
from __future__ import annotations

import numpy as np
import torch


class BackboneUnavailable(RuntimeError):
    """Raised with an actionable message when weights, code or hardware are missing."""


class Backbone:
    kind = "base"
    # True when a frame's embedding does not depend on the other frames, which
    # lets temporal diagnostics reuse cached per-frame embeddings.
    frame_independent = False

    def identity(self) -> dict:
        """Everything that changes embeddings; part of every cache key."""
        raise NotImplementedError

    def provenance(self) -> dict:
        """Human-readable record (code commit, weights, licences, settings)."""
        return self.identity()

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def encode_text(self, prompts: list[list[str]]) -> torch.Tensor:
        raise NotImplementedError


def normalise(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x / np.linalg.norm(x, axis=axis, keepdims=True)


def video_embedding(frame_embeddings: np.ndarray) -> np.ndarray:
    """[V, T, D] -> [D]: L2-normalise each frame, average all frames of all views, renormalise.

    The ranking equals Open-VCLIP's native score (sum over views of the mean
    per-frame cosine similarity times logit_scale), because the final
    renormalisation only rescales every class score of a video by the same
    positive factor.
    """
    mean = normalise(frame_embeddings.astype(np.float64)).reshape(-1, frame_embeddings.shape[-1]).mean(0)
    return normalise(mean)


def video_scores(frame_embeddings: np.ndarray, class_embeddings: np.ndarray) -> np.ndarray:
    """Cosine similarity between the video embedding and each class embedding: [C]."""
    return (video_embedding(frame_embeddings) @ class_embeddings.astype(np.float64).T).astype(np.float32)


def template_average(per_template: list[torch.Tensor]) -> torch.Tensor:
    """Normalise each [C, D] template embedding, average over templates, renormalise."""
    stacked = torch.stack([t / t.norm(dim=-1, keepdim=True) for t in per_template])
    mean = stacked.mean(0)
    return mean / mean.norm(dim=-1, keepdim=True)
