"""Baseline A: frozen CLIP (OpenAI weights through open_clip) with per-frame encoding.

open_clip's plain "ViT-B-32"/"ViT-B-16" configs use standard GELU, but the
OpenAI weights were trained with QuickGELU, so QuickGELU is forced and checked.
Weights are loaded from the local Hugging Face cache only, unless downloads
are explicitly allowed.
"""
from __future__ import annotations

from pathlib import Path

import torch

from ovar.backbones.base import Backbone, BackboneUnavailable, template_average
from ovar.cache import sha256_file

WEIGHT_FILES = ("open_clip_model.safetensors", "open_clip_pytorch_model.bin")
APPROX_DOWNLOAD_MB = {"ViT-B-32": 605, "ViT-B-16": 599}


class FrozenCLIP(Backbone):
    kind = "clip_frozen"
    frame_independent = True

    def __init__(self, cfg: dict, device: str, cache_dir, allow_download: bool = False):
        import open_clip
        from open_clip import pretrained as oc_pretrained

        self.arch = cfg["arch"]
        self.pretrained_tag = cfg.get("pretrained", "openai")
        self.device = device
        pcfg = oc_pretrained.get_pretrained_cfg(self.arch, self.pretrained_tag)
        if not pcfg:
            raise BackboneUnavailable(f"open_clip has no pretrained tag {self.pretrained_tag!r} for {self.arch}")
        self.hf_repo = pcfg["hf_hub"].rstrip("/")
        self.original_url = pcfg.get("url", "")
        self.weights_path = self._resolve_weights(allow_download)
        self.weights_sha256 = sha256_file(self.weights_path, memo_dir=Path(cache_dir) / "sha256")

        model, _, _ = open_clip.create_model_and_transforms(
            self.arch, pretrained=str(self.weights_path),
            force_quick_gelu=cfg.get("force_quick_gelu", True))
        self.activation = type(model.visual.transformer.resblocks[0].mlp.gelu).__name__
        if cfg.get("force_quick_gelu", True) and self.activation != "QuickGELU":
            raise BackboneUnavailable(f"expected QuickGELU in {self.arch}, found {self.activation}")
        self.model = model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(self.arch)
        self.open_clip_version = open_clip.__version__

    def _resolve_weights(self, allow_download: bool) -> Path:
        from huggingface_hub import hf_hub_download

        errors = []
        for filename in WEIGHT_FILES:
            try:
                return Path(hf_hub_download(self.hf_repo, filename, local_files_only=not allow_download))
            except Exception as exc:  # noqa: BLE001 - report every attempt
                errors.append(f"{filename}: {type(exc).__name__}")
        size = APPROX_DOWNLOAD_MB.get(self.arch, "?")
        raise BackboneUnavailable(
            f"CLIP {self.arch} ({self.pretrained_tag}) weights are not in the local Hugging Face cache "
            f"({'; '.join(errors)}). Download them explicitly (about {size} MB) with "
            f"`python -m ovar fetch-clip --arch {self.arch}`.")

    def identity(self) -> dict:
        return {"backbone": self.kind, "arch": self.arch, "pretrained": self.pretrained_tag,
                "weights_sha256": self.weights_sha256, "activation": self.activation}

    def provenance(self) -> dict:
        return {
            **self.identity(),
            "implementation": f"open_clip {self.open_clip_version} (verified-compatible re-implementation "
                              "of OpenAI CLIP)",
            "weights_file": str(self.weights_path),
            "weights_source": f"https://huggingface.co/{self.hf_repo} (OpenAI weights converted by "
                              "timm/open_clip)",
            "original_release": self.original_url,
            "training_data": "OpenAI WIT-400M image-text pairs (not publicly documented in detail); "
                             "no task-specific action training",
            "licence": "OpenAI CLIP code and weights: MIT; open_clip code: MIT",
        }

    @torch.inference_mode()
    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        views, length = frames.shape[:2]
        out = self.model.encode_image(frames.flatten(0, 1).to(self.device))
        return out.float().view(views, length, -1)

    @torch.inference_mode()
    def encode_text(self, prompts: list[list[str]]) -> torch.Tensor:
        per_template = []
        for k in range(len(prompts[0])):
            tokens = self.tokenizer([p[k] for p in prompts]).to(self.device)
            per_template.append(self.model.encode_text(tokens).float())
        return template_average(per_template).cpu()
