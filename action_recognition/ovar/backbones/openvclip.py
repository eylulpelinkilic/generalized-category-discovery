"""Baseline B: pretrained Open-VCLIP through the official model code (pinned clone).

The official `TemporalClipVideo` wrapper and test loop are not used, because
they hard-code `.cuda()` and need the full PySlowFast stack. This adapter
imports the official `WCLIP`/`build_model`, `TemporalVisionTransformer` and
tokenizer unchanged, then reproduces the documented test procedure:

1. Build WCLIP from the original OpenAI CLIP weights (as `load("ViT-B/16")` does).
2. Weight patching (tools/test_net.py): every original CLIP key becomes
   `ratio * clip_original + (1 - ratio) * checkpoint`; keys only present in the
   checkpoint are copied.
3. Class embeddings: the 28-template ensemble (`achieve_csf_matrix`) unless the
   protocol selects another template set.
4. Frame embeddings from `encode_image` on [views * T] frames; scoring as in
   `ovar.backbones.base.video_scores` (same ranking as the official logits).

Only the PySlowFast model registry (`slowfast/models/build.py`, which pulls in
fvcore) is replaced by a no-op stub, plus a `pkg_resources` shim when
setuptools no longer ships it.
"""
from __future__ import annotations

import copy
import importlib
import importlib.util
import sys
import types
from pathlib import Path

import torch

from ovar.backbones.base import Backbone, BackboneUnavailable, template_average
from ovar.cache import sha256_file
from ovar.config import project_path

PACKAGE = "ovar_vendored_openvclip"
JIT_ONLY_KEYS = ("input_resolution", "context_length", "vocab_size")


def import_official_modules(repo: Path):
    """Import Open-VCLIP's model modules from `repo` without running PySlowFast's __init__."""
    if PACKAGE in sys.modules:
        return sys.modules[f"{PACKAGE}.temporalclip_video_model"], sys.modules[f"{PACKAGE}.clip.clip"]
    models_dir = repo / "slowfast" / "models"
    if not (models_dir / "temporalclip_video_model.py").exists():
        raise BackboneUnavailable(f"Open-VCLIP code not found under {models_dir}; clone it with "
                                  "`python -m ovar fetch-repos`.")
    package = types.ModuleType(PACKAGE)
    package.__path__ = [str(models_dir)]
    sys.modules[PACKAGE] = package

    registry = types.ModuleType(f"{PACKAGE}.build")

    class _NoOpRegistry:
        def register(self, *args, **kwargs):
            return lambda obj: obj

    registry.MODEL_REGISTRY = _NoOpRegistry()
    sys.modules[f"{PACKAGE}.build"] = registry

    shimmed = importlib.util.find_spec("pkg_resources") is None
    if shimmed:
        import packaging.version

        sys.modules["pkg_resources"] = types.SimpleNamespace(packaging=packaging)
    try:
        temporal = importlib.import_module(f"{PACKAGE}.temporalclip_video_model")
        clip_module = importlib.import_module(f"{PACKAGE}.clip.clip")
    except Exception as exc:
        for name in [m for m in sys.modules if m.startswith(PACKAGE)]:
            del sys.modules[name]
        raise BackboneUnavailable(
            f"official Open-VCLIP model code failed to import with torch {torch.__version__}: "
            f"{type(exc).__name__}: {exc}. Use the environment documented in envs/openvclip.md.") from exc
    finally:
        if shimmed:
            del sys.modules["pkg_resources"]
    return temporal, clip_module


def load_clip_state(path: Path) -> tuple[dict, str]:
    """Original CLIP weights: OpenAI JIT archive (preferred, as in the official scripts) or a state dict."""
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path)), "safetensors_state_dict"
    try:
        state = torch.jit.load(str(path), map_location="cpu").eval().state_dict()
        for key in JIT_ONLY_KEYS:
            state.pop(key, None)
        return dict(state), "openai_jit_archive"
    except RuntimeError:
        return torch.load(str(path), map_location="cpu", weights_only=True), "state_dict"


def patch_weights(clip_state: dict, checkpoint_state: dict, ratio: float) -> tuple[dict, list[str]]:
    """Weight patching from tools/test_net.py (raw CLIP archive branch), without the 'model.' prefix."""
    patched = {}
    for key, value in clip_state.items():
        if f"model.{key}" not in checkpoint_state:
            raise BackboneUnavailable(f"checkpoint lacks model.{key}; not an Open-VCLIP checkpoint?")
        patched[key] = value * ratio + checkpoint_state[f"model.{key}"] * (1 - ratio)
    extra = sorted(set(checkpoint_state) - {f"model.{k}" for k in clip_state})
    for key in extra:
        if key.startswith("model."):
            patched[key[len("model."):]] = checkpoint_state[key]
    return patched, extra


class OpenVCLIP(Backbone):
    kind = "openvclip"
    frame_independent = False

    def __init__(self, cfg: dict, device: str, cache_dir, num_frames: int):
        self.cfg = cfg
        self.device = device
        self.repo = project_path(cfg["repo"])
        self.num_frames = num_frames
        memo = Path(cache_dir) / "sha256"

        clip_path = project_path(cfg["clip_original"])
        if not clip_path.is_file():
            raise BackboneUnavailable(
                f"original CLIP weights not found at {clip_path} (needed to build the model and for "
                f"weight patching). See docs/backbones.md for the OpenAI download URL and size.")
        checkpoint = cfg.get("checkpoint")
        checkpoint_path = project_path(checkpoint) if checkpoint else None
        if checkpoint_path is None and not cfg.get("allow_without_checkpoint", False):
            raise BackboneUnavailable("no Open-VCLIP checkpoint configured")
        if checkpoint_path is not None and not checkpoint_path.is_file():
            raise BackboneUnavailable(
                f"Open-VCLIP checkpoint not found at {checkpoint_path}. Download it from the official "
                "Google Drive folder (docs/backbones.md) and set checkpoints.* in configs/paths.yaml.")

        temporal, clip_module = import_official_modules(self.repo)
        self.tokenize = clip_module.tokenize
        clip_state, self.clip_format = load_clip_state(clip_path)
        self.clip_sha256 = sha256_file(clip_path, memo_dir=memo)
        model = temporal.build_model(
            copy.deepcopy(clip_state), T=num_frames,
            temporal_modeling_type=cfg.get("temporal_modeling_type", "expand_temporal_view"),
            context_length=77)
        model.float()

        self.checkpoint_sha256, self.extra_keys = None, []
        if checkpoint_path is not None:
            self.checkpoint_sha256 = sha256_file(checkpoint_path, memo_dir=memo)
            state = self._load_checkpoint(checkpoint_path)
            patched, self.extra_keys = patch_weights(clip_state, state, cfg.get("patching_ratio", 0.5))
            missing, unexpected = model.load_state_dict(patched, strict=False)
            if missing or unexpected:
                raise BackboneUnavailable(f"checkpoint/model mismatch: missing={missing[:5]} "
                                          f"unexpected={unexpected[:5]}")
        self.model = model.to(device).eval()
        self.checkpoint_path = checkpoint_path

    def _load_checkpoint(self, path: Path) -> dict:
        try:
            checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
        except Exception as exc:  # noqa: BLE001
            if not self.cfg.get("trust_pickle", False):
                raise BackboneUnavailable(
                    f"{path.name} contains pickled Python objects ({type(exc).__name__}). After checking "
                    "its SHA-256 against docs/backbones.md, set `model.trust_pickle=true` to load it with "
                    "weights_only=False.") from exc
            checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        return checkpoint["model_state"]

    def identity(self) -> dict:
        return {
            "backbone": self.kind, "arch": self.cfg["arch"],
            "temporal_modeling_type": self.cfg.get("temporal_modeling_type", "expand_temporal_view"),
            "num_frames_T": self.num_frames, "patching_ratio": self.cfg.get("patching_ratio", 0.5),
            "checkpoint_sha256": self.checkpoint_sha256, "clip_original_sha256": self.clip_sha256,
            "clip_original_format": self.clip_format,
        }

    def provenance(self) -> dict:
        return {
            **self.identity(),
            "code": "github.com/wengzejia1/Open-VCLIP @ 047a817e94b88c02c56fde2f1dd0fbf03f3be638 "
                    "(official model code imported unchanged)",
            "checkpoint_file": str(self.checkpoint_path) if self.checkpoint_path else None,
            "extra_checkpoint_keys": self.extra_keys,
            "training_data": "Kinetics-400 (all 400 classes), initialised from OpenAI CLIP",
            "trained_num_frames": 8,
            "licence": "Open-VCLIP repository: no licence file (all rights reserved by default); "
                       "PySlowFast: Apache-2.0; CLIP: MIT",
        }

    @torch.inference_mode()
    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        views, length = frames.shape[:2]
        if length != self.num_frames:
            raise ValueError(f"model built for T={self.num_frames}, got {length} frames")
        out = self.model.encode_image(frames.flatten(0, 1).to(self.device).float())
        return out.float().view(views, length, -1)

    @torch.inference_mode()
    def encode_text(self, prompts: list[list[str]]) -> torch.Tensor:
        per_template = []
        for k in range(len(prompts[0])):
            tokens = torch.cat([self.tokenize(p[k]) for p in prompts]).to(self.device)
            per_template.append(self.model.encode_text(tokens).float())
        return template_average(per_template).cpu()
