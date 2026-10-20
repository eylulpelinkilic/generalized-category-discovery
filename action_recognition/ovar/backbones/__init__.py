"""Backbone registry."""
from __future__ import annotations

from ovar.backbones.base import Backbone, BackboneUnavailable


def build_backbone(model_cfg: dict, protocol: dict, device: str, cache_dir,
                   allow_download: bool = False) -> Backbone:
    kind = model_cfg["backbone"]
    if kind == "clip_frozen":
        from ovar.backbones.clip_frozen import FrozenCLIP

        return FrozenCLIP(model_cfg, device, cache_dir, allow_download=allow_download)
    if kind == "openvclip":
        from ovar.backbones.openvclip import OpenVCLIP

        return OpenVCLIP(model_cfg, device, cache_dir, num_frames=protocol["sampling"]["num_frames"])
    if kind == "tcclip":
        from ovar.backbones import tcclip

        return tcclip.build()
    if kind == "fake_for_tests":
        from ovar.backbones.fake import FakeBackbone

        return FakeBackbone(seed=model_cfg.get("seed", 0))
    raise BackboneUnavailable(f"unknown backbone {kind!r}")


__all__ = ["Backbone", "BackboneUnavailable", "build_backbone"]
