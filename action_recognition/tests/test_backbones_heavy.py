"""Tests that load real CLIP weights (~600 MB). Opt in with OVAR_HEAVY_TESTS=1."""
import copy
import os
import tempfile
import unittest
from pathlib import Path

import torch

from tests.official_source import THIRD_PARTY, require

HEAVY = os.environ.get("OVAR_HEAVY_TESTS") == "1"


def cached_b32():
    from huggingface_hub import hf_hub_download

    try:
        return Path(hf_hub_download("timm/vit_base_patch32_clip_224.openai", "open_clip_model.safetensors",
                                    local_files_only=True))
    except Exception:  # noqa: BLE001
        raise unittest.SkipTest("CLIP ViT-B/32 weights not cached")


@unittest.skipUnless(HEAVY, "set OVAR_HEAVY_TESTS=1")
class TestRealWeights(unittest.TestCase):
    def test_frozen_clip_uses_quickgelu(self):
        from ovar.backbones.clip_frozen import FrozenCLIP

        cached_b32()
        with tempfile.TemporaryDirectory() as tmp:
            model = FrozenCLIP({"arch": "ViT-B-32", "pretrained": "openai"}, "cpu", tmp)
        self.assertEqual(model.activation, "QuickGELU")
        self.assertEqual(len(model.weights_sha256), 64)

    def test_official_openvclip_code_reproduces_clip_without_temporal_modelling(self):
        import open_clip

        from ovar.backbones.openvclip import import_official_modules, load_clip_state

        weights = cached_b32()
        temporal, _ = import_official_modules(require(THIRD_PARTY / "Open-VCLIP"))
        state, _ = load_clip_state(weights)
        ref = open_clip.create_model_and_transforms("ViT-B-32", pretrained=str(weights),
                                                    force_quick_gelu=True)[0].eval()
        ours = temporal.build_model(copy.deepcopy(state), T=4, temporal_modeling_type=None).float().eval()
        x = torch.randn(4, 3, 224, 224, generator=torch.Generator().manual_seed(0))
        with torch.inference_mode():
            diff = (ours.encode_image(x) - ref.encode_image(x)).abs().max().item()
        self.assertLess(diff, 1e-3)


if __name__ == "__main__":
    unittest.main()
