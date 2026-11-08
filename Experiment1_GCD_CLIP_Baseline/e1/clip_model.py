"""Frozen CLIP (open_clip) with explicit QuickGELU, and verification against OpenAI's package."""
from __future__ import annotations

import numpy as np
import torch

OPENAI_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_STD = (0.26862954, 0.26130258, 0.27577711)


def load_clip(model_cfg: dict, device):
    """`model_cfg`: open_clip_name (must end in -quickgelu for OpenAI weights), pretrained."""
    import open_clip

    name, pretrained = model_cfg["open_clip_name"], model_cfg.get("pretrained")
    if pretrained == "openai" and not name.endswith("-quickgelu"):
        raise ValueError(f"{name} with OpenAI weights builds GELU; use '{name}-quickgelu' (AUDIT.md §2, item 9)")
    if pretrained is None:            # smoke only: a fixed random model, identical on every load
        with torch.random.fork_rng():
            torch.manual_seed(0)
            model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=None, precision="fp32")
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained, precision="fp32")
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, preprocess, open_clip.get_tokenizer(name)


def describe(model, preprocess, tokenizer, model_cfg: dict) -> dict:
    import open_clip
    from torchvision import transforms as T

    act = sorted({type(m).__name__ for m in model.modules() if "GELU" in type(m).__name__})
    norm = next(t for t in preprocess.transforms if isinstance(t, T.Normalize))
    resize = next(t for t in preprocess.transforms if isinstance(t, T.Resize))
    info = {
        "open_clip_version": open_clip.__version__, "open_clip_name": model_cfg["open_clip_name"],
        "pretrained": model_cfg.get("pretrained"), "activations": act,
        "embed_dim": int(model.visual.output_dim), "image_size": list(model.visual.image_size),
        "context_length": int(model.context_length), "vocab_size": int(model.vocab_size),
        "preprocess": {"resize": int(resize.size if isinstance(resize.size, int) else resize.size[0]),
                       "interpolation": str(resize.interpolation), "mean": [float(x) for x in norm.mean],
                       "std": [float(x) for x in norm.std],
                       "transforms": [type(t).__name__ for t in preprocess.transforms]},
        "training_mode": bool(model.training), "dtype": str(next(model.parameters()).dtype),
    }
    if model_cfg.get("pretrained") == "openai":
        cfg = open_clip.pretrained.get_pretrained_cfg(model_cfg["open_clip_name"].replace("-quickgelu", ""), "openai")
        info["pretrained_source"] = {k: cfg.get(k) for k in ("url", "hf_hub")}
    info["checks"] = {
        "quick_gelu": act == ["QuickGELU"] if model_cfg.get("pretrained") == "openai" else "not applicable",
        "openai_normalisation": np.allclose(info["preprocess"]["mean"], OPENAI_MEAN) and
        np.allclose(info["preprocess"]["std"], OPENAI_STD),
        "eval_mode": not info["training_mode"], "context_length_77": info["context_length"] == 77,
    }
    return info


@torch.no_grad()
def verify_against_openai(model_cfg: dict, image_paths: list, texts: list[str], device="cpu") -> dict:
    """Compare open_clip (QuickGELU) with the official `clip` package on real images and prompts.

    Both run in fp32 on `device`. Also measures the GELU variant used by the pilot.
    """
    import clip  # official OpenAI package
    from PIL import Image

    arch = {"ViT-B-32-quickgelu": "ViT-B/32", "ViT-B-16-quickgelu": "ViT-B/16"}[model_cfg["open_clip_name"]]
    ref, ref_pre = clip.load(arch, device=device, jit=False)
    ref = ref.float().eval()
    ours, pre, tok = load_clip(model_cfg, device)
    gelu, _, _ = __import__("open_clip").create_model_and_transforms(model_cfg["open_clip_name"].replace("-quickgelu", ""),
                                                                     pretrained="openai", precision="fp32")
    gelu.eval().to(device)
    imgs = [Image.open(p).convert("RGB") for p in image_paths]
    x_ref = torch.stack([ref_pre(im) for im in imgs]).to(device)
    x_ours = torch.stack([pre(im) for im in imgs]).to(device)
    t_ref, t_ours = clip.tokenize(texts).to(device), tok(texts).to(device)
    n = lambda z: torch.nn.functional.normalize(z.float(), dim=-1)  # noqa: E731
    e_ref, e_ours, e_gelu = n(ref.encode_image(x_ref)), n(ours.encode_image(x_ours)), n(gelu.encode_image(x_ours))
    w_ref, w_ours, w_gelu = n(ref.encode_text(t_ref)), n(ours.encode_text(t_ours)), n(gelu.encode_text(t_ours))
    cos = lambda a, b: float((a * b).sum(1).min())  # noqa: E731
    result = {
        "reference": "official openai/CLIP package, clip.load(jit=False), fp32",
        "n_images": len(imgs), "n_texts": len(texts),
        "preprocess_max_abs_diff": float((x_ref - x_ours).abs().max()),
        "tokens_identical": bool(torch.equal(t_ref.cpu(), t_ours.cpu())),
        "image_min_cosine_quickgelu_vs_reference": cos(e_ours, e_ref),
        "image_max_abs_diff_quickgelu_vs_reference": float((e_ours - e_ref).abs().max()),
        "text_min_cosine_quickgelu_vs_reference": cos(w_ours, w_ref),
        "text_max_abs_diff_quickgelu_vs_reference": float((w_ours - w_ref).abs().max()),
        "image_min_cosine_gelu_vs_reference": cos(e_gelu, e_ref),
        "text_min_cosine_gelu_vs_reference": cos(w_gelu, w_ref),
        "zero_shot_agreement_quickgelu_vs_reference": float(((e_ours @ w_ours.T).argmax(1) ==
                                                             (e_ref @ w_ref.T).argmax(1)).float().mean()),
        "zero_shot_agreement_gelu_vs_reference": float(((e_gelu @ w_gelu.T).argmax(1) ==
                                                        (e_ref @ w_ref.T).argmax(1)).float().mean()),
    }
    result["passed"] = (result["tokens_identical"] and result["preprocess_max_abs_diff"] < 1e-5 and
                        result["image_min_cosine_quickgelu_vs_reference"] > 0.9999 and
                        result["text_min_cosine_quickgelu_vs_reference"] > 0.9999)
    return result
