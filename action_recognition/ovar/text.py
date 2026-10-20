"""Prompt templates. Class texts come from data/vocab/<dataset>.csv (`text_name`)."""
from __future__ import annotations

TEMPLATE_SETS = {
    # Single fixed template (XOV-Action paper, Sec. III-B).
    "video_person": ["a video of a person {}."],
    # Open-VCLIP's native 28-template ensemble, copied verbatim from
    # slowfast/models/temporalclip_video_model.py (TemporalClipVideo.text_prompt).
    "openvclip28": [
        "a photo of {}.", "a photo of a person {}.", "a photo of a person using {}.",
        "a photo of a person doing {}.", "a photo of a person during {}.",
        "a photo of a person performing {}.", "a photo of a person practicing {}.",
        "a video of {}.", "a video of a person {}.", "a video of a person using {}.",
        "a video of a person doing {}.", "a video of a person during {}.",
        "a video of a person performing {}.", "a video of a person practicing {}.",
        "a example of {}.", "a example of a person {}.", "a example of a person using {}.",
        "a example of a person doing {}.", "a example of a person during {}.",
        "a example of a person performing {}.", "a example of a person practicing {}.",
        "a demonstration of {}.", "a demonstration of a person {}.",
        "a demonstration of a person using {}.", "a demonstration of a person doing {}.",
        "a demonstration of a person during {}.", "a demonstration of a person performing {}.",
        "a demonstration of a person practicing {}.",
    ],
    # Synthetic fixtures only (software checks).
    "fixture_color": ["a photo of the color {}."],
}


def build_prompts(class_texts: list[str], template_set: str) -> list[list[str]]:
    """Prompts per class: [C][K]."""
    templates = TEMPLATE_SETS[template_set]
    return [[t.format(text) for t in templates] for text in class_texts]
