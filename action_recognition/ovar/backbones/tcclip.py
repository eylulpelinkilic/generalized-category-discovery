"""Baseline C (optional): TC-CLIP. Not runnable in-process yet.

TC-CLIP's text side is video-conditional (VP module + learned context), so its
class embeddings cannot be computed independently of each video and do not
fit the frozen-text interface used here. The supported route is the official
evaluation code on a compatible machine, fed with manifests exported by
`python -m ovar export-native --backbone tcclip`.
"""
from ovar.backbones.base import BackboneUnavailable

MESSAGE = (
    "TC-CLIP is evaluated with its official code (github.com/naver-ai/tc-clip @ 8728fe4c). "
    "Use the zero-shot Kinetics-400 checkpoint only (not the few-shot or base-to-novel ones, which are "
    "trained on UCF101/HMDB51). Export annotation files with `python -m ovar export-native --backbone "
    "tcclip` and follow docs/backbones.md#tc-clip."
)


def build(*_args, **_kwargs):
    raise BackboneUnavailable(MESSAGE)
