# Backbones: provenance and settings

Checksums marked *pending* can only be recorded after the file is downloaded; every run also
stores the SHA-256 of the exact files it loaded in `run.json`.

## A. Frozen CLIP

| Field | Value |
|---|---|
| Code | open_clip 3.3.0 (MIT), a verified-compatible re-implementation of OpenAI CLIP; official OpenAI repo `openai/CLIP` (MIT) |
| Weights | OpenAI CLIP, converted by timm/open_clip: `huggingface.co/timm/vit_base_patch32_clip_224.openai` (`open_clip_model.safetensors`, 605 MB, SHA-256 `e6d1bd7789aa45192b3bf90570a789b478bae1b74ebcce7eddd908e83a2b7c31`, cached here) and `timm/vit_base_patch16_clip_224.openai` (about 0.6 GB, *pending*) |
| Original release | `openaipublic.azureedge.net/clip/models/40d36571…/ViT-B-32.pt`, `…/5806e77c…/ViT-B-16.pt` (the URL path is the SHA-256 of the original file) |
| Training data | OpenAI's 400M image–text pairs (not publicly documented in detail); no action-specific training |
| Architecture | ViT-B/32 or ViT-B/16 image encoder, 12-layer text transformer, **QuickGELU** |
| Inference | frames encoded independently; aggregation as in docs/protocol.md |
| Licence | code MIT; weights MIT (OpenAI CLIP repository) |

**QuickGELU.** `open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")` in
open_clip 3.3.0 builds the model with standard GELU (it only warns), although the OpenAI
weights were trained with QuickGELU. The adapter forces QuickGELU and refuses to run otherwise.

Verified here: weights load offline; QuickGELU is active; on synthetic solid-colour clips the
full pipeline (decode → sample → preprocess → encode → score) labels every clip correctly,
which rules out channel-order and normalisation errors. Measured cost on the Apple M2 CPU:
0.26 s to encode 16 frames with ViT-B/32 (decode time excluded; peak process memory 1.2 GB).

## B. Open-VCLIP

| Field | Value |
|---|---|
| Code | `github.com/wengzejia1/Open-VCLIP` @ `047a817e94b88c02c56fde2f1dd0fbf03f3be638`; no licence file (all rights reserved by default; research use as published), built on PySlowFast (Apache-2.0) and CLIP (MIT) |
| Checkpoints | official Google Drive folder `drive.google.com/drive/folders/1VhwPFESkrr9Ed40yU5NEPIkrGIf6WQ9N`; ViT-B/16 file `openvclip-b16/swa_2_22.pth` (also B/32 and L/14); size and SHA-256 *pending* (fp32 ViT-B/16 weights alone are about 0.6 GB) |
| Also needed | original OpenAI JIT archive `ViT-B-16.pt` (weight patching; about 0.35 GB) |
| Training data | Kinetics-400, all 400 classes, from CLIP initialisation (interpolated weight optimisation + SWA) |
| Architecture | CLIP ViT-B/16 whose self-attention also attends to the previous and next frame's tokens (`expand_temporal_view`, zero-padded at clip ends); no extra parameters |
| Native test configuration | `configs/Kinetics/TemporalCLIP_vitb16_8x16_STAdapter.yaml` + `script/testing/temporal_test_b16/*.sh`: 8 frames, sampling rate 16 at 30 fps target, 3 temporal views (offset sampling) × 1 centre crop, shorter side 224 (bilinear), 28 prompt templates averaged (`TemporalClipVideo.text_prompt`), weight patching `0.5 · CLIP + 0.5 · checkpoint`, views summed (raw logits) |
| Decoding | PyAV (README, UCF scripts) or torchvision (YAML, HMDB script): inconsistent in the official repo |

Adapter (`ovar/backbones/openvclip.py`): imports the official model modules unchanged (only
the PySlowFast registry is stubbed, and `pkg_resources` is shimmed for new setuptools),
rebuilds the patched weights exactly as `tools/test_net.py`, and scores with the same ranking
as the official logits. Checkpoints containing pickled objects are only loaded with
`model.trust_pickle=true`, after you have checked the SHA-256.

Verified here (torch 2.14, CPU): the official code imports; with temporal modelling disabled
it reproduces open_clip's CLIP ViT-B/32 image embeddings (max |Δ| 2.5e-5) and text embeddings
exactly; the temporal model runs. The SlowFast frame sampler and resize/crop replicas match the
official functions in unit tests. **Not verified:** the released checkpoint (not downloaded)
and agreement with the official test script on real videos (`envs/openvclip.md`).

Settings that differ under the common protocol: T = 16 (trained with 8; no weights change,
neighbouring frames become closer in time), 1 view, single prompt, CLIP spatial transform.
Report those results separately from the native ones.

## C. TC-CLIP (optional)

| Field | Value |
|---|---|
| Code | `github.com/naver-ai/tc-clip` @ `8728fe4ca2b27f00e7235ed8f81f59cb933c8ed9`; CC BY-NC 4.0 (non-commercial) |
| Checkpoint | **zero-shot** model trained on Kinetics-400 only (README "Zero-shot action recognition", Drive folder `1gAB3LtNzVerkVerifgJFRL0WciPvrAlR`). Do not use the few-shot or base-to-novel checkpoints (trained on UCF101/HMDB51) or the "(LLM)" variant (FROSTER-rephrased class names) for this benchmark |
| Architecture | CLIP ViT-B/16 with temporal contextualisation tokens and video-conditional prompting (text embeddings depend on the video) |
| Native test configuration | `configs/zero_shot.yaml`: 16 frames, shorter side 256 then centre crop 224, learned prompt context initialised from "a photo of a"; final test uses 2 temporal views built with the training-mode sampler, i.e. randomly jittered offsets |

Because of video-conditional prompting it is not wrapped in-process. Export the manifests with
`python -m ovar export-native --backbone tcclip --out <dir>` and run the official evaluation on a
compatible machine (envs/openvclip.md, TC-CLIP section).

## Comparability

| | Frozen CLIP | Open-VCLIP native | Open-VCLIP common | TC-CLIP native |
|---|---|---|---|---|
| Backbone | ViT-B/16 (and B/32) | ViT-B/16 | ViT-B/16 | ViT-B/16 |
| Frames × views × crops | 16 × 1 × 1 | 8 × 3 × 1 | 16 × 1 × 1 | 16 × 2 × 1 |
| Frames encoded per video | 16 | 24 | 16 | 32 |
| Prompts | 1 template | 28 templates | 1 template | learned + video-conditional |
