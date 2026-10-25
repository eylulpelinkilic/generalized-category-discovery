# Status (2026-10-25)

## Real-data results so far (Kaggle session 1, code `7281db8`)

Frozen CLIP on the full XOV UCF101 test list (3,783 videos, 0 failures; data: Kaggle copy
`lifailifai/ucf101`, every listed file present, 300 decoded in validation). Top-1 % over all 101
classes with 95% cluster-bootstrap CIs; recomputed independently from `predictions.csv`.

| Model | Prompts | Overall | Familiar | Unfamiliar |
|---|---|---|---|---|
| ViT-B/32 | 1 template | 68.17 [65.2, 71.1] | 66.21 | 70.12 |
| ViT-B/32 | 28 templates | 69.89 [66.8, 72.9] | 71.84 | 67.96 |
| ViT-B/16 | 1 template | 71.16 [68.2, 74.3] | 74.23 | 68.12 |
| ViT-B/16 | 28 templates | 74.86 [72.0, 77.8] | 77.21 | 72.54 |

Diagnostics ran on 202 development videos (2 per class). The outputs are kept outside Git
(`ovar_outputs/`). XOV reports 63.15 (ViT-B/16) and 59.67 (ViT-B/32) for CLIP on UCF101; the
suspected cause (raw vs spaced class names) is tested by the side analysis in the next step.


## Implemented and verified on this machine

* **Official sources.** Open-VCLIP `047a817e`, XOV-Action `f90e6f7a`, TC-CLIP `8728fe4c`
  cloned to `third_party/`. Full file trees with blob hashes are in `third_party/*.tree.txt`;
  training-only files removed for disk space are listed in `third_party/PRUNED.txt`.
* **Manifests.** Built from the XOV files for UCF101 (3,783), HMDB51 (1,530), ARID (823),
  NEC-Drone (891), plus a UCF101 development pool (9,537). Validated:
  - labels agree with the paths;
  - membership and labels match the official lists;
  - the closed/open lists partition the full list;
  - class counts equal Table I;
  - no duplicates;
  - every vocabulary class has videos.

  The HMDB51 label-order difference with Open-VCLIP is handled and tested.
* **Sampling and preprocessing.** The TSN midpoint sampler equals the official mmaction
  `SampleFrames` test mode. The SlowFast-dense sampler equals Open-VCLIP's `get_start_end_idx` +
  `temporal_sampling`. `clip_center` equals open_clip's transform, and `slowfast_center` equals
  PySlowFast's resize/crop. Open-VCLIP's 28 prompt templates are verbatim. Each equality is a
  unit test against the pinned source.
* **Baseline A, frozen CLIP ViT-B/32.**
  - Loads offline with QuickGELU (weights SHA-256 `e6d1bd77…`).
  - End to end on synthetic fixtures: all solid-colour clips are labelled correctly, the
    single-frame variant is applied, shuffling is invariant as expected, and failures stay in
    the denominators.
  - Cost: 0.26 s per 16-frame video on the M2 CPU, peak 1.2 GB RSS.
* **Baseline B, Open-VCLIP adapter.**
  - The official model code imports under torch 2.14.
  - With temporal modelling disabled it reproduced open_clip's CLIP ViT-B/32 embeddings
    (max |Δ| 2.5e-5) in a manual check this session.
  - The temporal model runs.
  - Weight patching follows `tools/test_net.py`.
* **Pipeline.**
  - **Metrics:** overall/familiar/unfamiliar, per class, confusion, top-5, unweighted AVG vs
    pooled, cluster-bootstrap CIs.
  - **Runtime:** per-video outputs, caching keyed on checkpoint/preprocessing/sampling/transform/
    file identity, timing and memory.
  - **Analysis:** diagnostics (temporal, actor crop, degradation), error analysis, results report
    (synthetic runs excluded), exports to the official Open-VCLIP and TC-CLIP formats
    (subset-aligned with `infer`), CLI with explicit errors.
* **Tests:** 38 pass and 2 are skipped (`python -m unittest discover -s tests -t .`). The 2 opt-in heavy tests
  (`OVAR_HEAVY_TESTS=1`) were not run as tests; their checks were run manually once.
* **Category audit.** Lexical screen run; 50 unfamiliar classes listed for review; 15 tagged
  near-familiar in an unverified review, with a secondary metric.

## Implemented, not yet verified

* Open-VCLIP with the released checkpoint, and agreement with the official test script
  (`compare-native`).
* Actor-crop diagnostics on real videos (code tested on fixtures with synthetic boxes).
* `audit-categories --method clip`.
* The MPS device path (every run so far used the CPU).

## Blocked

| Blocker | Effect | Resolution |
|---|---|---|
| Local disk: about 2 GB free of 460 GB | no local dataset or checkpoint downloads | runs happen on Kaggle (kaggle/*.ipynb) |
| No NVIDIA GPU locally | the official Open-VCLIP/TC-CLIP test scripts need CUDA | Kaggle GPU (envs/openvclip.md, "On Kaggle") |
| NEC-Drone access is by request; ARID v1 must be confirmed | two of the four domains cannot be prepared from public links alone | request access; validate file names |
| No person detector | spatial diagnostics skip with "not run: no actor boxes" | docs/diagnostics.md |

## Not run

Everything except frozen CLIP on UCF101: Baselines B and C; HMDB51, ARID and NEC-Drone; actor-crop
diagnostics.

## Open items needing a human decision

* Confirm or reject `data/audit/category_review.csv` (second annotator recommended).
* Confirm ARID and NEC-Drone source-group semantics (`*_assumed_*` bases); they only affect
  confidence intervals.

## Next step

Push the current commit, then run `kaggle/ovar_ucf101_openvclip_check.ipynb` on a Kaggle T4
session: the Open-VCLIP native check on the 202-video subset (official `tools/run_net.py` vs the
adapter, `compare-native`) and the CLIP ViT-B/16 raw-class-name side analysis. Bring back
`ovar_outputs_2.zip`.
