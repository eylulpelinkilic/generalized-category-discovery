# Status (2026-10-26)

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

Diagnostics ran on 202 development videos (2 per class). The outputs are committed at the repository root
(`../ovar_outputs/`). XOV reports 63.15 (ViT-B/16) and 59.67 (ViT-B/32) for CLIP on UCF101; the
suspected cause (raw vs spaced class names) is tested by the side analysis in the next step.


## Kaggle session 2 (code `2c25661`, outputs in `../ovar_outputs_2/`)

* **Reproducibility.** The ViT-B/16 primary run repeated in a new session gives identical scores
  for all 3,783 videos (max |Δ| = 0), so the accuracy is again 71.16.
* **Side analysis, XOV raw class names** (`common_16x1x1_xovnames`): 64.10 [60.8, 67.5]; familiar
  68.12, unfamiliar 60.13. XOV reports 63.15 (closed 66.71, open 59.53), inside our intervals. The
  gap to XOV is therefore consistent with class-name formatting alone. XOV's code is unpublished,
  so this remains an explanation, not a confirmation. The name change alters 922 of 3,783
  predictions (428 correct→wrong, 161 wrong→correct). Per-class effects reach ±80 pp (UnevenBars
  89→0, FloorGymnastics 6→86). Classes whose own name is unchanged also move (−4.6 pp mean),
  because their competitors' texts changed. The primary setting is unchanged.
* **Open-VCLIP adapter, native 8×3×1, 202-video subset:** 86.14 [80.4, 90.2]; familiar 92.0,
  unfamiliar 80.4. On the same videos frozen CLIP ViT-B/16 (common protocol) gets 74.3. **Not yet
  a Baseline B result:** the official-script comparison did not run, because PyAV 10.0.0 has no
  Python 3.8 wheel (fixed in the notebook: PyAV 12.3.0, pytorchvideo `eb04d1b`, protobuf 3.20.3).
* Checkpoint SHA-256 recorded in docs/backbones.md.

## Kaggle session 3 (code `812a278`, outputs in `../ovar_outputs_3/`)

**Native Open-VCLIP check: the official test script and the adapter agree.** The setting was the
official UCF101 configuration (`script/testing/temporal_test_b16/ucf.sh`: PyAV decoding, 8×16 at
30 fps, 3 views × 1 crop, patching 0.5, fp32), run on the same 202 videos and class texts. The
official environment (torch 1.11.0+cu113, Python 3.8) is frozen in `runs/official_env_freeze.txt`.
The official run loaded all checkpoint keys and scored 606 clips (202 × 3).

| | Top-1 | Top-5 |
|---|---|---|
| Official `tools/run_net.py` | 85.64 | 96.04 |
| Adapter (`ovar infer`) | 86.14 | 95.54 |

* **Top-1 agreement:** 200 of 202 videos (99.0%), with 0 adapter failures. The two
  disagreements are near-ties in the official scores, where the official top-1 and top-2 classes
  differ by 0.39 and 0.12 logits on a scale of about 70–80. These are the 4th-smallest and the
  smallest official margins of the 202 videos.

  | Video | True class | Official | Adapter |
  |---|---|---|---|
  | `Hammering/v_Hammering_g01_c01` | Hammering | ShavingBeard | Hammering |
  | `PommelHorse/v_PommelHorse_g04_c04` | PommelHorse | BoxingSpeedBag | Drumming |
* **Score level.** Official scores are the sum over views of 100 × the mean per-frame cosine.
  Adapter scores are the cosine of the renormalised mean, which gives the same ranking when the
  inputs are identical.
  - Per-video Spearman correlation: median 0.996, minimum 0.968.
  - Top-5 sets identical for 76% of videos.
  - After a per-video scale fit, the median max residual is 0.007 cosine units, against a median
    top-1/top-2 gap of 0.041.
  - Some fitted scale factors exceed 1, so the inputs differ slightly, not only the numerics.
    The cause anticipated in envs/openvclip.md is PyAV selective decoding versus our full
    decoding (frame windows can shift by a frame). It has not been checked frame by frame.
* **Adapter reproducibility.** The adapter matches its session-2 run exactly (max |Δscore| = 0).
* **What this licenses.** The adapter is accepted as Open-VCLIP's native inference for UCF101
  (PyAV). The 202-video accuracies are subset numbers, not Baseline B results; those need the
  full 3,783-video list. HMDB51's official script decodes with torchvision, so HMDB51 needs its
  own check.

## Kaggle session 4 (code `57c722b`, outputs in `../ovar_outputs_4/`): Baseline B on UCF101

The run covered all 3,783 XOV UCF101 test videos with 0 failures. It used the checkpoint SHA-256 from
session 3, and every stage completed. Accuracies (%) were recomputed from `predictions.csv`. The 95%
cluster-bootstrap intervals are over the 707 source groups.

| Model | Protocol | Overall | Familiar | Unfamiliar |
|---|---|---|---|---|
| Frozen CLIP ViT-B/16 (session 1) | common 16×1×1 | 71.16 [68.2, 74.3] | 74.23 | 68.12 |
| Open-VCLIP ViT-B/16, adapter | common 16×1×1 | 83.08 [80.6, 85.5] | 90.06 [87.0, 92.7] | 76.17 [72.0, 80.2] |
| Open-VCLIP ViT-B/16, adapter | native 8×3×1 | 84.46 [82.2, 86.9] | 90.97 [88.4, 93.3] | 78.01 [73.9, 81.9] |
| Open-VCLIP ViT-B/16, official `run_net.py` | native 8×3×1 | 84.19 | 90.91 | 77.54 |

Native and common rows answer different questions and are not to be pooled. XOV's Open-VCLIP
number (83.00) comes from a model retrained by XOV and is a labelled reference only.

**Paired differences on the same videos.** These are cluster bootstrap estimates with 5,000
resamples, computed ad hoc from the saved predictions, not by `ovar evaluate`.

* Open-VCLIP minus frozen CLIP, both on the common protocol:
  - overall +11.92 [9.7, 14.2];
  - familiar +15.83 [12.4, 19.5];
  - unfamiliar +8.05 [5.2, 11.0].

  The gain on familiar classes exceeds the gain on unfamiliar classes by 7.79 pp [3.4, 12.4]. On
  UCF101, Kinetics training helps both groups, but helps familiar classes about twice as much.
* Open-VCLIP native minus common: overall +1.37 [0.2, 2.6], familiar +0.90 [−0.7, 2.5],
  unfamiliar +1.84 [0.1, 3.7].

**Native check on the full list** (`runs/native_check/compare_ucf101_full.json`).

* **Agreement.** Top-1 agreement is 97.54%, with 93 of 3,783 videos disagreeing:
  - 25 where only the official script is right;
  - 35 where only the adapter is right;
  - 33 where both are wrong.

  The net accuracy difference is +0.26 pp for the adapter. 2 of the 202 subset videos disagree,
  as in session 3. A 2.5% rate would give about 5 on 202 videos, and seeing 2 or fewer has a binomial
  probability of about 0.13. The subset's 99.0% is therefore consistent with the full-list rate.
* **The disagreements are near-ties.** The official top-1/top-2 margin at the disagreements has
  a median of 0.0024 cosine units, against 0.039 over all videos. The median disagreement lies at
  the 4th percentile of all margins, and the largest at the 24th.
* **Inputs.** 573 videos (15%) give identical scores in both pipelines (scale-fit residual 0), so
  colour conversion, resize, normalisation and the model agree. The rest differ by a median of
  0.007. This points to different frames being decoded for those videos (PyAV timestamp-based
  selective decoding vs our frame-index decoding), the cause anticipated in envs/openvclip.md. It
  has not been verified frame by frame.
* **Determinism.** On the 202 subset videos, both pipelines reproduce session 3 exactly (max
  |Δ| = 0).
* **Decision.** The adapter's native numbers are reported as Baseline B, with the official
  script's numbers next to them. The agreement rate and the 93 disagreeing videos are reported
  with them.

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
* **Tests:** 39 pass and 2 are skipped (`python -m unittest discover -s tests -t .`). The 2 opt-in heavy tests
  (`OVAR_HEAVY_TESTS=1`) were not run as tests; their checks were run manually once.
* **Category audit.** Lexical screen run; 50 unfamiliar classes listed for review; 15 tagged
  near-familiar in an unverified review, with a secondary metric.

## Implemented, not yet verified

* Open-VCLIP agreement with the official test script on HMDB51 (torchvision decoding); UCF101
  with PyAV decoding agrees on 97.5% of the full list (session 4).
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

Everything except UCF101 with frozen CLIP and Open-VCLIP: HMDB51, ARID and NEC-Drone (so no
cross-domain average yet), Baseline C, Open-VCLIP diagnostics, and actor-crop diagnostics.

## Open items needing a human decision

* Confirm or reject `data/audit/category_review.csv` (second annotator recommended).
* Confirm ARID and NEC-Drone source-group semantics (`*_assumed_*` bases); they only affect
  confidence intervals.

## Next step

Second domain: HMDB51, with `kaggle/ovar_hmdb51_baselines.ipynb` (about 2 hours on a T4; add the
dataset `kodurujagruthaaditya/hmdb51`, because the official archive link is dead):
1. A gate run of 102 videos (2 per class).
2. Frozen CLIP, both backbones, 1 and 28 templates, common protocol.
3. Open-VCLIP adapter, native and common.
4. The official script with torchvision decoding (INSTALL.md conda environment, because the pip
   torchvision 0.12 wheel has no video reader), then `compare-native`.

Bring back `ovar_outputs_5.zip`. Request NEC-Drone access now: it is by request and has lead time.
