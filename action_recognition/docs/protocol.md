# Experimental protocol

Sources: XOV-Action paper (arXiv:2403.01560v3, TPAMI 2026; local copy read in full) and the
official benchmark repository `KunyuLin/XOV-Action` at commit
`f90e6f7ab2f06beb19d09450f59811bcdfbef131` (file hashes in `data/manifests/*.meta.json`).

## Task

Training: labelled Kinetics videos (here: an existing Kinetics400-trained checkpoint); action
names are encoded by a pretrained text encoder. Testing: videos from an unseen target domain
and that domain's full candidate vocabulary, which contains the correct action; the model
predicts one action. No rejection, abstention or "unknown" output is part of this task.

## Source datasets and training categories

| Dataset | Categories | Use here |
|---|---|---|
| Kinetics400 | 400 (`labels/Kinetics400_labels.csv`) | training set of every checkpoint used |
| Kinetics150 | 150-class subset; contains every closed-set category of the four target domains | **not used**: no compatible checkpoint; our K400 results are not K150 results |

## Target datasets and evaluation lists

| Domain | Gap (paper) | Evaluation list (XOV) | Videos | Classes | Familiar classes / videos | Unfamiliar classes / videos | Source groups |
|---|---|---|---|---|---|---|---|
| UCF101 | small | `ucf_splits/video_testlist01.txt` (= official split-1 test list) | 3,783 | 101 | 50 / 1,882 | 51 / 1,901 | 707 |
| HMDB51 | moderate | `hmdb_splits/video_testlist01.txt` (= split-1 test, 30 per class) | 1,530 | 51 | 33 / 990 | 18 / 540 | 478 |
| ARID | large (dark) | `aridv1_splits/video_testlist01.txt` | 823 | 11 | 6 / 493 | 5 / 330 | 16 (assumed) |
| NEC-Drone | large (drone) | `necdrone16_splits/NEC_Drone_16_video_valtestlist.txt` (val + test) | 891 | 16 | 7 / 434 | 9 / 457 | 36 (assumed) |

Class counts equal the paper's Table I. `python -m ovar validate` checks every row against
the official lists: closed and open lists are disjoint and together equal the full list, each
label agrees with the class encoded in the path (folder name, or NEC-Drone action code
A0001–A0010 → 0–9, B0001–B0006 → 10–15), and every vocabulary class has videos.

Candidate labels are the full label files: `ucf_101_labels.csv`, `hmdb51_labels_corrected.csv`,
`arid_v1_labels.csv`, `necdrone16_labels.csv`. The text fed to the encoder is
`data/vocab/<dataset>.csv:text_name`: XOV's names, except UCF101, whose CamelCase names are
replaced by Open-VCLIP's natural-language names (`zs_label_db/ucf101-index2cls.json`,
id-aligned and checked automatically).

Side analysis `common_16x1x1_xovnames` (`prompts.class_names: official_name`) uses XOV's raw
label names instead; only UCF101 changes (77 of 101 classes). It tests whether name formatting
explains the gap to XOV's reported CLIP numbers (XOV's evaluation code is unpublished). Its
results appear under "Side analyses" in the report and never replace the primary setting.

## Closed-set (familiar) and open-set (unfamiliar) categories

Paper, Sec. IV-B: closed-set categories "share similar meanings and have a common lexicon with
the categories in the training domain"; open-set categories are the rest, identified against
the Kinetics400 categories (Table I caption). The status is a property of the ground-truth
class; recognition always uses the full vocabulary. Evidence that the paper does the same: in
Table III the reported All accuracy equals the video-weighted mean of Closed and Open for the
CLIP row on HMDB51 (44.27), ARID (27.69) and NEC-Drone (16.59), which would not hold if the
candidate set had been restricted per subgroup (UCF101 differs by 0.05 pp).

The official status is used unchanged. `docs/category_audit.md` lists officially unfamiliar
classes that have close Kinetics400 counterparts (for example UCF101 JumpRope vs "skipping
rope"); they feed a secondary sensitivity metric only.

## Metrics

Per domain: top-1 accuracy over all videos ("All"), over familiar-class videos ("Closed") and
over unfamiliar-class videos ("Open"); AVG is the unweighted mean of the four domain
accuracies (checked against Table III: e.g. Open-VCLIP All 83.00, 54.24, 42.67, 18.81 → 49.68).
Added here: pooled accuracy over all videos (reported separately from AVG), top-5, mean class
accuracy, per-class accuracy with counts, confusion matrices, 95% intervals from a percentile
bootstrap that resamples source groups (UCF101 class×group, HMDB51 source title; ARID and
NEC-Drone groupings are inferred from file names and deliberately coarse), inference time and
memory.

## Inference settings

| Setting | XOV-Action paper (Sec. V-A) | Common protocol here (`common_16x1x1`) |
|---|---|---|
| Frames | T = 16 | 16, midpoint of 16 equal segments (mmaction2 test-mode `SampleFrames`, as in ViFi-CLIP/TC-CLIP) |
| Views | one temporal clip, centre-cropped frames | 1 clip × 1 centre crop |
| Spatial | 224 × 224 | CLIP test transform: shorter side 224 (bicubic), centre crop 224 |
| Prompt | "a video of a person [doing something]." | `a video of a person {}.` (28-template ensemble is a separate configuration) |
| Aggregation | not stated for baselines | L2-normalise frames, average, renormalise, cosine with normalised class embeddings |
| Backbone | ViT-B/16 for Kinetics400 models | ViT-B/16 for every model (ViT-B/32 also provided) |

Native configurations are documented in `docs/backbones.md` and reported separately.

## Differences from the paper

1. **Checkpoints.** The paper retrained each compared model on Kinetics400 with T = 16 and the
   same recipe. We use the official released checkpoints (Open-VCLIP: 8 frames). Our numbers are
   not reproductions of Table III.
2. **Model selection.** Tables II/III select, for every model, "the checkpoint with the best
   average overall ACC" over the four test domains, which is selection on target test data. We
   do no selection on target data: checkpoints are the released ones; prompts, frame counts
   and thresholds are fixed in advance; no thresholds exist in this task.
3. **Preprocessing.** XOV's code is not released, so frame positions, resize interpolation and
   prompt handling for the baselines are inferred (ViFi-CLIP / Open-VCLIP lineage) or chosen
   here and stated above.
4. **Development data.** Diagnostics use the UCF101 split-1 training list (disjoint from the
   evaluation list). HMDB51, ARID and NEC-Drone development pools need their official training
   splits, which ship with the datasets.
5. **Uncertainty.** The paper reports no intervals; we add cluster-bootstrap intervals.
6. **Decoding.** ffmpeg (presentation-order frame indices, rotation metadata ignored). The
   official code uses PyAV/torchvision/decord; pixel values can differ slightly.
