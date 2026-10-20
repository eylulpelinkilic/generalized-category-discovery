# Data preparation

Nothing here is downloaded automatically. Set each root in `configs/paths.yaml`, then run
`python -m ovar validate --check-files --check-decode`; every missing or undecodable file is
listed in `runs/validation/<dataset>_<split>/failures.csv`.

**This machine currently has about 0.2 GB of free disk space, so none of the steps below can be
run here until space is freed.** Sizes marked "not established" were not verifiable from the
sources read for this setup.

| Dataset | Source | Download size | Layout expected under the root | Evaluation videos |
|---|---|---|---|---|
| UCF101 | crcv.ucf.edu/data/UCF101.php (`UCF101.rar`) or the MMAction2 download script referenced by Open-VCLIP | about 6.5 GB archive (extraction needs roughly twice that temporarily) | `ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi` | 3,783 |
| HMDB51 | serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database (`hmdb51_org.rar`, per-class RARs inside; splits in `test_train_splits.rar`) | about 2 GB | `brush_hair/brush_hair/<clip>.avi` (what extracting each class RAR into a folder of its name gives); set `hmdb51_single_class_folder: true` for `brush_hair/<clip>.avi` | 1,530 |
| ARID (v1) | xuyu0010.github.io/arid.html | not established | `Drink/Drink_10_1.mp4` | 823 |
| NEC-Drone | NEC Labs project page (access by request) | not established | `NEC-Phantom-001_A0001_P0014_V0001.mp4` (flat) | 891 |

XOV-Action uses ARID **v1** names (`aridv1_splits`); check that your download is v1 (not
v1.5) by running the validator, which reports every listed clip that is missing.

## Development pools for diagnostics

* UCF101: `data/manifests/ucf101_dev_train1.csv`, from the official `trainlist01.txt` (9,537
  videos; disjoint groups from the test list). Already built.
* HMDB51 / ARID / NEC-Drone: use the official training splits that come with each dataset
  (HMDB51 `*_test_split1.txt` entries marked 1; ARID split-1 train list; NEC-Drone train list).
  Add an adapter in `ovar/benchmark.py` once the files are available; keep them disjoint from
  the evaluation lists.

## Derived storage

| Item | Size |
|---|---|
| Manifests and vocabularies (committed) | 3.5 MB |
| Embedding cache, 16 frames × 512-d float32 per video | 32 KB per video, about 0.23 GB for the four evaluation lists per configuration |
| Embedding cache, Open-VCLIP native (3 × 8 frames) | 48 KB per video, about 0.34 GB per configuration |
| Diagnostics on the UCF101 dev subset (2 per class, 9 variants) | about 0.06 GB per backbone |
| Per-run predictions and scores | about 1 MB plus 0.4 MB of scores per dataset |

## Compute estimate

Measured on this Apple M2 (CPU): 0.26 s to encode 16 frames with ViT-B/32. ViT-B/16 costs
roughly 4× more per frame, so the 7,027 evaluation videos need on the order of 0.5 h (B/32) or
2 h (B/16) of encoding on this CPU, plus decoding (ffmpeg decodes each uncached video twice:
once to count frames, once to extract the sampled ones). These are estimates from one
measurement, not benchmarks; `run.json` records actual times.
