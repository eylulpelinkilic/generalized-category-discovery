# Cross-domain open-vocabulary action recognition: baselines and diagnostics

Reproducible baselines for **generalizable open-vocabulary action recognition** on
[XOVABench](https://github.com/KunyuLin/XOV-Action) (XOV-Action, arXiv:2403.01560):
a model trained on Kinetics400 (through an existing checkpoint) predicts one action from
the **full candidate vocabulary** of an unseen target domain (UCF101, HMDB51, ARID,
NEC-Drone). Accuracy is reported for all videos, for **familiar** categories (XOV
closed-set: shares a meaning with a Kinetics400 training category) and for
**unfamiliar** categories (XOV open-set). "Unfamiliar" refers to the task-specific action
training categories only, not to CLIP's pretraining corpus.

This directory is self-contained and independent of the GCD code in the parent repository.
No novel method is implemented here.

| Baseline | Status on this machine |
|---|---|
| A. Frozen CLIP (OpenAI weights via open_clip, QuickGELU) | full UCF101 test list run on Kaggle (ViT-B/32, ViT-B/16) |
| B. Open-VCLIP (official checkpoint, official model code) | full UCF101 test list run on Kaggle, native and common protocol; top-1 agreement with the official test script 97.5% |
| C. TC-CLIP (optional) | official code + exported manifests on a compatible machine |

See **STATUS.md** for what is verified, blocked and not yet run, and
**reports/results.md** for results (none on real videos yet).

## Layout

```
configs/      benchmark definition, models, protocols (native vs common), experiments, diagnostics
data/         manifests + vocabularies built from the official XOV files (committed, small)
docs/         protocol, backbone provenance, data preparation, diagnostics, category audit
envs/         environment for running the official Open-VCLIP / TC-CLIP code
ovar/         package: dataset adapters, backbones, sampling, inference, metrics, diagnostics, reports
reports/      generated results report (real runs only) and hypotheses
tests/        unit tests (several compare against the pinned official source code)
third_party/  official repositories at pinned commits (git-ignored)
runs/, cache/ outputs and embedding cache (git-ignored)
```

## Commands

All commands run from this directory with the Python environment that has the requirements.

### 0. Kaggle (GPU, no local downloads)

`kaggle/ovar_ucf101_clip.ipynb` runs steps 1–6 for frozen CLIP on UCF101: import it into a Kaggle
notebook, enable GPU and Internet, add the dataset `lifailifai/ucf101` (full UCF101, all XOV
test and development-pool videos present), run all cells, download `ovar_outputs.zip`. If the
dataset fails to mount, remove it: the notebook then downloads the official `UCF101.rar` from
crcv.ucf.edu and records its SHA-256 in `runs/DATA_SOURCE.json`.

`kaggle/ovar_ucf101_openvclip_check.ipynb` (second session): the Open-VCLIP native check on the
2-videos-per-class subset (official test script vs adapter, see envs/openvclip.md), and the
frozen-CLIP ViT-B/16 side analysis with XOV's raw class names. Download `ovar_outputs_2.zip`.

`kaggle/ovar_ucf101_openvclip_full.ipynb` (fourth session) runs Baseline B on the full UCF101 test
list: the adapter in the native configuration, the official test script on the same videos with
`compare-native`, and the adapter in the common protocol. Download `ovar_outputs_4.zip`.

`kaggle/ovar_hmdb51_baselines.ipynb` (fifth session) runs frozen CLIP (4 configurations) and
Open-VCLIP (native and common) on HMDB51, plus the official Open-VCLIP script with torchvision
decoding. Add the dataset `kodurujagruthaaditya/hmdb51` and download `ovar_outputs_5.zip`.

### 1. Environment

```bash
python3 -m venv .venv && . .venv/bin/activate        # or reuse an existing environment
pip install -r requirements.txt                      # ffmpeg/ffprobe must also be installed
python -m ovar fetch-repos                           # official repos at pinned commits -> third_party/
cp configs/paths.example.yaml configs/paths.yaml     # then edit dataset roots and checkpoint paths
python -m unittest discover -s tests -t .            # 41 tests; 2 that load CLIP run only with OVAR_HEAVY_TESTS=1
python -m ovar smoke-test                            # end to end on synthetic fixtures (needs CLIP ViT-B/32)
python -m ovar smoke-test --fake                     # same, without any model weights
```

Large downloads are never started implicitly. CLIP weights: `python -m ovar fetch-clip --arch
ViT-B-16 --yes` (about 0.6 GB; checks free disk first). Datasets and Open-VCLIP/TC-CLIP
checkpoints: see docs/data_preparation.md and docs/backbones.md.

### 2. Manifest validation

```bash
python -m ovar build-manifests                       # regenerate data/ from the XOV files (already committed)
python -m ovar validate                              # labels, official split membership, Table I counts, duplicates
python -m ovar validate --check-files --check-decode # + every path exists and decodes (needs the videos)
python -m ovar validate --datasets ucf101 --split dev_train1   # diagnostics development pool
python -m ovar audit-categories --method lexical     # screen official familiar/unfamiliar status vs Kinetics400
```

Reports go to `runs/validation/<dataset>_<split>/` (`report.json`, `failures.csv`).

### 3. Small-subset inference

```bash
python -m ovar infer clip_vitb32__common_16x1x1 --per-class 2               # 2 videos per class, seeded
python -m ovar infer openvclip_vitb16__openvclip_native_8x3x1 --per-class 2 --set model.trust_pickle=true
```

Native Open-VCLIP must first be checked against the official test script on the same subset
(envs/openvclip.md: `export-native`, official `tools/run_net.py`, then `compare-native`).

### 4. Full baseline evaluation

```bash
python -m ovar infer clip_vitb32__common_16x1x1
python -m ovar infer clip_vitb16__common_16x1x1
python -m ovar infer clip_vitb16__common_16x1x1_ens28            # prompt ensemble, separate configuration
python -m ovar infer openvclip_vitb16__openvclip_native_8x3x1     # native configuration
python -m ovar infer openvclip_vitb16__common_16x1x1              # common comparison protocol
python -m ovar evaluate runs/<run_name>                           # recompute metrics (e.g. --n-boot 5000)
```

Each run writes, per dataset: `predictions.csv` (ground truth, prediction, status, top-5,
scores of the true and predicted class, familiar/unfamiliar, model and configuration ids),
`scores.npy` (cosine similarity to every candidate), `classes.csv`, `failures.csv`,
`metrics.json`, `per_class.csv`, `confusion.csv`; plus `run.json` (resolved config,
checksums, environment, timing, memory) and `summary.json` (unweighted domain average and
pooled accuracy, each with 95% cluster-bootstrap intervals).

### 5. Diagnostics (development pool, not the evaluation lists)

```bash
python -m ovar diagnose clip_vitb32__common_16x1x1__ucf_dev
python -m ovar diagnose openvclip_vitb16__openvclip_native_8x3x1__ucf_dev
python -m ovar diagnose <name> --summarise-only                   # recompute comparisons from saved runs
```

Temporal order (ordered / shuffled / single repeated frame), actor crops (needs an actor-box
file, see docs/diagnostics.md) and synthetic degradations (brightness, downsampling, blur).

### 6. Reports

```bash
python -m ovar report                                             # reports/results.md (synthetic runs excluded)
python -m ovar error-analysis runs/<run_name> --diagnostics runs/diagnostics/<name>
```

## Conventions that matter for interpretation

* Labels always come from XOV's label files. Open-VCLIP's HMDB51 index order differs for ten
  classes; reading XOV labels through it would silently mislabel them.
* Familiar/unfamiliar status is the official XOV partition. An unverified review
  (data/audit/category_review.csv) lists officially unfamiliar classes with close Kinetics400
  counterparts; a secondary metric excludes them, as a sensitivity analysis only.
* Failed videos stay in every denominator and count as errors; they are listed per run.
* Native and common-protocol results are reported in separate sections; XOV's published
  numbers appear only as labelled reference values.
* Scores are cosine similarities. They are not calibrated probabilities.
