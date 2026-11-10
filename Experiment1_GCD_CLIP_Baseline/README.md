# Experiment 1: GCD discovery + frozen CLIP grounding baseline

Real generalized category discovery (the repository's GCD method) finds clusters; a separate
frozen CLIP model names each cluster from a candidate vocabulary, or rejects it (UNMAPPED), under
complete and incomplete vocabularies. Benchmarks: full ImageNet-100 (primary) and CUB-200-2011.

* `AUDIT.md`: repository audit, upstream defects found and how they are handled.
* `PROTOCOL.md`: the exact, prespecified protocol (splits, label permissions, settings, metrics).
* `HANDOFF_MEANSHIFT.md`: the exported artifacts for the later cosine mean-shift experiment.
* `../experiments/ORACLE_PILOT.md`: the earlier 2,000-image oracle-cluster pilot, kept unchanged.

## Status

| Part | State |
|---|---|
| Code for every stage (data → GCD training → features → SS-k-means/K estimate → CLIP → matrices → episodes → grounding → calibration → evaluation → report) | implemented |
| Tests (`python -m unittest discover -s tests -t .`) | 25 tests pass locally (CPU, synthetic data and miniature dataset layouts) |
| Smoke run (`configs/smoke.yaml`) | passes end to end; synthetic images and random weights, **software check only** |
| ImageNet-100 and CUB research runs | **not run**: need a GPU and the datasets (Kaggle notebooks in `kaggle/`) |

Nothing in this directory reports a research result yet. The report generator marks every stage as
completed or pending from the files on disk.

## Commands

From this directory, with Python ≥ 3.10, PyTorch, torchvision, `open_clip_torch==3.3.0`, pandas,
scikit-learn, scipy, PyYAML (and `git+https://github.com/openai/CLIP.git` for `verify-clip`):

```bash
E1="python -m e1 --config configs/cub.yaml --set dataset.root=/data/cub --set gcd_train.dino_checkpoint=/w/dino_vitbase16_pretrain.pth"

$E1 prepare-data --download /data/cub --upstream-check   # manifest + label tables; replays the upstream partition
python -m e1 --config configs/cub.yaml fetch-dino --dest /w # 343,242,485 bytes, sha256 recorded
$E1 benchmark --train-seed 0 --iters 20                     # throughput/memory, fp32 and AMP
$E1 train --train-seed 0 [--time-budget-hours 10.5]         # resumes from checkpoints/last.pt
$E1 extract-gcd --train-seed 0                              # final-epoch CLS features by sample id
$E1 cluster --train-seed 0 --k-mode known --cluster-seed 0  # SS-k-means + v2/NMI/ARI evaluation
$E1 cluster --train-seed 0 --k-mode estimated --cluster-seed 0
$E1 verify-clip --clip vitb32                               # open_clip QuickGELU vs official OpenAI package
$E1 extract-clip --clip vitb32
$E1 ground --train-seed 0 --k-mode known --cluster-seed 0 --clip vitb32
$E1 report                                                  # runs/<dataset>/report/BASELINE_REPORT.md
$E1 export-diagnostics --train-seed 0 --k-mode known --cluster-seed 0 --clip vitb32 --population dev --n 50 --seed 42
                       # 50 incorrectly + 50 correctly clustered images with similarity context, under
                       # <grounding run>/diagnostics/clustering_examples/ (+ .zip); see that folder's README.md
```

For ImageNet-100 use `configs/imagenet100.yaml` with `dataset.root` pointing at a directory that
contains `ILSVRC/Data/CLS-LOC/{train,val}`, `LOC_synset_mapping.txt` and `LOC_val_solution.csv`
(the Kaggle competition `imagenet-object-localization-challenge`). Software check on any machine:

```bash
python -m e1 --config configs/smoke.yaml smoke
```

## Kaggle (T4)

`kaggle/e1_cub.ipynb` and `kaggle/e1_imagenet100.ipynb` (regenerate with
`python kaggle/make_notebooks.py`). Each run restores the previous session's `e1_runs` when that
output is attached, benchmarks once, trains within a time budget with a full checkpoint every
epoch, and runs every downstream stage once training reaches its final epoch. Batches are 128
images (256 views); there is no gradient accumulation, so the contrastive negative set is exactly
the upstream one. Change `TRAIN_SEED` to 1 and 2 for the other two training seeds.

What a full run needs that this machine does not have:

| Stage | Needs |
|---|---|
| ImageNet-100 manifest | the ILSVRC-2012 CLS-LOC train and val images: join `imagenet-object-localization-challenge` on Kaggle and attach it |
| CUB manifest | internet (the notebook downloads the official archive) |
| GCD training, feature extraction, CLIP extraction | a CUDA GPU; the benchmark stage gives the measured hours |
| `verify-clip` | internet for the OpenAI weights and the official `clip` package |

## Outputs

`runs/<dataset>/` (git-ignored): `data/` (immutable manifest, separate label tables, checks),
`gcd/seed<S>_<hash>/` (checkpoints, training log, run record, features, clustering per K mode and
clustering seed), `clip/<model>_<hash>/` (embedding shards, model description, reference check),
`grounding/seed<S>_<K>_c<C>/<model>_<hash>/` (matrices, episodes, selections, thresholds,
predictions, metrics, bootstrap, failure cases), `report/BASELINE_REPORT.md`. Directory names
carry the hash of the configuration that produced them; paths and loader settings are excluded
from the hash so runs resume across machines.
