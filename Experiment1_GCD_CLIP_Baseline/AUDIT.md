# Audit: repository state, reusable code, required changes

Audited working version: commit `d9a1f05fdfe6195d48612a211913b6964935b92d` (same as the reviewed
commit). The working tree has uncommitted edits only under `action_recognition/`, which is an
unrelated project and is not touched by this experiment. No `AGENTS.md` exists in the repository.

## 1. What exists

| Component | State | Use in Experiment 1 |
|---|---|---|
| `data/imagenet.py` | Upstream GCD. ImageNet-100 = `np.random.seed(0)`, `np.random.choice(range(1000), 100, replace=False)`, sorted; GCD ids 0–99 in that order; known = ids 0–49; labelled = `subsample_instances` (seed 0, proportion 0.5) over the known-class rows in ImageFolder order | Partition rule reproduced in `e1/manifests.py`; cross-checked on Kaggle against the upstream functions (`subsample_classes`, `subsample_instances`, `subsample_dataset`) |
| `data/cub.py`, `data/ssb_splits/cub_osr_splits.pkl` | Upstream. 100 known classes from the SSB split; novel = Hard + Medium + Easy; GCD id = position in `known + novel`; labelled = seed-0 half of known-class training rows in `images.txt` order | Reused through the same manifest builder and cross-checked with `get_cub_datasets` |
| `data/get_datasets.py`, `data/data_utils.py` | Upstream; `MergedDataset` returns true labels for unlabelled images (used only under `mask_lab`) | Replaced by a manifest dataset that returns −1 for unlabelled images, so their labels cannot enter training |
| `methods/contrastive_training/contrastive_training.py` | Upstream GCD training: `SupConLoss` (τ = 0.07), `info_nce_logits` (τ = `--temperature`, default 1.0), two views, loss `(1 − 0.35)·unsup + 0.35·sup`, SGD lr 0.1, momentum 0.9, wd 5e-5 (`bash_scripts/contrastive_train.sh`), cosine schedule to lr·1e-3, blocks ≥ 11 trainable, DINO head 768→2048→2048→256→65536, `WeightedRandomSampler` balancing labelled/unlabelled, batch 128, 200 epochs | Loss functions, view generator, head and backbone are loaded verbatim from the source file by `e1/upstream.py`; the loop is re-hosted for resumable checkpoints |
| `methods/clustering/extract_features.py` | Writes one `.npy` per image under a directory named by the **true label**; loads `*_best.pt` by default | Replaced by sample-id-keyed feature arrays from the final-epoch checkpoint |
| `methods/clustering/k_means.py` + `faster_mix_k_means_pytorch.py` | Semi-supervised k-means (100 inits, 200 iterations in `bash_scripts/k_means.sh`) | Repaired copy in `e1/sskmeans.py` (see §2) |
| `methods/estimate_k/estimate_k.py` | sklearn k-means on all training features, accuracy on the **labelled** subset only, Brent search over K | Same objective in `e1/estimate_k.py` (GPU k-means; deviation recorded) |
| `project_utils/cluster_and_log_utils.py` | `split_cluster_acc_v2`: one Hungarian assignment on the whole population, Old/New from it | Loaded verbatim by `e1/upstream.py`; tested to use a single assignment |
| Root pilot scripts (`extract_*_embeddings.py`, `create_clusters.py`, `generate_*vocabularies.py`, `construct_matrices.py`, `evaluate_*.py`, `risk_coverage.py`) and `embeddings/`, `clusters/`, `experiments/` | **Oracle-cluster pilot**: 2,000 ImageNet validation images (20 per class, from `ILSVRC/imagenet-1k` on Hugging Face), clusters = ground-truth groups of 10, 10-concept paired vocabularies | Kept unchanged and identified as the oracle pilot (`experiments/ORACLE_PILOT.md`). Scoring definitions (mean similarity, margin, MSP at T = 0.01, voting) are carried over |

## 2. Defects found and how Experiment 1 handles them

| # | Where | Defect | Handling |
|---|---|---|---|
| 1 | `faster_mix_k_means_pytorch.fit_mix_once` | An empty cluster's centre becomes the mean of nothing (NaN); NaN distances then capture every point in later iterations | Empty clusters keep their previous centre; every empty event is counted and exported |
| 2 | same | `torch.nonzero(...).squeeze()` on one member gives a 0-d index | Vectorised update with `index_add_`; singletons handled and counted |
| 3 | same | Returned labels (assigned to the old centres) and returned centres (updated after the assignment) come from different states | After the loop, the final centres are used for one final assignment; the exported labels, centres and inertia come from that one state, and any label change is reported |
| 4 | same | Returns `i + 1` where `i` is the last labelled index, not the iteration count | Iterations and convergence are recorded explicitly |
| 5 | `kpp` | If the random draw exceeds the last cumulative probability (rounding), a stale index from the previous step is reused | `searchsorted` with the last index as the bound |
| 6 | `k_means.py` | Correctness depends on the merged loader being labelled-first, because `fit_mix` concatenates `[labelled, unlabelled]` | The adapter passes explicit index arrays and maps results back to sample ids |
| 7 | `contrastive_training.py` | The "best" checkpoint is chosen by Old accuracy on the labelled test set; each epoch also runs k-means on the unlabelled set with its true labels (logging only) | Primary checkpoint = final epoch (prespecified). Per-epoch label-based monitoring is off |
| 8 | `extract_features.py` | Feature files are stored under the true-label directory | Arrays are keyed by sample id |
| 9 | Pilot CLIP extraction | `open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")` with open_clip 3.3.0 (the version now in `.venv`) builds **GELU**; OpenAI weights were trained with **QuickGELU**, and open_clip only warns | Experiment 1 uses `ViT-B-32-quickgelu` and verifies against the official OpenAI TorchScript model. The size of the pilot's error is measured, not assumed |
| 10 | Pilot extraction scripts | Device choice is MPS or CPU only | Explicit `cuda`/`mps`/`cpu` selection with fallback |
| 11 | `data/cub.py` | Both download URLs return 404 | Official Caltech record `data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1`, MD5 `97eceeb1…` (the upstream constant) |
| 12 | `project_utils/*`, `methods/*` | Import `tensorboard` at module level | `e1/upstream.py` executes only the needed definitions from the source files |
| 13 | Upstream partitions | Unlabelled order comes from iterating a Python `set` | Every artifact joins through sample ids; positional order is never relied on |

## 3. Required new pieces

Immutable manifests with separate label tables per permission level; resumable GCD training with
full state; sample-id-keyed GCD and CLIP features; repaired SS-k-means and assignment of held-out
images to fixed centres; vocabulary/prompt tables; shared-vocabulary episodes; R, A, B matrices;
grounding baselines; development, calibration and discovery evaluation; cluster bootstrap; report;
Kaggle workflow; tests. The exact protocol is in `PROTOCOL.md`.
