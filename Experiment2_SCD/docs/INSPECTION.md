# Inspection: the SCD paper, the official code and the Experiment 1 artifacts

"SCD" is the paper's term (Semantic Category Discovery) and also our folder name (`SCD/` = our
re-implementation; this folder = the experiment). Inspected 2026-11-10/12.

## 1. Sources

| Source | Version |
|---|---|
| Paper | arXiv:2304.02364 **v2** (last revised 27 Jul 2024), CVPR 2024 Workshops (CVinW) |
| Official code | <https://github.com/Visual-AI/SCD>, commit **`2ce64f78b2fef35af9c397e45efb125bafc6abe2`** (2024-08-30), licence CC BY-NC-SA 4.0 (README) |
| Official assets (Google Drive links in the README) | `zeroshot_weights.zip` 151,216,331 B (sha256 `99ba7c38…76f5f4`); `GCD_pretrained_weights_VIT16.zip` 1,916,396,688 B; `extracted_features.zip` 9,037,041,820 B (listed via HTTP range requests, not downloaded) |

Files read: `main_unsup.py`, `main_ptsup.py`, `local_utils/clip_lang_util.py`, `local_utils/sskm_constrained.py`,
`local_utils/k_means_constrained/` (vendored package), `gcd/project_utils/{cluster_utils,cluster_and_log_utils}.py`,
`gcd/data/*.py`, `scripts/evaluate_{unsupervised,partial_supervised}.sh`, `README.md`, `requirements.txt`.
There is no separate evaluation script: evaluation is inline in the two `main_*.py` files.
No AGENTS.md exists in this repository.

## 2. The algorithm as the code implements it

| Step | Official code (commit 2ce64f7) | Paper |
|---|---|---|
| CLIP | `clip.load("ViT-B/16")` on CUDA (fp16 weights) | ViT-B/16 |
| Preprocessing | `test_transform = preprocess` (CLIP's resize-224 bicubic, centre crop, CLIP mean/std) for **every** feature model, including DINO and GCD | not stated |
| Initial clustering, unsupervised | `sklearn.cluster.KMeans(n_clusters, random_state=0)` on L2-normalised DINO features of the **unlabelled GCD split** (sklearn 1.0.2: `n_init=10`, `algorithm="auto"`→elkan) | k-means on DINO features |
| Initial clustering, partially supervised | CSS-k-means (`sskm_constrained.py`): labelled anchors fixed; unlabelled points assigned by min-cost flow with cluster sizes in [50, 1000] counted over **unlabelled** points only; arc cost `round(1000·‖x−c‖)` (Euclidean, not squared); `n_init=10`, `max_iterations=10`, `random_state=None` | CSS-k-means with a "loose" minimum size; values not given |
| Vocabulary | `get_nouns()` reads `/disk/work/xhhuang/.../wordnet_all_noun.txt` (not distributed); entries normalised with `lower().replace('-','_')` | "68k concepts" of WordNet |
| Text weights | `zeroshot_weights_all_nouns_vit_b_16.pt`, built by `zeroshot_classifier`: 80 OpenAI ImageNet templates, mean of normalised embeddings, renormalised | Eq. 1 uses the noun embeddings; no templates mentioned |
| Per-image names | logits `100·v̂·T` over all nouns (unsup: softmax, monotone), `topk(TOP_K)`; unsup `TOP_K=args.topk`; ptsup `TOP_K=5` hard-coded, then sliced `[:, :args.topk]` | nearest neighbour, Eq. 1 (no top-k) |
| Votes | per cluster `Counter` of members' top-k noun indices (unsup slices `[:, :5]` of a top-`topk` tensor: no effect) | "m-NN" frequencies (Sec. 3.2.3); "top-k voting" (Fig. 2); m not given |
| Candidate pool | union of each cluster's `most_common(num_common_vote)`, as `list(set(...))` | not stated |
| Distinct names | `assign_name`: D×D count matrix (D = max(pool, clusters)), each cluster contributes only its `most_common(num_common_linear)`; `linear_assignment(w.max() − w)` with sklearn's Munkres; names = first `num_unlab_classes` rows | Hungarian, "cost equal to the proportion of instances" (sign not stated) |
| Reassignment | every unlabelled image to `argmax` over the selected names' weights (`nouns.index(name)` → first column of that name) | re-cluster with the linear classifier |
| Stopping | `while set(cur_voted_names) != set(prev_voted_names)`; no iteration limit | "until convergence" |
| Known names (ptsup) | `lab_names` = names of the labelled classes; excluded from votes; `cand_names = sorted(set(voted + lab_names))`; slots `n_cluster − num_lab` | the known categories reduce the search space |
| sACC | `sACC_all` = image-weighted exact string match (paper Eq. 3); `sACC_avg` = class-averaged | Eq. 3 |
| Soft-sACC | per image `max lch_similarity` over all synset pairs whose normalised first lemma is the predicted / target name; divided by **the largest score observed in the evaluated subset** (All, Old and New each normalised separately) | Eq. 4, "dividing with the maximum possible score" |
| ACC | `split_cluster_acc_v2` (one Hungarian assignment, Munkres) | Eq. 5 |
| ImageNet target names | `wnid_to_name[wnid]` = normalised first lemma of the class synset | — |

Parameters. Scripts: `--topk 3 --num_common_vote 10 -num_common_linear 2`. Argparse help text: ImageNet 5/20/4,
CUB 3/10/2, Stanford Dogs 2/5/2 (unsup) or 1/5/2 (ptsup). Defaults: 5/20/4. The paper gives none of these.

## 3. Discrepancies and defects

**Paper vs code**
1. The paper describes nearest-neighbour naming without prompts; the code uses top-k votes over an 80-template ensemble.
2. The voting depth m, the pool size and the Hungarian sign are unspecified in the paper; the code uses the values in §2.
3. Soft-sACC: the paper normalises by the maximum possible score; the code by the maximum observed in each subset
   (identical whenever the subset contains at least one exact match; otherwise the code's value is inflated).
4. The paper's claimed gains are loosely stated (e.g. "roughly 50% improvement"; "can usually aid clustering
   accuracy" while Table 2 shows lower ACC on Stanford Dogs and CUB); Table 4 reuses Table 2's GCD baselines.

**README vs code / paper**
5. Both evaluation commands point to `script/evaluate_unsupervised.sh` (folder is `scripts/`; the partially supervised
   script is never referenced).
6. Both scripts pass `-num_common_linear 2` (single dash). Argparse rejects it (`unrecognized arguments`, checked with
   the same option set), so **the published scripts do not run as written**; the published numbers were not produced
   by these exact commands.
7. `evaluate_partial_supervised.sh` uses `--feat_model dino_vit`; the paper's partially supervised setting uses GCD
   features.
8. README tables: the CUB Soft-sACC columns repeat the Stanford Dogs values (the paper reports no CUB Soft-sACC);
   README Table 2 lists CUB sACC 35.3 for both "Ours" and "Ours w/TE" (paper: 28.2 and 35.3); README Table 2's ImageNet-100
   ACC 78.7 is the paper's *CSS-k-means* row (the paper's semantic-naming ACC is 83.0).
9. `requirements.txt` omits `nltk`; `k_means_constrained` is vendored with prebuilt Linux/py3.9 binaries.

**Code defects (each is an explicit option in `SCD/scd/naming.py` or `scd/clustering.py`)**
10. *Known-name filter in the wrong index space (ptsup).* From iteration 2, `known_name_idx = lab_class_index` holds
    positions in `cand_names` but is compared with vocabulary indices: known names are no longer excluded from votes
    and the vocabulary entries 0..~99 ('entity', 'physical_entity', …) are excluded instead. Option `known_filter`.
11. *Duplicate columns.* 712 normalised names occur twice: 707 case variants (e.g. 'Dalmatian'/'dalmatian',
    'Cardigan'/'cardigan') whose stored vectors are identical (CLIP's tokenizer lower-cases), and 5 hyphen/underscore
    spellings with different vectors (cosine 0.974–0.994, e.g. 'knife_edge'/'knife-edge'). Votes for one name split
    across two indices; both copies can be assigned to different clusters, giving duplicate names and an empty
    cluster after reassignment; in ptsup only the first copy of a known name is filtered, so the second copy of a known
    name (two ImageNet-100 known classes are affected: 'dalmatian', 'cardigan') can be voted. Option
    `merge_duplicate_names` (repair; off = official).
12. *Insufficient candidates.* If the pool is smaller than the number of clusters, a cluster row within the slots
    receives a padding column and `newidx_to_unameidx[x[1]]` raises `KeyError`. Repair: deepen the pool (recorded).
13. *Padding rows name clusters.* When fewer clusters than slots exist (empty clusters after reassignment), the
    padding rows of the square matrix still receive names (zero support); the code keeps them. Reproduced; counted.
14. *No iteration limit / cycles.* The loop can alternate between name sets forever. Repair: `max_iter` and cycle
    detection (recorded as `stop_reason`).
15. *Interpreter-dependent order.* The pool is `list(set(...))` (CPython set order depends on insertion sequence and
    table growth); ptsup iterates clusters in `set(cand_names) − set(lab_names)` order (string hashing, randomised
    per process). Both only affect Hungarian tie-breaking; we keep `list(set())` under the running interpreter and
    replace the string-set order by ascending positions.
16. *Precision.* Logits are fp16 on GPU (resolution ≈ 0.016 at scale 100), so near-ties are broken by the CUDA top-k
    kernel. We compute in fp32 with lowest-index ties, and run an fp16-emulation variant.
17. *CSS-k-means:* unseeded (`random_state=None`), reports `i + 1` from an unrelated loop as the iteration count, and
    computes the mean of an empty cluster (NaN) if one ever occurs (prevented by size_min ≥ 1).
18. *Feature extraction:* with `--feat_model dino_vit`, a missing `clip_<dataset>_all.pt` is extracted by calling the
    CLIP model's `forward` (which needs text) and crashes; the shipped features contain no ImageNet-100 CLIP features
    (`clip_imagenet_100_all.pt` is absent from `extracted_features.zip`).

## 4. The vocabulary, recovered and verified

The noun list is not distributed, but its embeddings are: `zeroshot_weights_all_nouns_vit_b_16.pt`, fp16 [512, **67,893**].
Candidate rules (WordNet 3.0, nltk 3.10.4, 82,115 noun synsets):

| Rule | Size |
|---|---|
| unique **raw (case-sensitive) first lemma of every noun synset**, first-occurrence order | **67,893** ✔ |
| same, `-`→`_` | 67,889 |
| lower-cased first lemma | 67,186 |
| official normalisation (`lower`, `-`→`_`) | 67,181 |
| synsets with `hypernyms()` only | 61,392 |

The embedded text is the raw lemma with underscores ("a photo of a physical_entity."): on a seeded sample, the
80-template weights recomputed with open_clip ViT-B/16 (QuickGELU, fp32) have minimum cosine **0.9996** with the stored
columns under first-occurrence order, against 0.63 under sorted order and 0.93 with underscores replaced by spaces.
`python -m e2 vocab` repeats this on 300 seeded positions plus duplicate columns (382 positions in all; minimum
cosine **0.99959**), runs a one-template retrieval check over all 67,893 positions (own column retrieved for
**99.97%**, top-5 99.996%) and a sorted-order negative control (minimum cosine 0.49, fails as expected), and refuses to
continue if the exact check fails (`outputs/vocabulary/*/alignment.json`; run on 2026-11-12, passed). The `all_name_desc` file in the same zip has 82,114 columns (= noun synsets
with a hypernym); it is not used. The 100 ImageNet-100 target names are all in the vocabulary; two of them
('dalmatian', 'cardigan') have duplicate columns (item 11).

## 5. Checkpoints and features

* `GCD_pretrained_weights_VIT16.zip` holds six 343,247,509-byte backbones; `imagenet100_model_best.pt` (CRC32 `696b229e`)
  is extracted alone by range requests (`e2/assets.py`).
* `extracted_features.zip` lists `dino_vit_imagenet_100_all.pt` and `gcd_imagenet_100_all.pt` (586 MB each) but no
  CLIP ImageNet-100 features. The pickles store features, masks and targets without sample ids, so reusing them would
  need a row-order check against recomputed features; protocol A recomputes all three feature sets instead.
* ImageNet-100 split: SCD's `get_imagenet_100_datasets` is identical to the repository's upstream GCD function
  except for an added `class_map` (checked function by function); Experiment 1's manifest replays that partition.
  SCD changes the CUB loader (targets not remapped to the SSB order), which matters only for CUB.

## 6. Experiment 1 artifacts used (verified from the files)

| Artifact | Evidence |
|---|---|
| `experiments/cub/grounding/seed0_known_c0/vitb32_831109500c95` | `status.json`: completed 2026-11-09, commit `811ebfb`, T4; only `__pycache__` was untracked. The current Experiment 1 configs recompute the same hash, so the run identity is current. |
| `experiments/cub/clip/vitb32_a2a00e25af6e/index.json` | cache identity `612879b237b9` (ViT-B-32-quickgelu, OpenAI weights, open_clip 3.3.0, fp32) |
| `experiments/cub/gcd/seed0_3a2d202f8444/clustering/known_c{0,1,2}_fcf46ef1931c` | completed SS-k-means runs (assignments, held-out assignments) |
| `selection.json` | dev-selected prompt `ensemble80`, T* = 0.02 |

The results zip excluded image embeddings and `R_*.npy`. Protocol B recomputes the embeddings for exactly
Experiment 1's 8,878 rows and accepts them only if the cache identity hash equals `612879b237b9`, the discovery
aggregates match the saved `A`, and Experiment 1's saved predictions are reproduced (`outputs/cub/inputs/*/validation.json`).
The README of Experiment 1 still says the research runs are pending; the artifacts above show the CUB seed-0 runs
completed. ImageNet-100 Experiment 1 runs do not exist locally.
