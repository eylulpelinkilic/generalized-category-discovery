# Experiment 1 protocol: GCD discovery + frozen CLIP grounding baseline

Every choice below is fixed before any research run. Configuration values live in
`configs/*.yaml`; each output directory stores the resolved configuration and its hash.

## 1. Data, identities and label permissions

| Dataset | Discovery population (GCD train split) | Known / novel classes | Labelled anchors | Held-out labelled images (extra annotation budget) |
|---|---|---|---|---|
| ImageNet-100 (primary) | all ILSVRC-2012 training images of the 100 upstream classes | 50 / 50 (GCD ids 0–49 / 50–99) | upstream rule: seed-0 sample of 50% of known-class training rows | ILSVRC-2012 validation images of the 50 known classes (50 per class) |
| CUB-200-2011 | all official training images (5,994) | 100 / 100 (SSB split) | same rule | official test images of the 100 known classes |

* Sample ids: `imagenet:train:<file stem>`, `imagenet:val:<file stem>`, `cub:<img_id>`.
* `manifest.csv` holds identities, relative paths, source split, GCD membership
  (`labelled`, `unlabelled`, `heldout_dev`, `heldout_calib`, `heldout_unused`) and dataset version.
  It contains **no labels**. Label tables are separate, one per permission level:

| Table | Rows | May be used for |
|---|---|---|
| `labels_labelled.csv` | labelled anchors | GCD training, SS-k-means anchors, K estimation |
| `labels_heldout.csv` | held-out known-class images | grounding development (dev pool) and threshold calibration (calibration pool) |
| `labels_eval.csv` | unlabelled discovery images | evaluation only |

* Held-out pools: per known class, the held-out images are shuffled with seed 0 and split in
  half (dev gets ⌊n/2⌋). Held-out images of novel classes are `heldout_unused` and never read.
* Class-selection and labelled-split seeds are the upstream ones (0) and never vary.

## 2. Discovery (GCD)

* Backbone ViT-B/16 initialised from `dino_vitbase16_pretrain.pth` (343,242,485 bytes); blocks
  ≥ 11 trainable; DINO projection head (3 layers, 65,536 outputs, last-layer weight norm frozen).
* Losses, view generator and augmentation are the upstream ones: two views, `imagenet` transform
  (resize 256 bicubic, random crop 224, flip, ColorJitter()), unsupervised InfoNCE over all 256
  views of a batch (τ = 1.0) and SupCon over labelled views (τ = 0.07), weight 0.35 on SupCon.
* SGD lr 0.1, momentum 0.9, weight decay 5e-5, cosine annealing to 1e-4 over 200 epochs;
  batch 128 images (256 views, one contrastive negative set per step; no gradient accumulation);
  `WeightedRandomSampler` with labelled weight 1 and unlabelled weight n_l/n_u.
* Unlabelled images enter training with label −1. Primary checkpoint: **final epoch (200)**.
* The projection head's random initialisation is seeded from the training seed (upstream defines
  `--seed` but never uses it). A batch without labelled images (practically impossible at batch
  128 with balanced sampling) contributes no SupCon term instead of dividing by zero.
* Training seeds 0, 1, 2 (seed 0 is primary). Each epoch's sampler and augmentation streams
  are seeded from (training seed, epoch), so a resumed run reproduces the same epoch inputs.
* Features: CLS token of the backbone with the upstream test transform (resize 256, centre crop
  224), L2-normalised before clustering.
* SS-k-means: K = number of classes (**known-K assumption**: 100 / 200); labelled anchors fixed
  to their class's cluster; k-means++ seeded with the labelled class means; 100 initialisations,
  ≤ 200 iterations, tolerance 1e-4 on the squared total centre shift; best final inertia.
  Clustering seeds 0, 1, 2 for each training seed; clustering seed 0 is primary.
* Estimated-K condition: upstream objective (accuracy of plain k-means on the **labelled** subset,
  all training features clustered), Brent search over K ∈ [n_known, 1000], then SS-k-means with
  the rounded estimate.
* GCD evaluation on the unlabelled population: upstream `v2` (one Hungarian assignment; All, Old,
  New from it), NMI, ARI, cluster sizes. The Hungarian map is used for nothing else.
* Held-out images are assigned to the fitted centres (nearest centre, same distance) without
  refitting; assignments form the dev and calibration groups.

## 3. CLIP representation

* Primary: OpenAI CLIP ViT-B/32 through `open_clip_torch==3.3.0` as `ViT-B-32-quickgelu`,
  `pretrained="openai"`, eval mode, fp32, OpenAI preprocessing (resize 224 bicubic, centre crop,
  OpenAI mean/std). Verified against the official OpenAI TorchScript checkpoint on real images and
  prompts before use. Robustness: ViT-B/16 the same way.
* Image and text embeddings are L2-normalised and cached by sample id.
* Vocabulary: one column per class, `class_id` = GCD id. ImageNet canonical name = first WordNet
  lemma of the synset (pilot convention); other lemmas are recorded as aliases. CUB canonical
  name = `classes.txt` name with underscores replaced by spaces.
* Prompt strategies: `single` = "a photo of a {name}." (pilot); `ensemble80` = OpenAI's 80
  ImageNet templates on the canonical name, averaged after normalisation, renormalised;
  `mcm` = "a photo of a {name}" (MCM's template). Removing a class removes every text of it.

## 4. Matrices

For the image set I and classes C: R = V̂ T̂ᵀ (|I| × |C|, raw cosines, stored once).
For cluster k with unlabelled members U_k:
A_kj = mean_{i∈U_k} R_ij; B_kj = ⟨m_k/‖m_k‖, t̂_j⟩ with m_k = mean_{i∈U_k} v̂_i; ‖m_k‖ stored.
A = ‖m_k‖·B, so the row rankings agree whenever ‖m_k‖ > 0, but the magnitudes differ:
max similarity and softmax scores on A shrink with cluster dispersion; on B they do not.
Labelled anchors are excluded from the primary aggregates and stay identifiable.

## 5. Vocabulary episodes

* Primary episodes share one vocabulary across all clusters. ImageNet-100: 100 (complete), 75,
  50, 25, 10 classes; CUB: 200, 150, 100, 50, 20. Incomplete sizes use vocabulary seeds 0–4,
  drawn uniformly from all classes. Each episode has its own id and stored class list.
* Secondary paired diagnostic: for target classes (all classes), 9 fixed distractors drawn with a
  seeded generator; three episodes per target: target present, target replaced by a random
  unrelated class, target replaced by the most text-similar remaining class (cosine of
  `single`-prompt text embeddings; this choice favours CLIP and is reported as such).
* Episode construction reads class ids only; grounding receives embeddings, memberships and texts.

## 6. Grounding baselines (cluster-level; a decision applies to every member image)

| Method | Score (higher = accept) |
|---|---|
| `forced_top1` | none (always accepts) |
| `max_mean_sim` | max_j A_kj |
| `margin` | top-1 minus top-2 of A_k |
| `msp_pilot` | max softmax(A_k / 0.01) (pilot temperature) |
| `msp_dev` | max softmax(A_k / T*), T* chosen on dev from {0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1} |
| `mcm_aggregate` | max softmax(A_k / 1) with the `mcm` prompts (MCM's published rule and template, applied to the cluster mean) |
| `mcm_image_mean` | mean over members of per-image MCM scores |
| `vote_fraction` | share of members whose own top-1 (within V) equals the aggregate top-1 |
| `neg_softmax_entropy` | 1 − H(softmax(A_k / T*)) / log|V| |
| `neg_vote_entropy` | 1 − H(member top-1 distribution) / log|V| |

Prediction = argmax_j A_kj (B gives the same argmax); exact ties go to the lowest column. The MCM
rows use MCM's own prompts for both label and score. MCM's rule was checked against its code
(`deeplearning-wisc/MCM`, `utils/detection_util.py`: L2-normalised features, raw cosines, no logit
scale, softmax at `--T` default 1, template "a photo of a {c}"); its cleaned ImageNet class-name
file was not used, so this is MCM's scoring rule with our class names, not a reproduction of its
results. `forced_top1` always accepts and is never calibrated. Representations: all unlabelled members
(primary); the 1, 3 and 5 members nearest to the GCD centre; 3 members drawn with seed 0. The
same selections are used for every method.

## 7. Development, calibration, evaluation

* Dev pool (dev groups, shared episodes): choose the prompt strategy (`single` vs `ensemble80`)
  by the AURC of `max_mean_sim`, then T* by the AURC of `msp_dev`; the criterion is the mean over
  vocabulary sizes of the mean AURC of that size's episodes.
* Calibration pool: per method, two thresholds: τ_e2e maximises end-to-end accuracy pooled over
  calibration episodes; τ_95 is the largest threshold that still accepts ≥ 95% of label-available
  calibration images. Candidates are the observed scores plus +∞ (reject all); ties in τ_e2e go
  to the largest threshold.
* Discovery evaluation: frozen choices and thresholds applied to the unlabelled population
  (transductive: discovery used these images without labels). Old and New are reported apart.

## 8. Metrics (image level; positive class for AUROC = "the image's true class is in V")

Forced accuracy (image-weighted and class-macro; complete vocabulary; Old/New); accuracy among
accepted; coverage; risk–coverage curve and AURC with whole-cluster acceptance (tied scores accepted
together); AUROC for label availability; AUROC for "available and correctly named";
false-acceptance rate of absent-label images; end-to-end accuracy (available, accepted and
correct, or absent and rejected) at τ_e2e and τ_95. Cluster diagnostics: composition, purity,
dominant class, strict-majority flag (mixed otherwise), share of members with labels in V.

## 9. Uncertainty

Bootstrap over discovered clusters (1,000 resamples; images and all episodes of a cluster move
together; episodes are averaged within each replicate, never treated as new images); percentile
95% intervals; paired method differences within replicates. Training-seed variation is reported
separately as mean ± SD over completed seeds.
