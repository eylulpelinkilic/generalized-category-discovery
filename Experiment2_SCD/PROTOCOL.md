# Experiment 2 protocol

Fixed before any research run. Configuration values live in `configs/*.yaml`; every output directory stores its
configuration hash and a `status.json` (code commit, environment). "SCD" = the paper's method (Semantic Category
Discovery); `../SCD/` = our re-implementation of it.

Three protocols, reported separately because they answer different questions and two of them change cluster
membership.

| | A. Reproduction | B. Fixed clusters | R. Refinement from GCD clusters |
|---|---|---|---|
| Dataset | ImageNet-100 (GCD split) | CUB-200-2011 (SSB split) | CUB-200-2011 |
| Initial clusters | DINO + k-means / authors' GCD + CSS-k-means | Experiment 1's GCD SS-k-means clusters, fixed | Experiment 1's GCD SS-k-means clusters |
| Vocabulary | 67,893 WordNet nouns (official, verified) | Experiment 1's episodes (200 / 150 / 100 / 50 / 20 names; paired) | the 200 CUB names |
| CLIP | ViT-B/16 (official package, fp16), 80 templates (official weights) | ViT-B/32 + Experiment 1's dev-selected prompts (ensemble80) | same as B |
| Memberships | change | never change | change |
| Decision | forced naming | forced, then UNMAPPED | forced |

## A. Reproduction (ImageNet-100)

A1. **Data.** Experiment 1's ImageNet-100 manifest (upstream partition replayed): 50 known / 50 novel classes,
labelled anchors = upstream 50% sample of known-class training images. SCD evaluates on the unlabelled training
images; so do we. Targets: the normalised first lemma of each class synset (official `wnid_to_name`).

A2. **Features.** CLIP ViT-B/16 via `clip.load` (fp16, CUDA); DINO ViT-B/16 (`dino_vitbase16_pretrain.pth`, the weights
torch.hub's `dino_vitb16` loads); the authors' `imagenet100_model_best.pt` GCD backbone. All with CLIP's preprocessing,
as the official code. L2-normalised. Order: labelled then unlabelled (upstream `MergedDataset`), unlabelled in manifest
order.

A3. **Initial clusterings.** Unsupervised: k-means (K = 100, n_init 10) on DINO features of the unlabelled images,
seeds 0 (official `random_state=0`), 1, 2. Partially supervised: CSS-k-means on GCD features (size 50–1000 over
unlabelled points, n_init 10, ≤ 10 iterations, tol 1e-4), seeds 0, 1, 2 (official: unseeded).

A4. **Naming variants** (all on every initial clustering):

| Variant | Behaviour | Fidelity label |
|---|---|---|
| `official` | top-k 3, pool 10/cluster, 2 names/cluster in the matrix, Munkres, `list(set())` pool, stop on an unchanged name set; fp32 logits; known filter as official (ptsup) | official-assets |
| `voting_once` | each cluster's most voted name, one reassignment (Table 5 "+ voting") | ablation |
| `voting_iterated` | same, iterated (Table 5 "+ iteration") | ablation |
| `help_defaults` | 5 / 20 / 4 (argparse help text for ImageNet) | deviation: parameters |
| `fp16_logits` | logits rounded to fp16 (official numerics), ties to the lowest index | sensitivity |
| `merged_duplicates` | one column per normalised name (repair of defect 11) | deviation: repair |
| `known_filter_fixed` | known names filtered in vocabulary space in every iteration (repair of defect 10; ptsup) | deviation: repair |

Repairs present in every variant and recorded per iteration: `max_iter = 50`, cycle detection, pool deepening when the
official code would raise `KeyError`. Partially supervised runs use only the labelled anchors' class names; evaluation
labels are read only by `repro evaluate`.

A5. **Metrics.** sACC (image-weighted, the paper's Eq. 3; class-averaged also stored), Soft-sACC (official observed-max
normalisation and max-possible normalisation), clustering ACC (Hungarian, clustering only), All / Old / New; for the
initial clustering (named by iteration-1 votes, before reassignment) and after every iteration; number of selected
names, name-set IoU, iterations, stop reason, repairs; zero-shot UB and Baseline rows. Published numbers are compared
only for the `official` variant with seed 0 k-means (unsupervised) and, with that caveat, seed-0 CSS-k-means.

## B. Fixed clusters (CUB, Experiment 1)

B1. **Inputs (by reference).** Experiment 1 run `seed0_known_c0/vitb32_831109500c95`: rows, cluster memberships,
episodes, dev-selected prompt strategy (ensemble80) and T* (0.02), per-class text embeddings. Image embeddings are
recomputed and accepted only if they reproduce Experiment 1 (identity hash, aggregates, predictions; criteria in
`e2/embed.py`). Representation: all unlabelled members.

B2. **Label rules.** *Mean*: argmax of the cluster-mean similarity (Experiment 1). *Plurality vote*: each member votes
for its top-1 available concept; the most-voted concept wins; image ties → lowest class id; cluster ties → higher mean
similarity, then lowest class id. *Top-k vote* (SCD's per-image top-k): each member votes for its k best available
concepts. Votes are recomputed within each episode's vocabulary; a removed class takes all its prompts with it.

B3. **Confidence of the vote label** (higher = accept; for its own label): `win_fraction` = v1/n, `margin` = (v1 − v2)/n,
`neg_entropy` = 1 − H(votes)/log|V|, `mean_sim` = mean similarity of the winner. Edge cases: empty cluster → no
prediction (excluded); |V| = 1 → v2 = 0 and neg_entropy = 1; a single-member cluster scores 1 on the first three.

B4. **Distinct names.** SCD's distinct-name assignment (top-3 votes, pool 10, 2 per cluster, CUB help values) is
evaluated only on the complete vocabulary (200 names, 200 clusters), forced. With an incomplete vocabulary the
constraint has no valid form (absent names cannot be shared out), so every rejection experiment uses independent
cluster decisions.

B5. **Development and calibration.** Dev pool: k of the top-k vote from {1, 3, 5} by the dev AURC of
`<vote>@win_fraction` (Experiment 1's criterion). Calibration pool: τ_e2e and τ_95 per method (Experiment 1's rule).
Experiment 1's own methods are recomputed with the same rows and checked against its saved metrics.

B6. **Evaluation.** Experiment 1's image-level metrics on the discovery population: forced accuracy (complete and
incomplete), coverage, accepted accuracy, AUROC (label available; available and correct), false acceptance of
absent-label images, end-to-end accuracy, risk–coverage, paired diagnostic; purity and composition; cluster bootstrap
(1,000) with paired differences against `max_mean_sim` and forced-accuracy differences against `forced_top1`.

## R. Refinement from GCD clusters (CUB)

Not a paper setting (closed 200-name vocabulary; Experiment 1 clusters; ViT-B/32). Modes: unsupervised (200 slots)
and partially supervised (100 known names from the anchors' classes, 100 voted slots), official parameters,
`known_filter = fixed`. Clustering seeds 0, 1, 2. Metrics: as A5 without Soft-sACC (CUB names are not WordNet
synsets). Uncertainty: spread over clustering seeds; paired image bootstrap of final − initial sACC conditional on
each run (memberships change, so Experiment 1's cluster bootstrap does not apply).

## E. Errors and failure patterns

Every evaluated image and cluster has a machine-readable decision (`errors/`, `image_predictions.csv.gz`); all
incorrect decisions are exported with sample/cluster ids, paths, labels, availability, prediction, scores, votes,
confidence, rejection outcome, composition and method. Iterative runs store names and assignments per iteration.
Patterns P1–P8 are defined in `e2/errors.py` (measurable definitions); galleries pair each shown error with the
correct cluster nearest in size and purity. Whether a confusion is visual, or a name a synonym in a looser sense than
WordNet's, is a hypothesis, marked as such.
