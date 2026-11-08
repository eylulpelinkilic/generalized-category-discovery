# Handoff: inputs for the cosine mean-shift experiment

The later experiment must measure **added value over this frozen baseline**, on the same images,
clusters, vocabularies, development pool, calibration pool and evaluation population. Nothing in it
may change the discovery stage or reuse evaluation labels for choices.

## Artifacts it consumes (all keyed by explicit identities)

In `runs/<dataset>/grounding/seed<S>_known_c<C>/<clip>_<hash>/`:

| File | Content |
|---|---|
| `rows.csv` | one row per image: `row`, `sample_id`, `population` (discovery / anchor / dev / calib), `cluster_id` (GCD cluster, or the fitted centre a held-out image was assigned to), `is_anchor`, `sq_dist_to_centre` |
| `R_<strategy>.npy` | image-level profiles R [rows × classes], raw cosines, stored once (no per-episode copies) |
| `columns.csv` | column order: `class_id`, canonical name, aliases, GCD known/novel status |
| `texts_<strategy>.csv`, `text_embeddings_<strategy>.npz` | every prompt with its class id; per-text and per-class unit text embeddings |
| `discovery_aggregates_<strategy>.npz` | cluster-level profiles A and B over all classes, mean-embedding norm, member counts |
| `episodes.csv`, `episodes/<episode_id>.npz` | each shared-vocabulary episode's class list and its cluster-by-label A/B matrices (rows = cluster ids 0..K−1, columns = the episode's class ids) |
| `cluster_predictions.csv.gz`, `image_predictions_complete_vocabulary.csv.gz` | baseline labels and scores per episode, method and representation |
| `selection.json`, `thresholds.json` | the baseline's dev choices and calibrated thresholds |

Original CLIP image embeddings: `runs/<dataset>/clip/<clip>_<hash>/image_shards/`, read in any
order with `e1.clip_extract.load_images(dir, sample_ids)`. GCD features, assignments, centres and
cluster sizes: `runs/<dataset>/gcd/seed<S>_<hash>/{features,clustering}/`.

## How a mean-shift method plugs in

1. **Geometry.** Work on the unit sphere of CLIP image embeddings (cosine kernel), in the full
   embedding dimension. Low-dimensional projections are for illustration only.
2. **Per cluster.** Start from the cluster's members (or its mean direction), run cosine mean shift
   over the cluster's own CLIP embeddings to a mode, and score labels at the mode against the same
   `text_embeddings_<strategy>.npz` columns as the baseline.
3. **Same protocol.** Choose any bandwidth or step size on the **dev** groups (rows with
   `population == dev`); fit thresholds on the **calib** groups; evaluate once on `discovery`
   with `e1.metrics` (whole-cluster decisions, tied scores accepted together, cluster bootstrap with
   paired differences against the baseline predictions in `cluster_predictions.csv.gz`).

## Candidate measurements (each reported against the baseline, per vocabulary size)

* density at the starting point and at the mode (kernel density estimate on the sphere);
* initial gradient magnitude of the density, distance travelled to the mode, iterations;
* curvature near the mode (e.g. Hessian eigenvalues of the log-density in the tangent space);
* change in the label ranking between the mean (baseline A/B) and the mode, and whether the change
  helps naming or rejection.

Two facts to keep in mind: the density gradient tends to zero at a converged mode, so gradient-based
confidence must be measured along the path or at the start, not at convergence; and a density peak
has no semantic meaning by itself; it acquires one only through the evaluation against labels.

**Current target:** a fully verified GCD + CLIP baseline. The mean-shift experiment starts only
after the baseline report shows completed real runs.
