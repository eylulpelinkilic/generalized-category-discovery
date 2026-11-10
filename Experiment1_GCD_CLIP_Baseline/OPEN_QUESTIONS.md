# Open research questions (Experiment 1)

## Q1. Do incorrectly clustered images look different from correctly clustered ones?

Status: answered (exploratory) on CUB, development pool, training seed 0. Report section 6;
`diagnostics/clustering_examples/comparison/`. Misclustered images are boundary images (farther from the
GCD centre, less typical of their cluster, flatter CLIP rows); clusters containing them do not differ.

## Q2. Do wrongly named clusters behave differently from correctly named ones?

Status: open (raised 2026-11-10).

Question: when CLIP gives a cluster the wrong name, how does that cluster differ from clusters named
correctly — in its similarity matrix, its scores, its GCD geometry and its composition?

Planned design (to fix before running):

* **Unit:** a cluster (or a development group: held-out images assigned to one fitted centre).
* **Named right:** the cluster's predicted name (forced top-1, complete vocabulary) equals its majority
  species; a second definition counts the share of member images whose species equals the name.
* **Population:** development groups first (labels permitted; known species only; groups are small, median
  a few images). Using the unlabelled discovery clusters means inspecting the evaluation population and
  needs an explicit decision.
* **Measures (label-free first):** max similarity, top-1 minus top-2 margin, softmax entropy, vote fraction
  (members whose own top name equals the cluster's), dispersion of member rows, mean-embedding norm,
  cluster size, mean distance to the GCD centre, text similarity between the predicted name and its nearest
  other name. Descriptive (need labels): purity, Old/New status of the majority species, rank of the true
  name in the cluster's mean row.
* **Comparison:** AUROC (wrongly vs correctly named) with cluster-bootstrap intervals; report how much the
  label-free measures add beyond the rejection scores already evaluated in section 5.
