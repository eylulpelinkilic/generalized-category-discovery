# SCD: semantic naming and refinement (re-implementation)

**Naming note.** "SCD" is used in two ways in this repository:

* **the paper's term**: *Semantic Category Discovery*, the task and method introduced in Han, Huang, Li, Vaze,
  Li and Jia, "What's in a Name? Beyond Class Indices for Image Recognition", CVPR 2024 Workshops
  (arXiv:2304.02364, v2, 27 Jul 2024);
* **our folder name**: this directory (`SCD/`), which holds our re-implementation, and
  `Experiment2_SCD/`, which holds the experiment that uses it.

Nothing in this folder is the authors' code. See `ATTRIBUTION.md` for what was read, what was re-implemented,
and the licences involved.

## What is here

The Python package `scd` (import with `SCD/` on `sys.path`). It contains no dataset-specific code and never
reads evaluation labels: label-dependent metrics are separate functions that the caller feeds.

| Module | Contents |
|---|---|
| `scd.vocabulary` | the official WordNet noun vocabulary: 67,893 columns, the unique case-sensitive first lemmas of all WordNet 3.0 noun synsets, in first-occurrence order (verified against the authors' embeddings, see `Experiment2_SCD/docs/INSPECTION.md`); normalisation (`lower`, `-`→`_`) and the duplicate columns it creates; WordNet dictionaries for Soft-sACC |
| `scd.text` | the 80 OpenAI ImageNet templates, zero-shot text weights (mean of normalised template embeddings, renormalised), loading the authors' weight file, alignment checks between a name list and a weight matrix |
| `scd.naming` | per-image top-k names over a large vocabulary (batched, deterministic ties), cluster vote counters, candidate pools, distinct-name assignment, CLIP reassignment, the iterative loop for the unsupervised and partially supervised settings, with explicit handling of empty clusters, insufficient candidates, cycles and iteration limits, and a per-iteration trace |
| `scd.assignment` | the Hungarian solver used by the official code (sklearn's former `linear_assignment`, BSD, vendored in `scd/third_party/`) and a SciPy alternative |
| `scd.clustering` | k-means as in the official unsupervised script; the constrained semi-supervised k-means (min-cost flow, size limits on unlabelled points) re-implemented with OR-Tools and explicit seeds |
| `scd.metrics` | sACC (exact name match), Soft-sACC (Leacock–Chodorow, two normalisations), name-set IoU, clustering accuracy with one global Hungarian assignment |

## Index spaces

`scd.naming` keeps four index spaces apart and names every array after its space:

* **vocabulary index** `v`: a column of the vocabulary and of its text-embedding matrix;
* **pool index** `s`: a position in the current candidate shortlist (columns of the assignment matrix);
* **name position** `p`: a position in the list of selected names that images are reassigned to;
* **cluster id** `c`: initial cluster ids, then name positions after each reassignment (as in the official code).

## Modes

Every deviation from the official code is a named option, recorded in each run's trace:

| Option | Official behaviour | Alternatives |
|---|---|---|
| `pool_order` | `list(set(...))` of vocabulary indices (CPython set order) | `first_appearance` |
| `solver` | sklearn Munkres (`munkres`) | `scipy` |
| `known_filter` (partially supervised) | `official`: from iteration 2 the known-name filter holds positions in the selected-name list but is compared with vocabulary indices, so known names are no longer excluded (and unrelated vocabulary indices are) | `fixed`: the filter always holds the vocabulary indices of every column whose normalised name is known |
| `merge_duplicate_names` | off: the 712 names that occur twice after normalisation keep both columns | on: one column per normalised name |
| `insufficient_pool` | the official code raises `KeyError` | `extend`: deepen each cluster's candidate list until enough names exist (recorded) |
| `max_iter` | none (the loop can cycle forever) | integer; cycles are detected and recorded |
| `logit_dtype` | fp16 on GPU (ties broken by the CUDA kernel) | `float32` (default) or `float16` emulation; ties always go to the lowest vocabulary index |

## Tests

```bash
cd SCD && ../.venv/bin/python -m unittest discover -s tests -t .
```
