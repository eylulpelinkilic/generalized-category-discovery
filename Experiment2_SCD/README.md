# Experiment 2: SCD reproduction and its naming component in the grounding protocol

**Naming.** "SCD" is the paper's term: *Semantic Category Discovery* (Han et al., "What's in a Name? Beyond Class
Indices for Image Recognition", CVPR-W 2024, arXiv:2304.02364). `../SCD/` is **our folder** holding our
re-implementation (no code copied from the authors; see `../SCD/ATTRIBUTION.md`). This folder is the experiment.

* `docs/INSPECTION.md`: paper, official code (commit `2ce64f7`) and README compared; defects; vocabulary recovery;
  provenance of the Experiment 1 artifacts used.
* `PROTOCOL.md`: the three protocols (A reproduction on ImageNet-100, B fixed clusters on CUB, R refinement from GCD
  clusters on CUB), fixed before any run.
* `reports/REPORT.md`: findings; `reports/GENERATED_TABLES.md`: every number, read from `outputs/`.

## Layout

```
e2/            stages: vocab_stage, embed, assets, fixed (B), voting, refine_gcd (R), repro (A), errors, report, smoke
configs/       base.yaml, cub.yaml (B, R), imagenet100.yaml (A)
kaggle/        make_notebooks.py -> e2_imagenet100.ipynb, e2_cub.ipynb
tests/         software checks (synthetic data only)
outputs/       (not in Git) <dataset>/<protocol>/<model>/<seed>/<config hash>/..., vocabulary/, cache/, assets/
reports/       committed report and figures
```

Experiment 1 artifacts are used by reference (`e1.runs_root`, default `../experiments`); nothing in
`Experiment1_GCD_CLIP_Baseline/` is modified.

## Commands

Python ≥ 3.10 with the Experiment 1 requirements plus `nltk` and `ortools` (protocol A also needs the official
`git+https://github.com/openai/CLIP.git`). From this folder:

```bash
# shell functions work in bash and zsh (zsh does not split "$VAR" commands)
e2cub()   { ../.venv/bin/python -m e2 --config configs/cub.yaml "$@"; }
e2in()    { ../.venv/bin/python -m e2 --config configs/imagenet100.yaml "$@"; }
e2smoke() { ../.venv/bin/python -m e2 --config configs/smoke.yaml "$@"; }

# software checks
(cd ../SCD && ../.venv/bin/python -m unittest discover -s tests -t .)
../.venv/bin/python -m unittest discover -s tests -t .
e2in smoke                                                    # synthetic protocol-A run, software check only
# protocols B and R through their entry points on Experiment 1's synthetic smoke run (software check only):
(cd ../Experiment1_GCD_CLIP_Baseline && ../.venv/bin/python -m e1 --config configs/smoke.yaml \
    --set paths.runs_root="$PWD/../Experiment2_SCD/outputs/e1_smoke" smoke)
e2smoke cub-embed && e2smoke fixed && e2smoke refine-gcd --mode unsupervised && e2smoke refine-gcd --mode partially_supervised

# vocabulary: rebuild the 67,893 WordNet names, download the authors' weights (151 MB, sha256-checked), verify order
e2cub vocab

# protocol B (CUB): embeddings for Experiment 1's rows (official archive, MD5-checked; validated against Experiment 1)
e2cub cub-embed --clip vitb32
e2cub fixed --clip vitb32                                     # dev choice, calibration, evaluation, errors, galleries

# protocol R (CUB): SCD refinement from Experiment 1's GCD clusters
for c in 0 1 2; do for m in unsupervised partially_supervised; do e2cub refine-gcd --cluster-seed $c --mode $m; done; done

# protocol A (ImageNet-100, GPU + ImageNet): kaggle/e2_imagenet100.ipynb, or on a GPU machine
R="--set=e1.runs_root=<Experiment 1 runs holding the imagenet100 manifest>"   # one token: safe in bash and zsh
e2in $R vocab && e2in $R repro features && e2in $R repro cluster && e2in $R repro name && e2in $R repro evaluate

e2cub report                                                  # reports/GENERATED_TABLES.md + figures
```

`--set` must come before the command name (it is an option of the main parser).

## Status

See `reports/REPORT.md` §1 for what ran, what is a software check, and what is pending.
