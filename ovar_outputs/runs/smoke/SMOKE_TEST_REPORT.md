# Error analysis: smoke_fixture

> **SYNTHETIC FIXTURE RUN — software verification only, not research evidence.**

## Observations

### fixture_colors

Overall 87.50 [66.7, 100.0] (n=8); familiar 83.33 [50.0, 100.0] (n=6); unfamiliar 100.00 [100.0, 100.0] (n=2).

Lowest-accuracy classes (n ≥ 5):

- no class has n ≥ 5

Most frequent confusions:

- none with count ≥ 2

### Diagnostics: smoke_fixture (development subset)

| Variant | Family | Dataset | Acc ref | Acc variant | Δ acc [95% CI] | Prediction change | Variant failures | Note |
|---|---|---|---|---|---|---|---|---|
| temporal_shuffled | temporal | fixture_colors | 87.5 | 87.5 | +0.0 [0.0, 0.0] | 0.0% of both-decoded | 1 | invariant by construction: frame-independent encoder with mean pooling |
| temporal_single_frame | temporal | fixture_colors | 87.5 | 75.0 | -12.5 [-25.0, 0.0] | 14.3% of both-decoded | 1 |  |
| dark_0.25 | degradation | fixture_colors | 87.5 | 87.5 | +0.0 [0.0, 0.0] | 0.0% of both-decoded | 1 |  |
| blur_2 | degradation | fixture_colors | 87.5 | 87.5 | +0.0 [0.0, 0.0] | 0.0% of both-decoded | 1 |  |
| down_0.25 | degradation | fixture_colors | 87.5 | 87.5 | +0.0 [0.0, 0.0] | 0.0% of both-decoded | 1 |  |
| actor_crop | spatial | fixture_colors | 87.5 | 75.0 | -12.5 [-29.1, 0.0] | 0.0% of both-decoded | 2 |  |
| actor_context | spatial | fixture_colors | 87.5 | 75.0 | -12.5 [-50.0, 0.0] | 0.0% of both-decoded | 2 |  |

Degradation variants are synthetic stress tests and are not evidence about real dark or drone domains. Temporal and crop effects are diagnostics: an action can be recognisable from static cues, and crops can remove evidence or add artefacts.

Classes with the highest prediction-change rate (videos decoded in both runs):

- temporal_single_frame / red (familiar): 1 of 3 changed, Δ acc -33 pp (n=3, variant failures 0)

## Hypotheses

_To be written by the researcher. Each hypothesis must cite the observation above that motivates it and name the experiment that would refute it. This report does not infer that any particular architecture is required._

