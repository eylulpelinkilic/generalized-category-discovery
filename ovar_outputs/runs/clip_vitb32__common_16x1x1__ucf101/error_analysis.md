# Error analysis: clip_vitb32__common_16x1x1__ucf101

## Observations

### UCF101

Overall 68.17 [65.2, 71.1] (n=3783); familiar 66.21 [61.8, 70.2] (n=1882); unfamiliar 70.12 [65.6, 74.5] (n=1901).

Lowest-accuracy classes (n ≥ 5):

- BalanceBeam (unfamiliar): 0.0% of 31; most often predicted as UnevenBars (28)
- PlayingDaf (unfamiliar): 0.0% of 41; most often predicted as Drumming (30)
- JumpingJack (unfamiliar): 0.0% of 37; most often predicted as Lunges (7)
- JugglingBalls (familiar): 0.0% of 40; most often predicted as YoYo (29)
- Nunchucks (unfamiliar): 0.0% of 35; most often predicted as FrisbeeCatch (11)

Most frequent confusions:

- Punch → BoxingPunchingBag: 38 (97.4% of the true class)
- ParallelBars → UnevenBars: 36 (97.3% of the true class)
- PlayingDaf → Drumming: 30 (73.2% of the true class)
- JugglingBalls → YoYo: 29 (72.5% of the true class)
- BalanceBeam → UnevenBars: 28 (90.3% of the true class)
- Kayaking → Rafting: 28 (77.8% of the true class)
- StillRings → UnevenBars: 27 (84.4% of the true class)
- FloorGymnastics → UnevenBars: 27 (75.0% of the true class)
- BasketballDunk → Basketball: 27 (73.0% of the true class)
- BreastStroke → FrontCrawl: 25 (89.3% of the true class)

### Diagnostics: clip_vitb32__common_16x1x1__ucf_dev (development subset)

| Variant | Family | Dataset | Acc ref | Acc variant | Δ acc [95% CI] | Prediction change | Variant failures | Note |
|---|---|---|---|---|---|---|---|---|
| temporal_shuffled | temporal | ucf101 | 64.4 | 64.4 | +0.0 [0.0, 0.0] | 0.0% of both-decoded | 0.0 | invariant by construction: frame-independent encoder with mean pooling |
| temporal_single_frame | temporal | ucf101 | 64.4 | 66.3 | +2.0 [-2.5, 6.3] | 16.8% of both-decoded | 0.0 |  |
| dark_0.5 | degradation | ucf101 | 64.4 | 65.8 | +1.5 [-1.5, 4.4] | 9.4% of both-decoded | 0.0 |  |
| dark_0.25 | degradation | ucf101 | 64.4 | 64.9 | +0.5 [-3.5, 4.4] | 15.3% of both-decoded | 0.0 |  |
| down_0.5 | degradation | ucf101 | 64.4 | 65.3 | +1.0 [-2.5, 4.5] | 10.4% of both-decoded | 0.0 |  |
| down_0.25 | degradation | ucf101 | 64.4 | 59.4 | -5.0 [-9.9, 0.0] | 21.3% of both-decoded | 0.0 |  |
| blur_1 | degradation | ucf101 | 64.4 | 65.8 | +1.5 [-1.5, 4.6] | 9.4% of both-decoded | 0.0 |  |
| blur_2 | degradation | ucf101 | 64.4 | 60.4 | -4.0 [-8.9, 1.0] | 24.3% of both-decoded | 0.0 |  |
| actor_crop | spatial | | | | not run: no actor boxes configured | | | |
| actor_context | spatial | | | | not run: no actor boxes configured | | | |

Degradation variants are synthetic stress tests and are not evidence about real dark or drone domains. Temporal and crop effects are diagnostics: an action can be recognisable from static cues, and crops can remove evidence or add artefacts.

Classes with the highest prediction-change rate (videos decoded in both runs):

- temporal_single_frame / UnevenBars (unfamiliar): 2 of 2 changed, Δ acc -100 pp (n=2, variant failures 0)
- temporal_single_frame / CricketBowling (unfamiliar): 2 of 2 changed, Δ acc +0 pp (n=2, variant failures 0)
- temporal_single_frame / Diving (familiar): 2 of 2 changed, Δ acc +50 pp (n=2, variant failures 0)
- dark_0.5 / Diving (familiar): 2 of 2 changed, Δ acc +50 pp (n=2, variant failures 0)
- temporal_single_frame / CleanAndJerk (familiar): 2 of 2 changed, Δ acc +100 pp (n=2, variant failures 0)
- dark_0.5 / Nunchucks (unfamiliar): 2 of 2 changed, Δ acc +0 pp (n=2, variant failures 0)
- dark_0.5 / ParallelBars (unfamiliar): 2 of 2 changed, Δ acc +0 pp (n=2, variant failures 0)
- blur_2 / Nunchucks (unfamiliar): 2 of 2 changed, Δ acc +0 pp (n=2, variant failures 0)

## Hypotheses

_To be written by the researcher. Each hypothesis must cite the observation above that motivates it and name the experiment that would refute it. This report does not infer that any particular architecture is required._

