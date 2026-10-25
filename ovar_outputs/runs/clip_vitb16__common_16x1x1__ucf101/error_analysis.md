# Error analysis: clip_vitb16__common_16x1x1__ucf101

## Observations

### UCF101

Overall 71.16 [68.2, 74.3] (n=3783); familiar 74.23 [70.2, 78.1] (n=1882); unfamiliar 68.12 [63.5, 73.0] (n=1901).

Lowest-accuracy classes (n ≥ 5):

- BodyWeightSquats (unfamiliar): 0.0% of 30; most often predicted as Lunges (19)
- CricketShot (unfamiliar): 0.0% of 49; most often predicted as CricketBowling (47)
- Mixing (unfamiliar): 0.0% of 45; most often predicted as CuttingInKitchen (41)
- JumpingJack (unfamiliar): 0.0% of 37; most often predicted as Lunges (20)
- PlayingDaf (unfamiliar): 0.0% of 41; most often predicted as PlayingTabla (26)

Most frequent confusions:

- CricketShot → CricketBowling: 47 (95.9% of the true class)
- Mixing → CuttingInKitchen: 41 (91.1% of the true class)
- Shotput → ThrowDiscus: 35 (76.1% of the true class)
- StillRings → UnevenBars: 29 (90.6% of the true class)
- BasketballDunk → Basketball: 29 (78.4% of the true class)
- Kayaking → Rafting: 28 (77.8% of the true class)
- ParallelBars → UnevenBars: 28 (75.7% of the true class)
- Punch → BoxingPunchingBag: 28 (71.8% of the true class)
- PushUps → WallPushups: 27 (90.0% of the true class)
- PlayingDaf → PlayingTabla: 26 (63.4% of the true class)

### Diagnostics: clip_vitb16__common_16x1x1__ucf_dev (development subset)

| Variant | Family | Dataset | Acc ref | Acc variant | Δ acc [95% CI] | Prediction change | Variant failures | Note |
|---|---|---|---|---|---|---|---|---|
| temporal_shuffled | temporal | ucf101 | 72.8 | 72.8 | +0.0 [0.0, 0.0] | 0.0% of both-decoded | 0.0 | invariant by construction: frame-independent encoder with mean pooling |
| temporal_single_frame | temporal | ucf101 | 72.8 | 69.3 | -3.5 [-7.6, 0.5] | 12.9% of both-decoded | 0.0 |  |
| dark_0.5 | degradation | ucf101 | 72.8 | 69.8 | -3.0 [-7.1, 0.5] | 8.4% of both-decoded | 0.0 |  |
| dark_0.25 | degradation | ucf101 | 72.8 | 71.3 | -1.5 [-5.5, 2.5] | 12.4% of both-decoded | 0.0 |  |
| down_0.5 | degradation | ucf101 | 72.8 | 68.3 | -4.5 [-8.0, -1.0] | 8.4% of both-decoded | 0.0 |  |
| down_0.25 | degradation | ucf101 | 72.8 | 61.9 | -10.9 [-15.8, -6.0] | 21.8% of both-decoded | 0.0 |  |
| blur_1 | degradation | ucf101 | 72.8 | 67.8 | -5.0 [-9.0, -1.0] | 10.9% of both-decoded | 0.0 |  |
| blur_2 | degradation | ucf101 | 72.8 | 62.4 | -10.4 [-15.7, -4.9] | 23.8% of both-decoded | 0.0 |  |
| actor_crop | spatial | | | | not run: no actor boxes configured | | | |
| actor_context | spatial | | | | not run: no actor boxes configured | | | |

Degradation variants are synthetic stress tests and are not evidence about real dark or drone domains. Temporal and crop effects are diagnostics: an action can be recognisable from static cues, and crops can remove evidence or add artefacts.

Classes with the highest prediction-change rate (videos decoded in both runs):

- temporal_single_frame / HammerThrow (familiar): 2 of 2 changed, Δ acc -100 pp (n=2, variant failures 0)
- dark_0.5 / HulaHoop (familiar): 2 of 2 changed, Δ acc -50 pp (n=2, variant failures 0)
- dark_0.5 / SoccerJuggling (familiar): 2 of 2 changed, Δ acc -50 pp (n=2, variant failures 0)
- temporal_single_frame / Nunchucks (unfamiliar): 2 of 2 changed, Δ acc +0 pp (n=2, variant failures 0)
- dark_0.5 / YoYo (unfamiliar): 2 of 2 changed, Δ acc -100 pp (n=2, variant failures 0)
- temporal_single_frame / FrontCrawl (unfamiliar): 2 of 2 changed, Δ acc +0 pp (n=2, variant failures 0)
- temporal_single_frame / HulaHoop (familiar): 2 of 2 changed, Δ acc +0 pp (n=2, variant failures 0)
- down_0.25 / Shotput (familiar): 2 of 2 changed, Δ acc +0 pp (n=2, variant failures 0)

## Hypotheses

_To be written by the researcher. Each hypothesis must cite the observation above that motivates it and name the experiment that would refute it. This report does not infer that any particular architecture is required._

