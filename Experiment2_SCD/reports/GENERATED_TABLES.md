# Experiment 2: generated tables

Every number below is read from `outputs/`. Narrative and interpretation: `REPORT.md`.

## Vocabulary verification
| columns | unique normalised names | duplicate columns | exact-check positions | min cosine (80 templates) | 1-template retrieval top-1 | top-5 | sorted-order control min cosine | duplicate pair min cosine | result |
|---|---|---|---|---|---|---|---|---|---|
| 67893 | 67181 | 712 | 382 | 0.99959 | 0.9997 | 0.99996 | 0.493 | 0.9744 | passed |

Source: `outputs/vocabulary/wordnet30_official_e4951377130e/alignment.json`.

## Protocol B: fixed clusters (CUB, ViT-B/32)
Embedding validation: aggregate max |ΔA| = 1.75e-05, member counts equal = True, Experiment 1 prediction agreement = 1.0000 over 97,860 (episode, cluster, method) rows; max score difference 1.08e-04.

Experiment 1 methods recomputed vs its saved metrics: forced_accuracy_available max |Δ| 1.1e-16, aurc max |Δ| 5.5e-05, auroc_label_available max |Δ| 6.2e-04, end_to_end_accuracy@tau_e2e max |Δ| 1.1e-16.

Dev selection: k* = 1 (k=1: 0.4653, k=3: 0.5200, k=5: 0.5710; dev AURC, lower is better).

**Forced naming, complete vocabulary (200 names), discovery population (4,496 images)**

| label rule | accuracy % | 95% CI | class-macro % | Old % | New % | Δ vs mean (paired) pp |
|---|---|---|---|---|---|---|
| distinct_top3_forced | 37.6 | [32.9, 42.9] | 40.7 | 49.3 | 31.7 | -2.4 [-5.6, +0.7] |
| forced_top1 | 40.0 | [35.2, 45.4] | 42.4 | 48.9 | 35.6 | – |
| plurality_forced | 39.6 | [35.0, 44.9] | 42.1 | 47.9 | 35.5 | -0.4 [-1.8, +1.1] |
| top3vote_forced | 37.2 | [32.5, 43.0] | 40.2 | 46.6 | 32.6 | -2.8 [-5.1, -0.6] |
| top5vote_forced | 34.9 | [29.9, 40.4] | 38.5 | 47.0 | 28.8 | -5.2 [-8.3, -2.8] |

**By vocabulary size (mean over episodes; thresholds τ_e2e from the calibration pool)**

| method | vocab size | forced acc (avail) % | AUROC avail | coverage % | accepted acc % | false accept absent % | end-to-end % |
|---|---|---|---|---|---|---|---|
| forced_top1 | 200 | 40.0 | – | 100.0 | 40.0 | – | 40.0 |
| forced_top1 | 150 | 45.2 | 0.500 | 100.0 | 33.6 | 100.0 | 33.6 |
| forced_top1 | 100 | 51.9 | 0.500 | 100.0 | 25.9 | 100.0 | 25.9 |
| forced_top1 | 50 | 63.9 | 0.500 | 100.0 | 15.9 | 100.0 | 15.9 |
| forced_top1 | 20 | 78.9 | 0.500 | 100.0 | 7.5 | 100.0 | 7.5 |
| plurality_forced | 200 | 39.6 | – | 100.0 | 39.6 | – | 39.6 |
| plurality_forced | 150 | 45.5 | 0.500 | 100.0 | 33.9 | 100.0 | 33.9 |
| plurality_forced | 100 | 51.5 | 0.500 | 100.0 | 25.7 | 100.0 | 25.7 |
| plurality_forced | 50 | 62.7 | 0.500 | 100.0 | 15.6 | 100.0 | 15.6 |
| plurality_forced | 20 | 76.9 | 0.500 | 100.0 | 7.3 | 100.0 | 7.3 |
| max_mean_sim | 200 | 40.0 | – | 64.7 | 44.9 | – | 29.0 |
| max_mean_sim | 150 | 45.2 | 0.633 | 56.9 | 40.7 | 43.6 | 37.6 |
| max_mean_sim | 100 | 51.9 | 0.654 | 46.1 | 36.6 | 36.9 | 48.5 |
| max_mean_sim | 50 | 63.9 | 0.749 | 29.2 | 32.9 | 20.8 | 69.0 |
| max_mean_sim | 20 | 78.9 | 0.820 | 15.2 | 29.9 | 11.4 | 84.4 |
| vote_fraction | 200 | 40.0 | – | 7.5 | 87.9 | – | 6.6 |
| vote_fraction | 150 | 45.2 | 0.545 | 8.9 | 75.3 | 6.0 | 30.8 |
| vote_fraction | 100 | 51.9 | 0.569 | 9.8 | 64.9 | 5.8 | 53.5 |
| vote_fraction | 50 | 63.9 | 0.631 | 14.4 | 36.4 | 11.8 | 71.3 |
| vote_fraction | 20 | 78.9 | 0.696 | 18.9 | 20.8 | 16.4 | 79.5 |
| plurality@win_fraction | 200 | 39.6 | – | 7.5 | 87.9 | – | 6.6 |
| plurality@win_fraction | 150 | 45.5 | 0.540 | 8.9 | 75.3 | 6.0 | 30.8 |
| plurality@win_fraction | 100 | 51.5 | 0.565 | 9.8 | 64.9 | 5.8 | 53.5 |
| plurality@win_fraction | 50 | 62.7 | 0.629 | 14.4 | 36.4 | 11.8 | 71.3 |
| plurality@win_fraction | 20 | 76.9 | 0.701 | 18.9 | 20.8 | 16.4 | 79.5 |
| plurality@margin | 200 | 39.6 | – | 13.5 | 82.9 | – | 11.2 |
| plurality@margin | 150 | 45.5 | 0.521 | 16.9 | 65.6 | 14.6 | 32.9 |
| plurality@margin | 100 | 51.5 | 0.543 | 18.7 | 52.5 | 14.9 | 52.4 |
| plurality@margin | 50 | 62.7 | 0.608 | 23.3 | 33.6 | 19.6 | 68.2 |
| plurality@margin | 20 | 76.9 | 0.686 | 30.9 | 16.5 | 28.1 | 70.1 |
| plurality@neg_entropy | 200 | 39.6 | – | 12.8 | 86.8 | – | 11.1 |
| plurality@neg_entropy | 150 | 45.5 | 0.543 | 13.0 | 73.1 | 9.0 | 32.8 |
| plurality@neg_entropy | 100 | 51.5 | 0.583 | 13.3 | 58.5 | 9.6 | 53.1 |
| plurality@neg_entropy | 50 | 62.7 | 0.643 | 14.8 | 35.9 | 12.2 | 71.2 |
| plurality@neg_entropy | 20 | 76.9 | 0.708 | 16.4 | 20.9 | 14.2 | 81.0 |
| plurality@mean_sim | 200 | 39.6 | – | 62.8 | 44.9 | – | 28.2 |
| plurality@mean_sim | 150 | 45.5 | 0.628 | 55.5 | 41.2 | 43.2 | 37.4 |
| plurality@mean_sim | 100 | 51.5 | 0.652 | 45.9 | 36.5 | 36.7 | 48.5 |
| plurality@mean_sim | 50 | 62.7 | 0.747 | 29.0 | 33.0 | 20.6 | 69.1 |
| plurality@mean_sim | 20 | 76.9 | 0.820 | 15.2 | 29.0 | 11.4 | 84.3 |
| top3vote@margin | 200 | 37.2 | – | 5.5 | 62.3 | – | 3.4 |
| top3vote@margin | 150 | 42.4 | 0.500 | 7.6 | 45.6 | 8.8 | 26.8 |
| top3vote@margin | 100 | 48.9 | 0.472 | 7.6 | 36.5 | 8.0 | 48.7 |
| top3vote@margin | 50 | 59.3 | 0.481 | 9.7 | 22.8 | 9.6 | 70.1 |
| top3vote@margin | 20 | 74.0 | 0.473 | 6.4 | 8.9 | 6.4 | 85.1 |
| top3vote@mean_sim | 200 | 37.2 | – | 28.9 | 58.6 | – | 16.9 |
| top3vote@mean_sim | 150 | 42.4 | 0.630 | 23.4 | 56.1 | 13.9 | 35.2 |
| top3vote@mean_sim | 100 | 48.9 | 0.648 | 18.2 | 54.1 | 10.2 | 54.8 |
| top3vote@mean_sim | 50 | 59.3 | 0.740 | 9.0 | 51.9 | 4.5 | 76.4 |
| top3vote@mean_sim | 20 | 74.0 | 0.814 | 4.4 | 51.5 | 2.3 | 90.5 |
| top3vote@neg_entropy | 200 | 37.2 | – | 40.0 | 52.1 | – | 20.8 |
| top3vote@neg_entropy | 150 | 42.4 | 0.548 | 30.0 | 45.8 | 25.4 | 32.8 |
| top3vote@neg_entropy | 100 | 48.9 | 0.583 | 24.2 | 37.5 | 17.6 | 50.0 |
| top3vote@neg_entropy | 50 | 59.3 | 0.637 | 8.2 | 24.1 | 6.5 | 72.2 |
| top3vote@neg_entropy | 20 | 74.0 | 0.633 | 3.1 | 11.9 | 3.0 | 88.0 |
| top3vote@win_fraction | 200 | 37.2 | – | 0.0 | – | – | 0.0 |
| top3vote@win_fraction | 150 | 42.4 | 0.533 | 0.0 | – | 0.0 | 25.6 |
| top3vote@win_fraction | 100 | 48.9 | 0.564 | 0.0 | – | 0.0 | 50.1 |
| top3vote@win_fraction | 50 | 59.3 | 0.625 | 0.0 | – | 0.0 | 75.1 |
| top3vote@win_fraction | 20 | 74.0 | 0.679 | 0.0 | – | 0.0 | 90.4 |
| top5vote@margin | 200 | 34.9 | – | 5.7 | 42.7 | – | 2.4 |
| top5vote@margin | 150 | 39.9 | 0.451 | 7.0 | 40.3 | 9.2 | 26.1 |
| top5vote@margin | 100 | 46.2 | 0.456 | 7.5 | 34.7 | 7.8 | 48.7 |
| top5vote@margin | 50 | 59.1 | 0.433 | 7.8 | 15.2 | 8.5 | 69.9 |
| top5vote@margin | 20 | 70.6 | 0.448 | 1.7 | 0.2 | 1.8 | 88.8 |
| top5vote@mean_sim | 200 | 34.9 | – | 27.6 | 59.9 | – | 16.5 |
| top5vote@mean_sim | 150 | 39.9 | 0.626 | 22.3 | 56.7 | 12.9 | 35.0 |
| top5vote@mean_sim | 100 | 46.2 | 0.644 | 18.3 | 53.7 | 10.3 | 54.8 |
| top5vote@mean_sim | 50 | 59.1 | 0.732 | 8.8 | 52.5 | 4.4 | 76.4 |
| top5vote@mean_sim | 20 | 70.6 | 0.804 | 4.4 | 51.5 | 2.3 | 90.5 |
| top5vote@neg_entropy | 200 | 34.9 | – | 54.0 | 42.2 | – | 22.8 |
| top5vote@neg_entropy | 150 | 39.9 | 0.536 | 44.0 | 35.8 | 40.9 | 30.9 |
| top5vote@neg_entropy | 100 | 46.2 | 0.566 | 32.3 | 27.9 | 26.6 | 45.6 |
| top5vote@neg_entropy | 50 | 59.1 | 0.622 | 9.1 | 21.0 | 7.5 | 71.2 |
| top5vote@neg_entropy | 20 | 70.6 | 0.572 | 0.0 | – | 0.0 | 90.4 |
| top5vote@win_fraction | 200 | 34.9 | – | 0.0 | – | – | 0.0 |
| top5vote@win_fraction | 150 | 39.9 | 0.527 | 0.0 | – | 0.0 | 25.6 |
| top5vote@win_fraction | 100 | 46.2 | 0.567 | 0.0 | – | 0.0 | 50.1 |
| top5vote@win_fraction | 50 | 59.1 | 0.617 | 0.0 | – | 0.0 | 75.1 |
| top5vote@win_fraction | 20 | 70.6 | 0.611 | 0.0 | – | 0.0 | 90.4 |

**Paired cluster-bootstrap differences against max_mean_sim** (e2e = end-to-end accuracy at τ_e2e)

| method | vocab size | metric | difference | 95% CI |
|---|---|---|---|---|
| plurality@margin | 100 | auroc | -0.112 | [-0.143, -0.082] |
| plurality@margin | 100 | e2e | +0.039 | [-0.003, +0.078] |
| plurality@margin | 150 | auroc | -0.113 | [-0.156, -0.066] |
| plurality@margin | 150 | e2e | -0.047 | [-0.083, -0.012] |
| plurality@margin | 200 | e2e | -0.177 | [-0.225, -0.131] |
| plurality@margin | 20 | auroc | -0.133 | [-0.174, -0.092] |
| plurality@margin | 20 | e2e | -0.143 | [-0.178, -0.107] |
| plurality@margin | 50 | auroc | -0.139 | [-0.173, -0.109] |
| plurality@margin | 50 | e2e | -0.009 | [-0.038, +0.025] |
| plurality@mean_sim | 100 | auroc | -0.002 | [-0.005, +0.001] |
| plurality@mean_sim | 100 | e2e | -0.000 | [-0.002, +0.002] |
| plurality@mean_sim | 150 | auroc | -0.005 | [-0.011, -0.000] |
| plurality@mean_sim | 150 | e2e | -0.002 | [-0.007, +0.001] |
| plurality@mean_sim | 200 | e2e | -0.008 | [-0.022, +0.000] |
| plurality@mean_sim | 20 | auroc | +0.000 | [-0.001, +0.001] |
| plurality@mean_sim | 20 | e2e | -0.001 | [-0.004, +0.000] |
| plurality@mean_sim | 50 | auroc | -0.002 | [-0.004, -0.000] |
| plurality@mean_sim | 50 | e2e | +0.001 | [+0.000, +0.002] |
| plurality@neg_entropy | 100 | auroc | -0.072 | [-0.099, -0.044] |
| plurality@neg_entropy | 100 | e2e | +0.045 | [+0.003, +0.084] |
| plurality@neg_entropy | 150 | auroc | -0.089 | [-0.126, -0.050] |
| plurality@neg_entropy | 150 | e2e | -0.048 | [-0.085, -0.013] |
| plurality@neg_entropy | 200 | e2e | -0.178 | [-0.226, -0.133] |
| plurality@neg_entropy | 20 | auroc | -0.111 | [-0.144, -0.076] |
| plurality@neg_entropy | 20 | e2e | -0.034 | [-0.067, -0.002] |
| plurality@neg_entropy | 50 | auroc | -0.104 | [-0.138, -0.075] |
| plurality@neg_entropy | 50 | e2e | +0.021 | [-0.009, +0.054] |
| plurality@win_fraction | 100 | auroc | -0.089 | [-0.117, -0.062] |
| plurality@win_fraction | 100 | e2e | +0.049 | [+0.009, +0.089] |
| plurality@win_fraction | 150 | auroc | -0.093 | [-0.134, -0.054] |
| plurality@win_fraction | 150 | e2e | -0.068 | [-0.107, -0.032] |
| plurality@win_fraction | 200 | e2e | -0.223 | [-0.270, -0.177] |
| plurality@win_fraction | 20 | auroc | -0.118 | [-0.155, -0.082] |
| plurality@win_fraction | 20 | e2e | -0.050 | [-0.088, -0.011] |
| plurality@win_fraction | 50 | auroc | -0.119 | [-0.151, -0.091] |
| plurality@win_fraction | 50 | e2e | +0.023 | [-0.007, +0.056] |
| top3vote@margin | 100 | auroc | -0.183 | [-0.226, -0.142] |
| top3vote@margin | 100 | e2e | +0.002 | [-0.046, +0.048] |
| top3vote@margin | 150 | auroc | -0.132 | [-0.188, -0.076] |
| top3vote@margin | 150 | e2e | -0.108 | [-0.151, -0.067] |
| top3vote@margin | 200 | e2e | -0.256 | [-0.306, -0.209] |
| top3vote@margin | 20 | auroc | -0.343 | [-0.399, -0.285] |
| top3vote@margin | 20 | e2e | +0.006 | [-0.028, +0.040] |
| top3vote@margin | 50 | auroc | -0.266 | [-0.317, -0.216] |
| top3vote@margin | 50 | e2e | +0.010 | [-0.028, +0.046] |
| top3vote@mean_sim | 100 | auroc | -0.006 | [-0.014, -0.001] |
| top3vote@mean_sim | 100 | e2e | +0.063 | [+0.033, +0.094] |
| top3vote@mean_sim | 150 | auroc | -0.002 | [-0.006, +0.001] |
| top3vote@mean_sim | 150 | e2e | -0.025 | [-0.056, +0.001] |
| top3vote@mean_sim | 200 | e2e | -0.121 | [-0.157, -0.088] |
| top3vote@mean_sim | 20 | auroc | -0.006 | [-0.014, -0.000] |
| top3vote@mean_sim | 20 | e2e | +0.060 | [+0.038, +0.083] |
| top3vote@mean_sim | 50 | auroc | -0.009 | [-0.017, -0.003] |
| top3vote@mean_sim | 50 | e2e | +0.073 | [+0.050, +0.099] |
| top3vote@neg_entropy | 100 | auroc | -0.072 | [-0.102, -0.040] |
| top3vote@neg_entropy | 100 | e2e | +0.015 | [-0.023, +0.053] |
| top3vote@neg_entropy | 150 | auroc | -0.084 | [-0.119, -0.049] |
| top3vote@neg_entropy | 150 | e2e | -0.048 | [-0.081, -0.016] |
| top3vote@neg_entropy | 200 | e2e | -0.081 | [-0.125, -0.037] |
| top3vote@neg_entropy | 20 | auroc | -0.187 | [-0.225, -0.148] |
| top3vote@neg_entropy | 20 | e2e | +0.036 | [+0.013, +0.059] |
| top3vote@neg_entropy | 50 | auroc | -0.112 | [-0.142, -0.084] |
| top3vote@neg_entropy | 50 | e2e | +0.031 | [-0.003, +0.068] |
| top3vote@win_fraction | 100 | auroc | -0.090 | [-0.118, -0.065] |
| top3vote@win_fraction | 100 | e2e | +0.015 | [-0.034, +0.063] |
| top3vote@win_fraction | 150 | auroc | -0.099 | [-0.137, -0.062] |
| top3vote@win_fraction | 150 | e2e | -0.120 | [-0.166, -0.077] |
| top3vote@win_fraction | 200 | e2e | -0.290 | [-0.344, -0.241] |
| top3vote@win_fraction | 20 | auroc | -0.141 | [-0.177, -0.103] |
| top3vote@win_fraction | 20 | e2e | +0.060 | [+0.034, +0.086] |
| top3vote@win_fraction | 50 | auroc | -0.123 | [-0.152, -0.093] |
| top3vote@win_fraction | 50 | e2e | +0.060 | [+0.022, +0.096] |
| top3vote_forced | 100 | auroc | -0.155 | [-0.192, -0.120] |
| top3vote_forced | 100 | e2e | -0.242 | [-0.276, -0.210] |
| top3vote_forced | 150 | auroc | -0.132 | [-0.170, -0.094] |
| top3vote_forced | 150 | e2e | -0.060 | [-0.090, -0.028] |
| top3vote_forced | 200 | e2e | +0.083 | [+0.046, +0.122] |
| top3vote_forced | 20 | auroc | -0.318 | [-0.349, -0.284] |
| top3vote_forced | 20 | e2e | -0.775 | [-0.808, -0.740] |
| top3vote_forced | 50 | auroc | -0.248 | [-0.284, -0.213] |
| top3vote_forced | 50 | e2e | -0.544 | [-0.589, -0.499] |
| top5vote@margin | 100 | auroc | -0.199 | [-0.248, -0.152] |
| top5vote@margin | 100 | e2e | +0.002 | [-0.045, +0.045] |
| top5vote@margin | 150 | auroc | -0.182 | [-0.239, -0.123] |
| top5vote@margin | 150 | e2e | -0.115 | [-0.160, -0.072] |
| top5vote@margin | 200 | e2e | -0.266 | [-0.314, -0.217] |
| top5vote@margin | 20 | auroc | -0.369 | [-0.420, -0.316] |
| top5vote@margin | 20 | e2e | +0.043 | [+0.013, +0.073] |
| top5vote@margin | 50 | auroc | -0.314 | [-0.367, -0.258] |
| top5vote@margin | 50 | e2e | +0.008 | [-0.037, +0.049] |
| top5vote@mean_sim | 100 | auroc | -0.010 | [-0.018, -0.004] |
| top5vote@mean_sim | 100 | e2e | +0.063 | [+0.033, +0.095] |
| top5vote@mean_sim | 150 | auroc | -0.007 | [-0.015, -0.001] |
| top5vote@mean_sim | 150 | e2e | -0.026 | [-0.057, -0.000] |
| top5vote@mean_sim | 200 | e2e | -0.125 | [-0.161, -0.091] |
| top5vote@mean_sim | 20 | auroc | -0.016 | [-0.029, -0.007] |
| top5vote@mean_sim | 20 | e2e | +0.060 | [+0.038, +0.083] |
| top5vote@mean_sim | 50 | auroc | -0.016 | [-0.027, -0.007] |
| top5vote@mean_sim | 50 | e2e | +0.074 | [+0.050, +0.099] |
| top5vote@neg_entropy | 100 | auroc | -0.089 | [-0.122, -0.054] |
| top5vote@neg_entropy | 100 | e2e | -0.029 | [-0.067, +0.011] |
| top5vote@neg_entropy | 150 | auroc | -0.096 | [-0.133, -0.058] |
| top5vote@neg_entropy | 150 | e2e | -0.067 | [-0.099, -0.033] |
| top5vote@neg_entropy | 200 | e2e | -0.063 | [-0.106, -0.021] |
| top5vote@neg_entropy | 20 | auroc | -0.247 | [-0.294, -0.204] |
| top5vote@neg_entropy | 20 | e2e | +0.060 | [+0.034, +0.086] |
| top5vote@neg_entropy | 50 | auroc | -0.127 | [-0.159, -0.096] |
| top5vote@neg_entropy | 50 | e2e | +0.021 | [-0.013, +0.054] |
| top5vote@win_fraction | 100 | auroc | -0.087 | [-0.117, -0.059] |
| top5vote@win_fraction | 100 | e2e | +0.015 | [-0.034, +0.063] |
| top5vote@win_fraction | 150 | auroc | -0.105 | [-0.139, -0.069] |
| top5vote@win_fraction | 150 | e2e | -0.120 | [-0.166, -0.077] |
| top5vote@win_fraction | 200 | e2e | -0.290 | [-0.344, -0.241] |
| top5vote@win_fraction | 20 | auroc | -0.207 | [-0.244, -0.170] |
| top5vote@win_fraction | 20 | e2e | +0.060 | [+0.034, +0.086] |
| top5vote@win_fraction | 50 | auroc | -0.131 | [-0.162, -0.101] |
| top5vote@win_fraction | 50 | e2e | +0.060 | [+0.022, +0.096] |
| top5vote_forced | 100 | auroc | -0.155 | [-0.192, -0.120] |
| top5vote_forced | 100 | e2e | -0.255 | [-0.289, -0.222] |
| top5vote_forced | 150 | auroc | -0.132 | [-0.170, -0.094] |
| top5vote_forced | 150 | e2e | -0.079 | [-0.108, -0.047] |
| top5vote_forced | 200 | e2e | +0.059 | [+0.018, +0.098] |
| top5vote_forced | 20 | auroc | -0.318 | [-0.349, -0.284] |
| top5vote_forced | 20 | e2e | -0.778 | [-0.810, -0.744] |
| top5vote_forced | 50 | auroc | -0.248 | [-0.284, -0.213] |
| top5vote_forced | 50 | e2e | -0.545 | [-0.590, -0.498] |
| vote_fraction | 100 | auroc | -0.085 | [-0.113, -0.058] |
| vote_fraction | 100 | e2e | +0.049 | [+0.009, +0.089] |
| vote_fraction | 150 | auroc | -0.087 | [-0.129, -0.046] |
| vote_fraction | 150 | e2e | -0.068 | [-0.107, -0.032] |
| vote_fraction | 200 | e2e | -0.223 | [-0.270, -0.177] |
| vote_fraction | 20 | auroc | -0.122 | [-0.164, -0.082] |
| vote_fraction | 20 | e2e | -0.050 | [-0.088, -0.011] |
| vote_fraction | 50 | auroc | -0.116 | [-0.147, -0.089] |
| vote_fraction | 50 | e2e | +0.023 | [-0.007, +0.056] |

**Failure patterns, pooled over the 21 shared episodes** (cluster decisions; images in parentheses for P1)

| method | cluster decisions | wrong | P1 weak support (images) | P2 override (strict majority) | P3 competition (label owned elsewhere) | P5 mixed: n, error mixed vs pure % | P6 related name |
|---|---|---|---|---|---|---|---|
| distinct_top3_forced | 200 | 111 | 65 (1485) | 46 (28) | 0 (0) | 68: 60.3 vs 53.0 | 49 (head word 45, text-NN 43) |
| forced_top1 | 4200 | 2991 | 511 (11385) | 60 (0) | 2796 (2096) | 1428: 74.1 vs 69.7 | 310 (head word 278, text-NN 241) |
| max_mean_sim | 4200 | 2991 | 511 (11385) | 60 (0) | 2796 (2096) | 1428: 74.1 vs 69.7 | 310 (head word 278, text-NN 241) |
| plurality@win_fraction | 4200 | 3003 | 552 (12655) | 31 (0) | 2821 (2117) | 1428: 74.7 vs 69.8 | 315 (head word 285, text-NN 230) |
| plurality_forced | 4200 | 3003 | 552 (12655) | 31 (0) | 2821 (2117) | 1428: 74.7 vs 69.8 | 315 (head word 285, text-NN 230) |

**P7/P8 (image level, pooled over episodes of a size)**

| method | vocab size | absent accepted % | absent images | available rejected % | rejected, label correct | rejected, label wrong |
|---|---|---|---|---|---|---|
| max_mean_sim | 200 | – | 0 | 35.3 | 494 | 1094 |
| max_mean_sim | 150 | 43.5 | 5755 | 38.5 | 2350 | 4086 |
| max_mean_sim | 100 | 36.8 | 11262 | 44.7 | 2024 | 2989 |
| max_mean_sim | 50 | 20.8 | 16881 | 45.5 | 1425 | 1121 |
| max_mean_sim | 20 | 11.4 | 20329 | 48.3 | 740 | 299 |
| plurality@win_fraction | 200 | – | 0 | 92.5 | 1485 | 2673 |
| plurality@win_fraction | 150 | 6.0 | 5755 | 90.1 | 6100 | 8963 |
| plurality@win_fraction | 100 | 5.9 | 11262 | 86.3 | 4352 | 5331 |
| plurality@win_fraction | 50 | 11.9 | 16881 | 78.0 | 2346 | 2022 |
| plurality@win_fraction | 20 | 16.4 | 20329 | 57.5 | 772 | 464 |

**Mean vs plurality label (forced), image level, pooled over episodes of a size**

| vocab size | images with different labels | image decisions | mean right, vote wrong | vote right, mean wrong |
|---|---|---|---|---|
| 200 | 631 | 4496 | 139 | 121 |
| 150 | 2588 | 22480 | 379 | 436 |
| 100 | 2735 | 22480 | 307 | 258 |
| 50 | 2462 | 22480 | 151 | 83 |
| 20 | 2135 | 22480 | 76 | 34 |

Source: `outputs/cub/fixed_cluster/vitb32/seed0_known_c0/de0ed3ecfe4b`.

## Protocol R: refinement from GCD clusters (CUB, closed vocabulary; not a paper setting)
| clustering | mode | initial sACC % | final sACC % | final − initial (paired image bootstrap) | initial ACC % | final ACC % | final sACC Old % | final sACC New % | names | iterations | stop | P4 flips correct→wrong / wrong→correct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| seed0_known_c0 | partially_supervised | 41.5 | 49.2 | +7.7 [+5.8, +9.5] | 50.8 | 51.2 | 53.4 | 47.0 | 192 | 2 | converged | 651 / 998 |
| seed0_known_c0 | unsupervised | 37.6 | 49.2 | +11.6 [+9.9, +13.1] | 50.8 | 51.2 | 53.4 | 47.0 | 192 | 2 | converged | 494 / 1015 |
| seed0_known_c1 | partially_supervised | 40.9 | 49.2 | +8.3 [+6.4, +10.0] | 49.7 | 51.2 | 53.4 | 47.0 | 192 | 3 | converged | 650 / 1022 |
| seed0_known_c1 | unsupervised | 36.8 | 49.2 | +12.4 [+10.8, +13.9] | 49.7 | 51.2 | 53.4 | 47.0 | 192 | 2 | converged | 450 / 1006 |
| seed0_known_c2 | partially_supervised | 38.9 | 49.2 | +10.3 [+8.3, +12.0] | 49.9 | 51.2 | 53.4 | 47.0 | 192 | 3 | converged | 656 / 1117 |
| seed0_known_c2 | unsupervised | 36.5 | 49.2 | +12.7 [+11.1, +14.3] | 49.9 | 51.2 | 53.4 | 47.0 | 192 | 2 | converged | 479 / 1049 |

Mean ± SD over clustering seeds: partially_supervised: sACC 40.4±1.3 → 49.2±0.0, ACC 50.1 → 51.2 (n = 3); unsupervised: sACC 37.0±0.6 → 49.2±0.0, ACC 50.1 → 51.2 (n = 3)

## Protocol A: reproduction (ImageNet-100)
**Pending (needs ImageNet and a GPU)**: run `kaggle/e2_imagenet100.ipynb`, or on a GPU machine

```bash
E2="python -m e2 --config configs/imagenet100.yaml --set e1.runs_root=<Experiment 1 runs>"
$E2 vocab && $E2 repro features && $E2 repro cluster && $E2 repro name && $E2 repro evaluate
```

