# Experiment 1 baseline report: cub

Generated 2026-11-09 05:56 UTC from `configs/cub.yaml`. Protocol: PROTOCOL.md. Every number below is read from the linked files.

## Data

Manifest sha256 `0c7be4f50dee1407…` ([manifest.csv](../data/manifest.csv)). Labelled 1498, unlabelled 4496 (known-class 1499, novel-class 2997), held-out dev 1433, calibration 1451. Files missing: 0; corrupt: 0. Upstream partition replay: passed.

## Run status

| Stage | Run | Status |
|---|---|---|
| GCD training | seed 0 | completed |
| GCD features | seed 0 | completed |
| SS-k-means (known K) | seed 0, clustering seed 0 | completed |
| SS-k-means (known K) | seed 0, clustering seed 1 | completed |
| SS-k-means (known K) | seed 0, clustering seed 2 | completed |
| SS-k-means (estimated K) | seed 0, clustering seed 0 | completed |
| SS-k-means (estimated K) | seed 0, clustering seed 1 | pending |
| SS-k-means (estimated K) | seed 0, clustering seed 2 | pending |
| GCD training | seed 1 | pending |
| GCD features | seed 1 | pending |
| SS-k-means (known K) | seed 1, clustering seed 0 | pending |
| SS-k-means (known K) | seed 1, clustering seed 1 | pending |
| SS-k-means (known K) | seed 1, clustering seed 2 | pending |
| SS-k-means (estimated K) | seed 1, clustering seed 0 | pending |
| SS-k-means (estimated K) | seed 1, clustering seed 1 | pending |
| SS-k-means (estimated K) | seed 1, clustering seed 2 | pending |
| GCD training | seed 2 | pending |
| GCD features | seed 2 | pending |
| SS-k-means (known K) | seed 2, clustering seed 0 | pending |
| SS-k-means (known K) | seed 2, clustering seed 1 | pending |
| SS-k-means (known K) | seed 2, clustering seed 2 | pending |
| SS-k-means (estimated K) | seed 2, clustering seed 0 | pending |
| SS-k-means (estimated K) | seed 2, clustering seed 1 | pending |
| SS-k-means (estimated K) | seed 2, clustering seed 2 | pending |
| CLIP vitb32 | ViT-B-32-quickgelu | extracted; reference check passed |
| Grounding | seed 0, known K, c0, vitb32 | completed |
| Grounding | seed 0, known K, c1, vitb32 | pending |
| Grounding | seed 0, known K, c2, vitb32 | pending |
| Grounding | seed 1, known K, c0, vitb32 | pending |
| Grounding | seed 1, known K, c1, vitb32 | pending |
| Grounding | seed 1, known K, c2, vitb32 | pending |
| Grounding | seed 2, known K, c0, vitb32 | pending |
| Grounding | seed 2, known K, c1, vitb32 | pending |
| Grounding | seed 2, known K, c2, vitb32 | pending |
| CLIP vitb16 | ViT-B-16-quickgelu | extracted; reference check passed |
| Grounding | seed 0, known K, c0, vitb16 | completed |
| Grounding | seed 0, known K, c1, vitb16 | pending |
| Grounding | seed 0, known K, c2, vitb16 | pending |
| Grounding | seed 1, known K, c0, vitb16 | pending |
| Grounding | seed 1, known K, c1, vitb16 | pending |
| Grounding | seed 1, known K, c2, vitb16 | pending |
| Grounding | seed 2, known K, c0, vitb16 | pending |
| Grounding | seed 2, known K, c1, vitb16 | pending |
| Grounding | seed 2, known K, c2, vitb16 | pending |

Pending (32): clustering seed 0/estimated/c1, clustering seed 0/estimated/c2, GCD training seed 1, clustering seed 1/known/c0, clustering seed 1/known/c1, clustering seed 1/known/c2, clustering seed 1/estimated/c0, clustering seed 1/estimated/c1, clustering seed 1/estimated/c2, GCD training seed 2, clustering seed 2/known/c0, clustering seed 2/known/c1, clustering seed 2/known/c2, clustering seed 2/estimated/c0, clustering seed 2/estimated/c1, clustering seed 2/estimated/c2, grounding seed 0/c1/vitb32, grounding seed 0/c2/vitb32, grounding seed 1/c0/vitb32, grounding seed 1/c1/vitb32, grounding seed 1/c2/vitb32, grounding seed 2/c0/vitb32, grounding seed 2/c1/vitb32, grounding seed 2/c2/vitb32, grounding seed 0/c1/vitb16, grounding seed 0/c2/vitb16, grounding seed 1/c0/vitb16, grounding seed 1/c1/vitb16, grounding seed 1/c2/vitb16, grounding seed 2/c0/vitb16, grounding seed 2/c1/vitb16, grounding seed 2/c2/vitb16

## Discovery (GCD) on the unlabelled population

v2: one Hungarian assignment over the whole population; All/Old/New from it (used only here).

| Training seed | K | Clustering seed | All | Old | New | NMI | ARI | Empty | Singleton | Converged | Files |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 200 (known) | 0 | 50.78 | 55.84 | 48.25 | 0.7825 | 0.3806 | 0 | 39 | True | [assignments.csv](../gcd/seed0_3a2d202f8444/clustering/known_c0_fcf46ef1931c/assignments.csv) |
| 0 | 200 (known) | 1 | 49.67 | 52.57 | 48.21 | 0.7816 | 0.3809 | 0 | 44 | True | [assignments.csv](../gcd/seed0_3a2d202f8444/clustering/known_c1_fcf46ef1931c/assignments.csv) |
| 0 | 200 (known) | 2 | 49.93 | 53.64 | 48.08 | 0.7818 | 0.3864 | 0 | 42 | True | [assignments.csv](../gcd/seed0_3a2d202f8444/clustering/known_c2_fcf46ef1931c/assignments.csv) |
| 0 | 199 (estimated) | 0 | 49.15 | 51.10 | 48.18 | 0.7839 | 0.3835 | 0 | 40 | True | [assignments.csv](../gcd/seed0_3a2d202f8444/clustering/estimated_c0_fcf46ef1931c/assignments.csv) |

## Grounding: training seed 0, known K, clustering seed 0, CLIP vitb32

Dev choices: prompt `ensemble80`, T* = 0.02 (pilot T = 0.01, MCM T = 1.0). Thresholds: [thresholds.json](../grounding/seed0_known_c0/vitb32_831109500c95/thresholds.json). Predictions: [cluster_predictions.csv.gz](../grounding/seed0_known_c0/vitb32_831109500c95/cluster_predictions.csv.gz), [image_predictions_complete_vocabulary.csv.gz](../grounding/seed0_known_c0/vitb32_831109500c95/image_predictions_complete_vocabulary.csv.gz).

Complete vocabulary, forced naming (image level, all unlabelled members represent each cluster):

| Method | Image-weighted | Class-macro | Old | New |
|---|---|---|---|---|
| forced_top1 (label = argmax selected prompts) | 40.04 | 42.38 | 48.90 | 35.60 |
| mcm_aggregate (label = argmax MCM prompts) | 40.19 | 43.57 | 52.43 | 34.07 |

By vocabulary size (mean over vocabulary seeds; AUROC positive class = true label in V; end-to-end = correct label accepted, or absent label rejected; thresholds from calibration):

| Size | Method | AUROC avail. | AURC | Forced acc. (avail.) | Coverage @τe2e | Acc. accepted @τe2e | False accept. absent @τe2e | End-to-end @τe2e | End-to-end @τ95 |
|---|---|---|---|---|---|---|---|---|---|
| 200 | forced_top1 | — | 0.5996 | 40.04 | 100.00 | 40.04 | — | 40.04 | 40.04 |
| 200 | margin | — | 0.4051 | 40.04 | 6.12 | 89.45 | — | 5.47 | 37.37 |
| 200 | max_mean_sim | — | 0.4505 | 40.04 | 64.68 | 44.91 | — | 29.05 | 38.50 |
| 200 | mcm_aggregate | — | 0.4738 | 40.19 | 0.00 | — | — | 0.00 | 22.18 |
| 200 | mcm_image_mean | — | 0.5188 | 40.19 | 0.00 | — | — | 0.00 | 24.13 |
| 200 | msp_dev | — | 0.3825 | 40.04 | 1.62 | 97.26 | — | 1.58 | 35.63 |
| 200 | msp_pilot | — | 0.3750 | 40.04 | 5.67 | 94.51 | — | 5.36 | 37.79 |
| 200 | neg_softmax_entropy | — | 0.4718 | 40.04 | 1.62 | 97.26 | — | 1.58 | 37.50 |
| 200 | neg_vote_entropy | — | 0.3916 | 40.04 | 12.83 | 86.83 | — | 11.14 | 35.28 |
| 200 | vote_fraction | — | 0.3919 | 40.04 | 7.52 | 87.87 | — | 6.61 | 38.41 |
| 150 | forced_top1 | 0.5000 | 0.6640 | 45.17 | 100.00 | 33.60 | 100.00 | 33.60 | 33.60 |
| 150 | margin | 0.5367 | 0.5035 | 45.17 | 8.20 | 71.09 | 6.79 | 29.69 | 33.94 |
| 150 | max_mean_sim | 0.6326 | 0.5156 | 45.17 | 56.90 | 40.71 | 43.61 | 37.61 | 35.77 |
| 150 | mcm_aggregate | 0.5835 | 0.5453 | 45.27 | 0.00 | — | 0.00 | 25.60 | 33.68 |
| 150 | mcm_image_mean | 0.6020 | 0.5729 | 45.27 | 0.00 | — | 0.00 | 25.60 | 33.68 |
| 150 | msp_dev | 0.5719 | 0.4729 | 45.17 | 1.40 | 97.15 | 0.05 | 26.94 | 36.42 |
| 150 | msp_pilot | 0.5575 | 0.4722 | 45.17 | 9.80 | 72.28 | 7.79 | 30.70 | 36.33 |
| 150 | neg_softmax_entropy | 0.5898 | 0.5397 | 45.17 | 1.44 | 94.72 | 0.09 | 26.94 | 35.12 |
| 150 | neg_vote_entropy | 0.5434 | 0.4886 | 45.17 | 13.05 | 73.10 | 8.96 | 32.83 | 35.94 |
| 150 | vote_fraction | 0.5455 | 0.4924 | 45.17 | 8.93 | 75.27 | 6.03 | 30.78 | 35.65 |
| 100 | forced_top1 | 0.5000 | 0.7411 | 51.94 | 100.00 | 25.89 | 100.00 | 25.89 | 25.89 |
| 100 | margin | 0.5530 | 0.5968 | 51.94 | 11.33 | 61.35 | 7.92 | 52.87 | 28.14 |
| 100 | max_mean_sim | 0.6545 | 0.5870 | 51.94 | 46.06 | 36.61 | 36.86 | 48.52 | 35.08 |
| 100 | mcm_aggregate | 0.6264 | 0.6021 | 53.23 | 0.00 | — | 0.00 | 50.10 | 26.53 |
| 100 | mcm_image_mean | 0.6422 | 0.6104 | 53.23 | 0.00 | — | 0.00 | 50.10 | 26.53 |
| 100 | msp_dev | 0.6130 | 0.5645 | 51.94 | 4.59 | 78.32 | 1.81 | 52.78 | 29.64 |
| 100 | msp_pilot | 0.5822 | 0.5723 | 51.94 | 13.55 | 61.13 | 9.27 | 53.63 | 29.92 |
| 100 | neg_softmax_entropy | 0.6297 | 0.5971 | 51.94 | 2.45 | 91.34 | 0.32 | 52.15 | 31.84 |
| 100 | neg_vote_entropy | 0.5825 | 0.5894 | 51.94 | 13.27 | 58.50 | 9.63 | 53.06 | 37.78 |
| 100 | vote_fraction | 0.5689 | 0.5939 | 51.94 | 9.76 | 64.90 | 5.83 | 53.47 | 29.84 |
| 50 | forced_top1 | 0.5000 | 0.8411 | 63.90 | 100.00 | 15.89 | 100.00 | 15.89 | 15.89 |
| 50 | margin | 0.5869 | 0.7377 | 63.90 | 17.67 | 34.83 | 14.92 | 69.93 | 19.99 |
| 50 | max_mean_sim | 0.7487 | 0.6951 | 63.90 | 29.19 | 32.90 | 20.77 | 69.04 | 44.42 |
| 50 | mcm_aggregate | 0.7201 | 0.7144 | 63.16 | 0.00 | — | 0.00 | 75.09 | 15.71 |
| 50 | mcm_image_mean | 0.7330 | 0.7155 | 63.16 | 0.00 | — | 0.00 | 75.09 | 15.71 |
| 50 | msp_dev | 0.6748 | 0.7094 | 63.90 | 8.31 | 47.56 | 5.68 | 74.76 | 16.08 |
| 50 | msp_pilot | 0.6272 | 0.7201 | 63.90 | 20.48 | 37.36 | 16.44 | 70.22 | 16.96 |
| 50 | neg_softmax_entropy | 0.7151 | 0.7100 | 63.90 | 5.25 | 51.81 | 3.23 | 75.37 | 23.18 |
| 50 | neg_vote_entropy | 0.6431 | 0.7383 | 63.90 | 14.83 | 35.91 | 12.22 | 71.17 | 41.49 |
| 50 | vote_fraction | 0.6309 | 0.7416 | 63.90 | 14.38 | 36.40 | 11.84 | 71.34 | 19.48 |
| 20 | forced_top1 | 0.5000 | 0.9247 | 78.88 | 100.00 | 7.53 | 100.00 | 7.53 | 7.53 |
| 20 | margin | 0.6821 | 0.8544 | 78.88 | 20.88 | 21.78 | 18.19 | 78.19 | 12.40 |
| 20 | max_mean_sim | 0.8203 | 0.8186 | 78.88 | 15.24 | 29.85 | 11.37 | 84.38 | 65.39 |
| 20 | mcm_aggregate | 0.7754 | 0.8394 | 78.53 | 0.12 | 84.00 | 0.04 | 90.48 | 7.49 |
| 20 | mcm_image_mean | 0.7768 | 0.8413 | 78.53 | 0.12 | 84.00 | 0.04 | 90.48 | 7.49 |
| 20 | msp_dev | 0.7601 | 0.8377 | 78.88 | 23.66 | 22.26 | 19.96 | 77.57 | 7.53 |
| 20 | msp_pilot | 0.7221 | 0.8452 | 78.88 | 28.38 | 18.84 | 25.27 | 72.72 | 7.90 |
| 20 | neg_softmax_entropy | 0.7751 | 0.8379 | 78.88 | 14.90 | 27.42 | 11.73 | 83.87 | 16.45 |
| 20 | neg_vote_entropy | 0.7085 | 0.8610 | 78.88 | 16.37 | 20.92 | 14.16 | 81.00 | 42.74 |
| 20 | vote_fraction | 0.6965 | 0.8619 | 78.88 | 18.95 | 20.84 | 16.43 | 79.47 | 9.24 |

Cluster-bootstrap 95% intervals (1000 resamples; clusters carry their images and episodes): see [bootstrap.json](../grounding/seed0_known_c0/vitb32_831109500c95/bootstrap.json).

| Method, size | End-to-end @τe2e | AUROC avail. |
|---|---|---|
| forced_top1|size=100 | 25.89 [22.8, 29.1] | 50.00 [50.0, 50.0] |
| forced_top1|size=150 | 33.60 [30.0, 37.4] | 50.00 [50.0, 50.0] |
| forced_top1|size=20 | 7.53 [6.3, 8.8] | 50.00 [50.0, 50.0] |
| forced_top1|size=200 | 40.04 [35.2, 45.4] | — |
| forced_top1|size=50 | 15.89 [14.1, 18.0] | 50.00 [50.0, 50.0] |
| margin|size=100 | 52.87 [50.4, 55.5] | 55.30 [51.4, 59.6] |
| margin|size=150 | 29.69 [27.0, 32.8] | 53.67 [49.3, 57.8] |
| margin|size=20 | 78.19 [75.5, 81.1] | 68.21 [62.9, 72.8] |
| margin|size=200 | 5.47 [2.9, 8.9] | — |
| margin|size=50 | 69.93 [66.9, 73.1] | 58.69 [54.5, 63.2] |
| max_mean_sim|size=100 | 48.52 [44.6, 52.5] | 65.45 [62.0, 69.2] |
| max_mean_sim|size=150 | 37.61 [33.8, 41.9] | 63.26 [59.4, 67.0] |
| max_mean_sim|size=20 | 84.38 [81.6, 87.5] | 82.03 [78.4, 84.9] |
| max_mean_sim|size=200 | 29.05 [24.1, 34.4] | — |
| max_mean_sim|size=50 | 69.04 [65.1, 73.0] | 74.87 [71.3, 78.4] |
| mcm_aggregate|size=100 | 50.10 [47.8, 52.4] | 62.64 [59.3, 66.1] |
| mcm_aggregate|size=150 | 25.60 [23.5, 27.7] | 58.35 [54.8, 62.0] |
| mcm_aggregate|size=20 | 90.48 [89.1, 91.8] | 77.54 [74.0, 80.5] |
| mcm_aggregate|size=200 | 0.00 [0.0, 0.0] | — |
| mcm_aggregate|size=50 | 75.09 [72.7, 77.4] | 72.01 [68.5, 75.3] |
| mcm_image_mean|size=100 | 50.10 [47.8, 52.4] | 64.22 [60.9, 68.0] |
| mcm_image_mean|size=150 | 25.60 [23.5, 27.7] | 60.20 [56.3, 64.0] |
| mcm_image_mean|size=20 | 90.48 [89.1, 91.8] | 77.68 [74.0, 80.7] |
| mcm_image_mean|size=200 | 0.00 [0.0, 0.0] | — |
| mcm_image_mean|size=50 | 75.09 [72.7, 77.4] | 73.30 [69.8, 76.5] |
| msp_dev|size=100 | 52.78 [50.4, 55.2] | 61.30 [57.9, 65.1] |
| msp_dev|size=150 | 26.94 [24.7, 29.3] | 57.19 [53.3, 61.1] |
| msp_dev|size=20 | 77.57 [73.8, 81.1] | 76.01 [72.4, 79.3] |
| msp_dev|size=200 | 1.58 [0.1, 3.4] | — |
| msp_dev|size=50 | 74.76 [72.0, 77.5] | 67.48 [64.0, 71.2] |
| msp_pilot|size=100 | 53.63 [51.0, 56.3] | 58.22 [54.4, 62.1] |
| msp_pilot|size=150 | 30.70 [27.8, 34.1] | 55.75 [52.0, 59.6] |
| msp_pilot|size=20 | 72.72 [69.2, 76.4] | 72.21 [67.9, 76.1] |
| msp_pilot|size=200 | 5.36 [2.9, 8.6] | — |
| msp_pilot|size=50 | 70.22 [67.2, 73.3] | 62.72 [59.1, 66.8] |
| neg_softmax_entropy|size=100 | 52.15 [49.8, 54.4] | 62.97 [59.6, 66.5] |
| neg_softmax_entropy|size=150 | 26.94 [24.7, 29.3] | 58.98 [55.4, 62.7] |
| neg_softmax_entropy|size=20 | 83.87 [81.0, 86.5] | 77.51 [74.0, 80.6] |
| neg_softmax_entropy|size=200 | 1.58 [0.1, 3.4] | — |
| neg_softmax_entropy|size=50 | 75.37 [72.9, 77.9] | 71.51 [68.2, 74.8] |
| neg_vote_entropy|size=100 | 53.06 [50.4, 55.9] | 58.25 [55.2, 61.5] |
| neg_vote_entropy|size=150 | 32.83 [29.7, 36.6] | 54.34 [50.7, 58.0] |
| neg_vote_entropy|size=20 | 81.00 [78.0, 83.7] | 70.85 [67.2, 74.3] |
| neg_vote_entropy|size=200 | 11.14 [7.4, 15.9] | — |
| neg_vote_entropy|size=50 | 71.17 [68.1, 74.1] | 64.31 [60.6, 68.0] |
| vote_fraction|size=100 | 53.47 [50.9, 56.0] | 56.89 [53.6, 60.4] |
| vote_fraction|size=150 | 30.78 [28.0, 34.1] | 54.55 [50.6, 58.4] |
| vote_fraction|size=20 | 79.47 [76.2, 82.5] | 69.65 [65.1, 73.7] |
| vote_fraction|size=200 | 6.61 [3.7, 10.1] | — |
| vote_fraction|size=50 | 71.34 [68.3, 74.2] | 63.09 [59.7, 66.7] |

Failure cases (max_mean_sim, tau_e2e; full list [failure_cases.json](../grounding/seed0_known_c0/vitb32_831109500c95/failure_cases.json)):

* Impure clusters: cluster 190 → 'Acadian Flycatcher' (score 0.313, purity 0.07; Rusty Blackbird:2; Great Crested Flycatcher:2; Chipping Sparrow:2); cluster 160 → 'Long tailed Jaeger' (score 0.299, purity 0.08; Pelagic Cormorant:1; American Crow:1; Long tailed Jaeger:1); cluster 199 → 'Red breasted Merganser' (score 0.286, purity 0.18; Gadwall:3; Pigeon Guillemot:2; Red breasted Merganser:2); cluster 50 → 'Acadian Flycatcher' (score 0.341, purity 0.19; Acadian Flycatcher:12; Great Crested Flycatcher:10; Western Wood Pewee:9)
* Confident wrong labels: cluster 101 → 'Carolina Wren' (score 0.375, purity 0.50; Field Sparrow:1; Carolina Wren:1); cluster 153 → 'Swainson Warbler' (score 0.367, purity 1.00; Red eyed Vireo:1); cluster 179 → 'Tropical Kingbird' (score 0.364, purity 1.00; Black billed Cuckoo:1); cluster 176 → 'Marsh Wren' (score 0.358, purity 1.00; Carolina Wren:1)
* Absent label replaced and accepted: shared_150_s3: cluster of 'Field Sparrow' named 'Carolina Wren' (score 0.375); shared_100_s1: cluster of 'Field Sparrow' named 'Carolina Wren' (score 0.375); shared_100_s4: cluster of 'Field Sparrow' named 'Carolina Wren' (score 0.375); shared_50_s0: cluster of 'Belted Kingfisher' named 'Ringed Kingfisher' (score 0.372)

Other files: [metrics_by_episode.csv](../grounding/seed0_known_c0/vitb32_831109500c95/metrics_by_episode.csv), [risk_coverage_curves.csv](../grounding/seed0_known_c0/vitb32_831109500c95/risk_coverage_curves.csv), [cluster_diagnostics.csv](../grounding/seed0_known_c0/vitb32_831109500c95/cluster_diagnostics.csv), [paired_diagnostic.csv](../grounding/seed0_known_c0/vitb32_831109500c95/paired_diagnostic.csv), [episodes.csv](../grounding/seed0_known_c0/vitb32_831109500c95/episodes.csv).

## Grounding: training seed 0, known K, clustering seed 0, CLIP vitb16

Dev choices: prompt `ensemble80`, T* = 0.02 (pilot T = 0.01, MCM T = 1.0). Thresholds: [thresholds.json](../grounding/seed0_known_c0/vitb16_831109500c95/thresholds.json). Predictions: [cluster_predictions.csv.gz](../grounding/seed0_known_c0/vitb16_831109500c95/cluster_predictions.csv.gz), [image_predictions_complete_vocabulary.csv.gz](../grounding/seed0_known_c0/vitb16_831109500c95/image_predictions_complete_vocabulary.csv.gz).

Complete vocabulary, forced naming (image level, all unlabelled members represent each cluster):

| Method | Image-weighted | Class-macro | Old | New |
|---|---|---|---|---|
| forced_top1 (label = argmax selected prompts) | 40.46 | 43.08 | 49.43 | 35.97 |
| mcm_aggregate (label = argmax MCM prompts) | 38.92 | 41.68 | 48.37 | 34.20 |

By vocabulary size (mean over vocabulary seeds; AUROC positive class = true label in V; end-to-end = correct label accepted, or absent label rejected; thresholds from calibration):

| Size | Method | AUROC avail. | AURC | Forced acc. (avail.) | Coverage @τe2e | Acc. accepted @τe2e | False accept. absent @τe2e | End-to-end @τe2e | End-to-end @τ95 |
|---|---|---|---|---|---|---|---|---|---|
| 200 | forced_top1 | — | 0.5954 | 40.46 | 100.00 | 40.46 | — | 40.46 | 40.46 |
| 200 | margin | — | 0.4003 | 40.46 | 7.50 | 88.43 | — | 6.63 | 36.97 |
| 200 | max_mean_sim | — | 0.4525 | 40.46 | 65.59 | 47.24 | — | 30.98 | 37.81 |
| 200 | mcm_aggregate | — | 0.4957 | 38.92 | 0.00 | — | — | 0.00 | 21.84 |
| 200 | mcm_image_mean | — | 0.5447 | 38.92 | 0.00 | — | — | 0.00 | 24.49 |
| 200 | msp_dev | — | 0.3850 | 40.46 | 0.65 | 100.00 | — | 0.65 | 35.14 |
| 200 | msp_pilot | — | 0.3745 | 40.46 | 8.01 | 89.17 | — | 7.14 | 37.41 |
| 200 | neg_softmax_entropy | — | 0.4799 | 40.46 | 3.29 | 81.08 | — | 2.67 | 36.57 |
| 200 | neg_vote_entropy | — | 0.3868 | 40.46 | 18.17 | 83.84 | — | 15.24 | 35.50 |
| 200 | vote_fraction | — | 0.3786 | 40.46 | 8.59 | 89.64 | — | 7.70 | 38.97 |
| 150 | forced_top1 | 0.5000 | 0.6664 | 44.86 | 100.00 | 33.36 | 100.00 | 33.36 | 33.36 |
| 150 | margin | 0.5232 | 0.5027 | 44.86 | 9.92 | 69.36 | 8.66 | 30.24 | 34.06 |
| 150 | max_mean_sim | 0.6350 | 0.5164 | 44.86 | 57.22 | 42.78 | 41.30 | 39.52 | 37.39 |
| 150 | mcm_aggregate | 0.5950 | 0.5631 | 43.41 | 0.00 | — | 0.00 | 25.60 | 32.29 |
| 150 | mcm_image_mean | 0.6269 | 0.5901 | 43.41 | 0.00 | — | 0.00 | 25.60 | 32.29 |
| 150 | msp_dev | 0.5700 | 0.4811 | 44.86 | 1.90 | 91.50 | 0.58 | 27.14 | 37.23 |
| 150 | msp_pilot | 0.5457 | 0.4820 | 44.86 | 10.67 | 72.79 | 8.65 | 31.14 | 36.49 |
| 150 | neg_softmax_entropy | 0.5920 | 0.5360 | 44.86 | 4.07 | 76.13 | 2.29 | 28.11 | 36.64 |
| 150 | neg_vote_entropy | 0.5386 | 0.4947 | 44.86 | 18.75 | 66.80 | 16.02 | 34.00 | 36.47 |
| 150 | vote_fraction | 0.5305 | 0.4935 | 44.86 | 11.01 | 72.26 | 8.91 | 31.26 | 35.51 |
| 100 | forced_top1 | 0.5000 | 0.7458 | 50.98 | 100.00 | 25.42 | 100.00 | 25.42 | 25.42 |
| 100 | margin | 0.5537 | 0.5925 | 50.98 | 13.48 | 57.03 | 10.26 | 52.46 | 28.39 |
| 100 | max_mean_sim | 0.6693 | 0.5807 | 50.98 | 48.02 | 37.56 | 36.97 | 49.61 | 40.50 |
| 100 | mcm_aggregate | 0.6433 | 0.6211 | 49.31 | 0.00 | — | 0.00 | 50.10 | 24.59 |
| 100 | mcm_image_mean | 0.6667 | 0.6336 | 49.31 | 0.00 | — | 0.00 | 50.10 | 24.59 |
| 100 | msp_dev | 0.6241 | 0.5642 | 50.98 | 3.07 | 86.44 | 0.83 | 52.29 | 29.98 |
| 100 | msp_pilot | 0.5882 | 0.5736 | 50.98 | 15.23 | 60.35 | 10.89 | 53.64 | 30.97 |
| 100 | neg_softmax_entropy | 0.6473 | 0.5936 | 50.98 | 6.98 | 63.83 | 4.49 | 51.98 | 33.39 |
| 100 | neg_vote_entropy | 0.5745 | 0.5965 | 50.98 | 18.63 | 52.62 | 14.65 | 52.37 | 36.57 |
| 100 | vote_fraction | 0.5518 | 0.6000 | 50.98 | 12.00 | 57.24 | 9.26 | 52.27 | 28.52 |
| 50 | forced_top1 | 0.5000 | 0.8436 | 62.93 | 100.00 | 15.64 | 100.00 | 15.64 | 15.64 |
| 50 | margin | 0.5967 | 0.7280 | 62.93 | 18.86 | 35.76 | 15.75 | 69.96 | 21.04 |
| 50 | max_mean_sim | 0.7612 | 0.6978 | 62.93 | 31.84 | 33.37 | 22.40 | 68.85 | 53.55 |
| 50 | mcm_aggregate | 0.7193 | 0.7293 | 62.03 | 0.00 | — | 0.00 | 75.09 | 15.41 |
| 50 | mcm_image_mean | 0.7407 | 0.7331 | 62.03 | 0.00 | — | 0.00 | 75.09 | 15.41 |
| 50 | msp_dev | 0.6787 | 0.7036 | 62.93 | 7.00 | 58.63 | 4.09 | 75.86 | 16.22 |
| 50 | msp_pilot | 0.6366 | 0.7144 | 62.93 | 22.64 | 34.06 | 19.16 | 68.42 | 18.10 |
| 50 | neg_softmax_entropy | 0.7134 | 0.7065 | 62.93 | 13.87 | 44.58 | 9.10 | 74.40 | 25.63 |
| 50 | neg_vote_entropy | 0.6374 | 0.7355 | 62.93 | 19.54 | 35.17 | 16.26 | 69.71 | 41.78 |
| 50 | vote_fraction | 0.6279 | 0.7368 | 62.93 | 16.04 | 35.40 | 13.42 | 70.69 | 17.86 |
| 20 | forced_top1 | 0.5000 | 0.9260 | 77.52 | 100.00 | 7.40 | 100.00 | 7.40 | 7.40 |
| 20 | margin | 0.7254 | 0.8351 | 77.52 | 25.57 | 21.50 | 21.94 | 75.89 | 11.97 |
| 20 | max_mean_sim | 0.8332 | 0.8208 | 77.52 | 13.53 | 31.18 | 9.60 | 85.78 | 72.71 |
| 20 | mcm_aggregate | 0.7879 | 0.8422 | 76.27 | 1.31 | 57.74 | 0.92 | 90.03 | 7.28 |
| 20 | mcm_image_mean | 0.7957 | 0.8422 | 76.27 | 3.67 | 50.30 | 2.49 | 89.44 | 7.28 |
| 20 | msp_dev | 0.7787 | 0.8299 | 77.52 | 17.68 | 27.41 | 13.94 | 82.59 | 7.40 |
| 20 | msp_pilot | 0.7498 | 0.8312 | 77.52 | 30.71 | 19.44 | 27.03 | 71.70 | 8.03 |
| 20 | neg_softmax_entropy | 0.7950 | 0.8307 | 77.52 | 20.48 | 25.23 | 16.40 | 80.77 | 16.22 |
| 20 | neg_vote_entropy | 0.7195 | 0.8555 | 77.52 | 18.65 | 21.25 | 16.02 | 79.86 | 43.07 |
| 20 | vote_fraction | 0.7218 | 0.8551 | 77.52 | 20.57 | 20.70 | 17.77 | 78.59 | 8.39 |

Cluster-bootstrap 95% intervals (1000 resamples; clusters carry their images and episodes): see [bootstrap.json](../grounding/seed0_known_c0/vitb16_831109500c95/bootstrap.json).

| Method, size | End-to-end @τe2e | AUROC avail. |
|---|---|---|
| forced_top1|size=100 | 25.42 [22.4, 28.8] | 50.00 [50.0, 50.0] |
| forced_top1|size=150 | 33.36 [29.7, 37.5] | 50.00 [50.0, 50.0] |
| forced_top1|size=20 | 7.40 [6.3, 8.6] | 50.00 [50.0, 50.0] |
| forced_top1|size=200 | 40.46 [35.7, 46.1] | — |
| forced_top1|size=50 | 15.64 [13.6, 17.8] | 50.00 [50.0, 50.0] |
| margin|size=100 | 52.46 [50.0, 55.1] | 55.37 [51.2, 59.6] |
| margin|size=150 | 30.24 [27.3, 33.5] | 52.32 [48.5, 56.2] |
| margin|size=20 | 75.89 [72.8, 78.7] | 72.54 [67.9, 77.1] |
| margin|size=200 | 6.63 [3.7, 10.3] | — |
| margin|size=50 | 69.96 [67.3, 72.9] | 59.67 [55.5, 63.8] |
| max_mean_sim|size=100 | 49.61 [45.9, 53.7] | 66.93 [63.7, 70.6] |
| max_mean_sim|size=150 | 39.52 [35.7, 44.0] | 63.50 [59.4, 67.5] |
| max_mean_sim|size=20 | 85.78 [83.3, 88.3] | 83.32 [80.0, 86.0] |
| max_mean_sim|size=200 | 30.98 [25.8, 36.7] | — |
| max_mean_sim|size=50 | 68.85 [64.7, 73.5] | 76.12 [72.9, 79.4] |
| mcm_aggregate|size=100 | 50.10 [47.8, 52.4] | 64.33 [61.0, 68.0] |
| mcm_aggregate|size=150 | 25.60 [23.5, 27.7] | 59.50 [55.8, 63.3] |
| mcm_aggregate|size=20 | 90.03 [88.5, 91.5] | 78.79 [75.2, 81.8] |
| mcm_aggregate|size=200 | 0.00 [0.0, 0.0] | — |
| mcm_aggregate|size=50 | 75.09 [72.7, 77.4] | 71.93 [68.7, 75.2] |
| mcm_image_mean|size=100 | 50.10 [47.8, 52.4] | 66.67 [63.4, 70.3] |
| mcm_image_mean|size=150 | 25.60 [23.5, 27.7] | 62.69 [58.6, 66.6] |
| mcm_image_mean|size=20 | 89.44 [87.4, 91.2] | 79.57 [75.9, 82.7] |
| mcm_image_mean|size=200 | 0.00 [0.0, 0.0] | — |
| mcm_image_mean|size=50 | 75.09 [72.7, 77.4] | 74.07 [70.6, 77.5] |
| msp_dev|size=100 | 52.29 [50.0, 54.6] | 62.41 [59.2, 66.0] |
| msp_dev|size=150 | 27.14 [24.9, 29.5] | 57.00 [53.2, 61.2] |
| msp_dev|size=20 | 82.59 [79.6, 85.5] | 77.87 [74.2, 81.1] |
| msp_dev|size=200 | 0.65 [0.0, 1.8] | — |
| msp_dev|size=50 | 75.86 [73.5, 78.2] | 67.87 [64.6, 71.3] |
| msp_pilot|size=100 | 53.64 [50.9, 56.6] | 58.82 [55.2, 62.6] |
| msp_pilot|size=150 | 31.14 [27.9, 34.7] | 54.57 [50.6, 58.9] |
| msp_pilot|size=20 | 71.70 [68.3, 75.1] | 74.98 [70.8, 78.9] |
| msp_pilot|size=200 | 7.14 [4.1, 10.7] | — |
| msp_pilot|size=50 | 68.42 [65.2, 71.8] | 63.66 [59.9, 67.5] |
| neg_softmax_entropy|size=100 | 51.98 [49.5, 54.3] | 64.73 [61.4, 68.0] |
| neg_softmax_entropy|size=150 | 28.11 [25.7, 30.9] | 59.20 [55.4, 63.0] |
| neg_softmax_entropy|size=20 | 80.77 [77.2, 84.0] | 79.50 [76.0, 82.5] |
| neg_softmax_entropy|size=200 | 2.67 [0.9, 4.8] | — |
| neg_softmax_entropy|size=50 | 74.40 [71.3, 77.5] | 71.34 [67.9, 74.6] |
| neg_vote_entropy|size=100 | 52.37 [49.5, 55.3] | 57.45 [54.4, 60.6] |
| neg_vote_entropy|size=150 | 34.00 [30.5, 37.8] | 53.86 [50.1, 57.6] |
| neg_vote_entropy|size=20 | 79.86 [76.7, 82.7] | 71.95 [68.3, 75.4] |
| neg_vote_entropy|size=200 | 15.24 [10.7, 20.8] | — |
| neg_vote_entropy|size=50 | 69.71 [66.3, 73.1] | 63.74 [60.3, 67.0] |
| vote_fraction|size=100 | 52.27 [49.7, 55.0] | 55.18 [51.9, 58.8] |
| vote_fraction|size=150 | 31.26 [28.3, 34.6] | 53.05 [49.2, 56.9] |
| vote_fraction|size=20 | 78.59 [75.7, 81.5] | 72.18 [68.2, 75.6] |
| vote_fraction|size=200 | 7.70 [4.8, 11.2] | — |
| vote_fraction|size=50 | 70.69 [67.4, 73.9] | 62.79 [59.3, 66.5] |

Failure cases (max_mean_sim, tau_e2e; full list [failure_cases.json](../grounding/seed0_known_c0/vitb16_831109500c95/failure_cases.json)):

* Impure clusters: cluster 190 → 'Acadian Flycatcher' (score 0.312, purity 0.07; Rusty Blackbird:2; Great Crested Flycatcher:2; Chipping Sparrow:2); cluster 160 → 'Pomarine Jaeger' (score 0.298, purity 0.08; Pelagic Cormorant:1; American Crow:1; Long tailed Jaeger:1); cluster 199 → 'Red breasted Merganser' (score 0.287, purity 0.18; Gadwall:3; Pigeon Guillemot:2; Red breasted Merganser:2); cluster 50 → 'Acadian Flycatcher' (score 0.345, purity 0.19; Acadian Flycatcher:12; Great Crested Flycatcher:10; Western Wood Pewee:9)
* Confident wrong labels: cluster 148 → 'Horned Grebe' (score 0.383, purity 1.00; Eared Grebe:1); cluster 181 → 'California Gull' (score 0.377, purity 1.00; Heermann Gull:1); cluster 153 → 'Bay breasted Warbler' (score 0.376, purity 1.00; Red eyed Vireo:1); cluster 101 → 'Carolina Wren' (score 0.372, purity 0.50; Field Sparrow:1; Carolina Wren:1)
* Absent label replaced and accepted: shared_150_s1: cluster of 'Eared Grebe' named 'Horned Grebe' (score 0.383); shared_100_s4: cluster of 'Eared Grebe' named 'Horned Grebe' (score 0.383); shared_50_s0: cluster of 'Eared Grebe' named 'Horned Grebe' (score 0.383); shared_20_s1: cluster of 'Eared Grebe' named 'Horned Grebe' (score 0.383)

Other files: [metrics_by_episode.csv](../grounding/seed0_known_c0/vitb16_831109500c95/metrics_by_episode.csv), [risk_coverage_curves.csv](../grounding/seed0_known_c0/vitb16_831109500c95/risk_coverage_curves.csv), [cluster_diagnostics.csv](../grounding/seed0_known_c0/vitb16_831109500c95/cluster_diagnostics.csv), [paired_diagnostic.csv](../grounding/seed0_known_c0/vitb16_831109500c95/paired_diagnostic.csv), [episodes.csv](../grounding/seed0_known_c0/vitb16_831109500c95/episodes.csv).

