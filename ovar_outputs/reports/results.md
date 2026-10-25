# Results report

Only experiments actually run on benchmark videos appear here. Accuracies are top-1 (%) over the full target vocabulary; brackets are 95% cluster-bootstrap intervals over source groups. Familiar/unfamiliar = XOV closed-set/open-set categories (relative to Kinetics400). Scores are cosine similarities, not calibrated probabilities.

## Common comparison protocol

| Run | Model | Protocol | Subset | Dataset | Overall | Familiar | Unfamiliar | Failed |
|---|---|---|---|---|---|---|---|---|
| clip_vitb16__common_16x1x1__ucf101 | clip_vitb16 | common_16x1x1 | full list | UCF101 | 71.16 [68.2, 74.3] (n=3783) | 74.23 [70.2, 78.1] (n=1882) | 68.12 [63.5, 73.0] (n=1901) | 0 |
| clip_vitb16__common_16x1x1_ens28__ucf101 | clip_vitb16 | common_16x1x1_ens28 | full list | UCF101 | 74.86 [72.0, 77.8] (n=3783) | 77.21 [73.2, 80.8] (n=1882) | 72.54 [68.1, 77.1] (n=1901) | 0 |
| clip_vitb32__common_16x1x1__ucf101 | clip_vitb32 | common_16x1x1 | full list | UCF101 | 68.17 [65.2, 71.1] (n=3783) | 66.21 [61.8, 70.2] (n=1882) | 70.12 [65.6, 74.5] (n=1901) | 0 |
| clip_vitb32__common_16x1x1__ucf101_subset2 | clip_vitb32 | common_16x1x1 | subset per_class=2 limit=None | UCF101 | 70.79 [64.3, 77.1] (n=202) | 73.00 [63.9, 81.2] (n=100) | 68.63 [58.7, 78.1] (n=102) | 0 |
| clip_vitb32__common_16x1x1_ens28__ucf101 | clip_vitb32 | common_16x1x1_ens28 | full list | UCF101 | 69.89 [66.8, 72.9] (n=3783) | 71.84 [67.5, 75.9] (n=1882) | 67.96 [63.4, 72.4] (n=1901) | 0 |

| Run | Device | Model load (s) | Total (s) | Encode (s) | Decode (s) | Memory |
|---|---|---|---|---|---|---|
| clip_vitb16__common_16x1x1__ucf101 | cuda | 8.81 | 2137.52 | 693.6 | 1425.6 | {"process_peak_rss_mb": 2033.6, "peak_gpu_memory_allocated_mb": 828.4} |
| clip_vitb16__common_16x1x1_ens28__ucf101 | cuda | 7.31 | 18.37 | 0.0 | 0.0 | {"process_peak_rss_mb": 2034.1, "peak_gpu_memory_allocated_mb": 833.7} |
| clip_vitb32__common_16x1x1__ucf101 | cuda | 7.26 | 3184.05 | 241.8 | 2926.2 | {"process_peak_rss_mb": 2046.8, "peak_gpu_memory_allocated_mb": 635.8} |
| clip_vitb32__common_16x1x1__ucf101_subset2 | cuda | 7.29 | 195.56 | 14.2 | 172.9 | {"process_peak_rss_mb": 2046.0, "peak_gpu_memory_allocated_mb": 835.6} |
| clip_vitb32__common_16x1x1_ens28__ucf101 | cuda | 7.32 | 18.14 | 0.0 | 0.0 | {"process_peak_rss_mb": 2046.8, "peak_gpu_memory_allocated_mb": 840.9} |

## Sensitivity: unfamiliar accuracy without near-familiar classes

Secondary analysis only. Excluded classes are officially unfamiliar but have a close Kinetics400 counterpart according to an unverified review (data/audit/category_review.csv).

| Run | Dataset | Unfamiliar (official) | Unfamiliar excluding near-familiar | Excluded classes |
|---|---|---|---|---|
| clip_vitb16__common_16x1x1__ucf101 | UCF101 | 68.12 [63.5, 73.0] (n=1901) | 70.16 [64.5, 75.5] (n=1307) | 15 |
| clip_vitb16__common_16x1x1_ens28__ucf101 | UCF101 | 72.54 [68.1, 77.1] (n=1901) | 73.45 [68.1, 78.7] (n=1307) | 15 |
| clip_vitb32__common_16x1x1__ucf101 | UCF101 | 70.12 [65.6, 74.5] (n=1901) | 70.24 [64.6, 75.5] (n=1307) | 15 |
| clip_vitb32__common_16x1x1__ucf101_subset2 | UCF101 | 68.63 [58.7, 78.1] (n=102) | 68.06 [57.1, 77.9] (n=72) | 15 |
| clip_vitb32__common_16x1x1_ens28__ucf101 | UCF101 | 67.96 [63.4, 72.4] (n=1901) | 65.57 [59.7, 71.3] (n=1307) | 15 |

## Reference numbers reported by XOV-Action (not reproduced here)

Transcribed from arXiv:2403.01560v3. Their Table III models were trained on Kinetics400 by the XOV authors with T=16 frames, one centre-cropped clip, and the checkpoint with the best average overall accuracy on the four test domains (test-set model selection). They are therefore not directly comparable with the official checkpoints evaluated here.

| Source | Training | Backbone | Model | Dataset | Closed | Open | All |
|---|---|---|---|---|---|---|---|
| XOV-Action arXiv:2403.01560v3 Table III | none (original CLIP) | ViT-B/16 | CLIP | UCF101 | 66.71 | 59.53 | 63.15 |
| XOV-Action arXiv:2403.01560v3 Table III | none (original CLIP) | ViT-B/16 | CLIP | HMDB51 | 49.19 | 35.26 | 44.27 |
| XOV-Action arXiv:2403.01560v3 Table III | none (original CLIP) | ViT-B/16 | CLIP | ARID | 37.76 | 12.65 | 27.69 |
| XOV-Action arXiv:2403.01560v3 Table III | none (original CLIP) | ViT-B/16 | CLIP | NEC-Drone | 10.42 | 22.44 | 16.59 |
| XOV-Action arXiv:2403.01560v3 Table III | none (original CLIP) | ViT-B/16 | CLIP | AVG | 41.04 | 32.47 | 37.92 |
| XOV-Action arXiv:2403.01560v3 Table III | Kinetics400 | ViT-B/16 | OpenVCLIP | UCF101 | 91.06 | 75.03 | 83.00 |
| XOV-Action arXiv:2403.01560v3 Table III | Kinetics400 | ViT-B/16 | OpenVCLIP | HMDB51 | 60.63 | 42.54 | 54.24 |
| XOV-Action arXiv:2403.01560v3 Table III | Kinetics400 | ViT-B/16 | OpenVCLIP | ARID | 52.25 | 28.35 | 42.67 |
| XOV-Action arXiv:2403.01560v3 Table III | Kinetics400 | ViT-B/16 | OpenVCLIP | NEC-Drone | 22.92 | 14.91 | 18.81 |
| XOV-Action arXiv:2403.01560v3 Table III | Kinetics400 | ViT-B/16 | OpenVCLIP | AVG | 56.71 | 40.21 | 49.68 |
| XOV-Action arXiv:2403.01560v3 Table III | Kinetics400 | ViT-B/16 | TC-CLIP | UCF101 | 90.91 | 67.23 | 79.01 |
| XOV-Action arXiv:2403.01560v3 Table III | Kinetics400 | ViT-B/16 | TC-CLIP | HMDB51 | 60.20 | 32.22 | 50.33 |
| XOV-Action arXiv:2403.01560v3 Table III | Kinetics400 | ViT-B/16 | TC-CLIP | ARID | 52.74 | 23.64 | 41.07 |
| XOV-Action arXiv:2403.01560v3 Table III | Kinetics400 | ViT-B/16 | TC-CLIP | NEC-Drone | 24.19 | 8.75 | 16.27 |
| XOV-Action arXiv:2403.01560v3 Table III | Kinetics400 | ViT-B/16 | TC-CLIP | AVG | 57.01 | 32.96 | 46.67 |
| XOV-Action arXiv:2403.01560v3 Table III | Kinetics400 | ViT-B/16 | XOV-Action | UCF101 | 91.97 | 80.08 | 86.00 |
| XOV-Action arXiv:2403.01560v3 Table III | Kinetics400 | ViT-B/16 | XOV-Action | HMDB51 | 65.28 | 43.83 | 57.72 |
| XOV-Action arXiv:2403.01560v3 Table III | Kinetics400 | ViT-B/16 | XOV-Action | ARID | 54.29 | 32.32 | 45.48 |
| XOV-Action arXiv:2403.01560v3 Table III | Kinetics400 | ViT-B/16 | XOV-Action | NEC-Drone | 24.31 | 16.67 | 20.39 |
| XOV-Action arXiv:2403.01560v3 Table III | Kinetics400 | ViT-B/16 | XOV-Action | AVG | 58.96 | 43.23 | 52.39 |
| XOV-Action arXiv:2403.01560v3 Table II | none (original CLIP) | ViT-B/32 | CLIP | UCF101 | 61.51 | 57.85 | 59.67 |
| XOV-Action arXiv:2403.01560v3 Table II | none (original CLIP) | ViT-B/32 | CLIP | HMDB51 | 42.92 | 38.62 | 41.40 |
| XOV-Action arXiv:2403.01560v3 Table II | none (original CLIP) | ViT-B/32 | CLIP | ARID | 32.04 | 12.04 | 24.02 |
| XOV-Action arXiv:2403.01560v3 Table II | none (original CLIP) | ViT-B/32 | CLIP | NEC-Drone | 18.52 | 6.44 | 12.33 |
| XOV-Action arXiv:2403.01560v3 Table II | none (original CLIP) | ViT-B/32 | CLIP | AVG | 38.75 | 28.74 | 34.35 |
