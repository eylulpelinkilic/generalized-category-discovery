# Results report

Only experiments actually run on benchmark videos appear here. Accuracies are top-1 (%) over the full target vocabulary; brackets are 95% cluster-bootstrap intervals over source groups. Familiar/unfamiliar = XOV closed-set/open-set categories (relative to Kinetics400). Scores are cosine similarities, not calibrated probabilities.

**No experiment on real benchmark videos has been run in this workspace yet.** See STATUS.md for the blockers and the commands that produce these results.

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
