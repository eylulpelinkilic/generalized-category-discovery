# Oracle-cluster pilot (kept as is)

The files in this folder, `embeddings/`, `clusters/` and the root scripts (`extract_*_embeddings.py`,
`create_clusters.py`, `generate_*vocabularies.py`, `construct_matrices.py`, `evaluate_*.py`,
`risk_coverage.py`, `analyze_matrix_failures.py`) form the **oracle-cluster pilot**:

* 2,000 ImageNet validation images, 20 per class of the GCD ImageNet-100 classes (Hugging Face
  `ILSVRC/imagenet-1k`, `download_imagenet100.py`);
* clusters are **ground-truth groups** of 10 images (two per class), not discovered by GCD;
* 10-concept paired vocabularies (true concept present / withheld), random and hard variants;
* CLIP ViT-B/32 via `open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")`.
  With open_clip 3.3.0 (the version in `.venv`) this builds GELU, while the OpenAI weights were trained
  with QuickGELU; see `Experiment1_GCD_CLIP_Baseline/AUDIT.md` item 9.

The real baseline with GCD-discovered clusters, full datasets and calibrated rejection is
`Experiment1_GCD_CLIP_Baseline/`. Pilot numbers are not comparable with it.
