# Attribution

## Method

Kai Han, Xiaohu Huang, Yandong Li, Sagar Vaze, Jie Li, Xuhui Jia. *What's in a Name? Beyond Class Indices for
Image Recognition.* CVPR 2024 Workshops (CVinW). arXiv:2304.02364 (v2, last revised 27 Jul 2024).

```bibtex
@inproceedings{han2024whats,
  title={What's in a Name? Beyond Class Indices for Image Recognition},
  author={Kai Han and Xiaohu Huang and Yandong Li and Vaze Sagar and Jie Li and Xuhui Jia},
  booktitle={CVPR Workshops},
  year={2024}
}
```

## Reference implementation that was read

* Repository: <https://github.com/Visual-AI/SCD>
* Commit read: `2ce64f78b2fef35af9c397e45efb125bafc6abe2` (2024-08-30, the latest at the time of reading,
  2026-11-10).
* Files read: `main_unsup.py`, `main_ptsup.py`, `local_utils/clip_lang_util.py`,
  `local_utils/sskm_constrained.py`, `local_utils/k_means_constrained/` (vendored package),
  `gcd/project_utils/cluster_utils.py`, `gcd/project_utils/cluster_and_log_utils.py`, `gcd/data/*.py`,
  `scripts/evaluate_unsupervised.sh`, `scripts/evaluate_partial_supervised.sh`, `README.md`,
  `requirements.txt`, and the three `download_link.txt` files.
* Licence of that repository: CC BY-NC-SA 4.0 (stated in its README; there is no LICENSE file).

**No code from that repository is copied here.** `scd/` re-implements the algorithm from the paper and from
reading the code, so that each behaviour (including the defects) is an explicit, documented option. Where the
reproduction depends on an exact behaviour of the official code (tie-breaking in `collections.Counter`, the
order of a Python `set`, the row order of the assignment output), the behaviour is re-created and tested, not
copied.

## Third-party code vendored here

| File | Origin | Licence |
|---|---|---|
| `scd/third_party/sklearn_linear_assignment.py` | scikit-learn ≤ 0.22, `sklearn/utils/linear_assignment_.py` (Brian M. Clapper, Gael Varoquaux, Lars Buitinck). The official SCD code ships the same module inside `gcd/project_utils/cluster_utils.py` and uses it for name assignment and clustering accuracy. Vendored so that tie-breaking matches. | BSD-3-Clause (header kept) |

The 80 prompt templates are OpenAI's ImageNet templates (CLIP repository, MIT licence), read from
`open_clip.zero_shot_metadata.OPENAI_IMAGENET_TEMPLATES`. A test checks that they produce exactly the template
list in the official SCD code (`local_utils/clip_lang_util.py`, same order).

## Assets from the authors (used, not redistributed)

Downloaded from the Google Drive links in the official README and kept outside Git
(`Experiment2_SCD/outputs/assets/scd_official/`):

| Asset | Size | Use |
|---|---|---|
| `zeroshot_weights.zip` (sha256 `99ba7c38…76f5f4`) | 151,216,331 bytes | `zeroshot_weights_all_nouns_vit_b_16.pt`: fp16 [512, 67,893] text weights of the WordNet vocabulary |
| `GCD_pretrained_weights_VIT16.zip` | 1,916,396,688 bytes | `imagenet100_model_best.pt`: the authors' GCD backbone for the partially supervised setting (only this member is extracted, by HTTP range requests) |
| `extracted_features.zip` | 9,037,041,820 bytes | not needed for the core run (it has no CLIP features for ImageNet-100); its DINO/GCD ImageNet-100 features can be compared with ours as an optional check |
