# Planned extension: textual enhancement (TE) with CC12M — not implemented

Status: documented only. The core reproduction (protocol A) runs without TE, as the paper's main "Ours" rows.

## What TE is (paper Sec. 3.4, 4.3, App. A.4)

For each image, retrieve the r = 10 CC12M captions with the highest CLIP image–text cosine, average their CLIP
text embeddings, and concatenate that vector with the image's visual feature (DINO in the unsupervised setting, GCD
in the partially supervised one). The concatenated feature is used **only for the initial clustering**; naming and
refinement are unchanged. Published effect on ImageNet-100: sACC 41.2 → 43.0 (unsupervised), 54.8 → 55.7
(partially supervised); ACC changes in both directions across datasets (see INSPECTION.md §3, item 4).

## Two routes

1. **Authors' features.** `extracted_features.zip` contains `img2text_dino_imagenet_100_all.pt` (943 MB) and
   `img2text_gcd_imagenet_100_all.pt` (943 MB), plus `*_relevant_*` and `*_clip_dino_*` variants. They store features,
   masks and targets **without sample ids**; reuse needs a row-order check (recompute the DINO/GCD half for a
   sample of rows and compare with the first block of columns) before any clustering. Members can be extracted alone
   with `e2/assets.py:extract_remote_member` (HTTP range requests).
2. **Recompute.** CC12M has ~12M captions: ViT-B/16 text embeddings in fp16 are ~12 GB, and nearest-caption search
   for 127k images needs an approximate index (e.g. FAISS IVF). Feasible on a Kaggle GPU session with the captions
   streamed; the paper does not specify caption cleaning, deduplication or the exact CC12M snapshot, so this route
   would be labelled `deviation: TE recomputed`.

## Where it would plug in

`e2/repro.py:cluster` with a feature option `dino+te` / `gcd+te` (concatenation, L2-normalised parts), recorded in the
initial-clustering hash. No change to `scd.naming`.

## Relation to the failure analysis

TE changes only memberships. It is relevant to P5 (mixed clusters) and P3 (several clusters competing for one name),
not to P1/P2 (aggregation of per-image votes). Run it after the protocol-A failure analysis shows how much of the
error comes from initial memberships.
