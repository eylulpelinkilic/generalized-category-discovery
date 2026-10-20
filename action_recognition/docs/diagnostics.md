# Diagnostics

Diagnostics run on a designated development pool (UCF101 split-1 training list, 2 videos per
class chosen by a seeded hash), never on the evaluation lists, and never to choose settings for
the final evaluation. Each variant changes one factor relative to the reference run; results
are paired per video (`runs/diagnostics/<name>/summary.csv`, `per_class.csv`,
`changed_videos.csv`).

Reported per variant: accuracy of reference and variant over all videos (failures count as
errors), accuracy change with a paired cluster-bootstrap interval, prediction-change rate among
videos decoded in both runs, correct→wrong and wrong→correct counts, failures of the variant.

## A. Temporal information

`ordered` (reference), `shuffled` (seeded permutation of the sampled frames within each view),
`single_frame` (the middle sampled frame repeated T times), optionally `reversed`.

* A frame-independent encoder with mean pooling (frozen CLIP) is invariant to shuffling by
  construction; the report says so instead of presenting it as a finding. Its single-frame
  variant measures how much evidence beyond one frame the average contains.
* For Open-VCLIP, shuffling changes the inputs to its cross-frame attention, so a change in
  predictions is informative. Unchanged predictions do not show that temporal modelling is
  useless: many actions are recognisable from static cues.

## B. Spatial context

Needs an actor-box file `dataset_name, video_id, frame_index, x1, y1, x2, y2` (pixels), set as
`actor_boxes:` in the diagnostic config. Variants: `actor_crop` (square crop around the box,
context 1.0) and `actor_context` (context 2.0). For every sampled frame the nearest annotated
box is used; videos without boxes fail and are counted.

No detector is installed here. A documented choice would be torchvision's COCO person detector
(e.g. Faster R-CNN ResNet-50 FPN v2, about 160 MB of weights, roughly 0.1–1 s per frame on
CPU; record model name, weights checksum and score threshold with the boxes). Boxes are only
reliable for single-actor clips; ARID darkness and NEC-Drone's small actors will reduce
detector recall, which must be reported with the results. Crops can remove action evidence or
introduce zero-padding artefacts, so crop effects do not by themselves prove scene bias.

## C. Synthetic degradation (supplementary stress tests)

`brightness` (linear scaling, e.g. 0.5 and 0.25), `downsample` (box downscale by 0.5/0.25 and
bilinear upscale back), `blur` (Gaussian, σ = 1, 2). They are reported separately from real
target domains and are not a substitute for ARID or NEC-Drone.

## D. Error analysis

`python -m ovar error-analysis runs/<run> --diagnostics runs/diagnostics/<name>` writes:
lowest-accuracy classes per domain, most frequent confusions, accuracy of concepts shared
across domains (`docs/reference/cross_domain_concepts.csv`, a curated mapping with match
quality), classes most affected by temporal and crop variants. Observations are computed; the
Hypotheses section is left for the researcher.
