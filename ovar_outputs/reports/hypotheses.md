# Hypotheses worth testing next

No experiment on real benchmark videos has been run in this workspace. The observations below
come from numbers **reported** in XOV-Action (arXiv:2403.01560v3, Tables II–III, transcribed in
`docs/reference/xov_reported_numbers.csv`) and from the category audit done here. They are
starting points, not findings; each names the local experiment that could refute it.

## Observations

* **O1. Drone domain is the hardest.** Every Kinetics400-trained model in Table III reaches at
  most 24.3% familiar and 16.7% unfamiliar accuracy on NEC-Drone, while the same models reach
  81.0–92.0% familiar accuracy on UCF101.
* **O2. Kinetics training can lower unfamiliar accuracy in large-gap domains.** On NEC-Drone,
  frozen CLIP ViT-B/16 scores 22.44% on unfamiliar classes, Open-VCLIP 14.91% and TC-CLIP 8.75%,
  while their familiar accuracy rises (10.42% → 22.92% / 24.19%).
* **O3. Frozen CLIP already shows a familiar–unfamiliar gap** without any task-specific
  training: UCF101 66.71 vs 59.53, HMDB51 49.19 vs 35.26, ARID 37.76 vs 12.65.
* **O4. NEC-Drone subgroup numbers are unstable.** Frozen CLIP is 18.52/6.44 (familiar/
  unfamiliar) with ViT-B/32 (Table II) but 10.42/22.44 with ViT-B/16 (Table III). There are 7 and
  9 classes and few recording groups.
* **O5. Part of UCF101's unfamiliar set has close Kinetics400 counterparts.** 15 of 51
  officially unfamiliar classes do (e.g. JumpRope / "skipping rope"; unverified review in
  `data/audit/category_review.csv`).

## Hypotheses

1. **Kinetics adaptation trades unfamiliar-category transfer for familiar accuracy when the
   domain gap is large (O2).** Test: frozen CLIP ViT-B/16 vs Open-VCLIP ViT-B/16 under the common
   protocol on NEC-Drone and ARID, per class, with paired cluster-bootstrap intervals. Refuted if
   Open-VCLIP is at least as accurate on unfamiliar classes in both domains.
2. **The familiar–unfamiliar gap mixes training exposure with class difficulty (O3).** Test:
   compare each trained model's per-class accuracy with frozen CLIP's on the same videos and
   report the gain over frozen CLIP separately for familiar and unfamiliar classes. Refuted if
   frozen CLIP shows no gap in our runs.
3. **Large-gap failures concentrate in a few confusable classes rather than degrading
   uniformly (O1).** Test: per-class accuracy, confusion pairs and the shared-concept table
   (walk/run/jump/sit/stand across HMDB51, ARID, NEC-Drone) from `ovar error-analysis`.
4. **Predictions rely mostly on static appearance (O1–O2).** Test: single-frame and shuffled
   diagnostics on development pools for both models. Refuted if single-frame accuracy drops
   substantially and Open-VCLIP is order-sensitive. A small drop would not by itself show a
   failure, since many classes are recognisable from one frame.
5. **Actor scale matters in the drone domain.** Test: actor-crop vs actor-with-context variants
   once person boxes exist, comparing the size of the effect on NEC-Drone with UCF101 (and
   reporting detector recall).
6. **Official unfamiliar accuracy on UCF101 overstates transfer to genuinely novel actions
   (O5).** Test: the secondary metric excluding near-familiar classes, after a human confirms
   the review.
7. **NEC-Drone differences of a few points are within noise (O4).** Test: cluster-bootstrap
   intervals for NEC-Drone; draw conclusions only where intervals separate.

None of these implies a particular architecture; they decide which failure, if any, a new
method should target.
