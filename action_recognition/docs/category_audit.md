# Familiar/unfamiliar category audit

The official XOV-Action closed/open lists define familiar/unfamiliar status in every primary
metric. This audit only looks for cases where that status may not match semantic overlap with
the Kinetics400 training categories, because such cases change how "unfamiliar" accuracy
should be read.

## Screening

* `python -m ovar audit-categories --method lexical` (no model): ranks Kinetics400 classes by
  shared word stems. It flags officially unfamiliar classes that share any word with a
  Kinetics400 class (50 of 179) and familiar classes that share none (0 after stemming).
  Output: `data/audit/category_audit_lexical.csv`.
* `--method clip`: the same screen with CLIP text-embedding similarity (loads CLIP; not run here
  because of memory pressure on this machine).

String overlap is neither necessary nor sufficient for equivalence (JumpRope vs "skipping rope"
shares only "rope"; "hammer throw" shares a word with Hammering but is a different action), so
every flag needs a human decision.

## Review (unverified)

`data/audit/category_review.csv` records the candidates checked against
`labels/Kinetics400_labels.csv` so far, with reviewer `claude-unverified`:

* 15 officially **unfamiliar** UCF101 classes with a close Kinetics400 counterpart, tagged
  `near_familiar`: JumpRope (skipping rope), TrampolineJumping (bouncing on trampoline),
  BodyWeightSquats (squat), BandMarching (marching), RockClimbingIndoor (rock climbing),
  SoccerPenalty (shooting goal (soccer)), BoxingSpeedBag (punching bag), GolfSwing (golf
  driving/chipping), SumoWrestling (wrestling), IceDancing (ice skating), TennisSwing (playing
  tennis), SalsaSpin (salsa dancing), CricketShot and CricketBowling (playing cricket),
  FloorGymnastics (gymnastics tumbling).
* An inconsistency: HMDB51 "golf" is familiar while UCF101 GolfSwing is unfamiliar.
* Officially **familiar** classes whose Kinetics400 counterpart is only a narrower class, tagged
  `weak_familiar`: "walk" in HMDB51/ARID/NEC-Drone (Kinetics400 has "walking the dog" but no
  plain walking) and HMDB51 "hit".

## Use in evaluation

`metrics.json` adds `secondary_unfamiliar_excluding_near_familiar` (accuracy on unfamiliar
videos after removing `near_familiar` classes), and the results report shows it in a separate
sensitivity table. Fill `reviewer_decision` (a second annotator, ideally two) before relying on
it. Never change the primary definition.
