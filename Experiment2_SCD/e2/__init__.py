"""Experiment 2: reproduction of SCD (Han et al.) and its naming component inside the Experiment 1 protocol.

"SCD" is both the paper's term (Semantic Category Discovery) and our folder name (../SCD, the re-implementation).
This package imports Experiment 1 (../Experiment1_GCD_CLIP_Baseline, `e1`) and the re-implementation (`scd`) by
reference; it does not modify either.
"""
import sys
from pathlib import Path

EXP2 = Path(__file__).resolve().parents[1]
REPO = EXP2.parent
for p in (REPO / "Experiment1_GCD_CLIP_Baseline", REPO / "SCD"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
