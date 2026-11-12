"""Linear assignment with the official tie-breaking (sklearn Munkres) or SciPy."""
from __future__ import annotations

import numpy as np

from scd.third_party.sklearn_linear_assignment import linear_assignment as _munkres


def linear_assignment(cost: np.ndarray, solver: str = "munkres") -> np.ndarray:
    """Minimum-cost assignment; returns (row, col) pairs sorted by row, like the official helper.

    `munkres` is the solver the official code uses (sklearn's former `linear_assignment`), so ties between
    equal-cost assignments are broken the same way. `scipy` is `linear_sum_assignment` (same optimum, possibly a
    different choice among tied optima)."""
    cost = np.asarray(cost)
    if solver == "munkres":
        return _munkres(cost.astype(float if cost.dtype.kind == "f" else np.int64))
    if solver == "scipy":
        from scipy.optimize import linear_sum_assignment

        r, c = linear_sum_assignment(cost)
        return np.stack([r, c], 1).astype(int)
    raise ValueError(f"unknown solver {solver!r}")


def max_weight_assignment(w: np.ndarray, solver: str = "munkres") -> np.ndarray:
    """Maximum-weight assignment on a square count matrix, as the official `assign_name`: cost = w.max() - w."""
    return linear_assignment(w.max() - w, solver)
