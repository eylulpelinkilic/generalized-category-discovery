
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------
# 1. Configuration
# ----------------------------------------

METHODS = [
    "max_similarity",
    "margin",
    "msp",
    "voting",
]

DATASETS = {
    "random": "experiments/baseline",
    "hard": "experiments/baseline_hard",
}

# ----------------------------------------
# 2. Risk-coverage calculation
# ----------------------------------------

def risk_coverage(df, method):
    """
    Higher score = stronger evidence of MAPPED.

    A prediction is correct only if:
      1. The true concept is available.
      2. The predicted concept is correct.

    An accepted UNMAPPED case is an error.
    """

    ranked = df.sort_values(
        method, ascending=False, kind="stable"
    ).reset_index(drop=True)

    correct = (
        (ranked["is_mapped"] == 1) &
        (ranked["correct_label"] == 1)
    ).to_numpy()

    errors = (~correct).astype(int)

    n = len(errors)

    coverage = np.arange(1, n + 1) / n

    cumulative_errors = np.cumsum(errors)

    risk = cumulative_errors / np.arange(1, n + 1)

    # Area under risk-coverage curve
    aurc = np.trapezoid(
        np.r_[0, risk],
        np.r_[0, coverage]
    )

    return coverage, risk, aurc


# ----------------------------------------
# 3. Evaluate random and hard conditions
# ----------------------------------------

results = []

for difficulty, folder in DATASETS.items():

    folder = Path(folder)

    df = pd.read_csv(
        folder / "scenario_scores.csv"
    )

    test = df[df["split"] == "test"].copy()

    assert len(test) == 80
    assert test["is_mapped"].sum() == 40

    plt.figure(figsize=(8, 6))

    for method in METHODS:

        coverage, risk, aurc = risk_coverage(
            test, method
        )

        results.append({
            "difficulty": difficulty,
            "method": method,
            "AURC": aurc,
        })

        plt.plot(
            coverage * 100,
            risk * 100,
            label=f"{method} (AURC={aurc:.3f})"
        )

    plt.xlabel("Coverage (%)")
    plt.ylabel("Risk (%)")

    plt.title(
        f"Risk-Coverage: {difficulty.capitalize()} Vocabulary"
    )

    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()

    plot_path = folder / "risk_coverage.png"

    plt.savefig(
        plot_path, dpi=200
    )

    plt.close()

    print(f"\n{difficulty.upper()} results:")
    for r in results:
        if r["difficulty"] == difficulty:
            print(
                f"{r['method']:18s} "
                f"AURC={r['AURC']:.4f}"
            )

# ----------------------------------------
# 4. Save combined results
# ----------------------------------------

summary = pd.DataFrame(results)

summary.to_csv(
    "experiments/risk_coverage_summary.csv",
    index=False
)

print("\nRisk-coverage evaluation complete!")
