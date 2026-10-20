import sys

DIFFICULTY = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "random"
)

assert DIFFICULTY in ["random", "hard"]

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve

# ----------------------------------------
# 1. Load matrices and scenarios
# ----------------------------------------

OUT = Path(f"experiments/baseline_{DIFFICULTY}")
OUT.mkdir(parents=True, exist_ok=True)

matrix_path = (
    "experiments/hard/similarity_matrices.npy"
    if DIFFICULTY == "hard"
    else "experiments/similarity_matrices.npy"
)

S_all = np.load(matrix_path)

scenario_path = (
    "experiments/hard/vocabulary_scenarios.json"
    if DIFFICULTY == "hard"
    else "experiments/vocabulary_scenarios.json"
)

with open(scenario_path) as f:
    scenarios = json.load(f)

assert S_all.shape == (400, 10, 10)
assert len(scenarios) == 400

# ----------------------------------------
# 2. Baseline methods
# ----------------------------------------

def compute_baselines(S, temperature=0.01):
    """
    S: shape (N_images, N_concepts)
    Returns the predicted concept column and
    four knownness scores.
    """

    # Average similarity for each concept
    mean_scores = S.mean(axis=0)

    # Forced top-1 prediction
    prediction = int(np.argmax(mean_scores))

    # Maximum similarity
    max_similarity = float(mean_scores.max())

    # Top-1/top-2 margin
    sorted_scores = np.sort(mean_scores)
    margin = float(
        sorted_scores[-1] - sorted_scores[-2]
    )

    # Maximum softmax probability
    logits = mean_scores / temperature
    logits = logits - logits.max()

    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()

    msp = float(probabilities.max())

    # Image voting:
    # Fraction of individual images that vote
    # for the cluster's forced top-1 concept.
    image_predictions = np.argmax(S, axis=1)

    voting = float(
        np.mean(image_predictions == prediction)
    )

    return {
        "prediction_col": prediction,
        "max_similarity": max_similarity,
        "margin": margin,
        "msp": msp,
        "voting": voting,
    }

# ----------------------------------------
# 3. Evaluate all scenarios
# ----------------------------------------

records = []

for S, scenario in zip(S_all, scenarios):

    result = compute_baselines(S)

    predicted_class = scenario["vocabulary_ids"][
        result["prediction_col"]
    ]

    is_mapped = scenario["status"] == "MAPPED"

    correct_label = (
        predicted_class == scenario["true_class_id"]
    )

    records.append({
        "cluster_id": scenario["cluster_id"],
        "split": scenario["split"],
        "status": scenario["status"],
        "is_mapped": int(is_mapped),
        "true_class_id": scenario["true_class_id"],
        "predicted_class_id": predicted_class,
        "correct_label": int(correct_label),
        **result,
    })

df = pd.DataFrame(records)

df.to_csv(
    OUT / "scenario_scores.csv",
    index=False
)

# ----------------------------------------
# 4. Compute AUROC
# ----------------------------------------

methods = [
    "max_similarity",
    "margin",
    "msp",
    "voting",
]

summary = []

# Use validation for cutoff selection
val = df[df["split"] == "validation"].copy()

# Use test for final reporting
test = df[df["split"] == "test"].copy()

print("\nFORCED TOP-1 GROUNDING")
print("----------------------")

for split in ["development", "validation", "test"]:
    subset = df[
        (df["split"] == split) &
        (df["is_mapped"] == 1)
    ]

    accuracy = subset["correct_label"].mean()

    print(f"{split}: {accuracy:.4f}")

print("\nREJECTION BASELINES")
print("-------------------")

for method in methods:

    # Higher score means stronger evidence
    # that the cluster is MAPPED.
    auc = roc_auc_score(
        test["is_mapped"],
        test[method]
    )

    # Accept approximately 95% of mapped
    # validation scenarios.
    mapped_val = val[val["is_mapped"] == 1]

    threshold = np.quantile(
        mapped_val[method],
        0.05
    )

    accepted = test[method] >= threshold

    # Correctly accepted mapped cases
    mapped_test = test["is_mapped"] == 1

    mapped_acceptance = np.mean(
        accepted[mapped_test]
    )

    # Incorrectly accepted UNMAPPED cases
    unmapped_test = test["is_mapped"] == 0

    false_acceptance = np.mean(
        accepted[unmapped_test]
    )

    # Semantic accuracy among accepted
    # mapped cases
    accepted_mapped = test[
        accepted & mapped_test
    ]

    accepted_accuracy = (
        accepted_mapped["correct_label"].mean()
        if len(accepted_mapped) else np.nan
    )

    summary.append({
        "method": method,
        "test_auroc": auc,
        "validation_threshold": threshold,
        "test_mapped_acceptance": mapped_acceptance,
        "test_false_acceptance": false_acceptance,
        "accepted_mapped_accuracy": accepted_accuracy,
    })

    print(
        f"{method:18s} "
        f"AUROC={auc:.4f}  "
        f"FAR={false_acceptance:.4f}"
    )

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    OUT / "summary.csv",
    index=False
)

# ----------------------------------------
# 5. Plot ROC curves
# ----------------------------------------

plt.figure(figsize=(7, 6))

for method in methods:

    fpr, tpr, _ = roc_curve(
        test["is_mapped"],
        test[method]
    )

    auc = roc_auc_score(
        test["is_mapped"],
        test[method]
    )

    plt.plot(
        fpr, tpr,
        label=f"{method} (AUC={auc:.3f})"
    )

plt.plot(
    [0, 1], [0, 1],
    "--", color="gray",
    label="Random"
)

plt.xlabel(
    "False positive rate (UNMAPPED accepted)"
)
plt.ylabel(
    "True positive rate (MAPPED accepted)"
)
plt.title("Mapped vs UNMAPPED Detection")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()

plt.savefig(
    OUT / "roc_curves.png",
    dpi=200
)

plt.close()

# ----------------------------------------
# 6. Plot knownness score distributions
# ----------------------------------------

fig, axes = plt.subplots(
    2, 2, figsize=(12, 8)
)

for ax, method in zip(axes.flat, methods):

    for status in ["MAPPED", "UNMAPPED"]:
        values = test[
            test["status"] == status
        ][method]

        ax.hist(
            values,
            bins=12,
            alpha=0.55,
            density=True,
            label=status
        )

    ax.set_title(method)
    ax.set_xlabel("Knownness score")
    ax.set_ylabel("Density")
    ax.legend()

plt.tight_layout()

plt.savefig(
    OUT / "score_distributions.png",
    dpi=200
)

plt.close()

print("\nResults saved to:", OUT)
print("\nBaseline evaluation complete!")
