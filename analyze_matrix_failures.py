
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------
# 1. Load existing experiment
# ---------------------------------------

OUT = Path("experiments/failure_analysis")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    "experiments/reliability/scenario_scores.csv"
)

V = np.load("embeddings/image_embeddings.npy")
T = np.load("embeddings/text_embeddings.npy")
text_ids = np.load("embeddings/text_class_ids.npy")

with open("clusters/clusters.json") as f:
    clusters = json.load(f)

with open("imagenet100_classes.json") as f:
    classes = json.load(f)

cluster_lookup = {
    int(c["cluster_id"]): c for c in clusters
}

text_lookup = {
    int(cid): i for i, cid in enumerate(text_ids)
}

names = {
    int(c["imagenet_id"]): c["name"].split(",")[0]
    for c in classes
}

# ---------------------------------------
# 2. Select MSP cutoff on validation
# ---------------------------------------

validation = df[df["split"] == "validation"]

mapped_val = validation[
    validation["is_mapped"] == 1
]

threshold = float(
    np.quantile(mapped_val["msp"], 0.05)
)

print(f"Validation MSP threshold: {threshold:.6f}")

# Explore DEVELOPMENT only
dev = df[df["split"] == "development"].copy()

dev["accepted"] = dev["msp"] >= threshold

# ---------------------------------------
# 3. Categorize the outcomes
# ---------------------------------------

def outcome(row):

    if row["is_mapped"] == 0:
        return (
            "correctly_rejected"
            if not row["accepted"]
            else "incorrectly_accepted"
        )

    if not row["accepted"]:
        return "incorrectly_rejected"

    if row["correct_label"] == 1:
        return "correctly_grounded"

    return "wrong_mapped_label"

dev["outcome"] = dev.apply(outcome, axis=1)

print("\nDevelopment outcome counts:")
print(dev["outcome"].value_counts())

dev.to_csv(
    OUT / "development_outcomes.csv",
    index=False
)

# ---------------------------------------
# 4. Reconstruct one similarity matrix
# ---------------------------------------

def reconstruct_matrix(row):
    cluster = cluster_lookup[int(row["cluster_id"])]

    image_indices = cluster["image_indices"]
    vocab_ids = json.loads(row["vocabulary_ids"])

    image_features = V[image_indices]

    text_features = T[
        [text_lookup[int(cid)] for cid in vocab_ids]
    ]

    S = image_features @ text_features.T

    return S, vocab_ids

# ---------------------------------------
# 5. Select representative examples
# ---------------------------------------

# For each group, choose an example near
# the group's median MSP, not an extreme.
examples = []

groups = [
    "correctly_grounded",
    "correctly_rejected",
    "incorrectly_accepted",
    "incorrectly_rejected",
    "wrong_mapped_label",
]

for group in groups:
    subset = dev[dev["outcome"] == group].copy()

    if subset.empty:
        print(f"No examples for: {group}")
        continue

    median = subset["msp"].median()

    index = (subset["msp"] - median).abs().idxmin()

    examples.append(dev.loc[index])

# ---------------------------------------
# 6. Plot the full similarity matrices
# ---------------------------------------

if examples:

    fig, axes = plt.subplots(
        1, len(examples),
        figsize=(5 * len(examples), 6),
        squeeze=False
    )

    axes = axes.ravel()

    # Shared cosine scale across examples.
    matrices = [
        reconstruct_matrix(row)[0]
        for row in examples
    ]

    low = min(S.min() for S in matrices)
    high = max(S.max() for S in matrices)

    for ax, row, S in zip(
        axes, examples, matrices
    ):
        vocab = json.loads(row["vocabulary_ids"])

        im = ax.imshow(
            S,
            cmap="Blues",
            vmin=low,
            vmax=high,
            aspect="auto"
        )

        ax.set_xticks(range(len(vocab)))
        ax.set_xticklabels(
            [names[int(cid)] for cid in vocab],
            rotation=65,
            ha="right",
            fontsize=8
        )

        ax.set_yticks(range(S.shape[0]))
        ax.set_yticklabels(
            [f"I{i+1}" for i in range(S.shape[0])]
        )

        ax.set_title(
            f"{row['outcome'].replace('_', ' ')}\n"
            f"True: {names[int(row['true_class_id'])]}\n"
            f"MSP: {row['msp']:.3f}",
            fontsize=10
        )

        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                ax.text(
                    j, i, f"{S[i,j]:.2f}",
                    ha="center", va="center",
                    fontsize=6
                )

    fig.subplots_adjust(
        bottom=0.34,
        top=0.82,
        wspace=0.6,
        right=0.9
    )

    fig.colorbar(
        im,
        ax=axes.tolist(),
        label="Cosine similarity",
        fraction=0.015,
        pad=0.02
    )

    plt.savefig(
        OUT / "failure_matrix_comparison.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()

# ---------------------------------------
# 7. Describe full matrix structure
# ---------------------------------------

def matrix_statistics(S):
    row_winners = np.argmax(S, axis=1)

    # Fraction of images supporting the
    # most frequently selected concept.
    counts = np.bincount(
        row_winners, minlength=S.shape[1]
    )

    agreement = counts.max() / S.shape[0]

    # Row-wise variation in similarity
    # profiles, before averaging.
    row_dispersion = np.mean(
        np.linalg.norm(S - S.mean(axis=0), axis=1)
    )

    # Spectral energy concentration.
    singular_values = np.linalg.svd(
        S, compute_uv=False
    )

    energy = singular_values ** 2

    spectral_concentration = (
        energy[0] / energy.sum()
        if energy.sum() > 0 else np.nan
    )

    return {
        "agreement": agreement,
        "row_dispersion": row_dispersion,
        "spectral_concentration": spectral_concentration
    }

stats = []

for _, row in dev.iterrows():
    S, _ = reconstruct_matrix(row)

    stats.append({
        "cluster_id": row["cluster_id"],
        "pair_id": row["pair_id"],
        "outcome": row["outcome"],
        **matrix_statistics(S)
    })

stats_df = pd.DataFrame(stats)

stats_df.to_csv(
    OUT / "matrix_statistics.csv",
    index=False
)

print("\nMedian structural properties by outcome:")
print(
    stats_df.groupby("outcome")[
        [
            "agreement",
            "row_dispersion",
            "spectral_concentration"
        ]
    ].median().round(4)
)

print("\nFailure analysis complete!")
print("Saved results to:", OUT)
