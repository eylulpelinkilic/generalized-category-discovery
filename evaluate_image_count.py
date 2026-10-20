
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# ---------------------------------------
# 1. Configuration
# ---------------------------------------

IMAGE_COUNTS = [1, 3, 5, 10]
FULL = max(IMAGE_COUNTS)

# One image subset per repeat. Matches the
# 20 vocabulary pairs of the reliability run.
N_SUBSETS = 20
N_BOOTSTRAPS = 1000
SEED = 42
TEMPERATURE = 0.01

METHODS = [
    "max_similarity",
    "margin",
    "msp",
    "voting",
]

rng = np.random.default_rng(SEED)

OUT = Path("experiments/image_count")
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------
# 2. Load existing embeddings and
#    vocabularies
# ---------------------------------------

V = np.load("embeddings/image_embeddings.npy")
T = np.load("embeddings/text_embeddings.npy")
text_ids = np.load("embeddings/text_class_ids.npy")

with open("clusters/clusters.json") as f:
    clusters = json.load(f)

with open("experiments/hard/vocabulary_scenarios.json") as f:
    hard_scenarios = json.load(f)

# Random vocabularies: reuse the exact 20 pairs
# per cluster from evaluate_reliability.py.
reliability = pd.read_csv(
    "experiments/reliability/scenario_scores.csv"
)

# Hard vocabularies: one pair per cluster.
hard_baseline = pd.read_csv(
    "experiments/baseline_hard/scenario_scores.csv"
)

assert V.shape == (2000, 512)
assert T.shape == (100, 512)
assert len(clusters) == 200
assert len(reliability) == 8000
assert len(hard_scenarios) == 400

text_lookup = {
    int(cid): i for i, cid in enumerate(text_ids)
}

cluster_lookup = {
    int(c["cluster_id"]): c for c in clusters
}

# ---------------------------------------
# 3. Draw nested image subsets
# ---------------------------------------

# For each cluster and repeat, shuffle its 10
# images once. The first k form the k-image
# subset, so 1 ⊂ 3 ⊂ 5 ⊂ 10 and only the
# image count changes between conditions.
# MAPPED and UNMAPPED scenarios of a pair
# see the same images.
orderings = {
    cid: [
        rng.permutation(FULL)
        for _ in range(N_SUBSETS)
    ]
    for cid in sorted(cluster_lookup)
}

def image_subset(cluster_id, repeat_id, k):
    cluster = cluster_lookup[cluster_id]
    order = orderings[cluster_id][repeat_id]

    # Sorting keeps the original row order, so
    # k=10 reproduces the existing matrices.
    positions = np.sort(order[:k])

    return np.array(cluster["image_indices"])[positions]

with open(OUT / "image_orderings.json", "w") as f:
    json.dump(
        {
            str(cid): [o.tolist() for o in perms]
            for cid, perms in orderings.items()
        },
        f
    )

# ---------------------------------------
# 4. Rejection baselines
#    (same as evaluate_reliability.py)
# ---------------------------------------

def compute_scores(S):
    mean = S.mean(axis=0)
    ordered = np.sort(mean)

    prediction = int(np.argmax(mean))

    logits = mean / TEMPERATURE
    logits -= logits.max()

    probs = np.exp(logits)
    probs /= probs.sum()

    votes = np.argmax(S, axis=1)

    return {
        "prediction_col": prediction,
        "max_similarity": float(ordered[-1]),
        "margin": float(ordered[-1] - ordered[-2]),
        "msp": float(probs.max()),
        # Always 1.0 for a single image
        "voting": float(np.mean(votes == prediction)),
    }

# ---------------------------------------
# 5. Collect vocabulary scenarios
# ---------------------------------------

scenarios = []

for row in reliability.itertuples(index=False):
    scenarios.append({
        "condition": "random",
        "cluster_id": int(row.cluster_id),
        "true_class_id": int(row.true_class_id),
        "repeat_id": int(row.pair_id),
        "split": row.split,
        "status": row.status,
        "vocabulary_ids": json.loads(row.vocabulary_ids),
    })

# The hard vocabulary is fixed per cluster, so
# repeats vary only the image subset.
for s in hard_scenarios:
    for repeat_id in range(N_SUBSETS):
        scenarios.append({
            "condition": "hard",
            "cluster_id": int(s["cluster_id"]),
            "true_class_id": int(s["true_class_id"]),
            "repeat_id": repeat_id,
            "split": s["split"],
            "status": s["status"],
            "vocabulary_ids": [
                int(cid) for cid in s["vocabulary_ids"]
            ],
        })

# ---------------------------------------
# 6. Score every scenario at every
#    image count
# ---------------------------------------

records = []

for k in IMAGE_COUNTS:
    for s in scenarios:

        image_features = V[
            image_subset(s["cluster_id"], s["repeat_id"], k)
        ]

        vocab = s["vocabulary_ids"]

        text_features = T[
            [text_lookup[cid] for cid in vocab]
        ]

        S = image_features @ text_features.T
        assert S.shape == (k, len(vocab))

        result = compute_scores(S)

        predicted_id = vocab[result["prediction_col"]]

        records.append({
            "condition": s["condition"],
            "n_images": k,
            "cluster_id": s["cluster_id"],
            "true_class_id": s["true_class_id"],
            "repeat_id": s["repeat_id"],
            "split": s["split"],
            "status": s["status"],
            "is_mapped": int(s["status"] == "MAPPED"),
            "correct_label": int(
                predicted_id == s["true_class_id"]
            ),
            **result,
        })

df = pd.DataFrame(records)

assert len(df) == len(scenarios) * len(IMAGE_COUNTS)

df.to_csv(
    OUT / "scenario_scores.csv",
    index=False
)

# ---------------------------------------
# 7. Verify k=10 against existing results
# ---------------------------------------

score_cols = ["prediction_col", "correct_label"] + METHODS

random_full = df[
    (df["condition"] == "random") &
    (df["n_images"] == FULL)
].reset_index(drop=True)

assert np.allclose(
    random_full[score_cols].to_numpy(float),
    reliability[score_cols].to_numpy(float)
), "k=10 random scores do not match experiments/reliability"

hard_full = df[
    (df["condition"] == "hard") &
    (df["n_images"] == FULL)
]

# Every repeat must equal the single
# hard-baseline scenario for its pair.
hard_check = hard_full.merge(
    hard_baseline,
    on=["cluster_id", "status"],
    suffixes=("", "_baseline")
)

assert len(hard_check) == len(hard_full)

for col in score_cols:
    assert np.allclose(
        hard_check[col], hard_check[f"{col}_baseline"]
    ), f"k=10 hard {col} does not match experiments/baseline_hard"

print("k=10 scores match existing reliability and hard-baseline results.")

# ---------------------------------------
# 8. Paired category-level bootstrap
# ---------------------------------------

test = df[df["split"] == "test"]
val = df[df["split"] == "validation"]

category_ids = np.sort(test["true_class_id"].unique())
assert len(category_ids) == 20

# One set of resampled categories, shared by
# every condition, image count and method, so
# differences between image counts are paired.
bootstrap_samples = rng.choice(
    category_ids,
    size=(N_BOOTSTRAPS, len(category_ids)),
    replace=True
)

summary = []
accuracy_summary = []

for condition in ["random", "hard"]:

    bootstrap_aucs = {}

    for k in IMAGE_COUNTS:

        cell = test[
            (test["condition"] == condition) &
            (test["n_images"] == k)
        ].reset_index(drop=True)

        cell_val = val[
            (val["condition"] == condition) &
            (val["n_images"] == k)
        ]

        assert len(cell) == 1600
        assert cell["is_mapped"].sum() == 800
        assert set(cell["true_class_id"]) == set(category_ids)

        rows_by_category = {
            cid: np.flatnonzero(cell["true_class_id"] == cid)
            for cid in category_ids
        }

        bootstrap_rows = [
            np.concatenate(
                [rows_by_category[cid] for cid in sample]
            )
            for sample in bootstrap_samples
        ]

        y = cell["is_mapped"].to_numpy()

        # Forced top-1 grounding on MAPPED cases
        mapped = cell["is_mapped"].to_numpy() == 1
        correct = cell["correct_label"].to_numpy()

        boot_acc = [
            correct[rows][mapped[rows]].mean()
            for rows in bootstrap_rows
        ]

        acc_lower, acc_upper = np.quantile(
            boot_acc, [0.025, 0.975]
        )

        accuracy_summary.append({
            "condition": condition,
            "n_images": k,
            "top1_accuracy": correct[mapped].mean(),
            "ci_lower": acc_lower,
            "ci_upper": acc_upper,
        })

        for method in METHODS:

            scores = cell[method].to_numpy()

            observed_auc = roc_auc_score(y, scores)

            bootstrap_aucs[k, method] = np.array([
                roc_auc_score(y[rows], scores[rows])
                for rows in bootstrap_rows
            ])

            lower, upper = np.quantile(
                bootstrap_aucs[k, method],
                [0.025, 0.975]
            )

            # Accept approximately 95% of mapped
            # validation scenarios.
            mapped_val = cell_val[cell_val["is_mapped"] == 1]

            threshold = np.quantile(
                mapped_val[method],
                0.05
            )

            accepted = scores >= threshold

            summary.append({
                "condition": condition,
                "n_images": k,
                "method": method,
                "auroc": observed_auc,
                "ci_lower": lower,
                "ci_upper": upper,
                "validation_threshold": threshold,
                "test_mapped_acceptance": accepted[mapped].mean(),
                "test_false_acceptance": accepted[~mapped].mean(),
            })

    # AUROC change relative to the full cluster
    for row in summary:
        if row["condition"] != condition:
            continue

        delta = (
            bootstrap_aucs[row["n_images"], row["method"]] -
            bootstrap_aucs[FULL, row["method"]]
        )

        row["delta_vs_full"] = row["auroc"] - next(
            r["auroc"] for r in summary
            if r["condition"] == condition
            and r["n_images"] == FULL
            and r["method"] == row["method"]
        )

        row["delta_ci_lower"], row["delta_ci_upper"] = (
            np.quantile(delta, [0.025, 0.975])
        )

summary_df = pd.DataFrame(summary)
accuracy_df = pd.DataFrame(accuracy_summary)

summary_df.to_csv(OUT / "summary.csv", index=False)
accuracy_df.to_csv(OUT / "grounding_accuracy.csv", index=False)

# ---------------------------------------
# 9. Report
# ---------------------------------------

for condition in ["random", "hard"]:

    print(f"\n{condition.upper()} VOCABULARY: TEST AUROC [95% CI]")
    print("-" * 46)

    subset = summary_df[summary_df["condition"] == condition]

    for method in METHODS:
        cells = []

        for k in IMAGE_COUNTS:
            r = subset[
                (subset["method"] == method) &
                (subset["n_images"] == k)
            ].iloc[0]

            cells.append(
                f"k={k:<2d} {r['auroc']:.3f} "
                f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]"
            )

        print(f"{method:15s} " + "  ".join(cells))

    print("\nForced top-1 grounding accuracy:")

    for r in accuracy_df[
        accuracy_df["condition"] == condition
    ].itertuples():
        print(
            f"  k={r.n_images:<2d} {r.top1_accuracy:.4f} "
            f"[{r.ci_lower:.4f}, {r.ci_upper:.4f}]"
        )

print(
    "\nNote: voting is constant (1.0) with one image, "
    "so its k=1 AUROC is 0.5 by construction."
)

# ---------------------------------------
# 10. Plot AUROC and accuracy vs. image count
# ---------------------------------------

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

STYLE = {
    "max_similarity": ("#2a78d6", "o"),
    "margin": ("#eb6834", "s"),
    "msp": ("#1baf7a", "D"),
    "voting": ("#eda100", "^"),
}

# Small horizontal offsets so error bars
# of different methods do not overlap.
OFFSETS = dict(zip(METHODS, [-0.24, -0.08, 0.08, 0.24]))

plt.rcParams.update({
    "font.size": 9,
    "text.color": INK,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": BASELINE,
})

fig, axes = plt.subplots(
    1, 3, figsize=(13, 4.2),
    gridspec_kw={"width_ratios": [1, 1, 0.8]}
)

fig.patch.set_facecolor(SURFACE)

def style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.set_xticks(IMAGE_COUNTS)
    ax.set_xlim(0.2, FULL + 0.8)
    ax.set_xlabel("Images per cluster")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

for ax, condition in zip(axes[:2], ["random", "hard"]):

    subset = summary_df[summary_df["condition"] == condition]

    # Chance level
    ax.axhline(0.5, color=BASELINE, linewidth=0.8)

    for method in METHODS:
        color, marker = STYLE[method]

        rows = subset[subset["method"] == method]
        x = rows["n_images"] + OFFSETS[method]

        ax.errorbar(
            x, rows["auroc"],
            yerr=[
                rows["auroc"] - rows["ci_lower"],
                rows["ci_upper"] - rows["auroc"]
            ],
            color=color,
            marker=marker,
            markersize=5,
            markeredgecolor=SURFACE,
            markeredgewidth=0.8,
            linewidth=1.5,
            elinewidth=1,
            capsize=0,
            label=method,
        )

    style_axis(ax)
    ax.set_ylim(0.45, 1.01)
    ax.set_yticks(np.arange(0.5, 1.01, 0.1))
    ax.set_ylabel("Test AUROC (MAPPED vs UNMAPPED)")
    ax.set_title(
        f"{condition.capitalize()} vocabulary",
        loc="left", fontsize=10
    )

# Accuracy uses ink, not method colors: it
# does not depend on the rejection score.
ax = axes[2]

for condition, color, marker, offset in [
    ("random", INK, "o", -0.1),
    ("hard", MUTED, "s", 0.1),
]:
    rows = accuracy_df[accuracy_df["condition"] == condition]
    x = rows["n_images"] + offset

    ax.errorbar(
        x, rows["top1_accuracy"],
        yerr=[
            rows["top1_accuracy"] - rows["ci_lower"],
            rows["ci_upper"] - rows["top1_accuracy"]
        ],
        color=color,
        marker=marker,
        markersize=5,
        markeredgecolor=SURFACE,
        markeredgewidth=0.8,
        linewidth=1.5,
        elinewidth=1,
        capsize=0,
        label=f"{condition} vocabulary",
    )

style_axis(ax)
ax.legend(loc="lower right", frameon=False)
ax.set_ylabel("Top-1 accuracy (MAPPED, test)")
ax.set_title("Forced top-1 grounding", loc="left", fontsize=10)

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles, labels,
    loc="upper center",
    ncol=len(METHODS),
    frameon=False,
    bbox_to_anchor=(0.36, 1.02),
)

fig.tight_layout(rect=(0, 0, 1, 0.93))

plt.savefig(
    OUT / "image_count_comparison.png",
    dpi=200,
    facecolor=SURFACE
)

plt.close()

# ---------------------------------------
# 11. Save experimental configuration
# ---------------------------------------

config = {
    "image_counts": IMAGE_COUNTS,
    "subsets_per_cluster": N_SUBSETS,
    "subset_scheme": "nested_prefix_of_random_ordering",
    "bootstrap_iterations": N_BOOTSTRAPS,
    "bootstrap_unit": "semantic_category",
    "bootstrap_pairing": "shared_across_conditions_counts_methods",
    "random_seed": SEED,
    "temperature": TEMPERATURE,
    "random_vocabularies": "experiments/reliability/scenario_scores.csv",
    "hard_vocabularies": "experiments/hard/vocabulary_scenarios.json",
    "model": "CLIP ViT-B/32",
}

with open(OUT / "config.json", "w") as f:
    json.dump(config, f, indent=2)

print("\nImage-count comparison complete!")
print("Results saved to:", OUT)
