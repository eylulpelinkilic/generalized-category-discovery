
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# ---------------------------------------
# 1. Configuration
# ---------------------------------------

N_PAIRS = 20
VOCAB_SIZE = 10
N_BOOTSTRAPS = 1000
SEED = 42
TEMPERATURE = 0.01

rng = np.random.default_rng(SEED)

OUT = Path("experiments/reliability")
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------
# 2. Load existing data
# ---------------------------------------

V = np.load("embeddings/image_embeddings.npy")
T = np.load("embeddings/text_embeddings.npy")
text_ids = np.load("embeddings/text_class_ids.npy")

with open("clusters/clusters.json") as f:
    clusters = json.load(f)

with open("class_splits.json") as f:
    splits = json.load(f)

assert V.shape == (2000, 512)
assert T.shape == (100, 512)
assert len(clusters) == 200

text_lookup = {
    int(cid): i for i, cid in enumerate(text_ids)
}

split_ids = {
    split: [int(c["imagenet_id"]) for c in subset]
    for split, subset in splits.items()
}

# ---------------------------------------
# 3. Compute rejection baselines
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
        "voting": float(np.mean(votes == prediction)),
    }

# ---------------------------------------
# 4. Generate 20 vocabulary pairs
#    for each cluster
# ---------------------------------------

records = []

for cluster in clusters:

    true_id = int(cluster["class_id"])
    cluster_id = int(cluster["cluster_id"])
    split = cluster["split"]

    image_features = V[cluster["image_indices"]]

    candidates = [
        cid for cid in split_ids[split]
        if cid != true_id
    ]

    for pair_id in range(N_PAIRS):

        selected = rng.choice(
            candidates,
            size=VOCAB_SIZE,
            replace=False
        ).tolist()

        shared = selected[:VOCAB_SIZE - 1]
        replacement = selected[-1]

        for status, vocab in [
            ("MAPPED", shared + [true_id]),
            ("UNMAPPED", shared + [replacement])
        ]:

            vocab = list(vocab)
            rng.shuffle(vocab)

            text_features = T[
                [text_lookup[cid] for cid in vocab]
            ]

            S = image_features @ text_features.T
            result = compute_scores(S)

            predicted_id = vocab[result["prediction_col"]]

            records.append({
                "cluster_id": cluster_id,
                "true_class_id": true_id,
                "pair_id": pair_id,
                "split": split,
                "status": status,
                "is_mapped": int(status == "MAPPED"),
                "correct_label": int(
                    predicted_id == true_id
                ),
                "vocabulary_ids": json.dumps(vocab),
                **result,
            })

df = pd.DataFrame(records)

assert len(df) == 8000

df.to_csv(
    OUT / "scenario_scores.csv",
    index=False
)

print("Total scenarios:", len(df))

print("\nScenarios per split:")
print(df.groupby("split").size())

# ---------------------------------------
# 5. Category-level bootstrap
# ---------------------------------------

methods = [
    "max_similarity",
    "margin",
    "msp",
    "voting",
]

test = df[df["split"] == "test"].copy()

category_ids = test["true_class_id"].unique()

assert len(category_ids) == 20
assert len(test) == 1600

# Pre-group observations by category
category_groups = {
    cid: test[test["true_class_id"] == cid]
    for cid in category_ids
}

summary = []

print("\nRELIABILITY RESULTS")
print("-------------------")

for method in methods:

    observed_auc = roc_auc_score(
        test["is_mapped"],
        test[method]
    )

    bootstrap_aucs = []

    for _ in range(N_BOOTSTRAPS):

        # Sample semantic categories, not
        # individual scenarios.
        sampled_ids = rng.choice(
            category_ids,
            size=len(category_ids),
            replace=True
        )

        # Repeated categories are retained
        # as repeated bootstrap observations.
        bootstrap_df = pd.concat(
            [category_groups[cid] for cid in sampled_ids],
            ignore_index=True
        )

        auc = roc_auc_score(
            bootstrap_df["is_mapped"],
            bootstrap_df[method]
        )

        bootstrap_aucs.append(auc)

    lower, upper = np.quantile(
        bootstrap_aucs,
        [0.025, 0.975]
    )

    summary.append({
        "method": method,
        "auroc": observed_auc,
        "ci_lower": lower,
        "ci_upper": upper,
    })

    print(
        f"{method:18s} "
        f"AUROC={observed_auc:.4f} "
        f"95% CI=[{lower:.4f}, {upper:.4f}]"
    )

pd.DataFrame(summary).to_csv(
    OUT / "summary.csv",
    index=False
)

# ---------------------------------------
# 6. Save experimental configuration
# ---------------------------------------

config = {
    "number_of_pairs_per_cluster": N_PAIRS,
    "vocabulary_size": VOCAB_SIZE,
    "bootstrap_iterations": N_BOOTSTRAPS,
    "random_seed": SEED,
    "temperature": TEMPERATURE,
    "bootstrap_unit": "semantic_category",
    "candidate_selection": "random_within_split",
    "model": "CLIP ViT-B/32",
}

with open(OUT / "config.json", "w") as f:
    json.dump(config, f, indent=2)

print("\nReliability evaluation complete!")
print("Results saved to:", OUT)
