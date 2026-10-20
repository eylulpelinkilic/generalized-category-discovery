
import json
from pathlib import Path

import numpy as np

# ----------------------------------------
# 1. Load existing data
# ----------------------------------------

with open("clusters/clusters.json") as f:
    clusters = json.load(f)

with open("class_splits.json") as f:
    splits = json.load(f)

with open("imagenet100_classes.json") as f:
    classes = json.load(f)

V = np.load("embeddings/image_embeddings.npy")
T = np.load("embeddings/text_embeddings.npy")
text_ids = np.load("embeddings/text_class_ids.npy")

class_names = {
    c["imagenet_id"]: c["name"]
    for c in classes
}

text_lookup = {
    int(class_id): i
    for i, class_id in enumerate(text_ids)
}

split_ids = {
    split: [c["imagenet_id"] for c in subset]
    for split, subset in splits.items()
}

cluster_lookup = {
    c["cluster_id"]: c for c in clusters
}

# ----------------------------------------
# 2. Text-text similarity
# ----------------------------------------

# Both text embeddings are normalized.
R = T @ T.T

VOCAB_SIZE = 10
rng = np.random.default_rng(42)

scenarios = []
matrices = []

# ----------------------------------------
# 3. Generate hard vocabularies
# ----------------------------------------

for cluster in clusters:

    true_id = int(cluster["class_id"])
    split = cluster["split"]

    true_text_idx = text_lookup[true_id]

    # Candidate concepts from same split,
    # excluding the true class.
    candidates = [
        cid for cid in split_ids[split]
        if cid != true_id
    ]

    # Rank other concepts by similarity
    # to the true concept.
    ranked = sorted(
        candidates,
        key=lambda cid: float(
            R[true_text_idx, text_lookup[cid]]
        ),
        reverse=True
    )

    # 9 hardest shared distractors
    shared = ranked[:VOCAB_SIZE - 1]

    # 10th hardest concept replaces true label
    replacement = ranked[VOCAB_SIZE - 1]

    mapped_vocab = shared + [true_id]
    unmapped_vocab = shared + [replacement]

    # Randomize column order
    rng.shuffle(mapped_vocab)
    rng.shuffle(unmapped_vocab)

    image_indices = cluster["image_indices"]
    image_features = V[image_indices]

    for status, vocab in [
        ("MAPPED", mapped_vocab),
        ("UNMAPPED", unmapped_vocab)
    ]:

        text_indices = [
            text_lookup[int(cid)]
            for cid in vocab
        ]

        text_features = T[text_indices]

        # Image-text similarity matrix
        S = image_features @ text_features.T

        assert S.shape == (10, 10)

        scenarios.append({
            "cluster_id": cluster["cluster_id"],
            "split": split,
            "status": status,
            "true_class_id": true_id,
            "true_class_name": class_names[true_id],
            "vocabulary_ids": [int(x) for x in vocab],
            "vocabulary_names": [
                class_names[int(cid)] for cid in vocab
            ]
        })

        matrices.append(S)

# ----------------------------------------
# 4. Save
# ----------------------------------------

out = Path("experiments/hard")
out.mkdir(parents=True, exist_ok=True)

matrices = np.stack(matrices).astype(np.float32)

with open(out / "vocabulary_scenarios.json", "w") as f:
    json.dump(scenarios, f, indent=2)

np.save(out / "similarity_matrices.npy", matrices)

# ----------------------------------------
# 5. Verify
# ----------------------------------------

assert len(scenarios) == 400
assert matrices.shape == (400, 10, 10)
assert np.isfinite(matrices).all()

for i in range(0, 400, 2):
    a, b = scenarios[i], scenarios[i + 1]

    assert a["cluster_id"] == b["cluster_id"]
    assert a["status"] == "MAPPED"
    assert b["status"] == "UNMAPPED"

    assert a["true_class_id"] in a["vocabulary_ids"]
    assert b["true_class_id"] not in b["vocabulary_ids"]

    assert len(set(a["vocabulary_ids"])) == 10
    assert len(set(b["vocabulary_ids"])) == 10

    # Nine distractors should be shared.
    assert len(
        set(a["vocabulary_ids"]) &
        set(b["vocabulary_ids"])
    ) == 9

print("Hard vocabulary generation complete!")
print("Total scenarios:", len(scenarios))
print("Mapped:", len(scenarios) // 2)
print("Unmapped:", len(scenarios) // 2)
print("Matrix shape:", matrices.shape)

# Display a few examples
for s in scenarios[:4]:
    print("\n", s["status"])
    print("True class:", s["true_class_name"])
    print("Vocabulary:")
    for name in s["vocabulary_names"]:
        print(" -", name.split(",")[0])
