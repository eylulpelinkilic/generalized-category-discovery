
import json
import numpy as np
from pathlib import Path

# ----------------------------------------
# 1. Load clusters and classes
# ----------------------------------------

with open("clusters/clusters.json") as f:
    clusters = json.load(f)

with open("class_splits.json") as f:
    splits = json.load(f)

with open("imagenet100_classes.json") as f:
    classes = json.load(f)

# Class names for metadata
class_names = {
    c["imagenet_id"]: c["name"]
    for c in classes
}

# Candidate vocabularies remain inside
# each experimental split
split_ids = {
    split: [c["imagenet_id"] for c in subset]
    for split, subset in splits.items()
}

# ----------------------------------------
# 2. Configuration
# ----------------------------------------

VOCAB_SIZE = 10
rng = np.random.default_rng(42)

scenarios = []

# ----------------------------------------
# 3. Generate paired vocabularies
# ----------------------------------------

for cluster in clusters:
    true_id = cluster["class_id"]
    split = cluster["split"]

    candidates = [
        cid for cid in split_ids[split]
        if cid != true_id
    ]

    # Select 9 shared distractors
    shared = rng.choice(
        candidates,
        size=VOCAB_SIZE - 1,
        replace=False
    ).tolist()

    # Select one additional distractor
    remaining = [
        cid for cid in candidates
        if cid not in shared
    ]

    replacement = int(rng.choice(remaining))

    # Correct label available
    mapped_vocab = shared + [true_id]

    # Correct label withheld
    unmapped_vocab = shared + [replacement]

    # Shuffle the column ordering
    rng.shuffle(mapped_vocab)
    rng.shuffle(unmapped_vocab)

    for status, vocab in [
        ("MAPPED", mapped_vocab),
        ("UNMAPPED", unmapped_vocab)
    ]:
        scenarios.append({
            "cluster_id": cluster["cluster_id"],
            "split": split,
            "status": status,
            "true_class_id": true_id,
            "true_class_name": class_names[true_id],
            "vocabulary_ids": vocab,
            "vocabulary_names": [
                class_names[cid] for cid in vocab
            ]
        })

# ----------------------------------------
# 4. Save scenarios
# ----------------------------------------

output_dir = Path("experiments")
output_dir.mkdir(exist_ok=True)

with open(
    output_dir / "vocabulary_scenarios.json", "w"
) as f:
    json.dump(scenarios, f, indent=2)

# ----------------------------------------
# 5. Verify
# ----------------------------------------

assert len(scenarios) == 400

for s in scenarios:
    has_true = (
        s["true_class_id"] in s["vocabulary_ids"]
    )

    assert len(s["vocabulary_ids"]) == VOCAB_SIZE
    assert len(set(s["vocabulary_ids"])) == VOCAB_SIZE

    if s["status"] == "MAPPED":
        assert has_true
    else:
        assert not has_true

print("Vocabulary generation complete!")
print("Total scenarios:", len(scenarios))
print("Mapped:", sum(
    s["status"] == "MAPPED" for s in scenarios
))
print("Unmapped:", sum(
    s["status"] == "UNMAPPED" for s in scenarios
))
print("Vocabulary size:", VOCAB_SIZE)
