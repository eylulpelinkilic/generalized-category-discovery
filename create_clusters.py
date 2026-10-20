
import json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

# ----------------------------------------
# 1. Load saved data
# ----------------------------------------

EMBED_DIR = Path("embeddings")
OUTPUT_DIR = Path("clusters")
OUTPUT_DIR.mkdir(exist_ok=True)

V = np.load(EMBED_DIR / "image_embeddings.npy")
labels = np.load(EMBED_DIR / "image_labels.npy")

with open(EMBED_DIR / "image_paths.json") as f:
    image_paths = json.load(f)

with open("class_splits.json") as f:
    class_splits = json.load(f)

assert len(V) == len(labels) == len(image_paths) == 2000
assert V.shape[1] == 512

# Map each category to its experimental split
split_lookup = {
    c["imagenet_id"]: split
    for split, classes in class_splits.items()
    for c in classes
}

# ----------------------------------------
# 2. Group images by ground-truth category
# ----------------------------------------

indices_by_class = defaultdict(list)

for i, label in enumerate(labels):
    indices_by_class[int(label)].append(i)

# ----------------------------------------
# 3. Construct clusters
# ----------------------------------------

CLUSTER_SIZE = 10
rng = np.random.default_rng(42)

clusters = []
cluster_embeddings = []

for class_id in sorted(indices_by_class):

    indices = np.array(indices_by_class[class_id])
    assert len(indices) == 20

    # Shuffle reproducibly
    rng.shuffle(indices)

    for cluster_num, group in enumerate(
        np.split(indices, 2)
    ):
        assert len(group) == CLUSTER_SIZE

        clusters.append({
            "cluster_id": len(clusters),
            "class_id": class_id,
            "split": split_lookup[class_id],
            "image_indices": group.tolist(),
            "image_paths": [
                image_paths[i] for i in group
            ]
        })

        cluster_embeddings.append(V[group])

# ----------------------------------------
# 4. Save clusters
# ----------------------------------------

with open(OUTPUT_DIR / "clusters.json", "w") as f:
    json.dump(clusters, f, indent=2)

cluster_embeddings = np.stack(cluster_embeddings)

np.save(
    OUTPUT_DIR / "cluster_embeddings.npy",
    cluster_embeddings
)

# ----------------------------------------
# 5. Verify
# ----------------------------------------

print("Total clusters:", len(clusters))
print("Cluster embeddings shape:", cluster_embeddings.shape)

split_counts = Counter(c["split"] for c in clusters)

print("\nClusters per split:")
for split, count in split_counts.items():
    print(f"{split}: {count}")

all_indices = [
    idx
    for c in clusters
    for idx in c["image_indices"]
]

assert len(clusters) == 200
assert cluster_embeddings.shape == (200, 10, 512)
assert len(all_indices) == 2000
assert len(set(all_indices)) == 2000

for c in clusters:
    assert all(
        int(labels[i]) == c["class_id"]
        for i in c["image_indices"]
    )

print("\nCluster verification PASSED!")
