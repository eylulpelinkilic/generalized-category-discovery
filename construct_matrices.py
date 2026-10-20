
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------
# 1. Load existing embeddings and metadata
# ----------------------------------------

V = np.load("embeddings/image_embeddings.npy")
T = np.load("embeddings/text_embeddings.npy")

text_ids = np.load("embeddings/text_class_ids.npy")

clusters = json.loads(
    Path("clusters/clusters.json").read_text()
)

scenarios = json.loads(
    Path("experiments/vocabulary_scenarios.json").read_text()
)

assert V.shape == (2000, 512)
assert T.shape == (100, 512)
assert len(scenarios) == 400

# Map original ImageNet class IDs to text rows
text_lookup = {
    int(class_id): i
    for i, class_id in enumerate(text_ids)
}

cluster_lookup = {
    c["cluster_id"]: c for c in clusters
}

# ----------------------------------------
# 2. Construct the similarity matrices
# ----------------------------------------

matrices = []

for scenario in scenarios:
    cluster = cluster_lookup[scenario["cluster_id"]]

    image_indices = cluster["image_indices"]
    vocab_ids = scenario["vocabulary_ids"]

    image_features = V[image_indices]

    text_indices = [
        text_lookup[int(cid)]
        for cid in vocab_ids
    ]

    text_features = T[text_indices]

    # Cosine similarity: embeddings are normalized
    S = image_features @ text_features.T

    assert S.shape == (10, 10)

    matrices.append(S)

matrices = np.stack(matrices).astype(np.float32)

# ----------------------------------------
# 3. Save all matrices
# ----------------------------------------

output_dir = Path("experiments")
output_dir.mkdir(exist_ok=True)

np.save(
    output_dir / "similarity_matrices.npy",
    matrices
)

print("Saved matrix shape:", matrices.shape)

# ----------------------------------------
# 4. Visualize one MAPPED/UNMAPPED pair
# ----------------------------------------

# First two scenarios belong to the same cluster
mapped = scenarios[0]
unmapped = scenarios[1]

assert mapped["cluster_id"] == unmapped["cluster_id"]
assert mapped["status"] == "MAPPED"
assert unmapped["status"] == "UNMAPPED"

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

for ax, index, scenario in [
    (axes[0], 0, mapped),
    (axes[1], 1, unmapped)
]:
    S = matrices[index]

    im = ax.imshow(
        S, cmap="Blues", vmin=-1, vmax=1,
        aspect="auto"
    )

    labels = [
        name.split(",")[0]
        for name in scenario["vocabulary_names"]
    ]

    ax.set_xticks(range(10))
    ax.set_xticklabels(
        labels,
        rotation=55,
        ha="right",
        fontsize=8
    )

    ax.set_yticks(range(10))
    ax.set_yticklabels(
        [f"Image {i+1}" for i in range(10)]
    )

    ax.set_xlabel("Candidate concepts")
    ax.set_ylabel("Cluster images")
    ax.set_title(
        f"{scenario['status']}\n"
        f"True class: {scenario['true_class_name'].split(',')[0]}"
    )

    for i in range(10):
        for j in range(10):
            ax.text(
                j, i, f"{S[i, j]:.2f}",
                ha="center", va="center",
                fontsize=7
            )

fig.colorbar(
    im, ax=axes.ravel().tolist(),
    label="Cosine similarity",
    fraction=0.025,
    pad=0.02
)

fig.subplots_adjust(
    bottom=0.30,
    top=0.84,
    left=0.07,
    right=0.88,
    wspace=0.40
)

plot_path = output_dir / "example_matrices.png"

plt.savefig(
    plot_path, dpi=200,
    bbox_inches="tight"
)

plt.show()

print("Saved visualization:", plot_path)

# ----------------------------------------
# 5. Final verification
# ----------------------------------------

assert matrices.shape == (400, 10, 10)
assert np.isfinite(matrices).all()
assert matrices.min() >= -1.0001
assert matrices.max() <= 1.0001

print("\nMatrix construction PASSED!")
