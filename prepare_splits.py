
import json
import hashlib
import numpy as np
from pathlib import Path
from collections import Counter

# ----------------------------------------
# 1. Load the 100 GCD categories
# ----------------------------------------

with open("imagenet100_classes.json") as f:
    classes = json.load(f)

assert len(classes) == 100
assert len({c["imagenet_id"] for c in classes}) == 100

# ----------------------------------------
# 2. Create reproducible class splits
# ----------------------------------------

rng = np.random.default_rng(42)

# Shuffle the classes reproducibly
shuffled = rng.permutation(classes).tolist()

splits = {
    "development": shuffled[:60],
    "validation": shuffled[60:80],
    "test": shuffled[80:100],
}

# Check that all splits are disjoint
ids = {
    name: {c["imagenet_id"] for c in subset}
    for name, subset in splits.items()
}

assert ids["development"].isdisjoint(ids["validation"])
assert ids["development"].isdisjoint(ids["test"])
assert ids["validation"].isdisjoint(ids["test"])

# Save splits
with open("class_splits.json", "w") as f:
    json.dump(splits, f, indent=2)

print("Class splits created:")
for name, subset in splits.items():
    print(f"  {name}: {len(subset)} classes")

# ----------------------------------------
# 3. Verify downloaded images
# ----------------------------------------

image_root = Path("imagenet100_images")

all_hashes = {}
duplicate_count = 0
total_images = 0
problems = []

for c in classes:
    class_id = c["imagenet_id"]
    class_dir = image_root / str(class_id)

    images = sorted(class_dir.glob("*.jpg"))
    total_images += len(images)

    if len(images) != 20:
        problems.append(
            f"Class {class_id}: expected 20, found {len(images)}"
        )

    for path in images:
        # SHA256 identifies exactly identical image files
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()

        if file_hash in all_hashes:
            duplicate_count += 1
            print(
                "Duplicate:",
                path,
                "matches",
                all_hashes[file_hash]
            )
        else:
            all_hashes[file_hash] = str(path)

print("\nDataset verification:")
print("Total images:", total_images)
print("Exact duplicates:", duplicate_count)
print("Classes with incorrect image counts:", len(problems))

for problem in problems:
    print(problem)

if (
    total_images == 2000
    and duplicate_count == 0
    and not problems
):
    print("\nDataset verification PASSED!")
else:
    print("\nDataset needs inspection.")
