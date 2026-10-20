
import json
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset

# Load the 100 GCD classes
with open("imagenet100_classes.json") as f:
    classes = json.load(f)

selected_ids = {c["imagenet_id"] for c in classes}
class_names = {
    c["imagenet_id"]: c["name"] for c in classes
}

# Save images into this directory
output_dir = Path("imagenet100_images")
output_dir.mkdir(exist_ok=True)

# Number of images per class
MAX_PER_CLASS = 20
counts = defaultdict(int)

dataset = load_dataset(
    "ILSVRC/imagenet-1k",
    split="validation",
    streaming=True
)

print("Starting ImageNet-100 extraction...", flush=True)

for example in dataset:
    label = example["label"]

    if label not in selected_ids:
        continue

    if counts[label] >= MAX_PER_CLASS:
        continue

    class_dir = output_dir / str(label)
    class_dir.mkdir(exist_ok=True)

    image = example["image"].convert("RGB")
    image.save(
        class_dir / f"{counts[label]:03d}.jpg",
        quality=95
    )

    counts[label] += 1

    if sum(counts.values()) % 100 == 0:
        print(f"Saved {sum(counts.values())} images", flush=True)

    if all(counts[c] >= MAX_PER_CLASS for c in selected_ids):
        break

print("\nExtraction complete!")
print("Total images:", sum(counts.values()))
print("Classes:", len([c for c in counts if counts[c] > 0]))
